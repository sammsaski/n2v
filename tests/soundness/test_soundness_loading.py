"""Loading and converting a model: the soundness and unit cells of
``n2v/utils/model_loader.py`` (``load_onnx`` with its opset retry and its fold
of shape subgraphs, ``load_pytorch``, ``get_model_summary``), of
``n2v/utils/quant_converters.py`` (the QuantizeLinear and DequantizeLinear
converters and the fold of constant Q/DQ chains), and of the vendored
onnx2torch converter that ``load_onnx`` calls.

The promise these cells check: a model that ``load_onnx`` returns computes
what onnxruntime computes on the file, for every input the file accepts, or
the load refuses (``NotImplementedError`` or ``ValueError``, or onnx's and
protobuf's own errors for a file that cannot be read); ``load_pytorch``
returns the model it was given, or the one the file holds, with the file's
weights and in eval mode, and refuses what it cannot load;
``get_model_summary`` reports each module's input and output shape,
parameter count and trainable flag. A converted model that differs from the
file is analysed as another network: every reach, falsification and witness
replay through it is about the wrong function. Where the load decides what a
reach sees (the fold of a shape subgraph, a float64 file, a Q/DQ model whose
quantization is folded or left to the runtime modules), a reach of the
loaded model through ``NeuralNetwork`` must hold 128 onnxruntime outputs by
the returned set's rows, or refuse with ``NotImplementedError``.

Oracles, never n2v: onnxruntime's CPU kernels on the file, with graph
optimizations off (with them on, onnxruntime fuses a weight DequantizeLinear
into an approximate quantized MatMul: 13.622047 where the operator's
definition gives 13.625); the definitions of QuantizeLinear and
DequantizeLinear written with integers in numpy (``_quantize``,
``_dequantize``); and values computed by hand. Inputs and parameters are
dyadic (multiples of small powers of two), so float32, float64 and the exact
values coincide and a comparison needs no slack; where a row computes a
transcendental function the slack is a stated float32 rounding budget.

Where the vendored converter gets an operator wrong, the settled promise is
that n2v either answers what onnxruntime answers or refuses at load time, so
those cells accept both and fail on a wrong answer or a crash
(``_assert_agrees_or_refuses``).
"""

import copy
import gzip
import os
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from google.protobuf.message import DecodeError
from onnx import TensorProto, helper, numpy_helper

from n2v.nn import NeuralNetwork
from n2v.sets import Box, Star, Zono
from n2v.utils import load_onnx, load_pytorch
from n2v.utils.model_loader import get_model_summary
from tests.cell_support import assert_covered, flat_box, samples
from tests.known_bugs import DID_NOT_RAISE, known_bug

# A load refuses with one of these; anything else raised at load is a crash.
_REFUSALS = (NotImplementedError, ValueError)

# Warnings the vendored converter and torch emit on legal files; the cells
# decide those files by comparing outputs, so the warnings are ignored by
# message on the rows that raise them.
_NEAREST_WARNING = pytest.mark.filterwarnings(
    "ignore:Pytorch's nearest neighbor interpolat:UserWarning")
_TRUNCATION_WARNING = pytest.mark.filterwarnings(
    "ignore:Truncating the start/stop/step of slice:UserWarning")


class SetupError(Exception):
    """A case that does not build what it states (onnxruntime answers a file
    the case says it refuses, or the reverse). No marker absorbs it."""


# --------------------------------------------------------------------------- #
# Files, references and the comparison                                         #
# --------------------------------------------------------------------------- #
def _value_info(spec):
    name, shape, *elem = spec
    return helper.make_tensor_value_info(
        name, elem[0] if elem else TensorProto.FLOAT, shape)


def _save(tmp_path, nodes, inputs, outputs, inits=None, opset=13,
          ir_version=10, name="m", infer=True, **save_kwargs):
    """Write a model and return its path as a ``str``. ``inputs`` and
    ``outputs`` are ``(name, shape)`` or ``(name, shape, element type)``;
    an output shape of ``None`` is filled by onnx's shape inference
    (``infer``). ``inits`` maps names to arrays (stored with their dtype) or
    to ``TensorProto`` objects. The model passes ``onnx.checker``."""
    tensors = [value if isinstance(value, onnx.TensorProto)
               else numpy_helper.from_array(np.asarray(value), key)
               for key, value in (inits or {}).items()]
    graph = helper.make_graph(nodes, name, [_value_info(s) for s in inputs],
                              [_value_info(s) for s in outputs],
                              initializer=tensors)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", int(opset))])
    model.ir_version = int(ir_version)
    if infer:
        model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    path = os.path.join(str(tmp_path), f"{name}.onnx")
    onnx.save(model, path, **save_kwargs)
    return path


def _session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def _ort(path, feeds, session=None):
    """onnxruntime's outputs on one input (a list with one array per graph
    input, in the file's order), each a float64 array of its own shape."""
    session = session or _session(path)
    names = [i.name for i in session.get_inputs()]
    types = {"tensor(float)": np.float32, "tensor(double)": np.float64}
    feed = {n: np.asarray(f).astype(types[i.type])
            for n, f, i in zip(names, feeds, session.get_inputs())}
    return [np.asarray(y, dtype=np.float64) for y in session.run(None, feed)]


def _run(module, feeds, dtype=None):
    """The outputs of ``module`` on one input, each a float64 array. With
    ``dtype=None`` the module runs as it was returned (its own dtype and
    training flag); with ``torch.float64`` a deep copy runs in float64. An
    output that is not a tensor (a tuple where a tensor is due) fails."""
    if dtype is not None:
        module = copy.deepcopy(module).to(dtype)
    tensors = [torch.as_tensor(np.asarray(f, dtype=np.float64)) for f in feeds]
    if dtype is None:
        params = list(module.parameters()) + list(module.buffers())
        floats = [p.dtype for p in params if p.is_floating_point()]
        cast = floats[0] if floats else torch.float32
        tensors = [t.to(cast) for t in tensors]
    else:
        tensors = [t.to(dtype) for t in tensors]
    with torch.no_grad():
        out = module(*tensors)
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    for k, o in enumerate(outs):
        assert isinstance(o, torch.Tensor), (
            f"output {k} of the loaded module is a {type(o).__name__}, not a "
            f"tensor")
    return [o.detach().cpu().numpy().astype(np.float64) for o in outs]


def _assert_same(got, want, what, tol=0.0):
    """``got`` equals onnxruntime's ``want`` output by output: the same
    number of outputs, the same shapes, and values within ``tol`` times
    max(1, |want|) (0 for an exact reference)."""
    assert len(got) == len(want), (
        f"{what}: {len(got)} outputs where onnxruntime gives {len(want)}")
    for k, (g, w) in enumerate(zip(got, want)):
        assert g.shape == w.shape, (
            f"{what}: output {k} has shape {g.shape}, onnxruntime's "
            f"{w.shape}")
        bad = np.abs(g - w) > tol * np.maximum(1.0, np.abs(w))
        assert not bad.any(), (
            f"{what}: output {k} differs from onnxruntime in {int(bad.sum())} "
            f"of {bad.size} entries, first at {np.argwhere(bad)[0].tolist()}: "
            f"{g[bad][:4]} where onnxruntime gives {w[bad][:4]}")


def _points(shape, n=16, seed=0):
    """``n`` dyadic points of ``shape``: multiples of 1/8 in [-3, 3], so
    entries lie beyond plus and minus 1 and on both sides of zero."""
    rng = np.random.default_rng(seed)
    return [rng.integers(-24, 25, size=shape) / 8.0 for _ in range(n)]


def _assert_agrees_or_refuses(path, points, what, ort_refuses=False, tol=0.0,
                              dtype=torch.float64):
    """The load refuses, or the loaded module agrees with onnxruntime on
    every point (each a list of feeds). Where onnxruntime refuses the file
    (``ort_refuses``), the load or the module's run must refuse as well: a
    module that answers a file onnxruntime rejects answers for a network the
    file does not define."""
    try:
        session = _session(path)
        refused = None
    except Exception as exc:  # noqa: BLE001 - onnxruntime's refusal
        session, refused = None, exc
    if (refused is not None) != ort_refuses:
        raise SetupError(f"{what}: onnxruntime "
                         f"{'refuses' if refused else 'answers'} the file, "
                         f"the case states the opposite ({refused})")
    try:
        module = load_onnx(path)
    except _REFUSALS:
        return
    if ort_refuses:
        try:
            _run(module, points[0], dtype)
        except (NotImplementedError, ValueError, RuntimeError):
            return
        pytest.fail(f"{what}: the loaded module answers a file onnxruntime "
                    f"refuses ({refused})")
    for feeds in points:
        _assert_same(_run(module, feeds, dtype), _ort(path, feeds, session),
                     what, tol)


def _module_classes(module):
    return {type(m).__name__ for m in module.modules()}


# --------------------------------------------------------------------------- #
# load_onnx: eval mode, the opset retry, the shape fold                         #
# --------------------------------------------------------------------------- #
def test_a_loaded_model_runs_in_inference_mode(tmp_path):
    """``load_onnx`` returns the module in eval mode: a BatchNormalization
    normalizes by the stored mean and variance and a Dropout passes its input
    on. Run as returned on a batch of four, the module must give onnxruntime's
    outputs; in training mode the BatchNorm would use the batch's own
    statistics and the Dropout would zero entries (catches a dropped
    ``eval()``). Epsilon 0 and variances 1/4 and 4 keep every value exact."""
    inits = {"g": np.float32([0.5, 2.0]), "b": np.float32([1.0, -1.0]),
             "m": np.float32([0.25, -0.5]), "v": np.float32([0.25, 4.0])}
    nodes = [helper.make_node("BatchNormalization", ["X", "g", "b", "m", "v"],
                              ["A"], epsilon=0.0),
             helper.make_node("Dropout", ["A"], ["Y"])]
    path = _save(tmp_path, nodes, [("X", ["N", 2, 3, 5])],
                 [("Y", ["N", 2, 3, 5])], inits)
    module = load_onnx(path)
    training = sorted(type(m).__name__ for m in module.modules() if m.training)
    assert not training, f"modules left in training mode: {training}"
    batch = np.stack([p for p in _points((2, 3, 5), n=4, seed=1)])
    _assert_same(_run(module, [batch]), _ort(path, [batch]),
                 "BatchNormalization and Dropout on a batch of four")


def _retry_files():
    """Files the first conversion refuses and the retry of ``load_onnx``
    (version conversion to opset 13, onnx-simplifier, a second conversion)
    loads: an old opset, the deprecated Upsample, a newer opset, and a node
    whose name collides with the name onnx2torch makes for an unnamed node."""
    W = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5) / 4
    B = np.float32([0.5, -0.25, 1.0, 2.0])
    Wc = (np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) - 12) / 8
    return {
        # 'Converter is not implemented' (Gemm since version 7)
        "gemm-opset8": dict(
            nodes=[helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)],
            shape=[1, 3], inits={"W": W, "B": B}, opset=8),
        # 'Converter is not implemented' (Upsample), rewritten to Resize
        "upsample-opset9-nearest": dict(
            nodes=[helper.make_node("Upsample", ["X", "S"], ["Y"],
                                    mode="nearest")],
            shape=[1, 2, 2, 3], inits={"S": np.float32([1, 1, 2, 2])},
            opset=9),
        # 'Converter is not implemented' (Conv since version 22): downgraded
        "conv-opset22": dict(
            nodes=[helper.make_node("Conv", ["X", "W"], ["Y"],
                                    pads=[1, 0, 1, 0], strides=[1, 2])],
            shape=[1, 2, 3, 5], inits={"W": Wc}, opset=22),
        # 'Got unexpected input value type': the second unnamed Constant is
        # named Constant_1 by onnx2torch, as is the Add, so one is dropped
        "name-collision": dict(
            nodes=[helper.make_node("Constant", [], ["c0"], value=numpy_helper
                                    .from_array(np.float32([[0.5, -1, 2]]))),
                   helper.make_node("Constant", [], ["c1"], value=numpy_helper
                                    .from_array(np.float32([[4, 0.25, -8]]))),
                   helper.make_node("Add", ["X", "c0"], ["a"],
                                    name="Constant_1"),
                   helper.make_node("Mul", ["a", "c1"], ["Y"])],
            shape=[1, 3], inits={}, opset=13),
    }


@pytest.mark.parametrize("case", [
    pytest.param(k, marks=_NEAREST_WARNING) if k.startswith("upsample") else k
    for k in sorted(_retry_files())])
def test_a_file_the_first_conversion_refuses_is_loaded_by_the_opset_retry(
        tmp_path, case):
    """The retry of ``load_onnx`` must take each recoverable conversion
    error, the missing per-opset converter and the dropped node of a name
    collision, and the loaded module must agree with onnxruntime on 16
    points. Catches a test of the error message that takes neither message
    (the file is then refused) or only one of them."""
    spec = _retry_files()[case]
    path = _save(tmp_path, spec["nodes"], [("X", spec["shape"])],
                 [("Y", None)], spec["inits"], opset=spec["opset"])
    module = load_onnx(path)
    for x in _points(spec["shape"], seed=2):
        _assert_same(_run(module, [x], torch.float64), _ort(path, [x]), case)


def test_a_file_the_opset_retry_cannot_convert_is_refused(tmp_path):
    """An opset-18 Split with ``num_outputs`` has no converter, and the
    version conversion to opset 13 fails: the load must raise
    ``NotImplementedError`` (the type of the first error, with both
    messages), not answer and not crash with another type."""
    nodes = [helper.make_node("Split", ["X"], ["A", "B"], axis=1,
                              num_outputs=2)]
    path = _save(tmp_path, nodes, [("X", [1, 5])], [("A", [1, 3]),
                                                    ("B", [1, 2])], opset=18)
    with pytest.raises(NotImplementedError, match="opset upgrade"):
        load_onnx(path)


def _shape_subgraph_files():
    W = np.full((6, 2), 0.25, np.float32)
    W[:, 1] = np.float32([1, -2, 0.5, 0, 3, -1])
    return {
        # the target of a Reshape computed from the batch size
        "shape-gather-concat-reshape": dict(
            nodes=[helper.make_node("Shape", ["X"], ["s"]),
                   helper.make_node("Gather", ["s", "i0"], ["b"], axis=0),
                   helper.make_node("Unsqueeze", ["b", "ax"], ["bu"]),
                   helper.make_node("Concat", ["bu", "m1"], ["t"], axis=0),
                   helper.make_node("Reshape", ["X", "t"], ["R"]),
                   helper.make_node("MatMul", ["R", "W"], ["Y"])],
            inits={"i0": np.array(0, np.int64), "ax": np.int64([0]),
                   "m1": np.int64([-1]), "W": W},
            in_shape=["N", 2, 3], out_shape=["N", 2]),
        # a constant of the input's shape
        "constant-of-shape": dict(
            nodes=[helper.make_node("Shape", ["X"], ["s"]),
                   helper.make_node("ConstantOfShape", ["s"], ["o"],
                                    value=numpy_helper.from_array(
                                        np.float32([-2.0]))),
                   helper.make_node("Mul", ["X", "o"], ["Y"])],
            inits={}, in_shape=["N", 2, 3], out_shape=["N", 2, 3]),
    }


@pytest.mark.parametrize("case", sorted(_shape_subgraph_files()))
def test_a_shape_subgraph_is_folded_away_and_the_model_is_reached(
        tmp_path, case):
    """A model that computes shapes with Shape or ConstantOfShape under a
    symbolic batch axis: ``load_onnx`` folds that subgraph into constants,
    so no Shape, ConstantOfShape, Gather or Concat module is left, the
    module agrees with onnxruntime, and a Star reach of the loaded model
    answers and holds 128 onnxruntime outputs by its rows. Catches a fold
    whose result is thrown away (the reach then refuses the Shape module)."""
    spec = _shape_subgraph_files()[case]
    path = _save(tmp_path, spec["nodes"], [("X", spec["in_shape"])],
                 [("Y", spec["out_shape"])], spec["inits"])
    module = load_onnx(path)
    left = _module_classes(module) & {"OnnxShape", "OnnxConstantOfShape",
                                      "OnnxGather", "OnnxConcat"}
    assert not left, f"{case}: modules of the shape subgraph left: {left}"
    lo, hi = flat_box(6, seed=3)
    lo, hi = lo * 8, hi * 8
    points = [p.reshape(1, 2, 3) for p in samples(lo, hi)]
    session = _session(path)
    want = [_ort(path, [p], session) for p in points]
    for p, w in zip(points[:16], want[:16]):
        _assert_same(_run(module, [p], torch.float64), w, case)
    star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    sets = NeuralNetwork(module).reach(star, method="approx",
                                       input_shape=(2, 3))
    assert sets, f"{case}: the reach returned no sets"
    assert_covered(sets, [w[0].reshape(-1) for w in want], case)


@pytest.mark.parametrize("length", [
    pytest.param(4, id="fixed-length-4"),
    pytest.param("L", id="symbolic-length-fed-4", marks=known_bug(
        12, "the shape fold of load_onnx sets every symbolic input axis to 1, "
            "not only the batch axis, so a model that reads the length of a "
            "symbolic axis is converted for length 1 and answers x + 1 on an "
            "input of length 4 where onnxruntime answers x + 4")),
])
def test_a_symbolic_axis_that_is_not_the_batch_axis_keeps_its_length(
        tmp_path, length):
    """Y = X + float(length of axis 1 of X) with axis 1 symbolic: on an input
    of length 4 the loaded module must answer X + 4, as onnxruntime does, or
    the load or the run must refuse. ``_simplify`` fixes every symbolic axis
    of the first input to 1 before it folds the Shape node, so the module
    adds 1. The file with the length fixed at 4 is the passing neighbour."""
    nodes = [helper.make_node("Shape", ["X"], ["s"]),
             helper.make_node("Gather", ["s", "i1"], ["d"], axis=0),
             helper.make_node("Cast", ["d"], ["df"], to=TensorProto.FLOAT),
             helper.make_node("Add", ["X", "df"], ["Y"])]
    path = _save(tmp_path, nodes, [("X", [1, length])], [("Y", [1, length])],
                 {"i1": np.array(1, np.int64)})
    _assert_agrees_or_refuses(path, [[p] for p in _points((1, 4), seed=4)],
                              f"length {length}")


def test_a_shape_subgraph_the_fold_cannot_remove_leaves_the_model_unfolded(
        tmp_path):
    """Y = X + len(X) + 2 len(Z) with the lengths of X (1, L) and Z (1, M)
    both symbolic and read by Shape nodes. ``_simplify`` fixes the axes of
    the first input only, so onnx-simplifier folds len(X) to 1 and keeps the
    Shape of Z; ``_fold_shape_subgraph`` must then drop the partial fold and
    convert the original graph, whose module reads both lengths at run time:
    on inputs of lengths 4 and 2 it must answer X + 8, as onnxruntime does.
    Catches a fold that is kept although Shape nodes remain (X + 5); the
    graph also has two inputs fed in the file's order."""
    i1 = np.array(1, np.int64)
    nodes = [helper.make_node("Shape", ["X"], ["sx"]),
             helper.make_node("Gather", ["sx", "i1"], ["lx"], axis=0),
             helper.make_node("Shape", ["Z"], ["sz"]),
             helper.make_node("Gather", ["sz", "i1"], ["lz"], axis=0),
             helper.make_node("Add", ["lx", "lz"], ["l1"]),
             helper.make_node("Add", ["l1", "lz"], ["l2"]),
             helper.make_node("Cast", ["l2"], ["lf"], to=TensorProto.FLOAT),
             helper.make_node("Add", ["X", "lf"], ["Y"])]
    path = _save(tmp_path, nodes, [("X", [1, "L"]), ("Z", [1, "M"])],
                 [("Y", [1, "L"])], {"i1": i1})
    points = [[x, z] for x, z in zip(_points((1, 4), seed=18),
                                     _points((1, 2), seed=19))]
    _assert_agrees_or_refuses(path, points, "two symbolic lengths")


# --------------------------------------------------------------------------- #
# load_onnx: input regimes and graph outputs                                   #
# --------------------------------------------------------------------------- #
def test_a_float64_file_is_loaded_with_its_float64_parameters(tmp_path):
    """A float64 Gemm whose weights are not float32 numbers (1e4 + 1/3,
    7 + 1e-9): the loaded module must keep them in float64 and give
    onnxruntime's float64 outputs exactly, at points beyond plus and minus 1.
    A loader that turns the module to float32 moves the outputs by about
    1e-3. The loaded module then goes to
    ``NeuralNetwork``: a Star reach over a box must hold 128 onnxruntime
    float64 outputs by the returned set's rows, so neither the load nor the
    model handle rounds the parameters to float32 on the way to the reach."""
    W = np.array([[1e4 + 1 / 3, -2.0], [0.1, 7.0 + 1e-9], [3.0, 1e5 + 0.1]])
    path = _save(tmp_path, [helper.make_node("Gemm", ["X", "W", "B"], ["Y"],
                                             transB=1)],
                 [("X", [1, 2], TensorProto.DOUBLE)],
                 [("Y", [1, 3], TensorProto.DOUBLE)],
                 {"W": W, "B": np.array([1 / 3, -1e4 - 0.1, 0.0])})
    module = load_onnx(path)
    for x in _points((1, 2), seed=5):
        got = _run(module, [x])
        assert module(torch.as_tensor(x)).dtype == torch.float64
        _assert_same(got, _ort(path, [x]), "float64 Gemm")
    lo, hi = np.array([-2.5, 0.75]), np.array([1.25, 3.0])
    session = _session(path)
    want = [_ort(path, [p.reshape(1, 2)], session)[0] for p in samples(lo, hi)]
    sets = NeuralNetwork(module).reach(
        Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1)),
        method="approx")
    assert_covered(sets, want, "float64 Gemm through NeuralNetwork.reach")


def test_an_external_data_file_given_as_a_path_object_is_loaded(tmp_path):
    """A file whose weights are stored as external data beside it, loaded
    from a ``pathlib.Path``: the module must carry those weights (catches a
    load that skips external data)."""
    W = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5) / 4
    path = _save(tmp_path, [helper.make_node("Gemm", ["X", "W"], ["Y"],
                                             transB=1)],
                 [("X", [1, 3])], [("Y", [1, 4])], {"W": W},
                 save_as_external_data=True, all_tensors_to_one_file=True,
                 location="weights.bin", size_threshold=0)
    assert os.path.exists(os.path.join(str(tmp_path), "weights.bin"))
    module = load_onnx(pathlib.Path(path))
    for x in _points((1, 3), seed=6):
        _assert_same(_run(module, [x], torch.float64), _ort(path, [x]),
                     "external data")


@pytest.mark.parametrize("case", ["initializer-that-is-a-graph-input",
                                  "two-inputs", "intermediate-as-output"])
def test_a_file_with_more_than_one_input_or_output_is_loaded(tmp_path, case):
    """Graph interfaces the converter maps one by one: an initializer that
    is also listed as a graph input (IR 4: a default value, fed by nobody),
    two graph inputs fed in the file's order (Sub, where a swap changes the
    result), and an intermediate value that is also a graph output. The
    module must take the same inputs and give onnxruntime's outputs, in the
    file's order."""
    W = (np.arange(12, dtype=np.float32).reshape(4, 3) - 5) / 4
    if case == "initializer-that-is-a-graph-input":
        path = _save(tmp_path, [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                     [("W", [3, 4]), ("X", [1, 3])], [("Y", [1, 4])],
                     {"W": W.T.copy()}, opset=9, ir_version=4)
        points = [[x] for x in _points((1, 3), seed=7)]
    elif case == "two-inputs":
        path = _save(tmp_path, [helper.make_node("Sub", ["X", "Z"], ["Y"])],
                     [("X", [1, 3]), ("Z", [1, 3])], [("Y", [1, 3])])
        points = [[x, z] for x, z in zip(_points((1, 3), seed=8),
                                         _points((1, 3), seed=9))]
    else:
        path = _save(tmp_path, [helper.make_node("Relu", ["X"], ["A"]),
                                helper.make_node("Neg", ["A"], ["Y"])],
                     [("X", [1, 3])], [("Y", [1, 3]), ("A", [1, 3])])
        points = [[x] for x in _points((1, 3), seed=10)]
    module = load_onnx(path)
    for feeds in points:
        _assert_same(_run(module, feeds, torch.float64), _ort(path, feeds),
                     case)


@pytest.mark.parametrize("case", [
    "outputs-of-two-nodes-in-reverse-order",
    pytest.param("outputs-of-one-split-node-in-reverse-order",
                 marks=known_bug(
                     23, "the vendored converter returns the whole tuple of a "
                         "multi-output node, in the node's order, for graph "
                         "outputs that are outputs of that node: outputs "
                         "listed as (S1, S0) of a Split come back as (S0, S1), "
                         "and one Split output beside another output comes "
                         "back as the tuple of every chunk")),
    pytest.param("a-graph-input-as-an-output",
                 marks=known_bug(
                     23, "the vendored converter looks up the producing node "
                         "of every graph output, so a graph output that is a "
                         "graph input (or an initializer), which the ONNX "
                         "checker accepts and onnxruntime answers, makes "
                         "load_onnx crash with KeyError",
                     raises=KeyError)),
])
def test_graph_outputs_come_back_in_the_files_order(tmp_path, case):
    """The module's outputs are the file's graph outputs in the file's order.
    onnx2torch builds the outputs from the producing nodes, deduplicated, so
    two outputs of one Split come back as that node's tuple, in the node's
    order: here swapped; and an output that no node produces (the graph
    input X listed as the second output) has no producing node to look up.
    Two outputs of two nodes listed in reverse order are the passing
    neighbour of both."""
    if case.startswith("outputs-of-two-nodes"):
        nodes = [helper.make_node("Relu", ["X"], ["S0"]),
                 helper.make_node("Neg", ["X"], ["S1"])]
        outputs = [("S1", [1, 4]), ("S0", [1, 4])]
    elif case.startswith("outputs-of-one-split"):
        nodes = [helper.make_node("Split", ["X"], ["S0", "S1"], axis=1)]
        outputs = [("S1", [1, 2]), ("S0", [1, 2])]
    else:
        nodes = [helper.make_node("Neg", ["X"], ["S0"])]
        outputs = [("S0", [1, 4]), ("X", [1, 4])]
    path = _save(tmp_path, nodes, [("X", [1, 4])], outputs)
    _assert_agrees_or_refuses(path, [[x] for x in _points((1, 4), seed=11)],
                              case)


def test_load_onnx_refuses_a_file_it_cannot_read(tmp_path):
    """Files that are not a valid ONNX model are refused with a clear error
    and never converted: a missing path (``FileNotFoundError``), a
    gzip-compressed model (the loader takes none; protobuf's
    ``DecodeError`` today), and a LeakyRelu whose ``alpha`` is misspelled,
    which onnx's checker and onnxruntime reject while the converter would
    load it with the default slope (onnx's ``ValidationError`` today;
    catches a load without the checker). Neither type derives from
    ``ValueError``, the settled type for an input that cannot be read, so
    each check accepts that type as well."""
    with pytest.raises(FileNotFoundError):
        load_onnx(os.path.join(str(tmp_path), "missing.onnx"))
    plain = _save(tmp_path, [helper.make_node("Relu", ["X"], ["Y"])],
                  [("X", [1, 3])], [("Y", [1, 3])], name="plain")
    with open(plain, "rb") as src, gzip.open(plain + ".gz", "wb") as dst:
        dst.write(src.read())
    with pytest.raises((DecodeError, ValueError)):
        load_onnx(plain + ".gz")
    graph = helper.make_graph(
        [helper.make_node("LeakyRelu", ["X"], ["Y"], alpah=0.5)], "typo",
        [_value_info(("X", [1, 3]))], [_value_info(("Y", [1, 3]))])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    typo = os.path.join(str(tmp_path), "typo.onnx")
    onnx.save(model, typo)
    with pytest.raises((onnx.checker.ValidationError, ValueError),
                       match="alpah"):
        load_onnx(typo)


# --------------------------------------------------------------------------- #
# The vendored converter: rows of the registry and attribute defaults          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("opset,height", [
    pytest.param(13, 4, id="opset13-height-4-divisible"),
    pytest.param(13, 3, id="opset13-height-3", marks=known_bug(
        "12b", "LOAD-5: the vendored Split module of opset 13 splits an even "
               "Split of a length that is not divisible by axis_len // "
               "num_splits, so load_onnx answers a file onnxruntime refuses",
        raises=DID_NOT_RAISE)),
    pytest.param(11, 3, id="opset11-height-3", marks=known_bug(
        "12b", "LOAD-5: the vendored Split module of opsets 2 and 11 splits "
               "an even Split of a length that is not divisible by axis_len "
               "// num_splits, so load_onnx answers a file onnxruntime "
               "refuses", raises=DID_NOT_RAISE)),
])
def test_a_split_onnxruntime_refuses_is_refused(tmp_path, opset, height):
    """An even Split of a height of 3 into two outputs is not a legal model:
    onnxruntime refuses it. The load, or a run of the loaded module, must
    refuse too; today the module splits into three chunks of 1 and answers
    -relu(row 1). The divisible height of 4 is the passing neighbour, on
    the same reference."""
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Split", ["A"], ["B", "C"], axis=2),
             helper.make_node("Neg", ["C"], ["Y"])]
    path = _save(tmp_path, nodes, [("X", [1, 2, height, 2])],
                 [("Y", [1, 2, height // 2, 2])], opset=opset, infer=False)
    _assert_agrees_or_refuses(
        path, [[x] for x in _points((1, 2, height, 2), seed=12)],
        f"Split of a height of {height} at opset {opset}",
        ort_refuses=height % 2 == 1)


_X4 = [1, 2, 3, 5]
_I64 = lambda *v: np.array(v, np.int64)  # noqa: E731
_F32 = lambda v: np.asarray(v, np.float32)  # noqa: E731
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max
_W = _F32((np.arange(24).reshape(2, 2, 2, 3) - 12) / 8)
# float32 rounding budget of a transcendental row computed in float32 by
# onnxruntime and in float64 by the module: 32 units of 2**-24 at the
# output's scale (an exp of at most one unit, a sum of at most 8 terms and
# a division), far below what a wrong axis or slope changes.
_SMOOTH = 32 * 2.0 ** -24

# (operator, node inputs, attributes, input shape, initializers, opset, slack)
_TABLE = [
    pytest.param("Flatten", ["X"], {"axis": 0}, _X4, {}, 13, 0.0,
                 id="Flatten-axis-0", marks=known_bug(
                     23, "LOAD-11: the vendored OnnxFlatten.forward with axis "
                         "0 flattens to one axis and then from axis 1, which "
                         "raises IndexError", raises=IndexError)),
    pytest.param("Flatten", ["X"], {"axis": -1}, _X4, {}, 13, 0.0,
                 id="Flatten-axis--1"),
    pytest.param("Flatten", ["X"], {}, _X4, {}, 13, 0.0,
                 id="Flatten-axis-left-out"),
    pytest.param("Softmax", ["X"], {}, [1, 2, 3], {}, 11, _SMOOTH,
                 id="Softmax-opset11-axis-left-out-rank3"),
    pytest.param("Softmax", ["X"], {"axis": 0}, [1, 2, 3], {}, 11, _SMOOTH,
                 id="Softmax-opset11-axis-0"),
    pytest.param("Softmax", ["X"], {}, [1, 2, 3], {}, 13, _SMOOTH,
                 id="Softmax-opset13-axis-left-out-rank3"),
    pytest.param("Softmax", ["X"], {"axis": 0}, [2, 3], {}, 13, _SMOOTH,
                 id="Softmax-opset13-axis-0"),
    pytest.param("Gemm", ["X", "W", "C"], {"transA": 1, "alpha": 0.25,
                                           "beta": 2.0}, [3, 1],
                 {"W": _F32(np.arange(12).reshape(3, 4) / 4),
                  "C": _F32([0.5])}, 13, 0.0, id="Gemm-transA-alpha-beta"),
    pytest.param("Gemm", ["X", "W"], {"transB": 1}, [1, 3],
                 {"W": _F32(np.arange(12).reshape(4, 3) / 4)}, 13, 0.0,
                 id="Gemm-C-left-out"),
    pytest.param("Concat", ["C", "X"], {"axis": -2}, [1, 2, 3],
                 {"C": _F32(np.full((1, 1, 3), 0.5))}, 13, 0.0,
                 id="Concat-axis--2-constant-first"),
    pytest.param("Gather", ["X", "i"], {}, [3, 4], {"i": _I64(2, 0)}, 13, 0.0,
                 id="Gather-axis-left-out"),
    pytest.param("Gather", ["X", "i"], {"axis": -1}, [1, 2, 5],
                 {"i": _I64(-1, 1)}, 13, 0.0,
                 id="Gather-axis--1-negative-index"),
    pytest.param("Transpose", ["X"], {}, [1, 2, 3], {}, 13, 0.0,
                 id="Transpose-perm-left-out"),
    pytest.param("Squeeze", ["X"], {}, [1, 3, 1, 2], {}, 11, 0.0,
                 id="Squeeze-opset11-axes-left-out"),
    pytest.param("Squeeze", ["X", "a"], {}, [1, 3, 1, 2], {"a": _I64(-2)}, 13,
                 0.0, id="Squeeze-opset13-axes--2"),
    pytest.param("Unsqueeze", ["X", "a"], {}, [1, 2, 3], {"a": _I64(-1, 1)},
                 13, 0.0, id="Unsqueeze-opset13-axes--1-1"),
    pytest.param("ReduceMean", ["X"], {}, [1, 2, 4], {}, 13, 0.0,
                 id="ReduceMean-axes-and-keepdims-left-out"),
    pytest.param("ReduceMax", ["X"], {"axes": [1, 2], "keepdims": 0},
                 [1, 2, 3, 2], {}, 13, 0.0, id="ReduceMax-axes-1-2-keepdims-0"),
    pytest.param("ReduceMax", ["X"], {"axes": [-2, -1], "keepdims": 0},
                 [1, 2, 3, 2], {}, 13, 0.0,
                 id="ReduceMax-axes--2--1-keepdims-0", marks=known_bug(
                     23, "the vendored OnnxReduceStaticAxes.forward reduces "
                         "ReduceMax, ReduceMin and ReduceProd one axis at a "
                         "time and, with keepdims 0, shifts each later axis "
                         "down by the number already removed, also when the "
                         "axes are negative, so axes [-2, -1] of a rank-4 "
                         "input reduce axes 2 and 1 and [-1, 1] reduce axes "
                         "3 and 0")),
    pytest.param("ReduceSum", ["X", "a"], {"noop_with_empty_axes": 1},
                 [1, 2, 3], {"a": np.zeros(0, np.int64)}, 13, 0.0,
                 id="ReduceSum-empty-axes-noop-1"),
    pytest.param("ReduceSum", ["X", "a"], {"keepdims": 0}, [1, 2, 3],
                 {"a": np.zeros(0, np.int64)}, 13, 0.0,
                 id="ReduceSum-empty-axes-keepdims-0", marks=known_bug(
                     23, "LOAD-12: the vendored OnnxReduceSumStaticAxes with an "
                         "empty axes constant and keepdims 0 calls "
                         "math_op_function, which only its sibling class "
                         "sets, and raises AttributeError",
                     raises=AttributeError)),
    pytest.param("ReduceSum", ["X"], {"axes": [0, -1], "keepdims": 0},
                 [1, 2, 3], {}, 11, 0.0, id="ReduceSum-opset11-axes-0--1"),
    pytest.param("Pad", ["X", "p"], {"mode": "reflect"}, _X4,
                 {"p": _I64(0, 0, 1, 2, 0, 0, 1, 1)}, 13, 0.0,
                 id="Pad-reflect"),
    pytest.param("Pad", ["X", "p"], {}, _X4,
                 {"p": _I64(0, 0, -1, 0, 0, 0, 0, -2)}, 13, 0.0,
                 id="Pad-mode-left-out-negative-pads"),
    pytest.param("AveragePool", ["X"], {"kernel_shape": [2, 2],
                                        "pads": [1, 1, 1, 1]}, _X4, {}, 13,
                 0.0, id="AveragePool-count_include_pad-left-out"),
    pytest.param("AveragePool", ["X"], {"kernel_shape": [2, 2],
                                        "pads": [1, 1, 1, 1],
                                        "count_include_pad": 1}, _X4, {}, 13,
                 0.0, id="AveragePool-count_include_pad-1"),
    pytest.param("AveragePool", ["X"], {"kernel_shape": [2, 2],
                                        "strides": [2, 2], "ceil_mode": 1},
                 _X4, {}, 13, 0.0, id="AveragePool-ceil_mode-1"),
    pytest.param("MaxPool", ["X"], {"kernel_shape": [2, 2],
                                    "strides": [2, 2], "ceil_mode": 1},
                 _X4, {}, 13, 0.0, id="MaxPool-ceil_mode-1"),
    pytest.param("MaxPool", ["X"], {"kernel_shape": [2, 2],
                                    "dilations": [2, 1]}, _X4, {}, 13, 0.0,
                 id="MaxPool-dilations"),
    pytest.param("Conv", ["X", "W", "B"], {"pads": [1, 0, 0, 2]}, _X4,
                 {"W": _W, "B": _F32([0.5, -0.25])}, 13, 0.0,
                 id="Conv-kernel_shape-left-out-asymmetric-pads"),
    pytest.param("Conv", ["X", "W"], {"group": 2, "dilations": [1, 2]}, _X4,
                 {"W": _F32(np.arange(12).reshape(4, 1, 1, 3) / 8)}, 13, 0.0,
                 id="Conv-group-2-dilations"),
    pytest.param("Conv", ["X", "W"], {"auto_pad": "SAME_UPPER",
                                      "strides": [2, 2]}, _X4, {"W": _W}, 13,
                 0.0, id="Conv-auto_pad-SAME_UPPER"),
    pytest.param("ConvTranspose", ["X", "W"], {"strides": [2, 2],
                                               "output_padding": [1, 0]},
                 _X4, {"W": _W}, 13, 0.0, id="ConvTranspose-output_padding"),
    pytest.param("BatchNormalization", ["X", "g", "b", "m", "v"], {}, _X4,
                 {"g": _F32([0.5, 2]), "b": _F32([1, -1]),
                  "m": _F32([0.25, -0.5]), "v": _F32([0.25, 4])}, 13,
                 _SMOOTH, id="BatchNormalization-epsilon-left-out"),
    pytest.param("LeakyRelu", ["X"], {}, [1, 6], {}, 13, _SMOOTH,
                 id="LeakyRelu-alpha-left-out"),
    pytest.param("Clip", ["X", "", "mx"], {}, [1, 6],
                 {"mx": np.array(0.5, np.float32)}, 13, 0.0,
                 id="Clip-min-left-out"),
    pytest.param("Clip", ["X"], {"min": -0.5, "max": 0.75}, [1, 6], {}, 6,
                 0.0, id="Clip-opset6-attributes"),
    pytest.param("Slice", ["X", "s", "e", "a", "st"], {}, _X4,
                 {"s": _I64(-5), "e": _I64(_INT64_MIN), "a": _I64(3),
                  "st": _I64(-1)}, 13, 0.0,
                 id="Slice-step--1-start-minus-the-length"),
    pytest.param("Slice", ["X", "s", "e", "a", "st"], {}, _X4,
                 {"s": _I64(-6), "e": _I64(_INT64_MIN), "a": _I64(3),
                  "st": _I64(-1)}, 13, 0.0,
                 id="Slice-step--1-start-below-minus-the-length",
                 marks=known_bug(
                     23, "the vendored Slice module turns a negative step into "
                         "a flip and the start s into -s - 1 without clamping "
                         "it, so a start below minus the axis length selects "
                         "nothing (an axis of length 0) where onnxruntime "
                         "clamps the start to entry 0 and selects that one "
                         "entry (the module side of GRAPH-20)")),
    pytest.param("Slice", ["X", "s", "e", "a", "st"], {}, _X4,
                 {"s": _I64(_INT64_MAX), "e": _I64(0), "a": _I64(-1),
                  "st": _I64(-2)}, 13, 0.0, id="Slice-step--2-start-INT64_MAX",
                 marks=_TRUNCATION_WARNING),
    pytest.param("Slice", ["X"], {"starts": [1], "ends": [3], "axes": [-1]},
                 _X4, {}, 9, 0.0, id="Slice-opset9-axis--1"),
    pytest.param("Reshape", ["X", "t"], {}, _X4, {"t": _I64(1, 0, -1)}, 13,
                 0.0, id="Reshape-0-copies-the-dimension"),
    pytest.param("Resize", ["X", "", "sc"], {}, _X4,
                 {"sc": _F32([1, 1, 2, 2])}, 13, 0.0,
                 id="Resize-mode-left-out-scales-2", marks=_NEAREST_WARNING),
    pytest.param("Round", ["X"], {}, [1, 6], {}, 13, 0.0,
                 id="Round-half-to-even"),
]


@pytest.mark.parametrize("op,ins,attrs,in_shape,inits,opset,tol", _TABLE)
def test_a_loaded_operator_agrees_with_onnxruntime_or_is_refused(
        tmp_path, op, ins, attrs, in_shape, inits, opset, tol):
    """One operator per row with the attribute value that the row names:
    left out where its default changed between opsets or differs from
    torch's, an axis of 0 or a negative axis, an optional input left out by
    an empty name. The loaded module, run in float64 on 16 dyadic points,
    must give onnxruntime's outputs (exactly, or within the float32 budget
    ``_SMOOTH`` for a transcendental row), or the load must refuse. Catches
    a converter that reads an attribute's default, a negative axis or a
    clamp of the operator wrongly."""
    nodes = [helper.make_node(op, ins, ["Y"], **attrs)]
    path = _save(tmp_path, nodes, [("X", in_shape)], [("Y", None)], inits,
                 opset=opset)
    points = [[p] for p in _points(in_shape, seed=13)]
    _assert_agrees_or_refuses(path, points, f"{op} {attrs}", tol=tol)


# --------------------------------------------------------------------------- #
# QuantizeLinear and DequantizeLinear                                          #
# --------------------------------------------------------------------------- #
_QRANGE = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767),
           "uint16": (0, 65535), "int4": (-8, 7)}
_NP = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16,
       "uint16": np.uint16}


def _per_axis(param, axis, ndim, block):
    """A scale or zero point broadcast along ``axis`` by the operator's
    definition: one value for every entry, one per index of ``axis``, or,
    for blocked quantization, each value repeated ``block`` times."""
    p = np.asarray(param, dtype=np.float64)
    if block:
        return np.repeat(p, block, axis=axis % ndim)
    if p.size == 1:
        return p.reshape(())
    shape = [1] * ndim
    shape[axis % ndim] = -1
    return p.reshape(shape)


def _quantize(x, scale, zero, axis, qtype, block=None):
    """QuantizeLinear by its definition: round x / scale half to even, add
    the zero point, saturate to the range of the output type. With
    power-of-two scales x / scale is exact."""
    lo, hi = _QRANGE[qtype]
    s = _per_axis(scale, axis, x.ndim, block)
    z = _per_axis(zero, axis, x.ndim, block)
    return np.clip(np.rint(np.asarray(x, np.float64) / s) + z, lo, hi)


def _dequantize(q, scale, zero, axis, block=None):
    """DequantizeLinear by its definition: (q - zero point) * scale."""
    s = _per_axis(scale, axis, q.ndim, block)
    z = _per_axis(zero, axis, q.ndim, block)
    return (q - z) * s


# x / 0.25 lands on the ties 0.5, 1.5, 2.5 ... of both signs
_TIES = _F32(np.array([[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5, 0.25,
                        -0.75, 6.0, -6.0]]) * 0.25)
# with scale 1 and a zero point: saturates int8 and uint8 at both ends, and
# holds ties
_WIDE = _F32([[-300, -133, -128.5, -123.5, -1, 0, 0.5, 120.5, 121.5, 122.5,
               127.5, 128, 248.5, 255.5, 256, 400]])
_PER_CHANNEL = _F32((np.random.default_rng(14).integers(-320, 321, (1, 2, 3, 5))
                     ) / 8)
_SQUARE = _F32([[-3.0, 1.5, 0.75], [2.25, -0.5, 9.0], [0.375, -6.0, 1.25]])
_WIDE16 = _F32([[-40000, -32770.5, -32768.5, -5, 0.5, 1.5, 32766.5, 32767.5,
                 40000]])
_WIDEU16 = _F32([[-5, 0.5, 1.5, 65524.5, 65525.5, 65530, 70000]])
_INT4 = _F32(np.array([[-20, -9, -8.5, -1, 0, 5, 6.5, 30]]) * 0.25)
_BLOCKED = _F32((np.arange(8).reshape(2, 4) - 3) * 1.5)


def _qdq(route, x, scale, zero, qtype, axis=None, opset=13, q_attrs=None,
         dq_attrs=None, block=None, dq_scale=None, dq_zero=None,
         from_constant=None, infer=True):
    """One Q/DQ case. ``route`` is ``runtime`` (X -> QuantizeLinear ->
    DequantizeLinear -> Y, converted to the runtime modules) or ``fold``
    (Y = X + DequantizeLinear(QuantizeLinear(C)) with the constant C, folded
    into an initializer before conversion, fed X = 0). ``zero`` is an array,
    ``None`` (input left out) or ``""`` (input named by an empty name).
    ``dq_scale``/``dq_zero`` give the DequantizeLinear other parameters than
    the QuantizeLinear's. ``from_constant`` names a parameter (``scale`` or
    ``zero``) that comes from a Constant node instead of an initializer.
    ``infer=False`` writes the file without the value types onnx's shape
    inference records for the intermediate values."""
    return dict(route=route, x=x, scale=scale, zero=zero, qtype=qtype,
                axis=axis, opset=opset, q_attrs=q_attrs or {},
                dq_attrs=dq_attrs or {}, block=block, dq_scale=dq_scale,
                dq_zero=dq_zero, from_constant=from_constant, infer=infer)


def _qdq_file(tmp_path, case):
    """The file of a Q/DQ case and the expected output, from ``_quantize``
    and ``_dequantize``."""
    x = np.asarray(case["x"], np.float32)
    axis = case["axis"]
    q_attrs, dq_attrs = dict(case["q_attrs"]), dict(case["dq_attrs"])
    if axis is not None:
        q_attrs["axis"] = axis
        dq_attrs["axis"] = axis
    inits, nodes = {}, []

    def tensor(name, value):
        if case["from_constant"] == name:
            nodes.append(helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(
                    np.asarray(value), name)))
        else:
            inits[name] = value
        return name

    zero = case["zero"]
    q_in = [tensor("scale", case["scale"])]
    if isinstance(zero, onnx.TensorProto):
        inits["zero"] = zero
        q_in.append("zero")
    elif zero is not None:
        q_in.append("" if isinstance(zero, str) else tensor("zero", zero))
    dq_scale = case["dq_scale"] if case["dq_scale"] is not None else None
    if dq_scale is None:
        dq_in = list(q_in)
    else:
        inits["dq_scale"] = dq_scale
        dq_in = ["dq_scale"]
        if case["dq_zero"] is not None:
            inits["dq_zero"] = case["dq_zero"]
            dq_in.append("dq_zero")
    source = "X" if case["route"] == "runtime" else "C"
    nodes += [helper.make_node("QuantizeLinear", [source] + q_in, ["q"],
                               **q_attrs),
              helper.make_node("DequantizeLinear", ["q"] + dq_in,
                               ["Y" if case["route"] == "runtime" else "d"],
                               **dq_attrs)]
    if case["route"] == "fold":
        inits["C"] = x
        nodes.append(helper.make_node("Add", ["X", "d"], ["Y"]))
    path = _save(tmp_path, nodes, [("X", list(x.shape))],
                 [("Y", list(x.shape))], inits, opset=case["opset"],
                 infer=case["infer"])
    feed = x if case["route"] == "runtime" else np.zeros_like(x)
    if isinstance(zero, onnx.TensorProto):
        zero_value = np.asarray(numpy_helper.to_array(zero), np.float64)
    elif zero is None or isinstance(zero, str):
        zero_value = np.zeros_like(np.asarray(case["scale"], np.float64))
    else:
        zero_value = np.asarray(zero, np.float64)
    q_axis = 1 if axis is None else axis
    q = _quantize(x, case["scale"], zero_value, q_axis, case["qtype"],
                  case["block"] if q_attrs.get("block_size") else None)
    if dq_scale is None:
        want = _dequantize(q, case["scale"], zero_value, q_axis,
                           case["block"] if dq_attrs.get("block_size")
                           else None)
    else:
        dq_zero = (np.zeros_like(np.asarray(dq_scale, np.float64))
                   if case["dq_zero"] is None
                   else np.asarray(case["dq_zero"], np.float64))
        want = _dequantize(q, dq_scale, dq_zero, q_axis,
                           case["block"] if dq_attrs.get("block_size")
                           else None)
    return path, feed, want


_I8 = np.int8
_U8 = np.uint8
_QDQ = {
    # rounding half to even, with an odd zero point (a tie of x / s + z then
    # rounds the other way) of both signs
    "runtime-ties-uint8-zero-point-3": _qdq("runtime", _TIES, _F32(0.25),
                                            _U8(3), "uint8"),
    "fold-ties-uint8-zero-point-3": _qdq("fold", _TIES, _F32(0.25), _U8(3),
                                         "uint8"),
    "runtime-ties-int8-zero-point--3": _qdq("runtime", _TIES, _F32(0.25),
                                            _I8(-3), "int8"),
    "fold-ties-int8-zero-point--3": _qdq("fold", _TIES, _F32(0.25), _I8(-3),
                                         "int8"),
    # saturation at both ends of int8 and of uint8, nonzero zero points
    "runtime-saturates-int8-zero-point--5": _qdq("runtime", _WIDE, _F32(1),
                                                 _I8(-5), "int8"),
    "fold-saturates-int8-zero-point--5": _qdq("fold", _WIDE, _F32(1), _I8(-5),
                                              "int8"),
    "runtime-saturates-uint8-zero-point-7": _qdq("runtime", _WIDE, _F32(1),
                                                 _U8(7), "uint8"),
    "fold-saturates-uint8-zero-point-7": _qdq("fold", _WIDE, _F32(1), _U8(7),
                                              "uint8"),
    # an absent zero point: zero, and the uint8 range
    "runtime-zero-point-left-out": _qdq("runtime", _WIDE, _F32(1), None,
                                        "uint8"),
    "fold-zero-point-left-out": _qdq("fold", _WIDE, _F32(1), None, "uint8"),
    "runtime-zero-point-named-empty": _qdq("runtime", _WIDE, _F32(1), "",
                                           "uint8"),
    # one scale and zero point per channel of a (1, 2, 3, 5) tensor
    "runtime-per-channel-axis--3": _qdq("runtime", _PER_CHANNEL,
                                        _F32([0.25, 0.5]), _I8([3, -2]),
                                        "int8", axis=-3),
    "fold-per-channel-axis--3": _qdq("fold", _PER_CHANNEL, _F32([0.25, 0.5]),
                                     _I8([3, -2]), "int8", axis=-3),
    # per-axis parameters with the axis attribute left out (default 1) on a
    # square tensor, where axis 0 gives another answer and no error
    "runtime-per-axis-axis-left-out": _qdq("runtime", _SQUARE,
                                           _F32([0.25, 0.5, 2.0]),
                                           _I8([1, 0, -1]), "int8"),
    "fold-per-axis-axis-left-out": _qdq("fold", _SQUARE,
                                        _F32([0.25, 0.5, 2.0]),
                                        _I8([1, 0, -1]), "int8"),
    # every registered version and a later one (downgraded by the retry)
    "runtime-opset10": _qdq("runtime", _TIES, _F32(0.25), _U8(3), "uint8",
                            opset=10),
    "runtime-opset19": _qdq("runtime", _TIES, _F32(0.25), _I8(-3), "int8",
                            opset=19),
    "runtime-opset21-int8": _qdq("runtime", _TIES, _F32(0.25), _I8(-3),
                                 "int8", opset=21),
    "runtime-opset23": _qdq("runtime", _TIES, _F32(0.25), _I8(-3), "int8",
                            opset=23),
    # 16-bit types of opset 21
    "runtime-opset21-int16": _qdq("runtime", _WIDE16, _F32(1),
                                  np.int16(-7), "int16", opset=21),
    "fold-opset21-int16": _qdq("fold", _WIDE16, _F32(1), np.int16(-7),
                               "int16", opset=21),
    "runtime-opset21-uint16": _qdq("runtime", _WIDEU16, _F32(1),
                                   np.uint16(9), "uint16", opset=21),
    # output_dtype of opset 21
    "runtime-opset21-output_dtype-uint8-with-zero-point": _qdq(
        "runtime", _WIDE, _F32(1), _U8(7), "uint8", opset=21,
        q_attrs={"output_dtype": TensorProto.UINT8}),
}
_QDQ_MARKED = {
    "runtime-opset21-output_dtype-int8-zero-point-left-out": (
        _qdq("runtime", _WIDE, _F32(1), None, "int8", opset=21,
             q_attrs={"output_dtype": TensorProto.INT8}),
        known_bug(23, "the runtime QuantizeLinear converter does not read the "
                      "opset-21 output_dtype attribute: with output_dtype int8 "
                      "and no zero point it saturates to the uint8 range "
                      "0..255 where the output type is int8")),
    "fold-opset21-output_dtype-int8-zero-point-left-out": (
        _qdq("fold", _WIDE, _F32(1), None, "int8", opset=21,
             q_attrs={"output_dtype": TensorProto.INT8}, infer=False),
        known_bug(23, "fold_constant_qdq does not read the opset-21 "
                      "output_dtype attribute: a constant QuantizeLinear with "
                      "output_dtype int8 and no zero point is folded into a "
                      "uint8 initializer saturated to 0..255, so the loaded "
                      "weights are 0 where -128 is due (a file that records "
                      "the int8 value type crashes instead, with onnx's "
                      "InferenceError inside the conversion)")),
    "runtime-opset21-int4-zero-point": (
        _qdq("runtime", _INT4, _F32(0.25),
             helper.make_tensor("zero", TensorProto.INT4, [], [-3]), "int4",
             opset=21),
        known_bug(23, "the runtime Q/DQ converters cannot read a 4-bit zero "
                      "point: onnx2torch's to_torch raises TypeError on the "
                      "ml_dtypes int4 array where an answer or a refusal is "
                      "due", raises=TypeError)),
    "fold-opset21-int4-zero-point": (
        _qdq("fold", _INT4, _F32(0.25),
             helper.make_tensor("zero", TensorProto.INT4, [], [-3]), "int4",
             opset=21),
        known_bug(23, "fold_constant_qdq has no range for 4-bit types: it "
                      "saturates an int4 QuantizeLinear to 0..255 and the "
                      "cast to int4 wraps, so the folded weights differ from "
                      "onnxruntime's (0.75 where -1.25 is due)")),
}


@pytest.mark.parametrize("case", sorted(_QDQ) + [
    pytest.param(k, marks=m) for k, (_, m) in sorted(_QDQ_MARKED.items())])
def test_a_loaded_qdq_model_computes_the_operator_definition(tmp_path, case):
    """A QuantizeLinear followed by a DequantizeLinear, converted to the
    runtime modules (``runtime-``) or folded into a constant before
    conversion (``fold-``): the loaded module, run as returned, must give
    the operators' definition computed with integers in numpy (rounding half
    to even, the zero point added after rounding, saturation to the range of
    the output type, per-axis parameters along their axis with the default
    axis 1), and onnxruntime must agree with that definition. Catches every
    slip of that arithmetic: rounding half up, the zero point inside the
    rounding or subtracted after scaling, a saturation range of the wrong
    type, a wrong default zero point, axis or saturation, an unsigned
    subtraction that wraps, a missing registration of a version."""
    spec = _QDQ[case] if case in _QDQ else _QDQ_MARKED[case][0]
    path, feed, want = _qdq_file(tmp_path, spec)
    reference = _ort(path, [feed])[0]
    if not np.array_equal(reference, want):
        raise SetupError(f"{case}: onnxruntime {reference.reshape(-1)} and "
                         f"the definition {want.reshape(-1)} differ")
    module = load_onnx(path)
    _assert_same(_run(module, [feed]), [want], case)


@pytest.mark.parametrize("case", [
    "runtime-blocked-dequantize", "runtime-blocked-quantize", "fold-blocked",
    "scale-from-a-constant-node", "zero-point-from-a-constant-node"])
def test_a_qdq_model_n2v_does_not_support_is_refused_or_answered_exactly(
        tmp_path, case):
    """Blocked quantization (``block_size``, opset 21) and parameters that
    are not initializers are not supported: the load must refuse
    (``NotImplementedError``) or the module must compute the operators'
    definition. A converter that drops the block check broadcasts the block
    scales as if they were per axis and answers another function."""
    blocked = dict(scale=_F32([[0.25, 0.5], [1.0, 2.0]]),
                   zero=np.zeros((2, 2), np.int8), qtype="int8", axis=1,
                   opset=21, block=2)
    spec = {
        "runtime-blocked-dequantize": _qdq(
            "runtime", _BLOCKED, _F32(0.25), _I8(0), "int8", axis=1,
            opset=21, dq_attrs={"block_size": 2}, block=2,
            dq_scale=_F32([[0.25, 0.5], [1.0, 2.0]]),
            dq_zero=np.zeros((2, 2), np.int8)),
        "runtime-blocked-quantize": _qdq(
            "runtime", _BLOCKED, q_attrs={"block_size": 2}, dq_scale=_F32(1),
            dq_zero=_I8(0), **blocked),
        "fold-blocked": _qdq("fold", _BLOCKED, q_attrs={"block_size": 2},
                             dq_attrs={"block_size": 2}, **blocked),
        "scale-from-a-constant-node": _qdq("runtime", _TIES, _F32(0.25),
                                           _U8(3), "uint8",
                                           from_constant="scale"),
        "zero-point-from-a-constant-node": _qdq("runtime", _TIES, _F32(0.25),
                                                _U8(3), "uint8",
                                                from_constant="zero"),
    }[case]
    path, feed, want = _qdq_file(tmp_path, spec)
    reference = _ort(path, [feed])[0]
    if not np.array_equal(reference, want):
        raise SetupError(f"{case}: onnxruntime {reference.reshape(-1)} and "
                         f"the definition {want.reshape(-1)} differ")
    try:
        module = load_onnx(path)
    except NotImplementedError:
        return
    _assert_same(_run(module, [feed]), [want], case)


@pytest.mark.parametrize("case", ["weights-folded", "runtime-quantize"])
def test_a_reach_of_a_loaded_qdq_model_holds_every_output_or_is_refused(
        tmp_path, case):
    """A reach of a loaded Q/DQ model. With only a weight-side
    DequantizeLinear (uint8 weights on both sides of a zero point given per
    output column, axis 1), folded into a float initializer before
    conversion, the model is an ordinary MatMul: a Star reach must hold 128
    onnxruntime outputs by the returned set's rows. With an activation
    QuantizeLinear and DequantizeLinear in front, converted to the runtime
    modules, the reach must refuse with ``NotImplementedError``, as the
    module docstring of ``quant_converters`` states, or hold those outputs,
    on Star, Zono and Box. The box has bounds that are not multiples of the
    quantization step 0.25 (1.875 rounds to 2.0, -1.375 to -1.5), so a
    reach that took the rounding step for the identity leaves out true
    outputs."""
    nodes = [helper.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["wd"],
                              axis=1),
             helper.make_node("MatMul", ["xd", "wd"], ["Y"])]
    inits = {"wq": np.uint8([[3, 250], [130, 1], [128, 4]]),
             "ws": _F32([0.5, 0.25]), "wz": np.uint8([128, 2])}
    if case == "weights-folded":
        nodes[1] = helper.make_node("MatMul", ["X", "wd"], ["Y"])
    else:
        nodes = [helper.make_node("QuantizeLinear", ["X", "as", "az"], ["xq"]),
                 helper.make_node("DequantizeLinear", ["xq", "as", "az"],
                                  ["xd"])] + nodes
        inits.update({"as": _F32(0.25), "az": np.uint8(128)})
    path = _save(tmp_path, nodes, [("X", [1, 3])], [("Y", [1, 2])], inits)
    module = load_onnx(path)
    lo, hi = np.array([-2.0, -1.375, 0.5]), np.array([1.875, 2.0, 3.0])
    session = _session(path)
    want = [_ort(path, [p.reshape(1, 3)], session)[0] for p in samples(lo, hi)]
    builders = ((Star.from_bounds, Zono.from_bounds, Box)
                if case == "runtime-quantize" else (Star.from_bounds,))
    for build in builders:
        try:
            sets = NeuralNetwork(module).reach(
                build(lo.reshape(-1, 1), hi.reshape(-1, 1)), method="approx")
        except NotImplementedError:
            if case == "runtime-quantize":
                continue
            raise
        assert_covered(sets, want, f"{case} on {build.__qualname__}")


# --------------------------------------------------------------------------- #
# load_pytorch                                                                 #
# --------------------------------------------------------------------------- #
def _small_net(seed=0):
    """Linear(3, 3), BatchNorm1d(3, eps=0), Dropout(0.5), with dyadic
    weights and running statistics (variances 1/4, 4 and 1)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3, eps=0.0),
                        nn.Dropout(0.5))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[0.5, -1.0, 2.0], [0.25, 0.0, -0.75],
                                          [1.5, 1.0, -0.5]]))
        net[0].bias.copy_(torch.tensor([0.125, -2.0, 1.0]))
        net[1].running_mean.copy_(torch.tensor([0.5, -1.0, 0.25]))
        net[1].running_var.copy_(torch.tensor([0.25, 4.0, 1.0]))
        net[1].weight.copy_(torch.tensor([2.0, 0.5, -1.0]))
        net[1].bias.copy_(torch.tensor([0.0, 1.0, -0.5]))
    return net


def _eval_reference(net, x):
    """The eval-mode output of ``_small_net`` by hand in float64: the
    affine map, then the stored statistics, then the identity."""
    W = net[0].weight.detach().double().numpy()
    b = net[0].bias.detach().double().numpy()
    bn = net[1]
    h = x @ W.T + b
    mean = bn.running_mean.double().numpy()
    var = bn.running_var.double().numpy()
    return ((h - mean) / np.sqrt(var) * bn.weight.detach().double().numpy()
            + bn.bias.detach().double().numpy())


def test_load_pytorch_returns_the_given_model_in_eval_mode():
    """``load_pytorch(model=...)`` with a model in training mode returns it
    in eval mode: on a batch of four the BatchNorm uses its stored mean and
    variance and the Dropout passes its input on, which the eval-mode
    formula computed by hand checks exactly (catches a dropped ``eval()``,
    after which the batch statistics and the dropout mask are used)."""
    net = _small_net()
    net.train()
    loaded = load_pytorch(model=net)
    training = [type(m).__name__ for m in loaded.modules() if m.training]
    assert not training, f"modules left in training mode: {training}"
    x = np.stack([p.reshape(-1) for p in _points((3,), n=4, seed=15)])
    with torch.no_grad():
        got = loaded(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    want = _eval_reference(net, x)
    assert np.array_equal(got, want), f"{got} where {want} is due"


def test_load_pytorch_checks_the_input_shape():
    """With ``input_shape`` the model is run once: a shape the model takes
    returns the model, one it cannot take raises ``ValueError`` (catches a
    check that builds the input and never runs the model)."""
    net = nn.Sequential(nn.Linear(3, 2))
    assert load_pytorch(model=net, input_shape=(3,)) is net
    with pytest.raises(ValueError, match="input_shape"):
        load_pytorch(model=net, input_shape=(4,))


@known_bug(5, "load_pytorch builds the dummy input of its input_shape check "
              "in float32, so a float64 model that takes the shape is refused "
              "with ValueError ('mat1 and mat2 must have the same dtype')",
           raises=ValueError)
def test_load_pytorch_checks_the_input_shape_in_the_models_dtype():
    """A float64 model that takes ``input_shape`` must be returned: the check
    runs the model on an input of its own dtype. The float32 model of
    ``test_load_pytorch_checks_the_input_shape`` is the passing neighbour."""
    net = nn.Sequential(nn.Linear(3, 2)).double()
    assert load_pytorch(model=net, input_shape=(3,)) is net


def test_load_pytorch_refuses_what_it_cannot_load(tmp_path):
    """Neither a model nor a path raises ``ValueError`` (catches a path
    branch that takes ``None``); a missing file raises
    ``FileNotFoundError``; a bare state dict without a model to load it into
    is refused (``ValueError``, today wrapped in ``RuntimeError``)."""
    with pytest.raises(ValueError, match="model_path or model"):
        load_pytorch()
    with pytest.raises(FileNotFoundError):
        load_pytorch(os.path.join(str(tmp_path), "missing.pt"))
    path = os.path.join(str(tmp_path), "state.pt")
    torch.save(_small_net().state_dict(), path)
    with pytest.raises((ValueError, RuntimeError), match="state dict"):
        load_pytorch(path)


@known_bug(5, "LOAD-3: load_pytorch called with model= and a path ignores the "
              "path, so the state dict in the file is never loaded")
def test_load_pytorch_loads_a_state_dict_into_the_given_model(tmp_path):
    """``load_pytorch(path, model=net)`` with a file that holds a state dict
    must return ``net`` with the file's weights: its outputs equal those of
    the saved model, computed by hand, and differ from the fresh model's."""
    saved = _small_net(seed=0)
    path = os.path.join(str(tmp_path), "state.pt")
    torch.save(saved.state_dict(), path)
    fresh = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3, eps=0.0),
                          nn.Dropout(0.5))
    loaded = load_pytorch(path, model=fresh)
    x = np.stack([p.reshape(-1) for p in _points((3,), n=4, seed=16)])
    with torch.no_grad():
        got = loaded(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    assert np.array_equal(got, _eval_reference(saved, x)), (
        "the returned model does not carry the weights of the file")


def test_load_pytorch_refuses_a_pickled_module_by_default(tmp_path):
    """A file that holds a pickled full module is refused by default, since
    unpickling runs code, with a message that names ``weights_only`` (the
    way to load it that DESIGN section 11 states)."""
    path = os.path.join(str(tmp_path), "full.pt")
    torch.save(_small_net(), path)
    with pytest.raises((ValueError, RuntimeError), match="weights_only"):
        load_pytorch(path)


@known_bug(5, "LOAD-2: load_pytorch has no way to load a pickled full "
              "nn.Module on torch 2.6 and later: torch.load defaults to "
              "weights_only=True and load_pytorch takes no weights_only "
              "argument", raises=TypeError)
def test_load_pytorch_loads_a_pickled_module_when_weights_only_is_off(
        tmp_path):
    """With ``weights_only=False`` (DESIGN section 11) a pickled full module
    loads and computes what the saved module computes, in eval mode. The
    default refusal of ``test_load_pytorch_refuses_a_pickled_module_by_default``
    is the passing neighbour."""
    saved = _small_net()
    path = os.path.join(str(tmp_path), "full.pt")
    torch.save(saved, path)
    loaded = load_pytorch(path, weights_only=False)
    x = np.stack([p.reshape(-1) for p in _points((3,), n=4, seed=17)])
    with torch.no_grad():
        got = loaded(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    assert np.array_equal(got, _eval_reference(saved, x))


@known_bug(5, "LOAD-2: load_pytorch reads every file with torch.load, which "
              "does not load a TorchScript archive under weights_only=True "
              "(re-raised as RuntimeError); DESIGN section 11 loads such an "
              "archive with torch.jit.load", raises=RuntimeError)
def test_load_pytorch_loads_a_torchscript_archive(tmp_path):
    """A TorchScript archive (``torch.jit.save`` of a scripted module in
    training mode) loads without ``model=`` and is returned in eval mode:
    its outputs on a batch of four equal the eval-mode formula computed by
    hand (DESIGN section 11). The pickled-module refusal of
    ``test_load_pytorch_refuses_a_pickled_module_by_default`` is the
    passing neighbour on the same file route."""
    saved = _small_net()
    saved.train()
    path = os.path.join(str(tmp_path), "scripted.pt")
    torch.jit.save(torch.jit.script(saved), path)
    loaded = load_pytorch(path)
    x = np.stack([p.reshape(-1) for p in _points((3,), n=4, seed=20)])
    with torch.no_grad():
        got = loaded(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    assert np.array_equal(got, _eval_reference(saved, x))


# --------------------------------------------------------------------------- #
# get_model_summary                                                            #
# --------------------------------------------------------------------------- #
class _SummaryNet(nn.Module):
    """A Conv2d and ReLU inside an ``nn.Sequential``, a Flatten and a frozen
    Linear without bias inside an ``nn.ModuleList``."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(2, 3, (2, 3)), nn.ReLU())
        self.tail = nn.ModuleList([nn.Flatten(), nn.Linear(18, 4, bias=False)])
        self.tail[1].requires_grad_(False)

    def forward(self, x):
        return self.tail[1](self.tail[0](self.body(x)))


_SUMMARY = {
    "Conv2d-1": {"input_shape": [1, 2, 3, 5], "output_shape": [1, 3, 2, 3],
                 "trainable": True, "nb_params": 2 * 3 * 2 * 3 + 3},
    "ReLU-2": {"input_shape": [1, 3, 2, 3], "output_shape": [1, 3, 2, 3],
               "nb_params": 0},
    "Flatten-3": {"input_shape": [1, 3, 2, 3], "output_shape": [1, 18],
                  "nb_params": 0},
    "Linear-4": {"input_shape": [1, 18], "output_shape": [1, 4],
                 "trainable": False, "nb_params": 18 * 4},
    "_SummaryNet-5": {"input_shape": [1, 2, 3, 5], "output_shape": [1, 4],
                      "nb_params": 0},
}


def test_model_summary_reports_shapes_parameters_and_trainable_flags():
    """``get_model_summary`` lists every module the forward runs, in call
    order, except the ``nn.Sequential`` and ``nn.ModuleList`` containers,
    with its input shape, output shape, parameter count (weight plus bias)
    and, for a module with a weight, whether it trains; the values are
    counted by hand on a (2, 3, 5) input, where input and output shapes
    differ at every layer that changes them. The forward hooks are removed
    afterwards (catches a count that keeps only the bias, an input shape
    read from the output, a container listed, a frozen layer reported as
    trainable, a missing bias read, hooks left on the model)."""
    net = _SummaryNet()
    summary = get_model_summary(net, (2, 3, 5))
    assert {k: dict(v) for k, v in summary.items()} == _SUMMARY
    left = [type(m).__name__ for m in net.modules() if m._forward_hooks]
    assert not left, f"forward hooks left on {left}"


@known_bug(5, "LOAD-4: get_model_summary builds its dummy input in float32, "
              "so a float64 model raises RuntimeError in its first layer "
              "('Input type (float) and bias type (double) should be the "
              "same')", raises=RuntimeError)
def test_model_summary_of_a_float64_model():
    """The summary of a float64 model is the summary of its float32 twin:
    the dummy input follows the model's dtype."""
    summary = get_model_summary(_SummaryNet().double(), (2, 3, 5))
    assert {k: dict(v) for k, v in summary.items()} == _SUMMARY


@known_bug(5, "LOAD-4: get_model_summary removes its forward hooks only after "
              "a forward that succeeds, not in finally, so a failing forward "
              "leaves them on the model")
def test_model_summary_removes_its_hooks_when_the_forward_fails():
    """An ``input_shape`` the model cannot take raises, and the model must
    carry no forward hook afterwards. A summary that succeeds, in
    ``test_model_summary_reports_shapes_parameters_and_trainable_flags``,
    is the passing neighbour."""
    net = _SummaryNet()
    with pytest.raises(RuntimeError):
        get_model_summary(net, (3, 3, 5))
    left = [type(m).__name__ for m in net.modules() if m._forward_hooks]
    assert not left, f"forward hooks left on {left}"
