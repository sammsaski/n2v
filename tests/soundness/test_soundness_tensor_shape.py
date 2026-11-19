"""Which tensor a set stands for, when a model walk must know it.

Before ``reach`` walks an ONNX graph, ``_propagate_shapes`` (``n2v/nn/reach.py``)
runs a placeholder tensor through the model and records the tensor shape at
every node; the shape-aware handlers (Softmax, Reduce, Concat, Slice, Gather,
Split, Transpose, MatMul) read their axis against that record. With
``input_shape`` it tries ``(1, *input_shape)`` first and keeps it whenever the
forward runs, else ``input_shape`` itself; without it an ``ImageStar`` gets
``(1, C, H, W)`` and every other set, ``ImageZono`` included, gets ``(1, dim)``,
else ``(dim,)``. The placeholder is built in the dtype of the model's first
floating parameter. Nothing checks the kept shape against the model's own
input, so a wrong placeholder silently moves every axis.

Two kinds of cell. The record cell
(:func:`test_the_tensor_shape_record_is_the_one_the_model_defines`) reads the
record the walk uses and compares it with the shapes torch computes when the
converted model runs on its declared input: the file's subject itself. Every
other cell checks the promise: the call raises ``NotImplementedError`` or
``ValueError`` (a refusal), or it returns a flat, non-empty list of sets with
the true number of entries and every onnxruntime output of a member point of
the input set is a member of a returned set, decided by the LP oracle on the
set's own rows (``assert_covered``). True outputs come from onnxruntime only,
always on the float64 twin of the graph (the same nodes with element type
DOUBLE): on the dyadic member points it is exact for row selection, sums and
Relu, and within float64 rounding (covered by ``LP_TOL``) for Softmax, means
over three or five entries and the L2 norm. The float32 file is what n2v
loads in the float32 cells; float32 onnxruntime rounds Softmax and means, and
that rounding is not this file's subject, so no slack for it is added. Flat
input boxes are the usual dyadic boxes times 8 (entries up to 16, widths up to
8), so a result that reads one entry in place of another misses the outputs.
For a graph that is a linear map, the member points include, for every output
entry that adds up several input entries, the points where it is largest and
smallest (box corners; for a Star the vertex of its predicate polytope, found
by an LP on the Star's own rows and checked exactly), where a wrong divisor or
a lost entry shows.

Defects of today's library marked here (registry ids of
``.claude/research/34-API-Redesign/test-audit/DEFECTS.md``):

- SHAPE-1: a model input with no batch axis (declared (2, 3), (3, 3), (4, 4)
  or (6,)) keeps the placeholder (1, *input_shape), so a non-negative axis
  names the axis to its left; a rank-1 model reached without ``input_shape``
  keeps (1, n) the same way; and an ``input_shape`` that disagrees with the
  declared input is answered (the registry's fix raises there).
- SHAPE-3: an ``ImageZono`` without ``input_shape`` gets the flat placeholder.
- SHAPE-2: a flat set on a model whose input has more than one axis besides
  the batch axis, without ``input_shape``, is read as the tensor (1, dim).
- GRAPH-8: a graph output that is directly one output of a Split comes back
  as a list holding lists.
- LOAD-1: an opset-18 Reduce node other than ReduceSum (axes given as a second
  input) cannot be loaded; the converter's ``TypeError`` comes out.
- SHAPE-9: a reach changes the model it is
  given, because the shape pass runs vendored forwards that store what they
  resolve; marked here for a ReduceMean over every axis (the row names the
  Transpose, marked in ``test_soundness_transpose.py``).

Every marked cell has passing neighbours built by the same builder, replayed
by the same onnxruntime call and checked by the same oracle: negative axes,
the same model with a batch axis, the same call with ``input_shape``, the
(1, 6) model without ``input_shape``, the same graph with a Relu after the
Split output, the attribute form of Reduce.

Not here: the handlers' own cells (their files: ``test_soundness_softmax.py``,
``test_soundness_reduce.py``, ``test_soundness_concat.py``,
``test_soundness_slice.py``, ``test_soundness_gather.py``,
``test_soundness_split.py``, ``test_soundness_transpose.py``,
``test_soundness_matmul.py``), with their exact cells and their refusals of
Hexatope and Octatope; a record lost because one node's forward raises at
the declared input (SHAPE-8, marked in those files).
"""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog

import n2v.nn.reach as reach_module
from n2v.nn.neural_network import NeuralNetwork
from n2v.sets import Box, HalfSpace, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles.membership import LP_TOL, hwc, rows
from tests.cell_support import (N_SAMPLES, assert_covered,
                                constrained_image_star, constrained_star,
                                flat_box, flat_set, hwc_index, image_set,
                                pixel_box, reach_model, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_SET_TYPES = (Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope)

# Flat and image input boxes are the dyadic boxes of cell_support times this.
_SCALE = 8.0

_SHAPE_1 = known_bug(
    12, "a model input with no batch axis is analysed through the placeholder "
        "(1, *input_shape), whose forward also runs, so every non-negative "
        "axis names the axis to its left: Softmax, Reduce, Concat, Slice, "
        "Gather and Split return sets that leave out the model's outputs or "
        "hold the wrong number of entries (registry SHAPE-1)")
_SHAPE_1_RANK_1 = known_bug(
    12, "a rank-1 model input (6,) reached with a flat set and no input_shape "
        "keeps the placeholder (1, 6), whose forward also runs, so axis 0 "
        "names a batch axis the model does not have: a Softmax over axis 0 "
        "returns all ones (registry SHAPE-1, through the flat guess)")
_SHAPE_1_DISAGREES = known_bug(
    12, "an input_shape that disagrees with the model's declared input (3, 2) "
        "for a declared (1, 2, 3) is not refused: the placeholder (1, 3, 2) "
        "runs and every axis reads the wrong tensor (registry SHAPE-1, whose "
        "fix raises when input_shape disagrees with the declared shape)",
    raises=DID_NOT_RAISE)
_SHAPE_3 = known_bug(
    12, "an ImageZono given without input_shape gets the flat placeholder "
        "(1, dim) instead of (1, C, H, W), so Reduce chains, Gather and "
        "Flatten then Reduce read the wrong tensor: wrong entry counts, and "
        "a three-Reduce chain of the right size that leaves out the outputs "
        "(registry SHAPE-3)")
_SHAPE_2 = known_bug(
    12, "a flat set on a model whose input has rank 3 or 4, given without "
        "input_shape, is analysed as the tensor (1, dim) with no error: "
        "Reduce, reversing Slice, Concat and Split return sets that leave out "
        "the outputs or hold the wrong number of entries (registry SHAPE-2)")
_SHAPE_2_RANK_2 = known_bug(
    12, "a flat set on a model whose input (2, 3) has no batch axis, given "
        "without input_shape, is analysed as the tensor (1, 6) with no error: "
        "a Softmax over the last axis is computed over all six entries "
        "(registry SHAPE-2, the same (1, dim) guess on a rank-2 input; the "
        "fix of SHAPE-1, which runs the declared shape, also removes it)")
_GRAPH_8 = known_bug(
    16, "a graph output that is directly one output of a Split node comes "
        "back as a list holding one list per Split output instead of the "
        "sets of the named output (registry GRAPH-8)")
_LOAD_1 = known_bug(
    23, "an opset-18 Reduce node with its axes given as a second input "
        "(ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, "
        "ReduceSumSquare, ReduceLogSum, ReduceLogSumExp) is converted with "
        "the version-13 converter, which raises TypeError instead of a clear "
        "refusal (registry LOAD-1)", raises=TypeError)


# --------------------------------------------------------------------------- #
# Graph builders: each returns (nodes, initializers) for the element type.    #
# --------------------------------------------------------------------------- #
def _i64(values):
    return np.asarray(values, dtype=np.int64)


def _softmax(axis):
    return lambda dt: ([helper.make_node("Softmax", ["X"], ["Y"], axis=axis)],
                       {})


def _reduce_attr(op, axes, keepdims):
    """Reduce with ``axes`` as an attribute (ReduceSum up to opset 12, the
    others up to opset 17)."""
    return lambda dt: ([helper.make_node(op, ["X"], ["Y"], axes=list(axes),
                                         keepdims=keepdims)], {})


def _reduce_input(op, axes, keepdims):
    """Reduce with ``axes`` as a second input (ReduceSum from opset 13, the
    others from opset 18)."""
    return lambda dt: ([helper.make_node(op, ["X", "axes"], ["Y"],
                                         keepdims=keepdims)],
                       {"axes": _i64(axes)})


def _chain(*steps):
    """Reduce nodes in a row, ``(op, axes, keepdims)`` each, axes as
    attributes."""
    def build(dt):
        names = ["X"] + [f"T{k}" for k in range(len(steps) - 1)] + ["Y"]
        return ([helper.make_node(op, [names[k]], [names[k + 1]],
                                  axes=list(axes), keepdims=kd)
                 for k, (op, axes, kd) in enumerate(steps)], {})
    return build


def _concat_double(axis):
    """Concat(X, X + X) along ``axis``."""
    return lambda dt: ([helper.make_node("Add", ["X", "X"], ["X2"]),
                        helper.make_node("Concat", ["X", "X2"], ["Y"],
                                         axis=axis)], {})


def _concat_relu_neg(axis):
    """Concat(Relu(X), Neg(X)) along ``axis``."""
    return lambda dt: ([helper.make_node("Relu", ["X"], ["A"]),
                        helper.make_node("Neg", ["X"], ["B"]),
                        helper.make_node("Concat", ["A", "B"], ["Y"],
                                         axis=axis)], {})


def _slice(axis, start, end):
    return lambda dt: ([helper.make_node("Slice", ["X", "st", "en", "ax"],
                                         ["Y"])],
                       {"st": _i64([start]), "en": _i64([end]),
                        "ax": _i64([axis])})


def _relu_slice_reversed(axis):
    """Relu, then the whole of ``axis`` read backwards (step -1)."""
    lowest = np.iinfo(np.int64).min
    return lambda dt: ([helper.make_node("Relu", ["X"], ["A"]),
                        helper.make_node("Slice",
                                         ["A", "st", "en", "ax", "sp"],
                                         ["Y"])],
                       {"st": _i64([-1]), "en": _i64([lowest]),
                        "ax": _i64([axis]), "sp": _i64([-1])})


def _gather(axis, indices=(0, 2)):
    return lambda dt: ([helper.make_node("Gather", ["X", "idx"], ["Y"],
                                         axis=axis)], {"idx": _i64(indices)})


def _split(axis, sizes, keep, after=None):
    """Split along ``axis`` into ``sizes``; part ``keep`` is the graph output
    itself (``after=None``) or goes through a node of type ``after``."""
    def build(dt):
        parts = [f"P{k}" for k in range(len(sizes))]
        if after is None:
            parts[keep] = "Y"
            return ([helper.make_node("Split", ["X", "sp"], parts,
                                      axis=axis)], {"sp": _i64(sizes)})
        return ([helper.make_node("Split", ["X", "sp"], parts, axis=axis),
                 helper.make_node(after, [parts[keep]], ["Y"])],
                {"sp": _i64(sizes)})
    return build


def _split_then_softmax(axis, sizes, keep, softmax_axis, reshape=None):
    """Split along ``axis``, part ``keep`` (reshaped to ``reshape`` when
    given), then Softmax over ``softmax_axis``: the Softmax reads the record
    of a ``getitem`` node (or of the Reshape behind one)."""
    def build(dt):
        parts = [f"P{k}" for k in range(len(sizes))]
        nodes = [helper.make_node("Split", ["X", "sp"], parts, axis=axis)]
        inits = {"sp": _i64(sizes)}
        src = parts[keep]
        if reshape is not None:
            nodes.append(helper.make_node("Reshape", [src, "shape"], ["R"]))
            inits["shape"] = _i64(reshape)
            src = "R"
        nodes.append(helper.make_node("Softmax", [src], ["Y"],
                                      axis=softmax_axis))
        return nodes, inits
    return build


def _gemm_reshape_softmax(target, axis):
    """Gemm by a dyadic 6 x 6 weight (an initializer of the file's own
    element type), Reshape to ``target``, Softmax over ``axis``."""
    weight = np.random.default_rng(5).integers(-4, 5, size=(6, 6)) / 4.0
    return lambda dt: ([helper.make_node("Gemm", ["X", "W"], ["M"], transB=1),
                        helper.make_node("Reshape", ["M", "shape"], ["R"]),
                        helper.make_node("Softmax", ["R"], ["Y"], axis=axis)],
                       {"W": weight.astype(dt), "shape": _i64(target)})


def _softmax_then_reduce(softmax_axis, axes, keepdims):
    """Softmax, then ReduceSum (axes as an attribute: opset 12 or lower)."""
    return lambda dt: ([helper.make_node("Softmax", ["X"], ["S"],
                                         axis=softmax_axis),
                        helper.make_node("ReduceSum", ["S"], ["Y"],
                                         axes=list(axes),
                                         keepdims=keepdims)], {})


def _transpose(perm):
    return lambda dt: ([helper.make_node("Transpose", ["X"], ["Y"],
                                         perm=list(perm))], {})


def _matmul(weight, weight_first):
    """MatMul(X, W) or MatMul(W, X) with a dyadic constant W."""
    def build(dt):
        operands = ["W", "X"] if weight_first else ["X", "W"]
        return ([helper.make_node("MatMul", operands, ["Y"])],
                {"W": np.asarray(weight, dtype=dt)})
    return build


def _flatten(axis):
    return lambda dt: ([helper.make_node("Flatten", ["X"], ["Y"], axis=axis)],
                       {})


def _flatten_then_sum(axis):
    """Flatten(axis), then ReduceSum over axis 1 without keepdims (axes as an
    attribute: opset 12 or lower)."""
    return lambda dt: ([helper.make_node("Flatten", ["X"], ["F"], axis=axis),
                        helper.make_node("ReduceSum", ["F"], ["Y"], axes=[1],
                                         keepdims=0)], {})


def _reshape_softmax(target, axis):
    return lambda dt: ([helper.make_node("Reshape", ["X", "shape"], ["R"]),
                        helper.make_node("Softmax", ["R"], ["Y"], axis=axis)],
                       {"shape": _i64(target)})


def _x_minus_mean(axes):
    """X - ReduceMean(X, axes, keepdims=1)."""
    return lambda dt: ([helper.make_node("ReduceMean", ["X"], ["M"],
                                         axes=list(axes), keepdims=1),
                        helper.make_node("Sub", ["X", "M"], ["Y"])], {})


# --------------------------------------------------------------------------- #
# Model files, built once per (graph, element type) and cached per module.    #
# --------------------------------------------------------------------------- #
_ELEMENT = {"float32": (TensorProto.FLOAT, np.float32),
            "float64": (TensorProto.DOUBLE, np.float64)}


def _ort_output_shape(nodes, inits, in_shape, opset, dtype):
    """The output shape, by onnxruntime on an unchecked copy of the graph
    whose output shape is left open."""
    elem, np_type = _ELEMENT[dtype]
    graph = helper.make_graph(
        nodes, "probe", [helper.make_tensor_value_info("X", elem,
                                                       list(in_shape))],
        [helper.make_tensor_value_info("Y", elem, None)],
        initializer=[numpy_helper.from_array(v, k) for k, v in inits.items()])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(),
                                   sess_options=options,
                                   providers=["CPUExecutionProvider"])
    return tuple(session.run(None, {"X": np.zeros(in_shape, np_type)})[0]
                 .shape)


class _Graph:
    """One test graph: ``path`` is the file n2v loads (float32 or float64),
    ``reference`` the float64 twin whose onnxruntime replay gives the true
    outputs, ``out_shape`` the output tensor shape (by onnxruntime)."""

    def __init__(self, path, reference, out_shape):
        self.path, self.reference, self.out_shape = path, reference, out_shape


class _Models:
    """ONNX files of this module, written once into a module temp dir."""

    def __init__(self, folder):
        self.folder = folder
        self.files = {}

    def _file(self, key, in_shape, opset, build, dtype):
        name = f"{key}-{dtype}".replace(" ", "_").replace(",", "_")
        if name not in self.files:
            nodes, inits = build(_ELEMENT[dtype][1])
            out_shape = _ort_output_shape(nodes, inits, in_shape, opset, dtype)
            elem = _ELEMENT[dtype][0]
            path = save_onnx(self.folder, nodes, in_shape, out_shape,
                             inits=inits, name=name, opset=opset,
                             in_type=elem, out_type=elem)
            self.files[name] = (path, out_shape)
        return self.files[name]

    def get(self, key, in_shape, opset, build, dtype="float32"):
        """The :class:`_Graph` of ``build`` on ``in_shape`` in ``dtype``."""
        path, out_shape = self._file(key, in_shape, opset, build, dtype)
        reference, ref_shape = self._file(key, in_shape, opset, build,
                                          "float64")
        if tuple(ref_shape) != tuple(out_shape):
            raise RuntimeError(f"{key}: the float64 twin has output shape "
                               f"{ref_shape}, the {dtype} graph {out_shape}")
        return _Graph(path, reference, out_shape)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return _Models(tmp_path_factory.mktemp("tensor_shape_models"))


# --------------------------------------------------------------------------- #
# Input sets, the call and the promise                                         #
# --------------------------------------------------------------------------- #
def _flat_box(n, seed):
    lo, hi = flat_box(n, seed=seed, zero_width=True)
    return lo * _SCALE, hi * _SCALE


def _flat_input(kind, n, seed):
    """A flat set of ``kind`` on dyadic bounds (the cell_support box times 8)
    with entries on both sides of zero and one of zero width, its bounds and
    member points computed without n2v (for a Star: a binding row that is
    not a box row)."""
    lo, hi = _flat_box(n, seed)
    if kind == "Star":
        star, points = constrained_star(lo, hi, n=N_SAMPLES, seed=seed)
        return star, points, lo, hi
    return flat_set(kind, lo, hi), samples(lo, hi, N_SAMPLES, seed=seed), lo, hi


def _image_input(kind, chw, seed):
    """An image set over a different dyadic interval per pixel (the
    cell_support pixel box times 8), its bounds and member points as
    (C, H, W) tensors (an ImageStar with a binding row)."""
    lo, hi = pixel_box(*chw, seed=seed)
    lo, hi = lo * _SCALE, hi * _SCALE
    if kind == "ImageStar":
        image, points = constrained_image_star(lo, hi, n=N_SAMPLES, seed=seed)
        return image, points, lo, hi
    return (image_set(kind, lo, hi), samples(lo, hi, N_SAMPLES, seed=seed),
            lo, hi)


def _extreme_members(reference, shape, kind, input_set, lo, hi):
    """Member points of ``input_set`` at which an output entry that adds up
    several input entries is largest and smallest, for a graph whose output
    is a linear map of its input with no constant term. The map is read from
    onnxruntime on the unit vectors. For a box set (Box, Zono, ImageZono) the
    points are the corners chosen by the signs of the entry's row; for a Star
    or an ImageStar each is a vertex of the predicate polytope that an LP
    over the set's own rows (scipy, never n2v) finds, snapped to the grid of
    1/1024 (the vertices of these dyadic polytopes lie on it) and checked
    exactly against every row. Points are in ``lo``'s shape and order."""
    tensor = np.shape(lo)
    n = int(np.prod(tensor))
    columns = replay_onnx(reference, list(np.eye(n)), shape)
    directions = [w for w in np.stack(columns, axis=1)
                  if np.count_nonzero(w) > 1]
    points = []
    if kind not in ("Star", "ImageStar"):
        low, high = np.reshape(lo, -1), np.reshape(hi, -1)
        for w in directions:
            for sign in (1.0, -1.0):
                points.append(np.where(sign * w > 0, high, low)
                              .reshape(tensor))
        return points
    c, G, C, d, lb, ub = rows(input_set)
    order = hwc_index(*tensor) if kind == "ImageStar" else np.arange(n)
    for w in directions:
        for sign in (1.0, -1.0):
            gain = G.T @ (sign * w)[order]
            solved = linprog(-gain, A_ub=C, b_ub=d,
                             bounds=list(zip(lb, ub)), method="highs")
            if solved.status != 0:
                pytest.fail(f"setup: the LP for an extreme member point "
                            f"stopped: {solved.message}")
            a = np.clip(np.round(solved.x * 1024.0) / 1024.0, lb, ub)
            if np.any(C @ a > d):
                pytest.fail("setup: the snapped vertex is not a member")
            stored = c + G @ a
            point = np.empty(n)
            point[order] = stored
            points.append(point.reshape(tensor))
    return points


def _reach(entry, model, input_set, **options):
    if entry == "reach_pytorch_model":
        return reach_model(model, input_set, "approx",
                           f"reach_pytorch_model of {type(input_set).__name__}",
                           **options)
    if entry == "NeuralNetwork.reach":
        return NeuralNetwork(model).reach(input_set, method="approx",
                                          **options)
    raise RuntimeError(f"test setup: unknown entry point {entry!r}")


def _keeps_the_promise(call, outputs, what, out_shape, tol=LP_TOL):
    """A refusal, or a flat non-empty list of sets that holds every output.
    Returns the sets, or None when the call refused."""
    try:
        sets = call()
    except (NotImplementedError, ValueError):
        return None
    assert isinstance(sets, (list, tuple)) and len(sets) > 0, (
        f"{what}: the result is {type(sets).__name__} of length "
        f"{len(sets) if hasattr(sets, '__len__') else '?'}, not a non-empty "
        f"list of sets (the input set is not empty)")
    stray = [type(s).__name__ for s in sets if not isinstance(s, _SET_TYPES)]
    assert not stray, (f"{what}: the result holds {stray}, not sets: a flat "
                       f"list of sets is due")
    out_chw = tuple(out_shape[1:]) if len(out_shape) == 4 else None
    far = _outside_every_outer_box(sets, outputs, out_chw, tol)
    assert not far, (
        f"{what}: {far} of {len(outputs)} true outputs lie outside the outer "
        f"box of every returned set, so they are in none of them; result: "
        f"{', '.join(sorted({type(s).__name__ for s in sets}))}")
    assert_covered(sets, outputs, what, out_chw=out_chw, tol=tol)
    return sets


def _outside_every_outer_box(sets, outputs, out_chw, tol):
    """How many outputs lie outside the box ``c + G [lb, ub]`` (the set's rows
    ``C a <= d`` dropped, so a superset of the set) of every returned set, by
    more than ``tol`` and a pad of 1e-9 of the box's size. A fast first pass
    that can only fail outputs the LP oracle would fail too; it returns 0
    (and leaves the verdict to ``assert_covered``) whenever a set has an open
    predicate bound or the sizes disagree."""
    boxes = []
    for s in sets:
        c, G, _, _, lb, ub = rows(s)
        if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
            return 0
        centre = c + G @ ((lb + ub) / 2)
        radius = np.abs(G) @ ((ub - lb) / 2)
        pad = 1e-9 * (np.abs(centre) + radius + 1.0)
        boxes.append((centre - radius - pad, centre + radius + pad,
                      out_chw is not None
                      and isinstance(s, (ImageStar, ImageZono))))
    far = 0
    for y in outputs:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        slack = tol * max(1.0, float(np.max(np.abs(y), initial=0.0)))
        inside = False
        for low, high, image in boxes:
            v = hwc(y, *out_chw) if image else y
            if v.size != low.size:
                return 0
            if np.all(v >= low - slack) and np.all(v <= high + slack):
                inside = True
                break
        far += not inside
    return far


# =========================================================================== #
# 0. The record itself                                                         #
# =========================================================================== #
class _TrueShapes(torch.fx.Interpreter):
    """The shape of every tensor-valued node of a converted model when torch
    runs it on a zero tensor of the model's declared input shape: the shapes
    the ONNX file defines, computed without n2v."""

    def __init__(self, module):
        super().__init__(module)
        self.shapes = {}

    def run_node(self, n):
        value = super().run_node(n)
        if isinstance(value, torch.Tensor):
            self.shapes[n.name] = tuple(int(d) for d in value.shape)
        return value


_IMG = (1, 3, 4, 5)

# id -> (declared input, opset, builder, dtype, set kind, options, whether the
# caller's information fixes the tensor, marks). The tensor is fixed when
# input_shape is given, when the set is an image set (it carries C, H, W),
# or when the declared input is (n,) or (1, n) and the set is flat.
_RECORDS = {
    "batch-axis-input_shape": (
        (1, 3, 3), 11, _reduce_attr("ReduceSum", [1], 0), "float32", "Box",
        {"input_shape": (3, 3)}, True, []),
    "batch-axis-input_shape-NeuralNetwork.reach": (
        (1, 3, 3), 11, _reduce_attr("ReduceSum", [1], 0), "float32", "Box",
        {"input_shape": (3, 3), "entry": "NeuralNetwork.reach"}, True, []),
    "no-batch-axis-prefixed-placeholder-runs": (
        (3, 3), 13, _slice(1, 0, 2), "float32", "Box",
        {"input_shape": (3, 3)}, True, [_SHAPE_1]),
    "no-batch-axis-prefixed-placeholder-raises": (
        (4, 4), 13, _split_then_softmax(0, [2, 2], 1, 0), "float32", "Box",
        {"input_shape": (4, 4)}, True, []),
    "split-part-getitem": (
        (1, 4, 4), 13, _split_then_softmax(1, [2, 2], 1, 1), "float32", "Box",
        {"input_shape": (4, 4)}, True, []),
    "float64-weights": (
        (1, 6), 13, _gemm_reshape_softmax([1, 2, 3], 1), "float64", "Box",
        {"input_shape": (6,)}, True, []),
    "row-vector-no-input_shape": (
        (1, 6), 11, _softmax_then_reduce(-1, [0], 1), "float32", "Box",
        {}, True, []),
    "rank-1-vector-fallback-no-input_shape": (
        (8,), 13, _split_then_softmax(0, [2, 6], 1, 1, reshape=[1, 2, 3]),
        "float32", "Box", {}, True, []),
    "rank-1-prefixed-placeholder-runs-no-input_shape": (
        (6,), 13, _softmax(0), "float32", "Box", {}, True, [_SHAPE_1_RANK_1]),
    "rank-3-flat-no-input_shape": (
        (1, 3, 4), 13, _x_minus_mean([-1]), "float32", "Box", {}, False,
        [_SHAPE_2]),
    "ImageStar-no-input_shape": (
        _IMG, 11, _chain(("ReduceMean", [3], 0), ("ReduceSum", [1], 0)),
        "float32", "ImageStar", {}, True, []),
    "ImageZono-no-input_shape": (
        _IMG, 11, _chain(("ReduceMean", [3], 0), ("ReduceSum", [1], 0)),
        "float32", "ImageZono", {}, True, [_SHAPE_3]),
    "ImageZono-input_shape": (
        _IMG, 11, _chain(("ReduceMean", [3], 0), ("ReduceSum", [1], 0)),
        "float32", "ImageZono", {"input_shape": _IMG[1:]}, True, []),
}


@pytest.mark.parametrize("key", [pytest.param(k, marks=v[7], id=k)
                                 for k, v in _RECORDS.items()])
def test_the_tensor_shape_record_is_the_one_the_model_defines(models,
                                                              monkeypatch, key):
    """The record the walk takes from ``_propagate_shapes`` (read by a spy on
    the module function, through the model entry), against the shapes torch
    computes when the converted model runs on its declared input. Promise:
    the reach refuses, or every node the record names holds its true shape,
    and, where the caller's information fixes the tensor, every tensor node
    is named. Catches a kept placeholder of the wrong shape (the unbatched
    shape tried first, a flat guess tried in the wrong order, height and
    width exchanged, a recorder that strips the leading axis, ``input_shape``
    lost on the way to the pass or by ``NeuralNetwork.reach``) and a record
    left empty or partial where a placeholder exists (no unbatched fallback,
    a float32 placeholder for a float64 model with weights, the ``getitem``
    nodes of a Split left out)."""
    shape, opset, build, dtype, kind, options, fixed, _ = _RECORDS[key]
    options = dict(options)
    entry = options.pop("entry", "reach_pytorch_model")
    graph = models.get(f"record-{key}", shape, opset, build, dtype)
    truth_model = load_onnx(graph.path)
    runner = _TrueShapes(truth_model)
    with torch.no_grad():
        runner.run(torch.zeros(shape, dtype=getattr(torch, dtype)))
    truth = runner.shapes
    output = [n for n in truth_model.graph.nodes if n.op == "output"][0]
    last = output.args[0]
    if truth.get(getattr(last, "name", None)) != tuple(graph.out_shape):
        pytest.fail(f"setup: the converted model's output shape "
                    f"{truth.get(getattr(last, 'name', None))} is not "
                    f"onnxruntime's {graph.out_shape}")
    if kind in ("ImageStar", "ImageZono"):
        input_set = _image_input(kind, shape[1:], seed=3)[0]
    else:
        input_set = _flat_input(kind, int(np.prod(shape)), seed=3)[0]
    seen = []
    original = reach_module._propagate_shapes

    def spy(*args, **kwargs):
        record = original(*args, **kwargs)
        seen.append(dict(record))
        return record
    monkeypatch.setattr(reach_module, "_propagate_shapes", spy)
    try:
        _reach(entry, load_onnx(graph.path), input_set, **options)
    except (NotImplementedError, ValueError):
        return
    if len(seen) != 1:
        pytest.fail(f"setup: the walk called the shape pass {len(seen)} "
                    f"times; this cell reads the record of one call")
    record = seen[0]
    wrong = {name: (tuple(got), truth[name]) for name, got in record.items()
             if name in truth and tuple(got) != truth[name]}
    assert not wrong, (f"{key}: the record the walk used gives wrong shapes "
                       f"(recorded, true): {wrong}")
    if fixed:
        missing = sorted(name for name in truth if name not in record)
        assert not missing, (f"{key}: the record names no shape for {missing}, "
                             f"which the model defines on its declared input")


# =========================================================================== #
# 1. A model input with no batch axis (SHAPE-1)                                #
# =========================================================================== #
_F32 = (("reach_pytorch_model", "float32", "Box"),
        ("reach_pytorch_model", "float32", "Star"),
        ("reach_pytorch_model", "float32", "Zono"))
_BOX = (("reach_pytorch_model", "float32", "Box"),)
_NN64 = (("NeuralNetwork.reach", "float64", "Box"),)

# id -> (declared input, opset, builder, fails today, calls). The "fails
# today" flag is exactly where the placeholder (1, *input_shape) runs and
# moves the axis. On a (2, 3) input axis -2 names the same tensor axis as 0
# and -1 the same as 1: the negative axes are the passing neighbours.
_NO_BATCH = {}


def _no_batch(key, shape, opset, build, fails, calls):
    _NO_BATCH[key] = (shape, opset, build, fails, calls)


_no_batch("softmax13-axis1", (2, 3), 13, _softmax(1), True, _F32)
_no_batch("softmax13-axis-2", (2, 3), 13, _softmax(-2), False, _F32)
_no_batch("softmax13-axis-1", (2, 3), 13, _softmax(-1), False, _F32 + _NN64)
# opset 11 flattens to 2-D at the axis: axis 0 is [1, 6] for both the declared
# (2, 3) and the placeholder (1, 2, 3), so only axis 1 moves. Axis 0 is also
# the value a ``value or default`` slip would replace by the default axis 1,
# which on (2, 3) selects other entries.
_no_batch("softmax11-axis0", (2, 3), 11, _softmax(0), False, _F32)
_no_batch("softmax11-axis1", (2, 3), 11, _softmax(1), True, _BOX)
_no_batch("softmax11-axis-1", (2, 3), 11, _softmax(-1), False, _F32)
for _ax in (0, 1):
    _no_batch(f"reducesum11-axis{_ax}-keepdims0", (2, 3), 11,
              _reduce_attr("ReduceSum", [_ax], 0), True, _F32)
    _no_batch(f"reducemean13-axis{_ax}-keepdims0", (3, 3), 13,
              _reduce_attr("ReduceMean", [_ax], 0), True, _F32)
_no_batch("reducesum11-axis-2-keepdims0", (2, 3), 11,
          _reduce_attr("ReduceSum", [-2], 0), False, _BOX)
_no_batch("reducesum11-axis-1-keepdims0", (2, 3), 11,
          _reduce_attr("ReduceSum", [-1], 0), False, _F32)
_no_batch("reducemean13-axis-2-keepdims0", (3, 3), 13,
          _reduce_attr("ReduceMean", [-2], 0), False, _BOX)
_no_batch("reducemean13-axis-1-keepdims0", (3, 3), 13,
          _reduce_attr("ReduceMean", [-1], 0), False, _F32)
_no_batch("reducemean13-axis-1-keepdims1", (3, 3), 13,
          _reduce_attr("ReduceMean", [-1], 1), False, _BOX)
_no_batch("concat-axis1", (2, 3), 13, _concat_double(1), True, _F32)
_no_batch("concat-axis-2", (2, 3), 13, _concat_double(-2), False, _BOX)
_no_batch("concat-axis-1", (2, 3), 13, _concat_double(-1), False, _F32)
_no_batch("slice-axis1", (3, 3), 13, _slice(1, 0, 2), True, _BOX)
_no_batch("slice-axis-2", (3, 3), 13, _slice(-2, 0, 2), False, _BOX)
_no_batch("slice-axis-1", (3, 3), 13, _slice(-1, 0, 2), False, _F32)
# Gather [0, 2] and Split [2, 2] over axis 0 cannot run on (1, 3, 3) or
# (1, 4, 4), so the declared shape is kept there: a record with no batch axis.
_no_batch("gather-axis0", (3, 3), 13, _gather(0), False, _BOX)
_no_batch("gather-axis1", (3, 3), 13, _gather(1), True, _F32 + _NN64)
_no_batch("gather-axis-1", (3, 3), 13, _gather(-1), False, _F32)
for _ax, _fails in ((0, False), (1, True), (-1, False)):
    _no_batch(f"split-relu-axis{_ax}", (4, 4), 13,
              _split(_ax, [2, 2], 1, after="Relu"), _fails, _F32)
_no_batch("rank1-softmax13-axis-1", (6,), 13, _softmax(-1), False, _BOX)
_no_batch("rank1-reducesum11-axis-1", (6,), 11,
          _reduce_attr("ReduceSum", [-1], 1), False, _BOX)
_no_batch("rank1-slice-axis0", (6,), 13, _slice(0, 0, 4), True, _BOX)
_no_batch("rank1-slice-axis-1", (6,), 13, _slice(-1, 0, 4), False, _BOX)
# Transpose [1, 0] cannot run on (1, 2, 3): the declared shape is kept.
_no_batch("transpose", (2, 3), 13, _transpose([1, 0]), False, _F32)
_no_batch("matmul-x-w", (2, 3), 13,
          _matmul([[0.5, -1.0, 0.25, 2.0], [1.0, 0.75, -0.5, 0.0],
                   [-0.25, 1.5, 1.0, -1.0]], weight_first=False), False,
          _F32 + _NN64)
_no_batch("matmul-w-x", (2, 3), 13,
          _matmul([[0.5, -1.0], [1.0, 0.75], [-0.25, 1.5], [2.0, -0.5]],
                  weight_first=True), False, _F32 + _NN64)
_no_batch("flatten-axis1", (2, 3), 13, _flatten(1), False, _BOX)
_no_batch("reshape-then-softmax", (2, 3), 13,
          _reshape_softmax([-1, 2, 3], 1), False, _F32)


def _no_batch_cells():
    cells = []
    for key, (shape, opset, build, fails, calls) in _NO_BATCH.items():
        for entry, dtype, kind in calls:
            cells.append(pytest.param(
                key, entry, dtype, kind, marks=[_SHAPE_1] if fails else [],
                id=f"{key}-{entry}-{dtype}-{kind}"))
    return cells


@pytest.mark.parametrize("key,entry,dtype,kind", _no_batch_cells())
def test_a_model_without_a_batch_axis_holds_every_output(models, key, entry,
                                                          dtype, kind):
    """An ONNX model whose declared input has no batch axis, (2, 3), (3, 3),
    (4, 4) or (6,), reached with ``input_shape`` equal to the declared shape
    (what the competition runner passes when the first dimension is not 1).
    Catches the placeholder ``(1, *input_shape)`` being kept, which moves
    every non-negative axis one place to the left (SHAPE-1), the unbatched
    fallback lost where the prefixed forward raises (Gather and Split over
    axis 0, Transpose: the record then has no batch axis, and a handler that
    takes axis 0 for a batch axis is wrong there), and any handler that reads
    its axis against the wrong recorded shape; the negative axes, Transpose,
    MatMul, Flatten and the graph that restores the batch axis first are the
    sound neighbours. ``NeuralNetwork.reach`` hands its options on."""
    shape, opset, build, _, _ = _NO_BATCH[key]
    graph = models.get(f"nobatch-{key}", shape, opset, build, dtype)
    input_set, points, _, _ = _flat_input(kind, int(np.prod(shape)),
                                          seed=len(key))
    outputs = replay_onnx(graph.reference, points, shape)
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach(entry, model, input_set, input_shape=shape), outputs,
        f"{key} on declared input {shape} ({entry}, {dtype}, {kind})",
        graph.out_shape)


# The same operations on a model that has the batch axis, reached the way the
# runner reaches it (input_shape without the batch axis): the axis that moves
# above is the axis + 1 here and names the same tensor axis.
_WITH_BATCH = {
    "softmax13-axis2": ((1, 2, 3), 13, _softmax(2)),
    "softmax11-axis2": ((1, 2, 3), 11, _softmax(2)),
    "reducesum11-axis2-keepdims0": ((1, 2, 3), 11,
                                    _reduce_attr("ReduceSum", [2], 0)),
    "reducesum11-axis1-keepdims1": ((1, 2, 3), 11,
                                    _reduce_attr("ReduceSum", [1], 1)),
    "reducemean13-axis2-keepdims0": ((1, 3, 3), 13,
                                     _reduce_attr("ReduceMean", [2], 0)),
    "reducemean13-axis1-keepdims1": ((1, 3, 3), 13,
                                     _reduce_attr("ReduceMean", [1], 1)),
    "concat-axis2": ((1, 2, 3), 13, _concat_double(2)),
    "slice-axis1": ((1, 3, 3), 13, _slice(1, 0, 2)),
    "slice-axis2": ((1, 3, 3), 13, _slice(2, 0, 2)),
    "gather-axis2": ((1, 3, 3), 13, _gather(2)),
    "split-relu-axis2": ((1, 4, 4), 13, _split(2, [2, 2], 1, after="Relu")),
    "rank1-softmax13-axis1": ((1, 6), 13, _softmax(1)),
    "rank1-slice-axis1": ((1, 6), 13, _slice(1, 0, 4)),
}


@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("key", sorted(_WITH_BATCH))
def test_the_same_model_with_a_batch_axis_holds_every_output(models, key,
                                                              kind):
    """The neighbour of the cells above: the batch axis is declared, so the
    placeholder ``(1, *input_shape)`` is the model's input. Catches the
    opposite slip (the unbatched shape tried first, so the recorded shapes
    are one axis short) and an axis normalisation that is off by one in the
    shape-aware handlers."""
    shape, opset, build = _WITH_BATCH[key]
    graph = models.get(f"batch-{key}", shape, opset, build)
    input_set, points, _, _ = _flat_input(kind, int(np.prod(shape)),
                                          seed=len(key))
    outputs = replay_onnx(graph.reference, points, shape)
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach("reach_pytorch_model", model, input_set,
                       input_shape=shape[1:]), outputs,
        f"{key} on declared input {shape} ({kind})", graph.out_shape)


# Without input_shape, on the models of the two dicts above: the flat set's
# placeholder (1, dim) is the model's input only for a (1, n) model.
_NO_INPUT_SHAPE = (
    # (graph family, key, calls, marks)
    ("nobatch", "rank1-softmax13-axis0", _BOX, [_SHAPE_1_RANK_1]),
    ("nobatch", "rank1-softmax13-axis-1", _BOX, []),
    ("nobatch", "softmax13-axis-1", (("reach_pytorch_model", "float32",
                                      "Star"),), [_SHAPE_2_RANK_2]),
    ("batch", "rank1-softmax13-axis1", _BOX, []),
    ("batch", "row-reducesum11-axis0-keepdims1",
     _F32 + (("NeuralNetwork.reach", "float32", "Box"),), []),
)
_EXTRA_GRAPHS = {
    "nobatch-rank1-softmax13-axis0": ((6,), 13, _softmax(0)),
    "batch-row-reducesum11-axis0-keepdims1": (
        (1, 6), 11, _reduce_attr("ReduceSum", [0], 1)),
}


def _graph_of(models, family, key, dtype="float32"):
    name = f"{family}-{key}"
    if name in _EXTRA_GRAPHS:
        shape, opset, build = _EXTRA_GRAPHS[name]
    elif family == "nobatch":
        shape, opset, build = _NO_BATCH[key][:3]
    else:
        shape, opset, build = _WITH_BATCH[key]
    return shape, models.get(name, shape, opset, build, dtype)


def _no_input_shape_cells():
    cells = []
    for family, key, calls, marks in _NO_INPUT_SHAPE:
        for entry, dtype, kind in calls:
            cells.append(pytest.param(family, key, entry, kind, marks=marks,
                                      id=f"{family}-{key}-{entry}-{kind}"))
    return cells


@pytest.mark.parametrize("family,key,entry,kind", _no_input_shape_cells())
def test_a_flat_set_without_input_shape_on_a_rank_1_or_2_input_refuses_or_holds_every_output(
        models, family, key, entry, kind):
    """A flat set reached without ``input_shape`` on a model whose declared
    input is (6,), (2, 3) or (1, 6). The (1, 6) model is the common route of
    a flat set and its placeholder (1, 6) is right; the (6,) model keeps the
    placeholder (1, 6) whenever its forward runs (SHAPE-1), and the (2, 3)
    model is read as (1, 6) (SHAPE-2). Catches the flat guess tried in the
    other order (the vector (6,) first: a ReduceSum over the batch axis of
    the (1, 6) model then sums every entry) and an axis 0 that a rank-1
    model reads as a batch axis."""
    shape, graph = _graph_of(models, family, key)
    input_set, points, _, _ = _flat_input(kind, int(np.prod(shape)),
                                          seed=len(key))
    outputs = replay_onnx(graph.reference, points, shape)
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach(entry, model, input_set), outputs,
        f"{key} on declared input {shape} without input_shape ({entry}, "
        f"{kind})", graph.out_shape)


# A declared (1, 2, 3) input reached with input_shape (3, 2): same number of
# entries, another tensor. onnxruntime rejects a (1, 3, 2) input.
_DISAGREES = {
    "softmax13-axis2": (_softmax(2), 13, "Box", [_SHAPE_1_DISAGREES]),
    "reducesum11-axis-1-keepdims0": (_reduce_attr("ReduceSum", [-1], 0), 11,
                                     "Star", [_SHAPE_1_DISAGREES]),
    "slice-axis2": (_slice(2, 0, 2), 13, "Zono", [_SHAPE_1_DISAGREES]),
    # Gather [0, 2] over axis 2 runs neither on (1, 3, 2) nor on (3, 2): no
    # record, and the Gather refuses.
    "gather-axis2": (_gather(2), 13, "Box", []),
}


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=v[3], id=f"{k}-{v[2]}") for k, v in _DISAGREES.items()])
def test_an_input_shape_that_disagrees_with_the_model_input_is_refused(
        models, key):
    """A model declared (1, 2, 3) reached with ``input_shape`` (3, 2), the
    same number of entries in another tensor. The network rejects that
    input, so the call must refuse (``ValueError`` or
    ``NotImplementedError``). Catches a placeholder built from the caller's
    shape without a look at the model's own input (SHAPE-1's fix raises
    here); the same graphs with the agreeing ``input_shape`` (2, 3) hold
    every output in
    ``test_the_same_model_with_a_batch_axis_holds_every_output``."""
    build, opset, kind, _ = _DISAGREES[key]
    graph = models.get(f"disagree-{key}", (1, 2, 3), opset, build)
    input_set = _flat_input(kind, 6, seed=len(key))[0]
    model = load_onnx(graph.path)
    with pytest.raises((ValueError, NotImplementedError)):
        _reach("reach_pytorch_model", model, input_set, input_shape=(3, 2))


# A node that reads the record of a node only the whole shape pass names: one
# output of a Split (a getitem node), the part of a rank-1 vector cut where the
# batched placeholder cannot run, a float64 model whose weights reject a
# float32 placeholder.
_RECORD_CONSUMERS = {
    "split-part-softmax": ((1, 4, 4), 13, _split_then_softmax(1, [2, 2], 1, 1),
                           "float32", {"input_shape": (4, 4)}),
    "no-batch-axis-split-part-softmax": (
        (4, 4), 13, _split_then_softmax(0, [2, 2], 1, 0), "float32",
        {"input_shape": (4, 4)}),
    "rank-1-split-part-reshape-softmax": (
        (8,), 13, _split_then_softmax(0, [2, 6], 1, 1, reshape=[1, 2, 3]),
        "float32", {}),
    "float64-gemm-reshape-softmax": (
        (1, 6), 13, _gemm_reshape_softmax([1, 2, 3], 1), "float64",
        {"input_shape": (6,)}),
}


@pytest.mark.parametrize("kind", ["Box", "Star"])
@pytest.mark.parametrize("key", sorted(_RECORD_CONSUMERS))
def test_a_softmax_behind_a_split_part_or_a_weight_holds_every_output(
        models, key, kind):
    """A Softmax over an inner axis whose record only the whole shape pass
    gives: behind one output of a Split (the ``getitem`` node), behind a
    Split of a model without a batch axis or of a rank-1 vector (the
    unbatched placeholder), behind the weights of a float64 file (the
    float64 placeholder). Without the record the Softmax falls back to one
    softmax over every entry. Catches the getitem nodes left out of the
    record, the unbatched fallbacks dropped and a float32 placeholder for a
    float64 model with weights."""
    shape, opset, build, dtype, options = _RECORD_CONSUMERS[key]
    graph = models.get(f"consumer-{key}", shape, opset, build, dtype)
    input_set, points, _, _ = _flat_input(kind, int(np.prod(shape)),
                                          seed=len(key))
    outputs = replay_onnx(graph.reference, points, shape)
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach("reach_pytorch_model", model, input_set, **options),
        outputs, f"{key} ({kind}, {dtype})", graph.out_shape)


_REACHED_BEFORE = known_bug(
    12, "a reach changes the model object it is given: the vendored Reduce "
        "forward (ReduceMean, ReduceSum up to opset 12, and ReduceSum 13 with "
        "an empty constant axes input) writes the axes it resolves for a "
        "Reduce over every axis with keepdims=1 into the module, and the "
        "shape pass runs that forward on its placeholder; after a reach "
        "without input_shape (placeholder (1, 12)) the module keeps axes "
        "[0, 1], and a second reach of the same object with input_shape "
        "returns 4 entries where the output has 1 (registry row SHAPE-9, which "
        "names the same write-back in the vendored Transpose)")

# id -> (opset, builder): a Reduce over every axis with keepdims=1, read by
# the flat handler through ``OnnxReduceStaticAxes.axes`` (ReduceMean) and
# through ``OnnxReduceSumStaticAxes._axes`` (ReduceSum with an empty constant
# axes input).
_EVERY_AXIS = {
    "ReduceMean-opset13-no-axes": (
        13, lambda dt: ([helper.make_node("ReduceMean", ["X"], ["Y"],
                                          keepdims=1)], {})),
    "ReduceSum-opset13-empty-axes": (
        13, lambda dt: ([helper.make_node("ReduceSum", ["X", "axes"], ["Y"],
                                          keepdims=1)],
                        {"axes": np.zeros(0, dtype=np.int64)})),
}


@pytest.mark.parametrize("key,first,second", [
    pytest.param(key, first, second, marks=marks, id=f"{key}-{order}")
    for key in _EVERY_AXIS
    for order, first, second, marks in (
        ("fresh-model", None, {"input_shape": (3, 4)}, []),
        ("reached-before-without-input_shape", {}, {"input_shape": (3, 4)},
         [_REACHED_BEFORE]),
        ("reached-before-with-input_shape-then-without",
         {"input_shape": (3, 4)}, {}, []))])
def test_a_model_reached_again_refuses_or_holds_every_output(models, key,
                                                             first, second):
    """One loaded model reached twice: a Reduce over every axis (no axes, or
    an empty axes input; keepdims=1) of a (1, 3, 4) input, the second reach
    checked against onnxruntime. The second answer must be the one a freshly
    loaded model gives, or a refusal. Catches a reach that leaves state on
    the model: the shape pass runs the vendored forward, which stores the
    axes of the placeholder's rank for good. The fresh model is the passing
    neighbour."""
    opset, build = _EVERY_AXIS[key]
    graph = models.get(f"reach-twice-{key}", (1, 3, 4), opset, build)
    input_set, points, lo, hi = _flat_input("Box", 12, seed=2)
    points = points + _extreme_members(graph.reference, (1, 3, 4), "Box",
                                       input_set, lo, hi)
    outputs = replay_onnx(graph.reference, points, (1, 3, 4))
    model = load_onnx(graph.path)
    if first is not None:
        try:
            _reach("reach_pytorch_model", model, input_set, **first)
        except (NotImplementedError, ValueError):
            pass
    _keeps_the_promise(
        lambda: _reach("reach_pytorch_model", model, input_set, **second),
        outputs, f"{key}, reached again with {second}", graph.out_shape)


def _true_output_box(path, shape, lo, hi):
    """The exact interval hull of the outputs over the box [lo, hi], for a
    graph whose every output is a nondecreasing function of one input entry
    (row selection, Relu): the outputs at ``lo`` and at ``hi``."""
    low, high = replay_onnx(path, [lo, hi], shape)
    return low, high


def _box_region(low, high):
    """The unsafe region ``low <= y <= high`` as ``G y <= g``."""
    n = low.size
    return HalfSpace(np.vstack([np.eye(n), -np.eye(n)]),
                     np.concatenate([high, -low]).reshape(-1, 1))


@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("batch", [
    pytest.param(False, marks=_SHAPE_1, id="no-batch-axis"),
    pytest.param(True, id="batch-axis"),
])
def test_gather_without_a_batch_axis_is_never_a_false_unsat(models, batch,
                                                             kind):
    """End to end: Gather([0, 2]) over the second axis of a (3, 3) input,
    then ``verify_specification`` against the unsafe region that is the exact
    interval hull of the true outputs. Every true output is unsafe, so UNSAT
    is a false proof. Catches SHAPE-1 turned into a verdict: the result picks
    rows where the model picks columns, and the region is missed."""
    if batch:
        shape, build, options = (1, 3, 3), _gather(2), {"input_shape": (3, 3)}
    else:
        shape, build, options = (3, 3), _gather(1), {"input_shape": (3, 3)}
    graph = models.get(f"verdict-gather-{batch}", shape, 13, build)
    lo, hi = flat_box(9, seed=4)
    input_set = (flat_set(kind, lo, hi) if kind != "Star"
                 else constrained_star(lo, hi, n=8, seed=4)[0])
    low, high = _true_output_box(graph.reference, shape, lo, hi)
    region = _box_region(low, high)
    outputs = replay_onnx(graph.reference, samples(lo, hi), shape)
    if not all(np.all(low <= y) and np.all(y <= high) for y in outputs):
        pytest.fail("setup: the region does not hold every true output")
    model = load_onnx(graph.path)
    try:
        sets = _reach("reach_pytorch_model", model, input_set, **options)
    except (NotImplementedError, ValueError):
        return
    assert verify_specification(sets, region).verdict != "UNSAT", (
        "UNSAT for an unsafe region that holds every true output")


# =========================================================================== #
# 2. An ImageZono without input_shape (SHAPE-3)                                #
# =========================================================================== #
_ZN = ("ImageZono", "reach_pytorch_model", False)
_ZNN = ("ImageZono", "NeuralNetwork.reach", False)
_ZI = ("ImageZono", "reach_pytorch_model", True)
_SN = ("ImageStar", "reach_pytorch_model", False)

# id -> (input tensor, opset, builder, fails today on an ImageZono without
# input_shape, calls). Image of 3 channels, height 4, width 5. Every graph is
# a linear map of the image.
_IMAGE_GRAPHS = {
    "mean-width-then-sum-channel": (_IMG, 11, _chain(
        ("ReduceMean", [3], 0), ("ReduceSum", [1], 0)), True,
        (_ZN, _ZI, _SN)),
    "mean-width-mean-channel-sum": (_IMG, 11, _chain(
        ("ReduceMean", [3], 0), ("ReduceMean", [1], 0),
        ("ReduceSum", [1], 0)), True, (_ZN, _ZI, _SN)),
    "gather-axis1": (_IMG, 13, _gather(1), True, (_ZN, _ZNN, _ZI, _SN)),
    "gather-axis-2": (_IMG, 13, _gather(-2), True, (_ZN, _ZNN, _ZI, _SN)),
    "gather-axis2": (_IMG, 13, _gather(2), False, (_ZN, _ZI, _SN)),
    "flatten2-then-sum": (_IMG, 11, _flatten_then_sum(2), True,
                          (_ZN, _ZI, _SN)),
    "flatten1-then-sum": (_IMG, 11, _flatten_then_sum(1), False,
                          (_ZN, _ZI, _SN)),
}


def _image_cells():
    cells = []
    for key, (_, _, _, fails, calls) in _IMAGE_GRAPHS.items():
        for kind, entry, given in calls:
            marked = fails and kind == "ImageZono" and not given
            cells.append(pytest.param(
                key, kind, entry, given, marks=[_SHAPE_3] if marked else [],
                id=f"{key}-{kind}-{entry}-"
                   f"{'input_shape' if given else 'no_input_shape'}"))
    return cells


@pytest.mark.parametrize("key,kind,entry,given", _image_cells())
def test_an_image_set_without_input_shape_holds_every_output(models, key,
                                                              kind, entry,
                                                              given):
    """A (3, 4, 5) image through Reduce chains, Gather and Flatten then
    ReduceSum, with and without ``input_shape``. Catches SHAPE-3 (an
    ImageZono gets the flat placeholder and a shape-aware handler reads the
    wrong tensor) and a placeholder for an ImageStar whose height and width
    are exchanged; the ImageStar and the given ``input_shape`` are the sound
    neighbours. The member points include the extreme ones of every output
    entry that adds up pixels, so a mean divided by the wrong count shows on
    the ImageStar too."""
    shape, opset, build, _, _ = _IMAGE_GRAPHS[key]
    graph = models.get(f"image-{key}", shape, opset, build)
    input_set, points, lo, hi = _image_input(kind, shape[1:], seed=len(key))
    points = points + _extreme_members(graph.reference, shape, kind,
                                       input_set, lo, hi)
    outputs = replay_onnx(graph.reference, points, shape)
    options = {"input_shape": shape[1:]} if given else {}
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach(entry, model, input_set, **options), outputs,
        f"{key} ({kind}, {entry}, input_shape {'given' if given else 'not given'})",
        graph.out_shape)


@pytest.mark.parametrize("kind,entry,given", [
    pytest.param("ImageZono", "reach_pytorch_model", False, marks=_SHAPE_3,
                 id="ImageZono-reach_pytorch_model-no_input_shape"),
    pytest.param("ImageZono", "reach_pytorch_model", True,
                 id="ImageZono-reach_pytorch_model-input_shape"),
    pytest.param("ImageStar", "reach_pytorch_model", False,
                 id="ImageStar-reach_pytorch_model-no_input_shape"),
])
def test_an_image_zono_reduce_chain_is_never_a_false_unsat(models, kind,
                                                           entry, given):
    """End to end: ReduceMean over the width, ReduceMean over the channels,
    ReduceSum over the height of a (3, 4, 5) image, then
    ``verify_specification`` against the unsafe region ``y >= v - 2**-20``,
    where ``v`` is the model's largest output over the input set, reached at
    a member point computed without n2v (the upper corner of the box for the
    ImageZono, the vertex of the ImageStar's predicate polytope that an LP
    on its own rows finds; the margin of ``2**-20`` keeps that point's real
    output inside the region whatever the float64 rounding of the
    reference). The region holds a true output, so UNSAT is a false proof.
    Catches SHAPE-3 turned into a verdict: on an ImageZono read through the
    placeholder (1, 60) the result's largest value is about a quarter of
    the true one."""
    shape = _IMG
    graph = models.get("image-mean-width-mean-channel-sum", shape, 11,
                       _IMAGE_GRAPHS["mean-width-mean-channel-sum"][2])
    lo, hi = pixel_box(*shape[1:], seed=5)
    input_set = (image_set(kind, lo, hi) if kind == "ImageZono"
                 else constrained_image_star(lo, hi, n=8, seed=5)[0])
    extremes = _extreme_members(graph.reference, shape, kind, input_set,
                                lo, hi)
    top = max(y[0] for y in replay_onnx(graph.reference, extremes, shape))
    outputs = replay_onnx(graph.reference, samples(lo, hi), shape)
    if kind == "ImageZono" and not all(y[0] <= top for y in outputs):
        pytest.fail("setup: the upper corner is not the largest output")
    region = HalfSpace(np.array([[-1.0]]), np.array([[-(top - 2.0 ** -20)]]))
    options = {"input_shape": shape[1:]} if given else {}
    model = load_onnx(graph.path)
    try:
        sets = _reach(entry, model, input_set, **options)
    except (NotImplementedError, ValueError):
        return
    assert verify_specification(sets, region).verdict != "UNSAT", (
        "UNSAT for an unsafe region that holds the model's largest output")


# =========================================================================== #
# 3. A flat set on a rank-3 or rank-4 model input without input_shape        #
#    (SHAPE-2)                                                                 #
# =========================================================================== #
_B_NO = ("Box", "reach_pytorch_model", False)
_S_NO = ("Star", "reach_pytorch_model", False)
_Z_NO = ("Zono", "reach_pytorch_model", False)
_GIVEN = (("Box", "reach_pytorch_model", True),
          ("Star", "reach_pytorch_model", True),
          ("Zono", "NeuralNetwork.reach", True))

# id -> (input tensor, opset, builder, fails today without input_shape, linear
# map, calls).
_FLAT_RANK = {
    # axis 2 of (1, 3, 4) cannot run on the placeholder (1, 12): refused.
    "x-minus-mean-axis2": ((1, 3, 4), 13, _x_minus_mean([2]), False, True,
                           (_B_NO,) + _GIVEN),
    "x-minus-mean-axis-1": ((1, 3, 4), 13, _x_minus_mean([-1]), True, True,
                            (_B_NO, _S_NO, _Z_NO) + _GIVEN),
    "x-minus-mean-axis1": ((1, 3, 4), 13, _x_minus_mean([1]), True, True,
                           (_B_NO, _S_NO, _Z_NO) + _GIVEN),
    "mean-axis1-then-sum-axis1": ((1, 3, 4), 11, _chain(
        ("ReduceMean", [1], 0), ("ReduceSum", [1], 0)), True, True,
        (_B_NO, _S_NO, _Z_NO) + _GIVEN),
    "relu-slice-reversed-axis2": (_IMG, 13, _relu_slice_reversed(2), True,
                                  False, (_B_NO,) + _GIVEN),
    "concat-relu-neg-axis-1": (_IMG, 13, _concat_relu_neg(-1), True, False,
                               (_B_NO, _S_NO, _Z_NO) + _GIVEN),
    "concat-relu-neg-axis1": (_IMG, 13, _concat_relu_neg(1), False, False,
                              (_B_NO,) + _GIVEN),
    "split-axis2-then-relu": (_IMG, 13, _split(2, [2, 2], 1, after="Relu"),
                              True, False, (_B_NO, _S_NO, _Z_NO) + _GIVEN),
}


def _flat_rank_cells():
    cells = []
    for key, (_, _, _, fails, _, calls) in _FLAT_RANK.items():
        for kind, entry, given in calls:
            marked = fails and not given
            cells.append(pytest.param(
                key, kind, entry, given, marks=[_SHAPE_2] if marked else [],
                id=f"{key}-{kind}-{entry}-"
                   f"{'input_shape' if given else 'no_input_shape'}"))
    return cells


@pytest.mark.parametrize("key,kind,entry,given", _flat_rank_cells())
def test_a_flat_set_on_a_rank3_or_rank4_input_holds_every_output(
        models, key, kind, entry, given):
    """A flat Box, Star or Zono that stands for a (1, 3, 4) or (1, 3, 4, 5)
    tensor, with and without ``input_shape``. Catches SHAPE-2 (the set is
    read as the tensor (1, dim) and every axis-dependent node computes on a
    different tensor) and, with ``input_shape`` given, any slip in how the
    shape-aware handlers place a non-negative or negative axis of a rank-3
    or rank-4 tensor, ``input_shape`` lost on its way to the shape pass, and
    a mean divided by the wrong count (the linear graphs add the extreme
    member points of every output entry)."""
    shape, opset, build, _, linear, _ = _FLAT_RANK[key]
    graph = models.get(f"flatrank-{key}", shape, opset, build)
    input_set, points, lo, hi = _flat_input(kind, int(np.prod(shape)),
                                            seed=len(key))
    if linear:
        points = points + _extreme_members(graph.reference, shape, kind,
                                           input_set, lo, hi)
    outputs = replay_onnx(graph.reference, points, shape)
    options = {"input_shape": shape[1:]} if given else {}
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach(entry, model, input_set, **options), outputs,
        f"{key} on input {shape} ({kind}, {entry}, input_shape "
        f"{'given' if given else 'not given'})", graph.out_shape)


@pytest.mark.parametrize("given,kind", [
    pytest.param(False, "Box", marks=_SHAPE_2, id="no-input_shape-Box"),
    pytest.param(True, "Box", id="input_shape-Box"),
    pytest.param(True, "Star", id="input_shape-Star"),
    pytest.param(True, "Zono", id="input_shape-Zono"),
])
def test_reversed_width_without_input_shape_is_never_a_false_unsat(
        models, given, kind):
    """End to end: Relu then the width axis of a (1, 3, 4, 5) input read
    backwards, then ``verify_specification`` against the exact interval hull
    of the true outputs. Catches SHAPE-2 turned into a false UNSAT: the flat
    set read as (1, 60) is reversed as a whole, the region is missed."""
    shape = _IMG
    graph = models.get("verdict-reversed-width", shape, 13,
                       _relu_slice_reversed(3))
    lo, hi = flat_box(60, seed=6)
    input_set = (flat_set(kind, lo, hi) if kind != "Star"
                 else constrained_star(lo, hi, n=8, seed=6)[0])
    low, high = _true_output_box(graph.reference, shape, lo, hi)
    outputs = replay_onnx(graph.reference, samples(lo, hi), shape)
    if not all(np.all(low <= y) and np.all(y <= high) for y in outputs):
        pytest.fail("setup: the region does not hold every true output")
    options = {"input_shape": shape[1:]} if given else {}
    model = load_onnx(graph.path)
    try:
        sets = _reach("reach_pytorch_model", model, input_set, **options)
    except (NotImplementedError, ValueError):
        return
    assert verify_specification(sets, _box_region(low, high)).verdict \
        != "UNSAT", "UNSAT for an unsafe region that holds every true output"


# =========================================================================== #
# 4. A graph output that is directly one output of a Split (GRAPH-8)          #
# =========================================================================== #
# id -> (input tensor, input_shape option, axis, sizes, kept part, set kinds)
_SPLITS = {
    "batch-1x4x4-axis2-part1": ((1, 4, 4), (4, 4), 2, [2, 2], 1,
                                ("Box", "Star", "Zono")),
    "image-axis1-part1": ((1, 4, 3, 5), None, 1, [1, 3], 1,
                          ("ImageStar", "ImageZono")),
}


def _split_cells():
    cells = []
    for key, (_, _, _, _, _, kinds) in _SPLITS.items():
        for after in (None, "Relu"):
            for kind in kinds:
                cells.append(pytest.param(
                    key, after, kind, "reach_pytorch_model",
                    marks=[_GRAPH_8] if after is None else [],
                    id=f"{key}-{'raw' if after is None else 'then-relu'}-"
                       f"{kind}-reach_pytorch_model"))
    return cells


@pytest.mark.parametrize("key,after,kind,entry", _split_cells())
def test_a_split_output_as_the_graph_output_is_a_flat_list(models, key,
                                                            after, kind,
                                                            entry):
    """The graph output is one part of a Split, as the graph's own output
    (raw) or after a Relu. Catches GRAPH-8 (a list holding one list per Split
    output) and a result that returns the other part: the entry count and
    every onnxruntime output of the named part are checked."""
    shape, given, axis, sizes, keep, _ = _SPLITS[key]
    graph = models.get(
        f"split-{key}-{after}", shape, 13, _split(axis, sizes, keep, after))
    if len(shape) == 4:
        input_set, points, _, _ = _image_input(kind, shape[1:], seed=len(key))
    else:
        input_set, points, _, _ = _flat_input(kind, int(np.prod(shape)),
                                              seed=len(key))
    outputs = replay_onnx(graph.reference, points, shape)
    options = {"input_shape": given} if given is not None else {}
    model = load_onnx(graph.path)
    _keeps_the_promise(
        lambda: _reach(entry, model, input_set, **options), outputs,
        f"Split part {keep} of {key} ({'raw' if after is None else after}, "
        f"{kind}, {entry})", graph.out_shape)


# =========================================================================== #
# 5. Reduce with its axes as a second input (LOAD-1)                           #
# =========================================================================== #
@pytest.mark.parametrize("op,keepdims", [
    pytest.param(op, kd, marks=_LOAD_1, id=f"{op}-{kd}")
    for op, kd in (("ReduceMean", 0), ("ReduceMean", 1), ("ReduceMax", 0),
                   ("ReduceL1", 1))])
def test_an_opset18_reduce_with_axes_as_input_is_refused_at_load(
        models, op, keepdims):
    """A legal opset-18 Reduce node, axes given as a second input, over the
    width of a (1, 2, 3, 4) input. The promise is a clear refusal at load
    (``NotImplementedError``); catches the converter's ``TypeError``
    ('AttributeProto' object is not iterable) escaping instead (LOAD-1).
    The nine Reduce operators other than ReduceSum share one converter
    function, so these four cases stand for all of them."""
    graph = models.get(f"opset18-{op}-{keepdims}", (1, 2, 3, 4), 18,
                       _reduce_input(op, [3], keepdims))
    with pytest.raises(NotImplementedError):
        load_onnx(graph.path)


_REDUCE_LOADS = {
    # key: (opset, builder, set kinds, linear): the forms that load today
    "ReduceMean-opset13-attribute": (13, _reduce_attr("ReduceMean", [3], 0),
                                     ("ImageStar", "Star", "Box"), True),
    "ReduceMean-opset13-attribute-keepdims": (
        13, _reduce_attr("ReduceMean", [2], 1),
        ("ImageStar", "Star", "Box"), True),
    "ReduceSum-opset18-input": (18, _reduce_input("ReduceSum", [3], 0),
                                ("ImageStar", "Star", "Box"), True),
    "ReduceSum-opset18-input-keepdims": (
        18, _reduce_input("ReduceSum", [2], 1),
        ("ImageStar", "Star", "Box"), True),
    "ReduceSum-opset13-input": (13, _reduce_input("ReduceSum", [1], 0),
                                ("ImageStar", "Star", "Box"), True),
    "ReduceMax-opset13-attribute": (13, _reduce_attr("ReduceMax", [3], 0),
                                    ("Star", "Box"), False),
    # ReduceMax is the identity over an axis of length 1 and ReduceL2 never
    # is, so the two stand for the two kinds of non-affine Reduce. The Star
    # case of ReduceL2 is tests/soundness/test_soundness_reduce.py::
    # test_non_affine_reduce_of_a_flat_set_through_the_model_is_refused
    # [Star-ReduceL2]; that file has no Box case of ReduceL2.
    "ReduceL2-opset13-attribute": (13, _reduce_attr("ReduceL2", [3], 0),
                                   ("Box",), False),
}


@pytest.mark.parametrize("key,kind", [
    pytest.param(key, kind, id=f"{key}-{kind}")
    for key in sorted(_REDUCE_LOADS) for kind in _REDUCE_LOADS[key][2]])
def test_a_reduce_that_loads_holds_every_output(models, key, kind):
    """The neighbours of the opset-18 refusal: the attribute form (opset 13)
    and ReduceSum with axes as an input (opsets 13 and 18) load. The loaded
    module computes onnxruntime's outputs (torch in float64, on the member
    points), and the reach result holds every onnxruntime output (or is
    refused: ReduceMax and ReduceL2 are refused by the kernel, so their
    cases check the load and the refusal). Catches a load path that reads
    the wrong axes or keepdims from either form, and a non-affine Reduce
    answered as a sum."""
    opset, build, _, linear = _REDUCE_LOADS[key]
    shape = (1, 2, 3, 4)
    graph = models.get(f"reduce-{key}", shape, opset, build)
    model = load_onnx(graph.path)
    if kind == "ImageStar":
        input_set, points, lo, hi = _image_input(kind, shape[1:],
                                                 seed=len(key))
    else:
        input_set, points, lo, hi = _flat_input(kind, 24, seed=len(key))
    if linear:
        points = points + _extreme_members(graph.reference, shape, kind,
                                           input_set, lo, hi)
    outputs = replay_onnx(graph.reference, points, shape)
    loaded = replay_torch(model, points, shape)
    worst = max(float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))
                for a, b in zip(loaded, outputs))
    assert worst <= 1e-12, (f"{key}: the loaded module differs from "
                            f"onnxruntime by {worst:.3g} (relative)")
    _keeps_the_promise(
        lambda: _reach("reach_pytorch_model", model, input_set,
                       input_shape=shape[1:]), outputs,
        f"{key} ({kind})", graph.out_shape)
