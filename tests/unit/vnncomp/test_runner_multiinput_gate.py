"""The runner gates multimodal / multi-input specs (>1 declared input tensor,
e.g. smart_turn's X1 + X2) to a sound ``unknown`` from the spec header, BEFORE
loading the model: sound reach over the concatenated joint input is intractable
(a ~1.27M-dim dense generator OOMs at 11.7 TiB). This pins the gate, that it is
precise (single-input specs are NOT gated), and that it fires without touching
the (124 MB) model. Mirrors NNV's header-level multimodal gate.
"""

import os
import re
import sys

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402
from n2v.config import config as n2v_config  # noqa: E402
from tests._oracles.membership import LP_TOL, hwc, uncovered  # noqa: E402
from tests.cell_support import assert_covered, replay_onnx  # noqa: E402
from tests.known_bugs import known_bug  # noqa: E402

CAT = "smart_turn_multimodal_2026"


def _prop(n_inputs):
    return {
        "lb": [0.0, 0.0], "ub": [1.0, 1.0], "pairs": [],
        "input_tensors": [{"name": f"X{i}", "shape": (1, 2)}
                          for i in range(n_inputs)],
        "output_tensors": [{"name": "Y", "shape": (1, 1)}],
    }


def test_multi_input_gated_to_unknown_without_loading_model(monkeypatch):
    monkeypatch.setattr(R, "load_vnnlib", lambda p: _prop(2))

    def _boom(*a, **k):
        raise AssertionError("load_onnx must NOT be called for a gated "
                             "multi-input spec")
    monkeypatch.setattr(R, "load_onnx", _boom)

    res = R.verify_instance("nonexistent.onnx", "spec.vnnlib", CAT)
    assert res["result"] == R.RESULT_UNKNOWN
    assert res.get("counterexample") is None


def test_single_input_not_gated(monkeypatch):
    """One declared input tensor must NOT trip the gate; the runner proceeds to
    load the model (proven here by load_onnx being reached)."""
    monkeypatch.setattr(R, "load_vnnlib", lambda p: _prop(1))

    reached = {"loaded": False}

    def _mark(*a, **k):
        reached["loaded"] = True
        raise RuntimeError("stop after gate check")
    monkeypatch.setattr(R, "load_onnx", _mark)

    with pytest.raises(RuntimeError, match="stop after gate"):
        R.verify_instance("x.onnx", "spec.vnnlib", CAT)
    assert reached["loaded"] is True


# =========================================================================== #
# The routes of verify_instance                                                #
# =========================================================================== #
# Below the gate: the input shape read from the ONNX file, the input set built
# from the parsed bounds, the reach stage and how it turns the regions and
# methods into one verdict, the nonlinear and ViT routes, and the global
# configuration around a call. Every instance is a small ONNX file and a
# VNN-LIB text written here; its true answer is known by construction (a
# counterexample replayed on onnxruntime by the test, or a bound that holds
# on the whole box), and the falsifier is replaced by one that finds nothing
# wherever the reach stage alone must decide.

def _save(path, nodes, inits, in_dims, out_dims, extra_inputs=()):
    graph = helper.make_graph(
        nodes, "g",
        list(extra_inputs)
        + [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                         list(in_dims))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                       list(out_dims))],
        [numpy_helper.from_array(np.asarray(v), k) for k, v in inits.items()])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _affine_net(path, W, b, in_dims=None):
    """``Y = X @ W.T + b`` on the input flattened row-major; X ``in_dims``
    (default [1, n]), Y [1, m]."""
    W = np.asarray(W, np.float32)
    nodes = [helper.make_node("Reshape", ["X", "sh"], ["f"]),
             helper.make_node("Gemm", ["f", "W", "B"], ["Y"], transB=1)]
    return _save(path, nodes,
                 {"sh": np.array([-1, W.shape[1]], np.int64), "W": W,
                  "B": np.asarray(b, np.float32)},
                 in_dims or [1, W.shape[1]], [1, W.shape[0]])


def _relu_net(path, W1, W2):
    """``Y = relu(X @ W1.T) @ W2.T``, X [1, n]."""
    W1, W2 = np.asarray(W1, np.float32), np.asarray(W2, np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W1", "B1"], ["h"], transB=1),
             helper.make_node("Relu", ["h"], ["r"]),
             helper.make_node("Gemm", ["r", "W2", "B2"], ["Y"], transB=1)]
    return _save(path, nodes,
                 {"W1": W1, "B1": np.zeros(W1.shape[0], np.float32),
                  "W2": W2, "B2": np.zeros(W2.shape[0], np.float32)},
                 [1, W1.shape[1]], [1, W2.shape[0]])


def _spec_v1(path, regions, outputs=(), n_out=1):
    """A VNN-LIB 1.0 spec. ``regions`` is a list of ``(lb, ub, own)``: one
    region without own atoms is written as top-level bounds; otherwise an
    ``(or (and ...))`` of boxes, each with its own output atoms ``own`` (a
    list of strings, the combined form) when given. ``outputs`` are
    top-level output asserts."""
    n = len(regions[0][0])
    lines = [f"(declare-const X_{i} Real)" for i in range(n)]
    lines += [f"(declare-const Y_{j} Real)" for j in range(n_out)]
    if len(regions) == 1 and not regions[0][2]:
        lb, ub, _ = regions[0]
        for i in range(n):
            lines += [f"(assert (>= X_{i} {float(lb[i])!r}))",
                      f"(assert (<= X_{i} {float(ub[i])!r}))"]
    else:
        lines.append("(assert (or")
        for lb, ub, own in regions:
            atoms = [f"(>= X_{i} {float(lb[i])!r}) (<= X_{i} {float(ub[i])!r})"
                     for i in range(n)] + list(own or [])
            lines.append("  (and " + " ".join(atoms) + ")")
        lines.append("))")
    lines += [f"(assert {a})" for a in outputs]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def _spec_v2(path, in_dims, out_dims, asserts):
    with open(path, "w") as f:
        f.write("(vnnlib-version <2.0>)\n(declare-network N\n"
                f"    (declare-input X float32 [{', '.join(map(str, in_dims))}])\n"
                f"    (declare-output Y float32 [{', '.join(map(str, out_dims))}])\n"
                ")\n" + "".join(f"(assert {a})\n" for a in asserts))
    return str(path)


def _config(monkeypatch, reach=(("approx", {}),), n_rand=100,
            method="random"):
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {
        "reach_methods": list(reach), "n_rand": n_rand,
        "falsify_method": method, "falsify_kwargs": {}})


@pytest.fixture
def no_counterexample_found(monkeypatch):
    """A falsifier that finds nothing, so the reach stage alone decides (a
    falsifier may always miss; the reach stage must not depend on it)."""
    monkeypatch.setattr(R, "falsify", lambda *a, **k: (2, None))


def _replay(path, x, in_dims):
    return replay_onnx(path, [np.asarray(x, dtype=np.float64)], in_dims)[0]


def _witness(text):
    """A written witness read as the checker reads it, in either format:
    ``((X_0 v) ... (Y_0 v))`` for 1.0, named blocks for 2.0."""
    if text.lstrip().startswith("("):
        pairs = re.findall(r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)", text)
        xs = {int(i): float(v) for k, i, v in pairs if k == "X"}
        ys = {int(i): float(v) for k, i, v in pairs if k == "Y"}
        return ([xs[i] for i in range(len(xs))],
                [ys[j] for j in range(len(ys))])
    blocks, name = {}, None
    for line in text.strip().splitlines():
        m = re.fullmatch(r"(\S+) (\S+) \[([0-9,\s]*)\]", line.strip())
        if m:
            name = m.group(1)
            blocks[name] = []
        else:
            blocks[name].append(float(line))
    return blocks["X"], blocks["Y"]


# The band instance: y = x0 + x1 on [0, 1]^2 with the unsafe band
# 0.749999999 <= y <= 0.750000001. x = (0.5, 0.25) gives y = 0.75 exactly (in
# float32 too): a real counterexample that no reach over the box rules out
# and that sampling does not hit. The safe region [0, 0.1]^2 has y <= 0.2.
_BAND = ["(>= Y_0 0.749999999)", "(<= Y_0 0.750000001)"]
_SAFE_BOX = ([0.0, 0.0], [0.1, 0.1], None)
_BAND_BOX = ([0.0, 0.0], [1.0, 1.0], None)


def _band_instance(tmp_path):
    onnx_p = _affine_net(tmp_path / "sum.onnx", [[1.0, 1.0]], [0.0])
    y = _replay(onnx_p, [0.5, 0.25], (1, 2))[0]
    if not 0.749999999 <= y <= 0.750000001:
        pytest.fail(f"the band's counterexample replays to {y!r}")
    return onnx_p


# --------------------------------------------------------------------------- #
# The input shape                                                             #
# --------------------------------------------------------------------------- #

def _shape_model(tmp_path, case):
    """A model of y = x0 + x1 + ... on the inputs of ``case``."""
    if case == "initializer listed before the input":
        W = np.ones((2, 1), np.float32)
        return _save(tmp_path / "m.onnx",
                     [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                     {"W": W}, [1, 2], [1, 1],
                     extra_inputs=[helper.make_tensor_value_info(
                         "W", TensorProto.FLOAT, [2, 1])])
    dims = {"batch 1": [1, 2], "dynamic batch": ["N", 2]}[case]
    return _save(tmp_path / "m.onnx",
                 [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                 {"W": np.ones((2, 1), np.float32)}, dims,
                 [dims[0], 1])


@pytest.mark.parametrize("dims,expected", [
    ([1, 2], (2,)),
    (["N", 2], (2,)),
    ([6], (6,)),
    ([2, 3], (2, 3)),
    ([1, 3, 2, 4], (3, 2, 4)),
    ([1, 1, 1, 5], (1, 1, 5)),
], ids=["batch-1", "dynamic-batch", "rank-1-no-batch", "leading-2-kept",
        "image", "acasxu"])
def test_input_shape_is_the_declared_shape_without_the_batch_dim(
        tmp_path, dims, expected):
    """``get_input_shape`` on an ONNX input declared as ``dims``: the
    declared shape with a leading batch dimension (1, or a symbolic one,
    read as 0) removed; a rank-1 input and a leading dimension above 1 are
    kept. Catches a dynamic batch dimension left in the shape and a stripped
    dimension that is not a batch dimension."""
    n = int(np.prod([d for d in dims if isinstance(d, int)]))
    path = _save(tmp_path / "m.onnx",
                 [helper.make_node("Identity", ["X"], ["Y"])], {}, dims,
                 dims)
    assert n > 0
    assert R.get_input_shape(path) == expected


def test_input_shape_skips_initializers_listed_as_graph_inputs(tmp_path):
    """An older export lists the weight ``W [2, 1]`` in ``graph.input``
    before the true input ``X [1, 2]``: the input shape is X's, (2,).
    Catches the first graph input taken whether or not it is an
    initializer. A graph whose only inputs are initializers has no input to
    read and is refused with ``ValueError``."""
    path = _shape_model(tmp_path, "initializer listed before the input")
    assert R.get_input_shape(path) == (2,)
    only_init = helper.make_graph(
        [helper.make_node("Identity", ["W"], ["Y"])], "g",
        [helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1])],
        [numpy_helper.from_array(np.ones((2, 1), np.float32), "W")])
    model = helper.make_model(only_init,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "w.onnx"))
    with pytest.raises(ValueError, match="no true input"):
        R._onnx_input_dims(str(tmp_path / "w.onnx"))


@pytest.mark.parametrize("case", ["batch 1", "dynamic batch",
                                  "initializer listed before the input"])
def test_instances_with_every_input_declaration_are_proved(
        tmp_path, monkeypatch, case):
    """The public route for the input shape: y = x0 + x1 on ``[0, 1]^2``,
    unsafe ``Y_0 >= 2.5``, proved by approx reach whatever way the ONNX file
    declares its input. Catches an input shape that keeps a dynamic batch
    dimension or reads an initializer, either of which makes every reshape
    of the box fail (the answer is then unknown)."""
    onnx_p = _shape_model(tmp_path, case)
    spec = _spec_v1(tmp_path / "s.vnnlib", [_BAND_BOX], ["(>= Y_0 2.5)"])
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "shape_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def test_a_rank_2_input_read_along_its_last_axis_is_proved(
        tmp_path, monkeypatch, no_counterexample_found):
    """X [1, 3, 4] (stripped (3, 4), a flat input set of 12 entries) and a
    first node that reads the last axis: Y [1, 3] = Flatten(X @ ones(4, 1)),
    the three row sums, each at most 4 on ``[0, 1]^12``, so ``Y_0 >= 4.5``
    is unreachable and approx reach proves it. The reach can place the
    MatMul only if it is told the input's tensor shape. Catches the input
    shape not handed to the reach (the MatMul then fails on a 12-entry
    vector and the instance is written unknown)."""
    onnx_p = _save(tmp_path / "rows.onnx",
                   [helper.make_node("MatMul", ["X", "W"], ["m"]),
                    helper.make_node("Flatten", ["m"], ["Y"], axis=1)],
                   {"W": np.ones((4, 1), np.float32)}, [1, 3, 4], [1, 3])
    spec = _spec_v1(tmp_path / "s.vnnlib", [([0.0] * 12, [1.0] * 12, None)],
                    ["(>= Y_0 4.5)"], n_out=3)
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "rows_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


# --------------------------------------------------------------------------- #
# The input set                                                               #
# --------------------------------------------------------------------------- #

def _entry_box(n):
    """A different interval per entry, both signs, beyond plus and minus 1
    and wider than 2, one entry of zero width."""
    k = np.arange(n, dtype=np.float64)
    lb = 3.0 * k - 7.0
    ub = lb + 0.5 + (k % 4)
    ub[n // 2] = lb[n // 2]
    return lb, ub


@pytest.mark.parametrize("shape,kind,stored", [
    ((2, 3, 5), "ImageStar", (3, 5, 2)),
    ((1, 3, 4), "ImageStar", (3, 4, 1)),
    ((2, 1, 3, 4), "ImageStar", (3, 4, 2)),
    ((2, 2, 3, 4), "ImageStar", (3, 4, 4)),
    ((1, 1, 5), "Star", None),
    ((3, 4), "Star", None),
    ((6,), "Star", None),
], ids=["chw-2x3x5", "grayscale-1x3x4", "singleton-inside-2x1x3x4",
        "rank-4-2x2x3x4", "acasxu-1x1x5", "rank-2-3x4", "flat-6"])
def test_the_input_set_is_exactly_the_box_in_its_stored_order(
        shape, kind, stored):
    """``create_input_set`` on bounds given in the ONNX tensor's row-major
    order: an image shape (two or more non-singleton dimensions, rank 3 or
    more) gives an ``ImageStar`` of that height, width and channel count,
    stored in (H, W, C) order, with the leading dimensions of a longer
    shape taken together as the channels; any other shape a flat ``Star`` in
    tensor order. Every corner-side sample of the box is a member, decided by an
    LP on the set's own rows after converting to the stored order, and no
    point one step outside any face is. Catches the (C, H, W) to (H, W, C)
    move left out, height and width swapped, a grayscale image built flat,
    and a degenerate shape such as ACAS Xu's (1, 1, 5) built as a 1x5
    image."""
    n = int(np.prod(shape))
    lb, ub = _entry_box(n)
    s = R.create_input_set(lb, ub, shape)
    assert type(s).__name__ == kind
    rng = np.random.default_rng(0)
    inside = [lb, ub] + [np.where(rng.random(n) < 0.5, lb, ub)
                         for _ in range(64)] + [
        lb + (ub - lb) * rng.random(n) for _ in range(62)]
    outside = []
    for i in range(n):
        for step in (-0.25, 0.25):
            x = (lb + ub) / 2
            x[i] = (lb[i] if step < 0 else ub[i]) + step
            outside.append(x)
    if stored is None:
        assert s.dim == n
        assert_covered([s], inside, f"flat input set {shape}")
        views = outside
    else:
        h, w, c = stored
        assert (s.height, s.width, s.num_channels) == stored
        assert_covered([s], [x.reshape(c, h, w) for x in inside],
                       f"image input set {shape}", out_chw=(c, h, w))
        views = [hwc(x, c, h, w) for x in outside]
    assert len(uncovered([s], views, LP_TOL)) == len(views)


def test_the_memory_guard_counts_only_perturbed_entries(monkeypatch):
    """The dense basis takes entries x (perturbed entries + 1) doubles, and
    a basis over the limit is refused with ``MemoryError``. With the limit
    at 16 KiB, as on a small host: a 64-entry box with 2 perturbed entries
    (1.5 KiB) is built, one with all 64 perturbed (33 KiB) is refused.
    Catches pinned entries counted as perturbed."""
    monkeypatch.setattr(R, "_MAX_INPUT_BASIS_BYTES", 16 * 1024)
    lb, ub = np.full(64, 0.25), np.full(64, 0.25)
    ub[:2] = 1.0
    s = R.create_input_set(lb, ub, (64,))
    assert s.dim == 64
    with pytest.raises(MemoryError):
        R.create_input_set(np.zeros(64), np.ones(64), (64,))


def test_the_physical_memory_falls_back_to_64_gib(monkeypatch):
    """Where ``sysconf`` cannot report the memory, the guard's limit is
    derived from 64 GiB. Catches a fallback in other units."""
    def no_sysconf(name):
        raise ValueError(name)
    monkeypatch.setattr(R.os, "sysconf", no_sysconf)
    assert R._physical_memory_bytes() == 64 * 2 ** 30


def _image_instance(tmp_path):
    """X [1, 3, 2, 4] (3 channels, height 2, width 4), one 2x2 convolution
    with weights 1 and one output channel: Y [1, 1, 1, 3]. Only the pixels
    of the first window (rows 0-1, columns 0-1) are free in [0, 1]; the
    rest are pinned to 0, so y_0 ranges over [0, 12] and y_1, y_2 over
    [0, 6] and [0, 0]. The band 9.999999 <= y_0 <= 10.000001 holds a real
    counterexample: the window with eight values 1 and four values 0.5."""
    C, H, W = 3, 2, 4
    onnx_p = _save(tmp_path / "conv.onnx",
                   [helper.make_node("Conv", ["X", "K"], ["Y"],
                                     kernel_shape=[2, 2])],
                   {"K": np.ones((1, C, 2, 2), np.float32)},
                   [1, C, H, W], [1, 1, 1, 3])
    lb, ub = np.zeros((C, H, W)), np.zeros((C, H, W))
    ub[:, 0:2, 0:2] = 1.0
    x = np.zeros((C, H, W))
    x[:, 0:2, 0:2] = 1.0
    x[0:2, 1, 0:2] = 0.5
    y = _replay(onnx_p, x.ravel(), (1, C, H, W))
    if not 9.999999 <= y[0] <= 10.000001:
        pytest.fail(f"the image counterexample replays to {y.tolist()}")
    return onnx_p, lb.ravel(), ub.ravel()


def test_an_image_instance_with_a_counterexample_is_not_proved(
        tmp_path, monkeypatch, no_counterexample_found):
    """The public route for the image layout: approx reach on the
    convolution instance with its band. The instance has a counterexample,
    so the answer must not be ``unsat``. Catches an input set whose pixels
    are not where the convolution reads them (height and width swapped, or
    the channel order not moved), on which the reach rules the band out."""
    onnx_p, lb, ub = _image_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib", [(lb, ub, None)],
                    ["(>= Y_0 9.999999)", "(<= Y_0 10.000001)"], n_out=3)
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "image_demo",
                             workers=1)["result"] != R.RESULT_UNSAT


def test_a_grayscale_image_instance_is_proved(tmp_path, monkeypatch,
                                              no_counterexample_found):
    """X [1, 1, 3, 4] through a 2x2 convolution with weights 1: every
    output is a sum of four pixels in [0, 1], at most 4, so ``Y_0 >= 4.5``
    is unreachable and approx reach proves it. Catches a grayscale input
    built as a flat set, which the convolution reach cannot take (the
    answer is then unknown)."""
    onnx_p = _save(tmp_path / "gray.onnx",
                   [helper.make_node("Conv", ["X", "K"], ["Y"],
                                     kernel_shape=[2, 2])],
                   {"K": np.ones((1, 1, 2, 2), np.float32)},
                   [1, 1, 3, 4], [1, 1, 2, 3])
    spec = _spec_v1(tmp_path / "s.vnnlib", [([0.0] * 12, [1.0] * 12, None)],
                    ["(>= Y_0 4.5)"], n_out=6)
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "gray_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


# --------------------------------------------------------------------------- #
# The reach stage: regions, methods and failures                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("regions", [
    pytest.param([_SAFE_BOX, _BAND_BOX], id="safe-region-first"),
    pytest.param([_BAND_BOX, _SAFE_BOX], id="band-region-first"),
])
def test_unsat_needs_every_region_proved(tmp_path, monkeypatch,
                                         no_counterexample_found, regions):
    """Two OR'ed input regions, one provably safe (y <= 0.2) and one with
    the band's counterexample, in both orders: approx reach proves one region
    only, so the answer is ``unknown``. Catches every region reached from the
    first region's box, a proof reset at each region (the last region
    decides), and an undecided region that does not clear the proof."""
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib", regions, _BAND)
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "regions_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


def test_a_spec_whose_regions_are_all_safe_is_proved(
        tmp_path, monkeypatch, no_counterexample_found):
    """Three OR'ed regions, each with y = x0 + x1 at most 1.5, unsafe
    ``Y_0 >= 1.75``: every region is proved, the answer is ``unsat``.
    Neighbour of the case above; catches a proof that never survives a
    second region."""
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [([0.0, 0.0], [0.5, 1.0], None),
                     ([-3.0, 0.5], [0.25, 0.75], None),
                     ([0.5, -2.0], [1.0, 0.5], None)], ["(>= Y_0 1.75)"])
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "regions_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def test_each_region_is_proved_against_its_own_property(
        tmp_path, monkeypatch, no_counterexample_found):
    """A combined-form 1.0 spec: region 1 ``[0,1]^2`` with the band (a real
    counterexample), region 2 ``[2,3] x [0,1]`` with ``Y_0 <= -100``
    (unreachable there). Read together the two properties are empty, read
    per region the first is undecided: the answer is ``unknown``. Catches
    the reach stage checking every region against the spec's top-level
    property."""
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [([0.0, 0.0], [1.0, 1.0], _BAND),
                     ([2.0, 0.0], [3.0, 1.0], ["(<= Y_0 -100.0)"])])
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    assert R.verify_instance(onnx_p, spec, "paired_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


# y = relu(x0+x1) + relu(x0-x1) + relu(-x0+x1) + relu(-x0-x1) on [-1, 1]^2 is
# 2 * max(|x0|, |x1|) <= 2; the triangle relaxation bounds it only by 4.
_W1 = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]


def test_a_later_method_proves_what_an_earlier_one_could_not(
        tmp_path, monkeypatch, no_counterexample_found):
    """Unsafe ``Y_0 >= 3`` on the four-ReLU net: approx alone leaves it
    undecided (checked here), the configured [approx, exact] proves it with
    exact reach: ``unsat``. Catches a proof flag that is not reset for each
    method, so an earlier method's undecided region outlives it."""
    onnx_p = _relu_net(tmp_path / "relu.onnx", _W1, [[1.0, 1.0, 1.0, 1.0]])
    spec = _spec_v1(tmp_path / "s.vnnlib", [([-1.0, -1.0], [1.0, 1.0], None)],
                    ["(>= Y_0 3.0)"])
    _config(monkeypatch, reach=[("approx", {})])
    if R.verify_instance(onnx_p, spec, "order_demo",
                         workers=1)["result"] != R.RESULT_UNKNOWN:
        pytest.fail("approx alone was expected to leave the instance open")
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    assert R.verify_instance(onnx_p, spec, "order_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def _failing_reach(monkeypatch, failing, error):
    """``NeuralNetwork`` replaced by one whose ``reach`` raises ``error`` for
    the methods in ``failing`` and runs the real reach for the others."""
    from n2v.nn import NeuralNetwork

    def make(model):
        real = NeuralNetwork(model)

        class Net:
            def reach(self, input_set, method, **kw):
                if method in failing:
                    raise error(f"{method} cannot run here")
                return real.reach(input_set, method=method, **kw)
        return Net()
    monkeypatch.setattr(R, "NeuralNetwork", make)


@pytest.mark.parametrize("error", [NotImplementedError, RuntimeError],
                         ids=["unsupported", "error"])
def test_a_method_that_cannot_run_proves_nothing(
        tmp_path, monkeypatch, no_counterexample_found, error):
    """The band instance with approx reach raising (an unsupported layer,
    or any other error): no region is proved, the answer is ``unknown``.
    Catches a failed method counted as having proved its regions."""
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib", [_BAND_BOX], _BAND)
    _config(monkeypatch)
    _failing_reach(monkeypatch, {"approx"}, error)
    assert R.verify_instance(onnx_p, spec, "failing_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


@pytest.mark.parametrize("error", [NotImplementedError, RuntimeError],
                         ids=["unsupported", "error"])
def test_a_method_that_cannot_run_hands_on_to_the_next(
        tmp_path, monkeypatch, no_counterexample_found, error):
    """Two safe regions (y <= 1.5 < 1.75) and [approx, exact] with approx
    raising: exact proves both regions, the answer is ``unsat``. Neighbour
    of the case above; catches a failure that ends the whole reach stage."""
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [([0.0, 0.0], [0.5, 1.0], None),
                     ([0.5, -2.0], [1.0, 0.5], None)], ["(>= Y_0 1.75)"])
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    _failing_reach(monkeypatch, {"approx"}, error)
    assert R.verify_instance(onnx_p, spec, "failing_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def test_the_reach_gets_each_methods_options_the_shape_and_the_prepass(
        tmp_path, monkeypatch, no_counterexample_found):
    """Unsafe ``Y_0 >= 3`` on the four-ReLU net (y <= 2, so the true answer
    is ``unsat``) with [approx with relax factor 0.5 and relax method
    'area', exact]: the relaxed approx reach leaves it open and exact proves
    it. Each method's reach is called with that method's configured options,
    the ONNX input shape without its batch dimension and the interval bound
    pre-pass the runner sets for every sound method, and nothing else.
    Catches a method's options dropped or handed to another method, the
    input shape left out, and the pre-pass left out."""
    from n2v.nn import NeuralNetwork
    calls = []

    def make(model):
        real = NeuralNetwork(model)

        class Net:
            def reach(self, input_set, method, **kw):
                calls.append((method, dict(kw)))
                return real.reach(input_set, method=method, **kw)
        return Net()
    monkeypatch.setattr(R, "NeuralNetwork", make)
    onnx_p = _relu_net(tmp_path / "relu.onnx", _W1, [[1.0, 1.0, 1.0, 1.0]])
    spec = _spec_v1(tmp_path / "s.vnnlib", [([-1.0, -1.0], [1.0, 1.0], None)],
                    ["(>= Y_0 3.0)"])
    _config(monkeypatch, reach=[
        ("approx", {"relax_factor": 0.5, "relax_method": "area"}),
        ("exact", {})])
    assert R.verify_instance(onnx_p, spec, "options_demo",
                             workers=1)["result"] == R.RESULT_UNSAT
    assert calls == [
        ("approx", {"relax_factor": 0.5, "relax_method": "area",
                    "input_shape": (2,), "precompute_bounds": "ibp"}),
        ("exact", {"input_shape": (2,), "precompute_bounds": "ibp"})]


def test_an_input_set_too_large_for_the_host_proves_nothing(
        tmp_path, monkeypatch, no_counterexample_found):
    """y = sum(x) on ``[0, 1]^64`` with the band 31.999999 <= y <= 32.000001
    (x = 0.5 everywhere is a counterexample), and the memory guard's limit
    lowered to 16 KiB as on a small host: the input set is refused with
    ``MemoryError``, nothing is proved, the answer is ``unknown``. Catches
    the refusal counted as a proof."""
    n = 64
    onnx_p = _affine_net(tmp_path / "sum64.onnx", np.ones((1, n)), [0.0])
    y = _replay(onnx_p, np.full(n, 0.5), (1, n))[0]
    if not 31.999999 <= y <= 32.000001:
        pytest.fail(f"the counterexample replays to {y!r}")
    spec = _spec_v1(tmp_path / "s.vnnlib", [([0.0] * n, [1.0] * n, None)],
                    ["(>= Y_0 31.999999)", "(<= Y_0 32.000001)"])
    _config(monkeypatch)
    monkeypatch.setattr(R, "_MAX_INPUT_BASIS_BYTES", 16 * 1024)
    assert R.verify_instance(onnx_p, spec, "guard_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


def test_an_input_set_with_pinned_entries_is_built_and_proved(
        tmp_path, monkeypatch, no_counterexample_found):
    """The same 64-entry net and limit, with 62 entries pinned to 2.5/62
    and two free in [0, 1]: the basis fits (64 x 3 doubles), approx reach
    proves y <= 4.5 < 5, the answer is ``unsat``. Neighbour of the case
    above; catches the guard counting pinned entries (it then refuses a set
    that fits and the answer is unknown)."""
    n = 64
    onnx_p = _affine_net(tmp_path / "sum64.onnx", np.ones((1, n)), [0.0])
    lb, ub = np.full(n, 2.5 / (n - 2)), np.full(n, 2.5 / (n - 2))
    lb[:2], ub[:2] = 0.0, 1.0
    spec = _spec_v1(tmp_path / "s.vnnlib", [(lb, ub, None)],
                    ["(>= Y_0 5.0)"])
    _config(monkeypatch)
    monkeypatch.setattr(R, "_MAX_INPUT_BASIS_BYTES", 16 * 1024)
    assert R.verify_instance(onnx_p, spec, "guard_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


@pytest.mark.parametrize("witness_ok", [True, False],
                         ids=["replay-in-region", "replay-outside"])
def test_a_reach_sat_is_written_only_with_a_replayed_witness(
        tmp_path, monkeypatch, no_counterexample_found, witness_ok):
    """``verify_specification`` never answers SAT today (its docstring says
    so), so the reach stage's SAT branch is reached here by a stub that
    answers SAT with a candidate input. y = x0 + x1 on ``[0, 1]^2``, unsafe
    ``Y_0 >= 1.5``: the candidate (1, 0.75) replays into the region and is
    written as ``sat`` with its replay as Y; the candidate (0.25, 0.25) does
    not and gives ``unknown``, never ``sat``. Catches a reach SAT written
    without the replay check or with an output that is not the replay."""
    from n2v.utils.verify_specification import VerificationResult
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib", [_BAND_BOX], ["(>= Y_0 1.5)"])
    candidate = [1.0, 0.75] if witness_ok else [0.25, 0.25]
    monkeypatch.setattr(R, "verify_specification", lambda sets, prop:
                        VerificationResult(verdict="SAT",
                                           counterexample_x=candidate))
    _config(monkeypatch)
    res = R.verify_instance(onnx_p, spec, "reach_sat_demo", workers=1)
    if not witness_ok:
        assert res["result"] == R.RESULT_UNKNOWN
        return
    assert res["result"] == R.RESULT_SAT
    x, y = _witness(res["counterexample"])
    assert x == candidate
    assert y == _replay(onnx_p, x, (1, 2)).tolist() and y[0] >= 1.5


@pytest.mark.parametrize("method", [
    pytest.param(("approx", {}), id="approx"),
    pytest.param(("probabilistic", {"m": 200, "surrogate": "naive",
                                    "seed": 0}), id="probabilistic",
                 marks=known_bug(
                     20, "verify_instance writes 'unsat' when a configured "
                         "'probabilistic' reach method answers UNSAT: a "
                         "coverage statement is written as a proof (reach "
                         "still accepts the alias, REACH-2, and "
                         "verify_specification answers it with a plain "
                         "UNSAT, PROB-5; no shipped configuration names the "
                         "method today)")),
])
def test_an_instance_with_a_counterexample_at_a_corner_is_never_proved(
        tmp_path, monkeypatch, no_counterexample_found, method):
    """y = x0 + x1 on ``[0, 1]^2``, unsafe ``Y_0 >= 1.99``: the corner
    (1, 1) is a counterexample (y = 2, replayed here), which a sampled
    coverage set misses. With approx reach (the passing neighbour) or a
    'probabilistic' reach method configured the answer must not be
    ``unsat``. The probabilistic case is latent: no shipped configuration
    selects the method (``test_no_probabilistic_anywhere``)."""
    onnx_p = _band_instance(tmp_path)
    if _replay(onnx_p, [1.0, 1.0], (1, 2))[0] < 1.99:
        pytest.fail("the corner is not a counterexample")
    spec = _spec_v1(tmp_path / "s.vnnlib", [_BAND_BOX], ["(>= Y_0 1.99)"])
    _config(monkeypatch, reach=[method])
    assert R.verify_instance(onnx_p, spec, "prob_demo",
                             workers=1)["result"] != R.RESULT_UNSAT


# --------------------------------------------------------------------------- #
# The global configuration around a call                                      #
# --------------------------------------------------------------------------- #

def _snapshot():
    c = n2v_config
    return (c.parallel_lp, c.n_workers, c.auto_parallel,
            c.parallel_threshold, str(getattr(c.lp_solver, "value",
                                              c.lp_solver)))


@pytest.mark.parametrize("ends", ["returns", "raises"])
def test_verify_instance_gives_back_the_callers_configuration(
        tmp_path, monkeypatch, ends):
    """The caller runs with parallel LP off, 3 workers, no auto-parallel, a
    threshold of 7 and the 'highs' solver; ``verify_instance`` switches to
    parallel LP and linprog for its run and must give every one of these
    back, when it returns and when it raises (here from the network
    construction, after the switch). Catches the decorator dropped, a field
    left out of the restore, and a restore that runs only on return."""
    import n2v
    onnx_p = _band_instance(tmp_path)
    spec = _spec_v1(tmp_path / "s.vnnlib", [_BAND_BOX], ["(>= Y_0 2.5)"])
    _config(monkeypatch)
    if ends == "raises":
        def broken(model):
            raise RuntimeError("network construction failed")
        monkeypatch.setattr(R, "NeuralNetwork", broken)
    try:
        n2v.set_parallel(False, n_workers=3)
        n2v_config.auto_parallel = False
        n2v_config.parallel_threshold = 7
        n2v.set_lp_solver("highs")
        before = _snapshot()
        if ends == "raises":
            with pytest.raises(RuntimeError, match="construction failed"):
                R.verify_instance(onnx_p, spec, "config_demo", workers=2)
        else:
            assert R.verify_instance(onnx_p, spec, "config_demo",
                                     workers=2)["result"] == R.RESULT_UNSAT
        assert _snapshot() == before
    finally:
        n2v_config.reset()


@pytest.mark.parametrize("workers,env,expected", [
    (3, "7", 3),
    (None, "7", 7),
    (None, "0", 1),
    (None, "seven", None),
    (None, None, None),
], ids=["explicit", "env", "env-floor-1", "env-unreadable", "unset"])
def test_the_worker_count(monkeypatch, workers, env, expected):
    """An explicit count wins; otherwise ``N2V_WORKERS`` with a floor of 1;
    an unreadable or missing value gives the machine's core count (None
    here). Catches the variable ignored, its floor dropped, and an explicit
    count overridden."""
    if env is None:
        monkeypatch.delenv("N2V_WORKERS", raising=False)
    else:
        monkeypatch.setenv("N2V_WORKERS", env)
    expected = os.cpu_count() if expected is None else expected
    assert R._resolve_workers(workers) == expected


# --------------------------------------------------------------------------- #
# The nonlinear route                                                         #
# --------------------------------------------------------------------------- #

def _sum_spec_v2(path, n, extra, in_dims=None, bounded=None):
    dims = in_dims or [1, n]
    idx = (lambda i: f"0,{i}") if len(dims) == 2 else (
        lambda i: ",".join(["0"] * (len(dims) - 1) + [str(i)]))
    asserts = []
    for i in range(n):
        if bounded is None or i in bounded:
            asserts += [f"(>= X[{idx(i)}] 0.0)", f"(<= X[{idx(i)}] 1.0)"]
    return _spec_v2(path, dims, [1, 1], asserts + list(extra))


@pytest.mark.parametrize("failing", [(), ("approx",)],
                         ids=["approx", "approx-unsupported-then-exact"])
def test_a_nonlinear_instance_is_proved_on_a_flat_input_set(
        tmp_path, monkeypatch, no_counterexample_found, failing):
    """ACAS Xu's input declaration [1, 1, 1, 5]: Flatten and a MatMul give
    y = sum(x) on ``[0, 1]^5``, and the nonlinear property ``Y * Y >= 30.25``
    is unreachable (y * y <= 25). The nonlinear route proves it (``unsat``),
    also when the first configured method is unsupported and the next one
    runs. Catches the degenerate shape built as a 1x5 image (the nonlinear
    reach then cannot decide) and a route that gives up after an
    unsupported method."""
    onnx_p = _save(tmp_path / "acas.onnx",
                   [helper.make_node("Flatten", ["X"], ["f"], axis=1),
                    helper.make_node("MatMul", ["f", "W"], ["Y"])],
                   {"W": np.ones((5, 1), np.float32)}, [1, 1, 1, 5], [1, 1])
    spec = _sum_spec_v2(tmp_path / "s.vnnlib", 5,
                        ["(>= (* Y[0,0] Y[0,0]) 30.25)"],
                        in_dims=[1, 1, 1, 5])
    assert R.load_vnnlib(spec)["format"] == "nonlinear"
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    if failing:
        _failing_reach(monkeypatch, set(failing), NotImplementedError)
    assert R.verify_instance(onnx_p, spec, "nonlinear_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def test_a_nonlinear_instance_the_reach_cannot_decide_is_unknown(
        tmp_path, monkeypatch):
    """y = x0 + x1 on ``[0, 1]^2`` and the nonlinear band
    ``0.5625 <= Y * Y <= 0.5625000001``: x = (0.5, 0.25) gives y * y =
    0.5625 exactly, a real counterexample the sampler does not hit and the
    reach cannot rule out, so the answer is ``unknown``. Catches the
    reach's undecided answer written as ``unsat``."""
    onnx_p = _band_instance(tmp_path)
    spec = _sum_spec_v2(tmp_path / "s.vnnlib", 2,
                        ["(>= (* Y[0,0] Y[0,0]) 0.5625)",
                         "(<= (* Y[0,0] Y[0,0]) 0.5625000001)"])
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    assert R.verify_instance(onnx_p, spec, "nonlinear_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


@pytest.mark.parametrize("half", [[], ["(>= X[0,1] 0.0)"],
                                  ["(<= X[0,1] 1.0)"]],
                         ids=["no-bound", "lower-bound-only",
                              "upper-bound-only"])
def test_a_nonlinear_instance_with_an_unbounded_input_is_unknown(
        tmp_path, monkeypatch, no_counterexample_found, half):
    """X[0,1] has no bound in the spec, or a bound on one side only: y = x0
    + x1 is unbounded and ``Y * Y >= 6.25`` is reachable, the box cannot be
    built, the answer is ``unknown``. Catches an infinite bound on either
    side handed to the falsifier and the reach."""
    onnx_p = _band_instance(tmp_path)
    spec = _sum_spec_v2(tmp_path / "s.vnnlib", 2,
                        half + ["(>= (* Y[0,0] Y[0,0]) 6.25)"], bounded={0})
    _config(monkeypatch, reach=[("approx", {}), ("exact", {})])
    assert R.verify_instance(onnx_p, spec, "nonlinear_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


def test_a_nonlinear_witness_must_hold_in_the_checkers_float32(
        tmp_path, monkeypatch):
    """y0 = x0 with x0 pinned to 0.1, y1 = 1, unsafe ``Y0 * Y1 > 0.1``. The
    replay gives y0 = float32(0.1), whose product with 1 exceeds 0.1 in
    float64 but not in float32, the precision the checker uses, nor in exact
    arithmetic (0.1 * 1 = 0.1): no counterexample, the answer must not be
    ``sat``. Catches the nonlinear sat gate evaluated in float64 only."""
    onnx_p = _affine_net(tmp_path / "pin.onnx", [[1.0], [0.0]], [0.0, 1.0])
    spec = _spec_v2(tmp_path / "s.vnnlib", [1, 1], [1, 2],
                    ["(<= X[0,0] 0.1)", "(>= X[0,0] 0.1)",
                     "(> (* Y[0,0] Y[0,1]) 0.1)"])
    _config(monkeypatch)
    assert R.verify_instance(onnx_p, spec, "nonlinear_demo",
                             workers=1)["result"] != R.RESULT_SAT


def test_a_nonlinear_counterexample_on_the_replay_boundary_is_found(
        tmp_path, monkeypatch):
    """Y = [Sigmoid(x), 1] with x pinned to -2.99609375 and unsafe
    ``Y0 * Y1 <= c``, c the onnxruntime value there; the converted torch
    model gives a larger value (checked here). The search must run on the
    replay the competition grades: ``sat``, and the written witness replays
    to a product at most c. Catches a search on the converted model."""
    x0 = -2.99609375
    onnx_p = _save(tmp_path / "sig.onnx",
                   [helper.make_node("Sigmoid", ["X"], ["s"]),
                    helper.make_node("Concat", ["s", "one"], ["Y"], axis=1)],
                   {"one": np.ones((1, 1), np.float32)}, [1, 1], [1, 2])
    c = float(_replay(onnx_p, [x0], (1, 1))[0])
    with torch.no_grad():
        t = float(R.load_onnx(onnx_p)(torch.tensor([[x0]])).reshape(-1)[0])
    if not t > c:
        pytest.fail(f"no divergence at {x0}: torch {t!r}, onnxruntime {c!r}")
    spec = _spec_v2(tmp_path / "s.vnnlib", [1, 1], [1, 2],
                    [f"(<= X[0,0] {x0!r})", f"(>= X[0,0] {x0!r})",
                     f"(<= (* Y[0,0] Y[0,1]) {c!r})"])
    _config(monkeypatch, reach=[])
    res = R.verify_instance(onnx_p, spec, "nonlinear_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness(res["counterexample"])
    replay = _replay(onnx_p, x, (1, 1))
    assert x == [x0] and y == replay.tolist()
    assert np.float32(replay[0]) * np.float32(replay[1]) <= np.float32(c)


@pytest.mark.parametrize("n_rand,expected", [(100, 2000), (5000, 5000)])
def test_the_nonlinear_search_draws_at_least_2000_samples(
        tmp_path, monkeypatch, n_rand, expected):
    """The nonlinear search draws ``max(n_rand, 2000)`` samples: the floor
    for a small configured budget, the configured budget above it. Catches
    the floor applied with ``min``."""
    import n2v.utils.verify_nonlinear as vn
    seen = []

    def spy(model, lb, ub, prop, input_shape, *, n_samples, seed, forward):
        seen.append(n_samples)
        return None
    monkeypatch.setattr(vn, "falsify_nonlinear", spy)
    onnx_p = _band_instance(tmp_path)
    spec = _sum_spec_v2(tmp_path / "s.vnnlib", 2,
                        ["(>= (* Y[0,0] Y[0,0]) 6.25)"])
    _config(monkeypatch, reach=[], n_rand=n_rand)
    R.verify_instance(onnx_p, spec, "nonlinear_demo", workers=1)
    assert seen == [expected]


# --------------------------------------------------------------------------- #
# The ViT route                                                               #
# --------------------------------------------------------------------------- #
# A category containing "vit" goes to ``verify_vit_instance``. Its model
# reconstruction and CROWN bounds belong to ``n2v.nn.vit_crown``; here they are
# replaced by a linear model y = A x + c over the flattened [1, 3, 32, 32]
# input (the ONNX file computes the same in float32) and by a bound that is
# exact for it, so the runner's own logic is what is tested: the candidates,
# the zero-tolerance judgement and how the half-spaces are handed to the bound.

_VIT_N = 3 * 32 * 32


def _vit_instance(tmp_path, monkeypatch, seed=0):
    """The linear ViT stand-in: ONNX file, A, c and a box centre."""
    rng = np.random.default_rng(seed)
    A = (rng.integers(-8, 9, size=(2, _VIT_N)) / 64.0).astype(np.float32)
    c = np.array([0.5, -0.25], np.float32)
    onnx_p = _affine_net(tmp_path / "vit.onnx", A, c, [1, 3, 32, 32])
    A64, c64 = A.astype(np.float64), c.astype(np.float64)

    class Twin(torch.nn.Module):
        def forward(self, x):
            x = x.reshape(x.shape[0], -1).to(torch.float64)
            return x @ torch.from_numpy(A64).T + torch.from_numpy(c64)

    def exact_bound(ops, lb, ub, G, g, **kw):
        G = np.atleast_2d(G)
        W = G @ A64
        low = W @ ((lb + ub) / 2) - np.abs(W) @ ((ub - lb) / 2)
        margins = low + G @ c64 - np.asarray(g).reshape(-1)
        return bool(np.all(margins > 0)), margins

    import n2v.nn.vit_crown as vc
    monkeypatch.setattr(vc, "load_vit_onnx", lambda p: Twin())
    monkeypatch.setattr(vc, "to_ops", lambda m: m)
    monkeypatch.setattr(vc, "verify_halfspace_group_safe", exact_bound)
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {"n_rand": 8})
    centre = np.round(rng.uniform(-1, 1, _VIT_N) * 64) / 64
    return onnx_p, A64, c64, centre


def test_the_vit_gradient_search_reaches_a_corner_counterexample(
        tmp_path, monkeypatch):
    """Box of plus and minus 1/64 around a centre, unsafe ``Y_0 >= t`` with t
    at 80 % of the way from y_0 at the centre to its maximum over the box
    (at the corner of the gradient's signs). Uniform samples stay near the
    centre; the gradient search reaches the corner: ``sat``, with a witness
    in the box whose replay meets t. Category ``ViT_2023`` (upper case).
    Catches the search's loss with its sign flipped (it then walks away
    from the region) and a category match that depends on case."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    lb, ub = centre - 1 / 64, centre + 1 / 64
    top = A[0] @ centre + c[0] + np.abs(A[0]).sum() / 64
    t = float(A[0] @ centre + c[0] + 0.8 * (top - A[0] @ centre - c[0]))
    spec = _spec_v1(tmp_path / "s.vnnlib", [(lb, ub, None)],
                    [f"(>= Y_0 {t!r})"], n_out=2)
    res = R.verify_instance(onnx_p, spec, "ViT_2023", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness(res["counterexample"])
    x = np.asarray(x)
    assert np.all(x >= lb) and np.all(x <= ub)
    replay = _replay(onnx_p, x, (1, 3, 32, 32))
    assert y == replay.tolist() and replay[0] >= t


def test_a_vit_or_group_is_open_while_any_of_its_half_spaces_is(
        tmp_path, monkeypatch):
    """One output group, the OR of ``Y_1 >= 1e6`` (unreachable) and a band
    of half-width 1e-5 around y_0 at the centre (the centre replays into
    it, a real counterexample the search does not hit). The group is not
    unreachable, so the answer is ``unknown``. Catches the bound handed only
    the first half-space of each OR group."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    lb, ub = centre - 1 / 64, centre + 1 / 64
    v = float(_replay(onnx_p, centre, (1, 3, 32, 32))[0])
    lo, hi = v - 1e-5, v + 1e-5
    spec = _spec_v1(tmp_path / "s.vnnlib", [(lb, ub, None)],
                    [f"(or (and (>= Y_1 1000000.0)) "
                     f"(and (>= Y_0 {lo!r}) (<= Y_0 {hi!r})))"], n_out=2)
    assert R.verify_instance(onnx_p, spec, "vit_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


def test_a_vit_candidate_is_judged_on_its_replay_at_zero_tolerance(
        tmp_path, monkeypatch):
    """A point box at the centre, y_0 = v there (replayed), unsafe band
    ``v + 5e-5 <= Y_0 <= v + 5e-5 + 1e-6``. The centre passes the batched
    screen (slack 1e-4) but is no counterexample, and there is no other
    point: the answer must not be ``sat``. Catches the batch-1 judgement
    made with the screen's slack."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    v = float(_replay(onnx_p, centre, (1, 3, 32, 32))[0])
    t = v + 5e-5
    spec = _spec_v1(tmp_path / "s.vnnlib", [(centre, centre, None)],
                    [f"(>= Y_0 {t!r})", f"(<= Y_0 {t + 1e-6!r})"], n_out=2)
    assert R.verify_instance(onnx_p, spec, "vit_demo",
                             workers=1)["result"] != R.RESULT_SAT


@pytest.mark.parametrize("regions,expected", [
    ("one safe region", R.RESULT_UNSAT),
    ("a safe region and an open one", R.RESULT_UNKNOWN),
    ("an open region and a safe one", R.RESULT_UNKNOWN),
])
def test_vit_unsat_needs_every_region_proved(tmp_path, monkeypatch, regions,
                                             expected):
    """Unsafe ``Y_0 >= t``, t one above the box maximum of the first region
    (proved by the bound). A second region, moved along the signs of the
    first row of A so that y_0 at its centre is 2.5 above the first
    centre's, reaches t only near a corner (an open region; the gradient
    search is left out and the uniform samples stay near its centre). One
    region gives ``unsat``, both give ``unknown`` in either order. Catches a
    ViT stage that never proves, one that stops after the first proved
    region and one where the last region decides."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "_vit_pgd",
                        lambda md, lb, ub, gp, rng: np.empty((0, lb.size)))
    top = float(A[0] @ centre + c[0] + np.abs(A[0]).sum() / 64)
    boxes = [(centre - 1 / 64, centre + 1 / 64, None)]
    if regions != "one safe region":
        shifted = centre + np.sign(A[0]) * 2.5 / np.abs(A[0]).sum()
        if not top + 1.0 < A[0] @ shifted + c[0] + np.abs(A[0]).sum() / 64:
            pytest.fail("the second region does not reach the unsafe region")
        boxes.append((shifted - 1 / 64, shifted + 1 / 64, None))
        if regions.startswith("an open"):
            boxes.reverse()
    spec = _spec_v1(tmp_path / "s.vnnlib", boxes, [f"(>= Y_0 {top + 1.0!r})"],
                    n_out=2)
    assert R.verify_instance(onnx_p, spec, "vit_demo",
                             workers=1)["result"] == expected


@pytest.mark.parametrize("broken", ["bound", "reconstruction"])
def test_a_vit_stage_that_fails_proves_nothing(tmp_path, monkeypatch, broken):
    """One region far from the unsafe region (``Y_0 >= top + 1``, provable),
    with the bound raising, or the model reconstruction raising: nothing is
    proved, the answer is ``unknown``. Catches an error of the bound counted
    as a proof and a failed reconstruction that is not conceded."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    import n2v.nn.vit_crown as vc

    def boom(*a, **k):
        raise RuntimeError(f"{broken} failed")
    monkeypatch.setattr(vc, {"bound": "verify_halfspace_group_safe",
                             "reconstruction": "load_vit_onnx"}[broken], boom)
    top = float(A[0] @ centre + c[0] + np.abs(A[0]).sum() / 64)
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [(centre - 1 / 64, centre + 1 / 64, None)],
                    [f"(>= Y_0 {top + 1.0!r})"], n_out=2)
    assert R.verify_instance(onnx_p, spec, "vit_demo",
                             workers=1)["result"] == R.RESULT_UNKNOWN


def _spec_v2_box(path, lb, ub, in_dims, out_dims, outputs):
    """A VNN-LIB 2.0 spec with the box ``[lb, ub]`` over X (row-major over
    ``in_dims``, one bound per entry) and ``outputs`` as separate asserts,
    which the parser reads as separate AND'ed groups."""
    asserts = []
    for k, idx in enumerate(np.ndindex(*in_dims)):
        name = "X[" + ",".join(map(str, idx)) + "]"
        asserts += [f"(>= {name} {float(lb[k])!r})",
                    f"(<= {name} {float(ub[k])!r})"]
    return _spec_v2(path, in_dims, out_dims, asserts + list(outputs))


def test_a_vit_region_is_safe_when_any_of_its_and_groups_is_unreachable(
        tmp_path, monkeypatch):
    """VNN-LIB 2.0 with two output asserts, two AND'ed groups: ``Y_0 >= t0``
    with t0 one below the box minimum of y_0 (met everywhere) and then
    ``Y_1 >= t1`` with t1 one above the box maximum of y_1 (met nowhere).
    The unsafe region is their conjunction, so no input reaches it and the
    bound proves the second group unreachable: ``unsat``. Catches a ViT
    stage that bounds only the first AND group (the instance is then
    written unknown)."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)
    lb, ub = centre - 1 / 64, centre + 1 / 64
    low0 = float(A[0] @ centre + c[0] - np.abs(A[0]).sum() / 64)
    top1 = float(A[1] @ centre + c[1] + np.abs(A[1]).sum() / 64)
    spec = _spec_v2_box(tmp_path / "s.vnnlib", lb, ub, [1, 3, 32, 32], [1, 2],
                        [f"(>= Y[0,0] {low0 - 1.0!r})",
                         f"(>= Y[0,1] {top1 + 1.0!r})"])
    groups = R._extract_halfspace_groups(R.load_vnnlib(spec)["pairs"][0]["prop"])
    if len(groups) != 2:
        pytest.fail(f"the spec was read as {len(groups)} AND group(s)")
    assert R.verify_instance(onnx_p, spec, "vit_demo",
                             workers=1)["result"] == R.RESULT_UNSAT


def test_a_vit_gradient_search_that_fails_leaves_the_uniform_samples(
        tmp_path, monkeypatch):
    """The gradient search raises; unsafe ``Y_0 >= t`` with t the box minimum
    of y_0, so every uniform sample is a counterexample: ``sat``, with a
    witness in the box whose replay meets t. Catches a failure of the
    gradient search that ends the ViT stage (the counterexamples among the
    uniform samples are then lost)."""
    onnx_p, A, c, centre = _vit_instance(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("the gradient search failed")
    monkeypatch.setattr(R, "_vit_pgd", boom)
    lb, ub = centre - 1 / 64, centre + 1 / 64
    t = float(A[0] @ centre + c[0] - np.abs(A[0]).sum() / 64)
    spec = _spec_v1(tmp_path / "s.vnnlib", [(lb, ub, None)],
                    [f"(>= Y_0 {t!r})"], n_out=2)
    res = R.verify_instance(onnx_p, spec, "vit_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness(res["counterexample"])
    x = np.asarray(x)
    assert np.all(x >= lb) and np.all(x <= ub)
    replay = _replay(onnx_p, x, (1, 3, 32, 32))
    assert y == replay.tolist() and replay[0] >= t
