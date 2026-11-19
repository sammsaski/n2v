"""Regressions for the VNN-COMP 2026 runner's SAT witnesses.

The 2026 checker grades a 2.0 witness by parsing the section-5.3 text format
strictly (header shape must equal the declared shape, blocks in declaration
order), replaying the input through onnxruntime at batch 1, and evaluating
every assertion at zero tolerance on the float32 values. Each test here pins a
case where the runner's own acceptance disagreed with that grading:

  * a scalar ``[]`` declaration written as ``[1]`` (malformed witness);
  * witness blocks routed by a ``Y`` name prefix instead of the declaration
    kind (malformed witness);
  * a strict ``<`` / ``>`` accepted on its closed lowering (a boundary point
    graded "spec not violated");
  * a ViT candidate accepted on its batched onnxruntime row, which can differ
    from the batch-1 replay;
  * an ONNX input without a batch dimension, on which every falsifier crashed.

From the section "The checker" on, the tests read what the runner writes the
way the competition does, with a checker of their own: the written witness is
parsed in the format of the spec's version, its input must lie in the box at
zero tolerance, its outputs must be the onnxruntime replay of that input, and
the replay must violate the property, decided in exact arithmetic from the
instance as the test wrote it down. They cover the witness writer
(``_snap_into_box``, ``_v2_ce_meta``, ``_format_ce_v2``,
``format_counterexample``), the sat gate (``_witness_satisfies_spec``,
``_assertions_hold``, ``in_unsafe_region``) and the results file
(``write_result``), directly and through ``verify_instance``, the nonlinear
and ViT routes and ``main``.
"""

import os
import re
import signal
import sys
from fractions import Fraction

import numpy as np
import pytest
import torch

import onnx
import onnxruntime  # noqa: F401 - a hard dependency; a broken install must fail
pytest.importorskip("onnx2torch")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402
from n2v.sets.halfspace import HalfSpace  # noqa: E402
from tests.cell_support import replay_onnx  # noqa: E402
from tests.known_bugs import known_bug  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v2_spec(path, inputs, outputs, asserts):
    """Write a VNNLIB 2.0 spec: ``inputs``/``outputs`` are (name, dims-string)."""
    decl = [f"    (declare-input  {n} float32 [{d}])" for n, d in inputs]
    decl += [f"    (declare-output {n} float32 [{d}])" for n, d in outputs]
    text = ("(vnnlib-version <2.0>)\n(declare-network N\n" + "\n".join(decl)
            + "\n)\n" + "\n".join(f"(assert {a})" for a in asserts) + "\n")
    path.write_text(text)
    return str(path)


def _save(nodes, inputs, outputs, inits, path):
    g = helper.make_graph(nodes, "g", inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return str(path)


def _relu_net(path, w, b, shift=0.0, in_name="X", out_name="Y"):
    """ONNX ``out = relu(in @ w + b) + shift`` with in [1, len(w)], out [1, 1]."""
    W = numpy_helper.from_array(np.asarray(w, np.float32).reshape(-1, 1), "W")
    B = numpy_helper.from_array(np.asarray([b], np.float32), "B")
    S = numpy_helper.from_array(np.asarray([[shift]], np.float32), "S")
    nodes = [helper.make_node("Gemm", [in_name, "W", "B"], ["h"]),
             helper.make_node("Relu", ["h"], ["r"]),
             helper.make_node("Add", ["r", "S"], [out_name])]
    return _save(nodes,
                 [helper.make_tensor_value_info(in_name, TensorProto.FLOAT,
                                                [1, len(w)])],
                 [helper.make_tensor_value_info(out_name, TensorProto.FLOAT,
                                                [1, 1])],
                 [W, B, S], path)


def _box_asserts(name, n, index=lambda i: f"0,{i}"):
    out = []
    for i in range(n):
        out += [f"(>= {name}[{index(i)}] 0.0)", f"(<= {name}[{index(i)}] 1.0)"]
    return out


@pytest.fixture
def runner_env(monkeypatch):
    """Falsification only (no reach methods). The runner's global
    ``set_parallel``/``set_lp_solver`` calls run for real: the runner restores
    the configuration on exit and the suite-wide fixture fails any leak."""
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {
        "reach_methods": [], "n_rand": 64, "falsify_method": "random+pgd"})


def _blocks(witness):
    """Parse a 2.0 witness the way the checker does: {name: (dims, values)}."""
    lines = [ln.strip() for ln in witness.splitlines() if ln.strip()]
    out, i = {}, 0
    while i < len(lines):
        name, _dtype, dims = lines[i].split(" ", 2)
        shape = [int(d) for d in dims.strip("[]").split(",") if d.strip()]
        n = int(np.prod(shape))
        out[name] = (shape, [float(v) for v in lines[i + 1:i + 1 + n]])
        i += 1 + n
    return out


# ---------------------------------------------------------------------------
# A scalar declaration keeps its [] header
# ---------------------------------------------------------------------------

def test_scalar_output_header_is_empty_brackets(tmp_path):
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "")],
                    _box_asserts("X", 2) + ["(>= Y 1.0)"])
    assert R._v2_ce_meta(spec) == [("input", "X", "float32", [1, 2]),
                                   ("output", "Y", "float32", [])]
    ce = R.format_counterexample([1.0, 0.5], [1.5], spec)
    assert ce.splitlines() == ["X float32 [1,2]", "1.0", "0.5",
                               "Y float32 []", "1.5"]


# ---------------------------------------------------------------------------
# Blocks are routed by declare-input / declare-output, not by name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_name,out_name", [("inp", "logits"),
                                              ("Yin", "out")])
def test_blocks_routed_by_declaration_kind(tmp_path, in_name, out_name):
    spec = _v2_spec(tmp_path / "s.vnnlib", [(in_name, "1, 2")],
                    [(out_name, "1, 1")],
                    _box_asserts(in_name, 2) + [f"(<= {out_name}[0,0] 0.0)"])
    blocks = _blocks(R.format_counterexample([0.25, 0.5], [0.75], spec))
    assert blocks == {in_name: ([1, 2], [0.25, 0.5]),
                      out_name: ([1, 1], [0.75])}


def test_short_witness_raises_instead_of_writing_malformed(tmp_path):
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 2")],
                    _box_asserts("X", 2) + ["(<= Y[0,0] 0.0)"])
    with pytest.raises(ValueError, match="expected 2"):
        R.format_counterexample([0.25, 0.5], [0.75], spec)


def test_runner_sat_with_custom_tensor_names(tmp_path, runner_env):
    """End to end: a spec naming its tensors ``inp``/``logits`` gets a witness
    with the output value under the output header."""
    onnx_p = _relu_net(tmp_path / "n.onnx", [1.0, 1.0], 0.0, shift=-0.5,
                       in_name="inp", out_name="logits")
    spec = _v2_spec(tmp_path / "s.vnnlib", [("inp", "1, 2")],
                    [("logits", "1, 1")],
                    _box_asserts("inp", 2) + ["(<= logits[0,0] 0.0)"])
    res = R.verify_instance(onnx_p, spec, "names_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    blocks = _blocks(res["counterexample"])
    x = np.array(blocks["inp"][1])
    assert blocks["logits"][0] == [1, 1]
    assert blocks["logits"][1][0] == pytest.approx(max(x.sum(), 0) - 0.5,
                                                   abs=1e-6)


# ---------------------------------------------------------------------------
# Strict comparisons are checked strictly before emitting sat
# ---------------------------------------------------------------------------

def test_parser_keeps_assertions_only_for_strict_specs(tmp_path):
    base = _box_asserts("X", 2)
    strict = R.load_vnnlib(_v2_spec(tmp_path / "a.vnnlib", [("X", "1, 2")],
                                    [("Y", "1, 1")],
                                    base + ["(< Y[0,0] 0.0)"]))
    closed = R.load_vnnlib(_v2_spec(tmp_path / "b.vnnlib", [("X", "1, 2")],
                                    [("Y", "1, 1")],
                                    base + ["(<= Y[0,0] 0.0)"]))
    # Same closed lowering for reach; only the strict spec keeps its asserts.
    hs, hc = strict["pairs"][0]["prop"][0]["Hg"], closed["pairs"][0]["prop"][0]["Hg"]
    assert np.array_equal(hs.G, hc.G) and np.array_equal(hs.g, hc.g)
    assert "assertions" not in closed
    assert len(strict["assertions"]) == 5
    combined = R.load_vnnlib(_v2_spec(
        tmp_path / "c.vnnlib", [("X", "1, 2")], [("Y", "1, 1")],
        ["(or (and (>= X[0,0] 0.0) (<= X[0,0] 1.0) (>= X[0,1] 0.0)"
         " (<= X[0,1] 1.0) (> Y[0,0] 0.5)))"]))
    assert len(combined["assertions"]) == 1


@pytest.mark.parametrize("text,strict", [
    ("(vnnlib-version <2.0>)\n(assert (<= Y[0] 0.1))", False),
    ("(assert (>= Y 1))", False),
    ("(assert (< Y[0] 0.1))", True),
    ("(assert (Y[0] > 0.1))", True),                  # infix
    ("(assert (>\nY 1))", True),
    ("(assert (<(+ Y 1) 2))", True),
])
def test_strict_comparison_detection(text, strict):
    from n2v.utils.vnnlib2 import _has_strict_comparison
    assert _has_strict_comparison(text) is strict


def test_strict_output_bound_on_boundary_is_not_sat(tmp_path, runner_env):
    """y = relu(-x0 - x1 - 1) is 0 on [0,1]^2, so ``Y < 0`` is unreachable,
    but every sample lies on the closed row ``Y <= 0``."""
    onnx_p = _relu_net(tmp_path / "n.onnx", [-1.0, -1.0], -1.0)
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 1")],
                    _box_asserts("X", 2) + ["(< Y[0,0] 0.0)"])
    res = R.verify_instance(onnx_p, spec, "strict_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN
    assert res["counterexample"] is None


def test_strict_output_bound_with_margin_is_sat(tmp_path, runner_env):
    onnx_p = _relu_net(tmp_path / "n.onnx", [-1.0, -1.0], -1.0)
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 1")],
                    _box_asserts("X", 2) + ["(< Y[0,0] 0.5)"])
    res = R.verify_instance(onnx_p, spec, "strict_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    assert _blocks(res["counterexample"])["Y"][1] == [0.0]


def test_strict_bound_at_float32_constant_is_not_sat(tmp_path, runner_env):
    """y = float32(0.1) everywhere. ``Y > 0.1`` holds in float64 but not in
    the float32 comparison the checker makes, so there is no counterexample."""
    onnx_p = _relu_net(tmp_path / "n.onnx", [0.0, 0.0], 0.1)
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 1")],
                    _box_asserts("X", 2) + ["(> Y[0,0] 0.1)"])
    res = R.verify_instance(onnx_p, spec, "strict_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN


def test_strict_gate_uses_checker_float32_semantics(tmp_path):
    """The checker compares the float32 replay output with the spec constant
    in float32: ``float32(0.1) > 0.1`` holds in float64 but not there."""
    prop = R.load_vnnlib(_v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")],
                                  [("Y", "1, 1")],
                                  _box_asserts("X", 2) + ["(> Y[0,0] 0.1)"]))
    groups = R._extract_halfspace_groups(prop["pairs"][0]["prop"])
    x = np.array([0.5, 0.5])
    y = np.array([float(np.float32(0.1))])
    assert y[0] > 0.1                                   # float64 says unsafe
    assert not R._witness_satisfies_spec(prop, groups, x, y)
    assert R._witness_satisfies_spec(prop, groups, x, np.array([0.2]))


def test_strict_input_bound_rejects_boundary_witness(tmp_path):
    prop = R.load_vnnlib(_v2_spec(
        tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 1")],
        ["(> X[0,0] 0.0)", "(<= X[0,0] 1.0)"] + _box_asserts("X", 2)[2:]
        + ["(<= Y[0,0] 0.0)"]))
    groups = R._extract_halfspace_groups(prop["pairs"][0]["prop"])
    y = np.array([-1.0])
    assert not R._witness_satisfies_spec(prop, groups, [0.0, 0.5], y)
    assert R._witness_satisfies_spec(prop, groups, [0.25, 0.5], y)


# Arithmetic is evaluated by the checker in float32, one rounding per
# operation, where the float64 row need not agree.

def _const_net(path, values, n_in=2):
    """ONNX ``Y = X @ 0 + values``: every input maps to the float32 ``values``."""
    W = numpy_helper.from_array(np.zeros((n_in, len(values)), np.float32), "W")
    B = numpy_helper.from_array(np.asarray(values, np.float32), "B")
    return _save([helper.make_node("Gemm", ["X", "W", "B"], ["Y"])],
                 [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                                [1, n_in])],
                 [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                                [1, len(values)])],
                 [W, B], path)


@pytest.mark.parametrize("text,arith", [
    ("(assert (<= X[0,0] -0.5))", False),             # negative literal
    ("(assert (<= X[0,0] 1e+05))", False),            # exponent sign
    ("(vnnlib-version <2.0>)\n(declare-network N)", False),
    ("(assert (<= (+ Y[0,0] Y[0,1]) 0))", True),
    ("(assert (>= (- Y[0,0]) 1))", True),
    ("(assert (<= ( * 0.3 Y) 1))", True),
    ("(assert (<= (/\nY 2) 1))", True),
])
def test_arithmetic_detection(text, arith):
    from n2v.utils.vnnlib2 import _has_arithmetic
    assert _has_arithmetic(text) is arith


def test_three_term_sum_checked_in_float32_is_not_sat(tmp_path, runner_env):
    """Y = [1 + 2^-23, 2^-24, -1] everywhere: the float64 sum is exactly
    c = 1.5 * 2^-23, but the checker's float32 sum rounds 1 + 2^-23 + 2^-24
    up to 1 + 2^-22 and gets 2^-22 > c, so there is no counterexample."""
    y = [1.0 + 2.0 ** -23, 2.0 ** -24, -1.0]
    onnx_p = _const_net(tmp_path / "n.onnx", y)
    c = np.format_float_positional(1.5 * 2.0 ** -23)
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 3")],
                    _box_asserts("X", 2)
                    + [f"(<= (+ Y[0,0] Y[0,1] Y[0,2]) {c})"])
    res = R.verify_instance(onnx_p, spec, "sum_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN
    prop = R.load_vnnlib(spec)
    groups = R._extract_halfspace_groups(prop["pairs"][0]["prop"])
    assert R.in_unsafe_region(np.array(y), groups, tol=0.0)  # float64 row
    assert len(prop["assertions"]) == 5


def test_coefficient_checked_in_float32():
    """0.3 * y <= c holds in float64 at y = float32(1.5942448377609253) and
    c = 0.4782734513282776, but not as float32(0.3) * y <= float32(c)."""
    y = float(np.float32(1.5942448377609253))
    c = 0.4782734513282776
    assert 0.3 * y <= c
    assert not np.float32(0.3) * np.float32(y) <= np.float32(c)
    text = ("(vnnlib-version <2.0>)\n(declare-network N\n"
            "    (declare-input  X float32 [1, 2])\n"
            "    (declare-output Y float32 [1, 1])\n)\n"
            + "\n".join(f"(assert {a})" for a in _box_asserts("X", 2))
            + f"\n(assert (<= (* 0.3 Y[0,0]) {c!r}))\n")
    from n2v.utils.vnnlib2 import load_vnnlib_v2
    prop = load_vnnlib_v2("coef.vnnlib", text=text)
    groups = R._extract_halfspace_groups(prop["prop"])
    x = np.array([0.5, 0.5])
    assert R.in_unsafe_region(np.array([y]), groups, tol=0.0)
    assert not R._witness_satisfies_spec(prop, groups, x, np.array([y]))
    assert R._witness_satisfies_spec(prop, groups, x, np.array([1.5]))


def test_float32_evaluation_does_not_rely_on_numpy2_promotion():
    """Constants and accumulator starts are float32 scalars, so the float32
    evaluation stays in float32 under NumPy 1.x promotion rules as well."""
    from n2v.utils.vnnlib2 import evaluate_ast
    x, y = np.zeros(1), np.array([0.25, 0.5])
    f32 = np.float32
    assert type(evaluate_ast(("const", 0.1), x, y, f32)) is f32
    total = ("+", ("var", "out", 0), ("var", "out", 1), ("const", 0.1))
    assert type(evaluate_ast(total, x, y, f32)) is f32
    assert type(evaluate_ast(("*", ("const", 3.0), ("var", "out", 0)),
                             x, y, f32)) is f32
    assert evaluate_ast(("const", 0.1), x, y) == 0.1       # float64 default


# ---------------------------------------------------------------------------
# ViT candidates are judged on their batch-1 replay
# ---------------------------------------------------------------------------

def _vit_stubs(monkeypatch, batched_shift):
    """Stub the ViT reconstruction/CROWN (stage 2 never proves) and give the
    onnxruntime forward a batched run whose rows read ``batched_shift`` off
    the batch-1 output of the same point."""
    import n2v.nn.vit_crown as vc
    monkeypatch.setattr(vc, "load_vit_onnx", lambda p: None)
    monkeypatch.setattr(vc, "to_ops", lambda m: None)
    monkeypatch.setattr(vc, "verify_halfspace_group_safe",
                        lambda *a, **k: (False, None))
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {"n_rand": 4})
    monkeypatch.setattr(R, "_vit_pgd",
                        lambda md, lb, ub, gp, rng: np.empty((0, lb.size)))

    def make_forward(_path):
        def fwd(X):
            X = np.atleast_2d(np.asarray(X, dtype=np.float64))
            y = X.sum(axis=1, keepdims=True)
            return y + batched_shift if X.shape[0] > 1 else y
        return fwd
    monkeypatch.setattr(R, "make_onnx_forward", make_forward)


def _vit_prop(x0, t):
    """Point box at ``x0``; unsafe region ``sum(x) <= t``."""
    hs = HalfSpace(np.array([[1.0]]), np.array([[t]]))
    return {"pairs": [{"lb": x0.copy(), "ub": x0.copy(),
                       "prop": [{"Hg": hs}]}]}


def test_vit_batched_hit_rejected_by_batch_one_replay(monkeypatch):
    _vit_stubs(monkeypatch, batched_shift=-1e-3)
    x0 = np.array([0.25, 0.5])
    res = R.verify_vit_instance("vit.onnx", _vit_prop(x0, 0.75 - 5e-4),
                                (2,), "vit_2023", 0.0)
    assert res["result"] == R.RESULT_UNKNOWN


def test_vit_batch_one_hit_emitted_with_batch_one_output(monkeypatch):
    # The batched rows sit just outside the region; the batch-1 replay is in.
    _vit_stubs(monkeypatch, batched_shift=+5e-5)
    x0 = np.array([0.25, 0.5])
    res = R.verify_vit_instance("vit.onnx", _vit_prop(x0, 0.75),
                                (2,), "vit_2023", 0.0)
    assert res["result"] == R.RESULT_SAT
    assert "(Y_0  0.75)" in res["counterexample"]


# ---------------------------------------------------------------------------
# An ONNX input without a batch dimension
# ---------------------------------------------------------------------------

def _unbatched_net(path):
    """ONNX with input X [6] and no batch dim: the first 4 entries are
    reshaped to [1, 4] (as cctsdb_yolo_2023 does with [12296] -> [1,3,64,64])
    and summed, Y [1, 1]. A batch-prefixed input fails at the Reshape."""
    starts = numpy_helper.from_array(np.array([0], np.int64), "st")
    ends = numpy_helper.from_array(np.array([4], np.int64), "en")
    axes = numpy_helper.from_array(np.array([0], np.int64), "ax")
    shape = numpy_helper.from_array(np.array([1, 4], np.int64), "sh")
    W = numpy_helper.from_array(np.ones((4, 1), np.float32), "W")
    B = numpy_helper.from_array(np.zeros(1, np.float32), "B")
    nodes = [helper.make_node("Slice", ["X", "st", "en", "ax"], ["s"]),
             helper.make_node("Reshape", ["s", "sh"], ["r"]),
             helper.make_node("Gemm", ["r", "W", "B"], ["Y"])]
    return _save(nodes,
                 [helper.make_tensor_value_info("X", TensorProto.FLOAT, [6])],
                 [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                                [1, 1])],
                 [starts, ends, axes, shape, W, B], path)


def test_unbatched_onnx_input_is_falsified(tmp_path, runner_env):
    onnx_p = _unbatched_net(tmp_path / "n.onnx")
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "6")], [("Y", "1, 1")],
                    _box_asserts("X", 6, index=str) + ["(>= Y[0,0] 2.5)"])
    res = R.verify_instance(onnx_p, spec, "unbatched_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    blocks = _blocks(res["counterexample"])
    x = np.array(blocks["X"][1])
    assert blocks["X"][0] == [6] and np.all((x >= 0) & (x <= 1))
    assert blocks["Y"][1][0] == pytest.approx(x[:4].sum(), abs=1e-5)
    assert blocks["Y"][1][0] >= 2.5


def test_falsify_model_leaves_batchable_models_alone(tmp_path):
    from n2v.utils.model_loader import load_onnx
    onnx_p = _relu_net(tmp_path / "n.onnx", [1.0, 1.0], 0.0)
    model = load_onnx(onnx_p)
    # Batch dim stripped from [1, 2]: the model is returned as is, unprobed.
    assert R._falsify_model(model, (2,), (1, 2), np.zeros(2)) is model
    # No batch dim declared, but the graph accepts a batch prefix anyway.
    lin = torch.nn.Linear(3, 1)
    assert R._falsify_model(lin, (3,), (3,), np.zeros(3)) is lin
    # No batch dim and a batch prefix is rejected: wrapped per sample.
    unb = load_onnx(_unbatched_net(tmp_path / "u.onnx"))
    wrapped = R._falsify_model(unb, (6,), (6,), np.full(6, 0.5))
    assert isinstance(wrapped, R._PerSampleModel)
    out = wrapped(torch.ones(3, 6))
    assert out.shape == (3, 1) and torch.allclose(out, torch.full((3, 1), 4.0))


# ---------------------------------------------------------------------------
# The checker: the test's own reading of what the runner wrote
# ---------------------------------------------------------------------------
#
# An instance is written down here, independently of n2v: the ONNX input
# tensor shape, one interval per input coordinate (row-major over the tensor),
# the output tensor shape, and the unsafe region as a list of asserts that are
# ANDed, each an OR of conjunctions of atoms (j, op, c) on the flat output,
# y_j op c. Its spec is written in either VNN-LIB version. The checker parses
# the written witness itself, demands the format of that version (variables
# numbered from 0 in 1.0; blocks named, typed and shaped as declared, in
# declaration order, in 2.0), checks the written input against the box and
# replays it on onnxruntime, and decides the property on that replay; box and
# property are compared in exact arithmetic on the decimal text of the spec.

_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


class _Instance:
    def __init__(self, W, b, in_shape, lb, ub, out_shape, unsafe):
        self.W, self.b = np.asarray(W, np.float32), np.asarray(b, np.float32)
        self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
        self.lb, self.ub = [float(v) for v in lb], [float(v) for v in ub]
        self.unsafe = unsafe

    def onnx(self, tmp_path):
        """X (in_shape) -> Reshape [1, d] -> Gemm -> Y [1, n_out]."""
        d = int(np.prod(self.in_shape))
        nodes = [helper.make_node("Reshape", ["X", "sh"], ["f"]),
                 helper.make_node("Gemm", ["f", "W", "B"], ["Y"], transB=1)]
        inits = [numpy_helper.from_array(np.array([1, d], np.int64), "sh"),
                 numpy_helper.from_array(self.W, "W"),
                 numpy_helper.from_array(self.b, "B")]
        return _save(nodes,
                     [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                                    list(self.in_shape))],
                     [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                                    list(self.out_shape))],
                     inits, tmp_path / "net.onnx")

    def names(self, version):
        if version == "1.0":
            return ([f"X_{i}" for i in range(len(self.lb))],
                    [f"Y_{j}" for j in range(int(np.prod(self.out_shape)))])
        return ([f"X[{','.join(map(str, k))}]"
                 for k in np.ndindex(*self.in_shape)],
                [f"Y[{','.join(map(str, k))}]"
                 for k in np.ndindex(*self.out_shape)])

    def spec(self, tmp_path, version):
        xs, ys = self.names(version)
        if version == "1.0":
            lines = [f"(declare-const {v} Real)" for v in xs + ys]
        else:
            lines = ["(vnnlib-version <2.0>)", "(declare-network N",
                     f"    (declare-input X float32 [{_dims(self.in_shape)}])",
                     f"    (declare-output Y float32 [{_dims(self.out_shape)}])",
                     ")"]
        for v, lo, hi in zip(xs, self.lb, self.ub):
            lines += [f"(assert (>= {v} {lo!r}))", f"(assert (<= {v} {hi!r}))"]
        for disjuncts in self.unsafe:
            conj = [[f"({op} {ys[j]} {c!r})" for j, op, c in atoms]
                    for atoms in disjuncts]
            if len(conj) > 1:
                lines.append("(assert (or " + " ".join(
                    f"(and {' '.join(a)})" for a in conj) + "))")
            elif version == "1.0" or len(conj[0]) == 1:
                lines += [f"(assert {a})" for a in conj[0]]
            else:
                lines.append(f"(assert (and {' '.join(conj[0])}))")
        path = tmp_path / f"spec_{version}.vnnlib"
        path.write_text("\n".join(lines) + "\n")
        return str(path)


def _dims(shape):
    return ", ".join(str(d) for d in shape)


def _value(text, what):
    """The exact value of one written number; anything else is malformed."""
    if not _NUMBER.fullmatch(text):
        pytest.fail(f"{what}: {text!r} is not a number")
    return Fraction(text)


def _read_v1(witness, n_in, n_out):
    """The 1.0 witness ``((X_0 v) ... (Y_0 v) ...)``: every input numbered
    0..n_in-1 and every output 0..n_out-1 exactly once; the written texts."""
    body = witness.strip()
    assert body.startswith("(") and body.endswith(")"), witness[:80]
    atom = r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)"
    assert re.sub(atom, "", body[1:-1]).strip() == "", witness[:200]
    found = {"X": {}, "Y": {}}
    for kind, i, v in re.findall(atom, body[1:-1]):
        assert int(i) not in found[kind], f"{kind}_{i} written twice"
        found[kind][int(i)] = v
    assert sorted(found["X"]) == list(range(n_in)), sorted(found["X"])
    assert sorted(found["Y"]) == list(range(n_out)), sorted(found["Y"])
    return ([found["X"][i] for i in range(n_in)],
            [found["Y"][j] for j in range(n_out)])


def _read_v2(witness, declared):
    """The 2.0 textual witness: for each declared (name, dtype, shape), in
    declaration order, the header ``name dtype [d0,d1,...]`` with the
    declared shape, then its values one per line; nothing else."""
    lines = [ln.strip() for ln in witness.splitlines() if ln.strip()]
    blocks, i = {}, 0
    for name, dtype, shape in declared:
        assert i < len(lines), f"no block for {name}"
        m = re.fullmatch(r"(\S+) (\S+) \[([0-9,\s]*)\]", lines[i])
        assert m, f"malformed header {lines[i]!r}"
        dims = [int(d) for d in m.group(3).split(",") if d.strip()]
        assert (m.group(1), m.group(2), dims) == (name, dtype, list(shape))
        n = int(np.prod(shape))
        blocks[name] = lines[i + 1:i + 1 + n]
        assert len(blocks[name]) == n, f"{name}: {len(blocks[name])} values"
        i += 1 + n
    assert i == len(lines), f"trailing lines {lines[i:i + 3]}"
    return blocks


def _read_witness(witness, version, inst):
    """The written input and output texts of the instance's witness."""
    if version == "1.0":
        return _read_v1(witness, len(inst.lb), int(np.prod(inst.out_shape)))
    blocks = _read_v2(witness, [("X", "float32", inst.in_shape),
                                ("Y", "float32", inst.out_shape)])
    return blocks["X"], blocks["Y"]


def _holds(y, op, c):
    y, c = Fraction(float(y)), Fraction(repr(float(c)))
    return {"<=": y <= c, ">=": y >= c, "<": y < c, ">": y > c}[op]


def _replayed_point(witness, version, inst, onnx_path):
    """Read the witness as the checker does: the written input inside the box
    (zero tolerance), the written outputs equal to the onnxruntime replay of
    the written input. Returns the replayed output."""
    x_text, y_text = _read_witness(witness, version, inst)
    x = [_value(t, f"input {i}") for i, t in enumerate(x_text)]
    for i, (v, lo, hi) in enumerate(zip(x, inst.lb, inst.ub)):
        assert Fraction(repr(lo)) <= v <= Fraction(repr(hi)), (
            f"input {i} written as {x_text[i]}, outside [{lo!r}, {hi!r}]")
    y = replay_onnx(onnx_path, [[float(v) for v in x]], inst.in_shape)[0]
    written = [float(_value(t, f"output {j}")) for j, t in enumerate(y_text)]
    assert written == y.tolist(), "written outputs are not the replay"
    return y


def _assert_accepted(witness, version, inst, onnx_path):
    """The competition's verdict on a sat witness: correct only when the
    replay of the written input violates the property."""
    y = _replayed_point(witness, version, inst, onnx_path)
    assert all(any(all(_holds(y[j], op, c) for j, op, c in atoms)
                   for atoms in disjuncts) for disjuncts in inst.unsafe), (
        f"replayed output {y.tolist()} is not in the unsafe region")


# ---------------------------------------------------------------------------
# A sat witness written by the runner is one the checker accepts
# ---------------------------------------------------------------------------

_ONE_TWO = [[1.0, 1.0], [1.0, -1.0]]
_IMAGE_W = np.zeros((2, 24))
_IMAGE_W[0, 5], _IMAGE_W[1, 17] = 1.0, -1.0

_WITNESS_INSTANCES = {
    # x0 pinned to 0.1, not a float32 value; X and Y of equal size, so a
    # block filled from the wrong vector is still well formed.
    "pinned at 0.1": _Instance(_ONE_TWO, [0.0, 0.0], (1, 2), [0.1, 0.0],
                               [0.1, 1.0], (1, 2), [[[(0, ">=", 0.5)]]]),
    # a bound that needs all 17 significant digits to be read back exactly
    # (any shorter printing, 9 or 16 digits, writes 0.3)
    "pinned at a 17-digit value": _Instance(
        [[1.0, 1.0]], [0.0], (1, 2), [0.30000000000000004, 0.0],
        [0.30000000000000004, 1.0], (1, 1), [[[(0, ">=", 0.75)]]]),
    # y = x0 >= 0.3 is met by the replay only at x0 = 0.3, the upper face,
    # which float32 rounds to 0.30000001192092896 (outside the box)
    "on the upper face at 0.3": _Instance(
        [[1.0, 0.0]], [0.0], (1, 2), [0.0, 0.0], [0.3, 1.0], (1, 1),
        [[[(0, ">=", 0.3)]]]),
    # a (2, 3, 4) image input, every entry with its own interval, so an
    # entry written in another position or under another number leaves it
    "image input with a box per entry": _Instance(
        _IMAGE_W, [0.0, 0.0], (1, 2, 3, 4), [10.0 * k for k in range(24)],
        [10.0 * k + 0.5 for k in range(24)], (1, 2),
        [[[(0, ">=", 50.25)]], [[(1, ">=", -170.25)]]]),
}


class TestTheWrittenWitnessIsAcceptedByTheChecker:
    @pytest.mark.parametrize("version", ["1.0", "2.0"])
    @pytest.mark.parametrize("case", list(_WITNESS_INSTANCES))
    def test_sat_witness_replays_inside_the_box(self, tmp_path, runner_env,
                                                case, version):
        """``verify_instance`` with the real falsifier on an instance with a
        counterexample: the answer is sat and the written witness passes the
        checker. Catches a witness snapped into the box and then rounded to
        float32, a value printed with too few digits (9 in 1.0, 16 in 2.0), 1.0
        inputs numbered from 1, a 2.0 block filled from the wrong vector, the
        wrong format for the spec's version, an entry of an image written in
        another position, and written outputs that are not the replay."""
        inst = _WITNESS_INSTANCES[case]
        onnx_p = inst.onnx(tmp_path)
        res = R.verify_instance(onnx_p, inst.spec(tmp_path, version),
                                "witness_demo", workers=1)
        assert res["result"] == R.RESULT_SAT
        _assert_accepted(res["counterexample"], version, inst, onnx_p)


def test_nonlinear_route_writes_a_witness_the_checker_accepts(tmp_path,
                                                              runner_env):
    """``verify_nonlinear_instance`` searches and replays on onnxruntime
    (``make_onnx_forward``, one batched call over a dynamic batch dim),
    snaps, checks ``_assertions_hold`` and writes the 2.0 witness. X [N, 2, 2]
    flattened row-major to r, y = (r1 - r2, r0 + r3); the box makes y0 lie in
    [0.5, 1], so every point violates ``y0 * y0 >= 0.25 and y0 >= 0``.
    Catches a batched run whose rows are reshaped column-major (it computes
    r2 - r1 < 0 and finds nothing) and a witness whose pinned coordinates
    (0.1, 0.30000000000000004) are not written exactly."""
    W = np.zeros((2, 4), np.float32)
    W[0, 1], W[0, 2], W[1, 0], W[1, 3] = 1.0, -1.0, 1.0, 1.0
    nodes = [helper.make_node("Reshape", ["X", "sh"], ["f"]),
             helper.make_node("Gemm", ["f", "W", "B"], ["Y"], transB=1)]
    onnx_p = _save(
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 2])],
        [numpy_helper.from_array(np.array([-1, 4], np.int64), "sh"),
         numpy_helper.from_array(W, "W"),
         numpy_helper.from_array(np.zeros(2, np.float32), "B")],
        tmp_path / "n.onnx")
    inst = _Instance(W, [0.0, 0.0], (1, 2, 2),
                     [0.1, 0.75, 0.0, 0.30000000000000004],
                     [0.1, 1.0, 0.25, 0.30000000000000004], (1, 2), [])
    spec = inst.spec(tmp_path, "2.0")
    with open(spec, "a") as f:
        f.write("(assert (>= (* Y[0,0] Y[0,0]) 0.25))\n"
                "(assert (>= Y[0,0] 0.0))\n")
    assert R.load_vnnlib(spec)["format"] == "nonlinear"
    res = R.verify_instance(onnx_p, spec, "nonlinear_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    y = _replayed_point(res["counterexample"], "2.0", inst, onnx_p)
    y0 = Fraction(float(y[0]))
    assert y0 * y0 >= Fraction("0.25") and y0 >= 0


# ---------------------------------------------------------------------------
# The sat gate decides on the onnxruntime replay at zero tolerance
# ---------------------------------------------------------------------------

_SUM = dict(W=[[1.0, 1.0]], b=[0.0], in_shape=(1, 2), lb=[0.0, 0.0],
            ub=[0.25, 0.25], out_shape=(1, 1))
_PAIR = dict(W=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0], in_shape=(1, 2),
             lb=[0.0, 0.0], ub=[1.0, 0.5], out_shape=(1, 2))

# (instance, the falsifier's candidate, its replay is a counterexample)
_GATE_CASES = {
    # max y = 0.5 < 0.5 + 2^-15: no counterexample; the candidate's replay
    # misses the region by less than the default slack of 1e-4
    "misses by 2^-15": (dict(_SUM, unsafe=[[[(0, ">=", 0.5 + 2.0 ** -15)]]]),
                        [0.25, 0.25], False),
    # the closed boundary: y = 0.5 at the corner is a counterexample
    "on the closed boundary": (dict(_SUM, unsafe=[[[(0, ">=", 0.5)]]]),
                               [0.25, 0.25], True),
    # two asserts y0 >= 0.75, y1 >= 0.75 (y1 <= 0.5: no counterexample);
    # 1.0 reads them as two rows of one half-space, 2.0 as two AND groups
    "two asserts, the first met": (
        dict(_PAIR, unsafe=[[[(0, ">=", 0.75)]], [[(1, ">=", 0.75)]]]),
        [1.0, 0.5], False),
    # y0 <= 0.25 and y1 <= 0.75 with y1 = 0.5 between the two bounds
    "two rows, the output between their bounds": (
        dict(_PAIR, unsafe=[[[(0, "<=", 0.25), (1, "<=", 0.75)]]]),
        [0.125, 0.5], True),
    # (or y0 >= 0.75, y1 >= 0.375): the candidate meets the second disjunct
    "the second disjunct met": (
        dict(_PAIR, unsafe=[[[(0, ">=", 0.75)], [(1, ">=", 0.375)]]]),
        [0.5, 0.5], True),
}


def _stub_falsifier(monkeypatch, candidate):
    """A falsifier that claims a counterexample at ``candidate``, as an
    attack on the converted model can where the replay disagrees."""
    def fake(model, lb, ub, prop, **kwargs):
        x = np.asarray(candidate, np.float64).reshape(np.shape(lb))
        return 0, (x, None)
    monkeypatch.setattr(R, "falsify", fake)


class TestTheSatGateDecidesOnTheReplayAtZeroTolerance:
    # The gate does not depend on the version; both are run only where the
    # two parsers group the asserts differently (rows of one half-space in
    # 1.0, AND groups in 2.0).
    @pytest.mark.parametrize("case,version", [
        ("misses by 2^-15", "1.0"),
        ("on the closed boundary", "2.0"),
        ("two asserts, the first met", "1.0"),
        ("two asserts, the first met", "2.0"),
        ("two rows, the output between their bounds", "2.0"),
        ("the second disjunct met", "1.0"),
    ])
    def test_sat_only_for_a_replayed_counterexample(self, tmp_path,
                                                    monkeypatch, runner_env,
                                                    case, version):
        """The falsifier claims a counterexample; the runner writes sat only
        when the onnxruntime replay of the candidate lies in the unsafe
        region at zero tolerance, and then with a witness the checker
        accepts; otherwise unknown (no reach method runs). Catches a gate
        with the default slack of 1e-4, an AND group counted as met after an
        earlier group was, a half-space met by one of its rows, each row
        compared with every right-hand side, a closed boundary read as
        strict, and an OR group that looks at its first disjunct only."""
        data, candidate, is_counterexample = _GATE_CASES[case]
        inst = _Instance(**data)
        onnx_p = inst.onnx(tmp_path)
        _stub_falsifier(monkeypatch, candidate)
        res = R.verify_instance(onnx_p, inst.spec(tmp_path, version),
                                "gate_demo", workers=1)
        if not is_counterexample:
            assert res["result"] == R.RESULT_UNKNOWN
            assert res["counterexample"] is None
            return
        assert res["result"] == R.RESULT_SAT
        _assert_accepted(res["counterexample"], version, inst, onnx_p)


def test_strict_sum_true_only_in_float32_is_not_sat(tmp_path, runner_env):
    """Y = [1, 2^-24, 2^-24] everywhere and ``Y0 + Y1 + Y2 < 1 + 2^-23``
    (written exactly). The sum is exactly 1 + 2^-23, so there is no
    counterexample in exact or float64 arithmetic; only the float32
    evaluation (1 + 2^-24 rounds to 1, twice) says the sum is below. The
    closed row accepts the point, so the decision is the assertions', and
    both evaluations must hold. Catches an assertion check that trusts the
    float32 evaluation alone (or either one of the two)."""
    y = [1.0, 2.0 ** -24, 2.0 ** -24]
    assert np.float32(y[0]) + np.float32(y[1]) + np.float32(y[2]) == 1.0
    onnx_p = _const_net(tmp_path / "n.onnx", y)
    spec = _v2_spec(tmp_path / "s.vnnlib", [("X", "1, 2")], [("Y", "1, 3")],
                    _box_asserts("X", 2)
                    + ["(< (+ Y[0,0] Y[0,1] Y[0,2]) 1.00000011920928955078125)"])
    assert Fraction("1.00000011920928955078125") == 1 + Fraction(1, 2 ** 23)
    res = R.verify_instance(onnx_p, spec, "sum_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN


# ---------------------------------------------------------------------------
# The ViT route snaps its batch of candidates into the float64 box
# ---------------------------------------------------------------------------

def test_vit_witness_is_snapped_into_the_float64_box(monkeypatch):
    """Box x0 in [0, 0.1], x1 in [0, 0.2], unsafe x0 + x1 >= 0.3: only the
    corner (0.1, 0.2) is a counterexample. The gradient search (stubbed)
    returns the corner as float32 rounds it, (0.10000000149..., 0.2000000029...),
    outside the box. Catches a snap that leaves the upper side unclipped and
    one that rounds the snapped batch to float32: the witness must be
    written as 0.1 and 0.2 exactly."""
    _vit_stubs(monkeypatch, batched_shift=0.0)
    corner32 = np.array([np.float32(0.1), np.float32(0.2)], np.float64)
    monkeypatch.setattr(R, "_vit_pgd",
                        lambda md, lb, ub, gp, rng: corner32[None, :])
    hs = HalfSpace(np.array([[-1.0]]), np.array([[-0.3]]))
    prop = {"pairs": [{"lb": np.array([0.0, 0.0]),
                       "ub": np.array([0.1, 0.2]), "prop": [{"Hg": hs}]}]}
    res = R.verify_vit_instance("vit.onnx", prop, (2,), "vit_2023", 0.0)
    assert res["result"] == R.RESULT_SAT
    x_text, _ = _read_v1(res["counterexample"], 2, 1)
    assert [float(t) for t in x_text] == [0.1, 0.2]


# ---------------------------------------------------------------------------
# _snap_into_box: every coordinate onto its own float64 bound
# ---------------------------------------------------------------------------

_SNAP_LB = [0.1, -0.7, 0.30000000000000004, 2.5, -1e10, 0.0]
_SNAP_UB = [0.1, 0.2, 0.30000000000000004, 3.75, 1e10, 0.3]
# float32 roundings just outside, a point below, one inside, one far above
_SNAP_X = [float(np.float32(0.1)), -0.75, float(np.float32(0.3)), 3.0, 2e10,
           float(np.float32(0.3))]
_SNAPPED = [0.1, -0.7, 0.30000000000000004, 3.0, 1e10, 0.3]


@pytest.mark.parametrize("shape", [(6,), (2, 3), (3, 6)],
                         ids=["flat", "shaped", "batch of rows"])
def test_snap_puts_each_coordinate_on_its_own_float64_bound(shape):
    """A flat witness, a tensor-shaped one (row-major over the box) and an
    (N, d) batch: each coordinate is clipped to its own interval, an inside
    value is left as it is, the result keeps the input's shape and stays
    float64, so 0.1 and 0.30000000000000004 come back exactly. Catches a
    snap rounded to float32, an upper or a lower side left unclipped, and a
    batch or a shaped witness that loses its shape."""
    rows = int(np.prod(shape)) // 6
    x = np.array(_SNAP_X * rows).reshape(shape)
    out = R._snap_into_box(x, _SNAP_LB, _SNAP_UB)
    assert out.shape == x.shape and out.dtype == np.float64
    assert out.reshape(-1).tolist() == _SNAPPED * rows


# ---------------------------------------------------------------------------
# The witness text: numbering, order and exact values
# ---------------------------------------------------------------------------

_TEXT_X = [0.1, 0.30000000000000004, 0.123456789123, -12345.678901234567,
           1.0 / 3.0, 5e-324]
_TEXT_Y = np.array([0.1, -2.5], np.float32)


@pytest.mark.parametrize("spec_kind", ["no path", "1.0 file"])
def test_v1_witness_numbers_inputs_from_zero_with_exact_values(tmp_path,
                                                               spec_kind):
    """A (2, 3) witness tensor and two float32 outputs in the 1.0 s-expression:
    X_0..X_5 in row-major order and Y_0, Y_1, each value read back exactly.
    Catches inputs numbered from 1, a column-major flatten, values printed
    with too few digits, and outputs written from the input vector."""
    path = None
    if spec_kind == "1.0 file":
        decls = ([f"(declare-const X_{i} Real)" for i in range(6)]
                 + ["(declare-const Y_0 Real)", "(declare-const Y_1 Real)"])
        path = tmp_path / "s.vnnlib"
        path.write_text("\n".join(decls) + "\n(assert (>= Y_0 0.0))\n")
    text = R.format_counterexample(np.array(_TEXT_X).reshape(2, 3), _TEXT_Y,
                                   None if path is None else str(path))
    x_text, y_text = _read_v1(text, 6, 2)
    assert [float(t) for t in x_text] == _TEXT_X
    assert [float(t) for t in y_text] == [float(v) for v in _TEXT_Y]


def test_v2_witness_gives_each_input_tensor_its_own_slice(tmp_path):
    """Two input tensors A [1, 2] and B [3] and an output Y [2, 1]: the blocks
    are A = x[0:2], B = x[2:5], Y = y, in declaration order, each value read
    back exactly. Catches a second tensor filled from the first one's slice
    (the input offset not advanced) and inexact value text."""
    spec = tmp_path / "s.vnnlib"
    spec.write_text("(vnnlib-version <2.0>)\n(declare-network N\n"
                    "    (declare-input A float32 [1, 2])\n"
                    "    (declare-input B float32 [3])\n"
                    "    (declare-output Y float32 [2, 1])\n)\n"
                    "(assert (>= Y[0,0] 0.0))\n")
    x = np.array(_TEXT_X[:5])
    text = R.format_counterexample(x, _TEXT_Y, str(spec))
    blocks = _read_v2(text, [("A", "float32", (1, 2)), ("B", "float32", (3,)),
                             ("Y", "float32", (2, 1))])
    assert [float(t) for t in blocks["A"]] == _TEXT_X[:2]
    assert [float(t) for t in blocks["B"]] == _TEXT_X[2:5]
    assert [float(t) for t in blocks["Y"]] == [float(v) for v in _TEXT_Y]


# ---------------------------------------------------------------------------
# The result file
# ---------------------------------------------------------------------------

def test_result_file_holds_the_verdict_and_then_the_witness(tmp_path):
    """``write_result`` replaces the file: a verdict without a witness is the
    whole file; ``sat`` is alone on the first line and the witness follows
    line for line. Catches the newline after ``sat`` dropped (the grader
    reads the first line as the verdict) and a file appended to instead of
    replaced (an earlier verdict would stay first)."""
    f = tmp_path / "result.txt"
    f.write_text("sat\nX float32 [1]\n0.5\n")
    R.write_result(str(f), R.RESULT_UNSAT)
    assert f.read_text().splitlines() == [R.RESULT_UNSAT]
    witness = "X float32 [1,2]\n0.1\n0.5\nY float32 [1,1]\n0.6000000238418579"
    R.write_result(str(f), R.RESULT_SAT, witness)
    assert f.read_text().splitlines() == [R.RESULT_SAT] + witness.splitlines()


@pytest.fixture
def alarm_restored():
    """``main`` installs a SIGALRM handler and arms the timer."""
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("no SIGALRM on this platform")
    saved = signal.getsignal(signal.SIGALRM)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, saved)


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_main_writes_a_result_file_the_checker_accepts(
        tmp_path, monkeypatch, capsys, runner_env, alarm_restored, version):
    """The public route: ``main`` with the competition's arguments on an
    instance with a counterexample (x0 pinned to 0.1). The results file's
    first line is ``sat`` and the rest is a witness the checker accepts."""
    inst = _WITNESS_INSTANCES["pinned at 0.1"]
    onnx_p = inst.onnx(tmp_path)
    results = tmp_path / "results.txt"
    monkeypatch.setenv("N2V_WORKERS", "1")
    monkeypatch.setattr(sys, "argv", [
        "vnncomp_runner.py", "witness_demo", onnx_p,
        inst.spec(tmp_path, version), str(results), "120"])
    assert R.main() == 0
    lines = results.read_text().splitlines()
    assert lines[0] == R.RESULT_SAT
    _assert_accepted("\n".join(lines[1:]), version, inst, onnx_p)


# ---------------------------------------------------------------------------
# The 2.0 witness follows the declarations, not the comments
# ---------------------------------------------------------------------------

@known_bug(20, "_v2_ce_meta reads the declarations from the raw spec text, "
               "comments included, so a commented-out (declare-input ...) adds "
               "a witness block that the parser does not have; "
               "format_counterexample then raises and the runner answers "
               "unknown on an instance with a counterexample")
def test_v2_witness_ignores_declarations_in_comments(tmp_path, runner_env):
    """The instance "pinned at 0.1" in 2.0 (its passing neighbour is
    ``test_sat_witness_replays_inside_the_box[pinned at 0.1-2.0]``) with a
    header comment that quotes an old declaration: the runner answers sat with
    a witness of the declared X and Y blocks only. VNN-LIB comments run from
    ';' to the end of the line and the parser drops them."""
    comment = ("; generated; an earlier version declared\n"
               "; (declare-input image float32 [1, 2])\n")
    inst = _WITNESS_INSTANCES["pinned at 0.1"]
    onnx_p = inst.onnx(tmp_path)
    spec = inst.spec(tmp_path, "2.0")
    with open(spec) as f:
        text = f.read()
    with open(spec, "w") as f:
        f.write(comment + text)
    res = R.verify_instance(onnx_p, spec, "comment_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    _assert_accepted(res["counterexample"], "2.0", inst, onnx_p)


@known_bug(6, "SPEC-5: the 1.0 parser stores a strict output comparison as "
              "the closed one and keeps no source assertion, so the runner's "
              "sat gate accepts an output on the boundary and writes a "
              "witness the checker rejects")
def test_strict_output_bound_on_boundary_is_not_sat_in_a_v1_spec(tmp_path,
                                                                 runner_env):
    """The 1.0 form of ``test_strict_output_bound_on_boundary_is_not_sat``
    (its passing neighbour): y = relu(-x0 - x1 - 1) is 0 on [0, 1]^2, so
    ``Y_0 < 0`` is unreachable, but every sample lies on the closed row."""
    onnx_p = _relu_net(tmp_path / "n.onnx", [-1.0, -1.0], -1.0)
    spec = tmp_path / "s.vnnlib"
    spec.write_text("(declare-const X_0 Real)\n(declare-const X_1 Real)\n"
                    "(declare-const Y_0 Real)\n"
                    "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n"
                    "(assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n"
                    "(assert (< Y_0 0.0))\n")
    res = R.verify_instance(onnx_p, str(spec), "strict_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN
    assert res["counterexample"] is None
