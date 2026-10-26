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
"""

import os
import sys

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
