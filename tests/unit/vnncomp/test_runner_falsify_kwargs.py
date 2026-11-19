"""Tests for the production runner's per-method falsify-budget whitelist.

The runner passes n_samples/method/seed to falsify() EXPLICITLY, so any
falsify_kwargs from a category config must be filtered to budget knobs only —
otherwise falsify() raises "multiple values for argument". This pins that
contract so a future config (or a new budget knob) can't silently break it.

From the section "The falsification stage of verify_instance" on, the file
tests stage 1 of ``verify_instance`` as a whole: what ``falsify`` is handed
per region (box, property, budget, knobs), and that a ``sat`` is written only
for a candidate whose onnxruntime replay is a counterexample of its own
region, with that replay as the written output.
"""

import os
import re
import sys

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

# The production runner lives under examples/Submission and inserts its own
# sys.path for benchmark_configs at import time.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402
from tests.cell_support import replay_onnx  # noqa: E402


def test_whitelist_keeps_budget_knobs():
    src = {'n_iters': 20000, 'n_restarts': 1, 'n_steps': 30,
           'batch': 128, 'p_init': 0.3, 'step_size': 0.01}
    assert R._whitelist_falsify_kwargs(src) == src


def test_whitelist_drops_explicit_and_unknown_keys():
    # n_samples/method/seed are passed explicitly to falsify() -> must be dropped
    # (else a duplicate-argument TypeError); unknown keys dropped too.
    out = R._whitelist_falsify_kwargs(
        {'n_iters': 5000, 'n_samples': 999, 'method': 'square',
         'seed': 7, 'bogus': 1})
    assert out == {'n_iters': 5000}


def test_whitelist_none_and_empty():
    assert R._whitelist_falsify_kwargs(None) == {}
    assert R._whitelist_falsify_kwargs({}) == {}


# --------------------------------------------------------------------------- #
# The falsification stage of verify_instance                                   #
# --------------------------------------------------------------------------- #
# Stage 1 of ``verify_instance`` searches every (region, property) pair of the
# spec for a counterexample with ``falsify`` and writes ``sat`` only for a
# candidate whose onnxruntime replay is a counterexample of that pair. The
# instances below are small ONNX files and VNN-LIB texts written here; the
# expected answer of each is known by construction, and a written witness is
# read back and replayed on onnxruntime by the test, not by the runner.

def _save(path, nodes, inits, in_dims, out_dims):
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(in_dims))],
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
    """``Y = X @ W.T + b``, the input flattened row-major; X ``in_dims``
    (default [1, n]), Y [1, m]."""
    W = np.asarray(W, np.float32)
    n = W.shape[1]
    nodes = [helper.make_node("Reshape", ["X", "sh"], ["f"]),
             helper.make_node("Gemm", ["f", "W", "B"], ["Y"], transB=1)]
    return _save(path, nodes,
                 {"sh": np.array([-1, n], np.int64), "W": W,
                  "B": np.asarray(b, np.float32)},
                 in_dims or [1, n], [1, W.shape[0]])


def _sigmoid_net(path):
    """``Y = Sigmoid(X)``, X and Y [1, 1]."""
    return _save(path, [helper.make_node("Sigmoid", ["X"], ["Y"])], {},
                 [1, 1], [1, 1])


def _spec_v1(path, regions, outputs=(), n_out=1):
    """A VNN-LIB 1.0 spec. ``regions`` is a list of ``(lb, ub, own)``: one
    region is written as top-level bounds; several as an ``(or (and ...))``
    of boxes, each with its own output atoms ``own`` (a list of strings, the
    combined form) when given. ``outputs`` are top-level output asserts."""
    n = len(regions[0][0])
    lines = [f"(declare-const X_{i} Real)" for i in range(n)]
    lines += [f"(declare-const Y_{j} Real)" for j in range(n_out)]
    if len(regions) == 1 and not regions[0][2]:
        lb, ub, _ = regions[0]
        for i in range(n):
            lines += [f"(assert (>= X_{i} {lb[i]!r}))",
                      f"(assert (<= X_{i} {ub[i]!r}))"]
    else:
        lines.append("(assert (or")
        for lb, ub, own in regions:
            atoms = [f"(>= X_{i} {lb[i]!r}) (<= X_{i} {ub[i]!r})"
                     for i in range(n)] + list(own or [])
            lines.append("  (and " + " ".join(atoms) + ")")
        lines.append("))")
    lines += [f"(assert {a})" for a in outputs]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def _config(monkeypatch, n_rand=100, method="random", kwargs=None,
            reach=()):
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {
        "reach_methods": list(reach), "n_rand": n_rand,
        "falsify_method": method, "falsify_kwargs": dict(kwargs or {})})


def _witness_v1(text):
    """The written 1.0 witness ``((X_0 v) ... (Y_0 v) ...)`` read as the
    checker reads it: the inputs and the outputs, numbered from 0."""
    pairs = re.findall(r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)", text)
    xs = {int(i): float(v) for k, i, v in pairs if k == "X"}
    ys = {int(i): float(v) for k, i, v in pairs if k == "Y"}
    assert sorted(xs) == list(range(len(xs))) and xs, text[:200]
    assert sorted(ys) == list(range(len(ys))) and ys, text[:200]
    return ([xs[i] for i in range(len(xs))], [ys[j] for j in range(len(ys))])


def _spy_falsify(monkeypatch):
    """Replace the runner's ``falsify`` by a recorder that finds nothing."""
    calls = []

    def spy(model, lb, ub, prop, **kwargs):
        calls.append({"lb": np.array(lb, dtype=np.float64),
                      "ub": np.array(ub, dtype=np.float64), "prop": prop,
                      "kwargs": kwargs})
        return 2, None
    monkeypatch.setattr(R, "falsify", spy)
    return calls


def _unsafe(prop, y):
    """Is the output ``y`` in the unsafe region a property states? The
    property is a list of AND'ed dicts, each with ``Hg`` a half-space or a
    list of OR'ed half-spaces; a half-space holds where every row of
    ``G y <= g`` holds. Read here from the stored arrays alone."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    groups = prop if isinstance(prop, list) else [prop]
    for group in groups:
        hs = group["Hg"] if isinstance(group, dict) else group
        hs = hs if isinstance(hs, list) else [hs]
        if not any(np.all(np.asarray(h.G) @ y <= np.asarray(h.g).reshape(-1, 1))
                   for h in hs):
            return False
    return True


_TWO_BOXES = [([0.0, -2.0], [1.0, -1.0], None), ([2.0, 0.5], [3.0, 4.0], None)]
_SEVEN_BOXES = [([float(k), 0.0], [k + 0.5, 1.0], None) for k in range(7)]


@pytest.mark.parametrize("regions,n_rand,per_region", [
    pytest.param(_TWO_BOXES, 5000, 2500, id="5000-over-2-regions"),
    pytest.param(_SEVEN_BOXES, 100, 20, id="100-over-7-regions-floor-20"),
])
def test_falsify_receives_its_region_and_the_configured_budget_only(
        tmp_path, monkeypatch, regions, n_rand, per_region):
    """``verify_instance`` calls ``falsify`` once per region, on that region's
    box (shaped as the model's input), with the configured method, the
    configured sample budget shared across the regions (at least 20 each),
    seed 42 and the configuration's whitelisted budget knobs, and nothing
    else. The configuration also carries keys the runner passes itself or
    does not know (``method``, ``seed``, ``n_samples``, ``bogus``). Catches a
    budget read from a key the configuration does not have, a share without
    its floor, a whitelisted knob dropped, a non-whitelisted key passed on
    (``falsify`` then raises on the duplicate ``method`` and is never
    reached), and a region searched on another region's box."""
    onnx_p = _affine_net(tmp_path / "n.onnx", [[1.0, 1.0]], [0.0])
    spec = _spec_v1(tmp_path / "s.vnnlib", regions, ["(>= Y_0 1000.0)"])
    _config(monkeypatch, n_rand=n_rand, method="random+square",
            kwargs={"n_iters": 7, "p_init": 0.2, "method": "pgd", "seed": 3,
                    "n_samples": 9, "bogus": 1})
    calls = _spy_falsify(monkeypatch)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert res["result"] == R.RESULT_UNKNOWN
    assert len(calls) == len(regions)
    for call, (lb, ub, _) in zip(calls, regions):
        assert call["lb"].tolist() == lb and call["ub"].tolist() == ub
        assert call["kwargs"] == {"method": "random+square",
                                  "n_samples": per_region, "seed": 42,
                                  "n_iters": 7, "p_init": 0.2}


def test_each_region_of_a_combined_spec_is_falsified_against_its_own_property(
        tmp_path, monkeypatch):
    """A combined-form 1.0 spec: region 1 with ``Y_0 >= 1.5``, region 2 with
    ``Y_0 <= -100``. The property ``falsify`` gets for a region must classify
    outputs as that region's own atoms do (checked on outputs on both sides
    of each bound), not as the conjunction of both regions' atoms, which is
    empty. Catches the spec's top-level property handed to every region."""
    onnx_p = _affine_net(tmp_path / "n.onnx", [[1.0, 1.0]], [0.0])
    spec = _spec_v1(tmp_path / "s.vnnlib", [
        ([0.0, 0.0], [1.0, 1.0], ["(>= Y_0 1.5)"]),
        ([2.0, 0.0], [3.0, 1.0], ["(<= Y_0 -100.0)"])])
    _config(monkeypatch)
    calls = _spy_falsify(monkeypatch)
    R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert len(calls) == 2
    first, second = (c["prop"] for c in calls)
    assert [_unsafe(first, [v]) for v in (1.4, 1.5, 1.6, -101.0)] == [
        False, True, True, False]
    assert [_unsafe(second, [v]) for v in (-101.0, -100.0, -99.0, 1.6)] == [
        True, True, False, False]


def test_a_counterexample_in_the_second_region_is_written_from_that_region(
        tmp_path, monkeypatch):
    """Combined form: region 1 ``[0,1]^2`` with ``Y_0 <= -100`` (no
    counterexample), region 2 ``[2,3] x [0,1]`` with ``Y_0 >= 2.5`` (y = x0 +
    x1 meets it on 7/8 of the box). The real falsifier finds it; the answer
    is ``sat`` and the written input lies in region 2 with a replay that
    meets ``Y_0 >= 2.5``. Catches a candidate clipped into the first
    region's box (it then fails the re-check) and a search against the
    spec's top-level property (the empty conjunction)."""
    onnx_p = _affine_net(tmp_path / "n.onnx", [[1.0, 1.0]], [0.0])
    spec = _spec_v1(tmp_path / "s.vnnlib", [
        ([0.0, 0.0], [1.0, 1.0], ["(<= Y_0 -100.0)"]),
        ([2.0, 0.0], [3.0, 1.0], ["(>= Y_0 2.5)"])])
    _config(monkeypatch, n_rand=200)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness_v1(res["counterexample"])
    assert 2.0 <= x[0] <= 3.0 and 0.0 <= x[1] <= 1.0
    replay = replay_onnx(onnx_p, [x], (1, 2))[0]
    assert y == replay.tolist() and replay[0] >= 2.5


# Sigmoid at x = -2.99609375: the converted torch model and onnxruntime
# differ in the last bits (0.04760265722870827 against 0.04760262370109558,
# measured); the exact value 0.0476026568... lies between them. A threshold
# at the torch value is met by the falsifier's own forward and missed by the
# replay the competition grades.
_X0 = -2.99609375


@pytest.fixture
def sigmoid_split(tmp_path):
    """The Sigmoid model, its onnxruntime value at ``_X0`` and the converted
    model's value there, which must be the larger (checked here, outside any
    expected-failure marker)."""
    onnx_p = _sigmoid_net(tmp_path / "sig.onnx")
    ort_y = float(replay_onnx(onnx_p, [[_X0]], (1, 1))[0][0])
    with torch.no_grad():
        torch_y = float(R.load_onnx(onnx_p)(
            torch.tensor([[_X0]], dtype=torch.float32)).reshape(-1)[0])
    if not torch_y > ort_y:
        pytest.fail(f"no divergence at {_X0}: torch {torch_y!r}, "
                    f"onnxruntime {ort_y!r}")
    return onnx_p, ort_y, torch_y


def test_a_candidate_the_replay_rejects_is_not_written_and_the_search_goes_on(
        tmp_path, monkeypatch, sigmoid_split):
    """Two regions, unsafe ``Y_0 >= t`` with t the torch value at ``_X0``.
    Region 1 pins x to ``_X0``: the falsifier's forward meets t, the
    onnxruntime replay does not, so that candidate is no counterexample.
    Region 2, ``x in [2, 3]`` (sigmoid above 0.88), is unsafe everywhere.
    The answer is ``sat`` with a witness from region 2 whose replay meets t.
    Catches a search that stops after the first rejected candidate, and a
    re-check that judges the falsifier's own output instead of the replay
    (it writes the region-1 point)."""
    onnx_p, _, t = sigmoid_split
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [([_X0], [_X0], None), ([2.0], [3.0], None)],
                    [f"(>= Y_0 {t!r})"])
    _config(monkeypatch)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, _ = _witness_v1(res["counterexample"])
    assert 2.0 <= x[0] <= 3.0
    assert replay_onnx(onnx_p, [x], (1, 1))[0][0] >= t


def test_every_and_group_is_checked_on_the_replay_before_sat(
        tmp_path, monkeypatch, sigmoid_split):
    """VNN-LIB 2.0, x pinned to ``_X0``, two output asserts (two AND'ed
    groups): ``Y >= 0`` and ``Y >= t`` with t the torch value. The replay
    meets the first and not the second, and x has no other value, so there
    is no counterexample: the answer must not be ``sat``. Catches a re-check
    of the first AND group only, and one that judges the falsifier's output."""
    onnx_p, _, t = sigmoid_split
    spec = tmp_path / "s.vnnlib"
    spec.write_text(
        "(vnnlib-version <2.0>)\n(declare-network N\n"
        "    (declare-input X float32 [1, 1])\n"
        "    (declare-output Y float32 [1, 1])\n)\n"
        f"(assert (>= X[0,0] {_X0!r}))\n(assert (<= X[0,0] {_X0!r}))\n"
        f"(assert (>= Y[0,0] 0.0))\n(assert (>= Y[0,0] {t!r}))\n")
    _config(monkeypatch)
    res = R.verify_instance(onnx_p, str(spec), "falsify_demo", workers=1)
    assert res["result"] != R.RESULT_SAT


def test_the_written_output_is_the_replay_of_the_written_input(
        tmp_path, monkeypatch, sigmoid_split):
    """x pinned to ``_X0`` and unsafe ``Y_0 >= 0.04`` (met by both forwards):
    ``sat``, and the written Y is exactly the onnxruntime replay of the
    written X, as the checker compares them. Catches the falsifier's own
    output (the torch value) written as Y."""
    onnx_p, ort_y, _ = sigmoid_split
    spec = _spec_v1(tmp_path / "s.vnnlib", [([_X0], [_X0], None)],
                    ["(>= Y_0 0.04)"])
    _config(monkeypatch)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness_v1(res["counterexample"])
    assert x == [_X0]
    assert y == replay_onnx(onnx_p, [x], (1, 1))[0].tolist() == [ort_y]


def _unbatched_net(path):
    """X [3] with no batch dimension: Y [1, 1] = x0 + x1 + x2 + 1, built as
    ``Concat(X, [1.0])`` (rank 1, axis 0), Reshape [1, 4] and a Gemm with
    weights 1. A batch-prefixed input (N, 3) fails at the Concat."""
    nodes = [helper.make_node("Concat", ["X", "one"], ["c"], axis=0),
             helper.make_node("Reshape", ["c", "sh"], ["r"]),
             helper.make_node("Gemm", ["r", "W", "B"], ["Y"])]
    return _save(path, nodes,
                 {"one": np.ones(1, np.float32),
                  "sh": np.array([1, 4], np.int64),
                  "W": np.ones((4, 1), np.float32),
                  "B": np.zeros(1, np.float32)}, [3], [1, 1])


def test_an_input_without_a_batch_dim_is_falsified_one_sample_at_a_time(
        tmp_path, monkeypatch):
    """ONNX input [3] with no batch dimension, y = x0 + x1 + x2 + 1 on
    ``[0, 1]^3``, unsafe ``Y_0 >= 3`` (one sixth of the box). The falsifier's
    batched calls must reach a per-sample wrapper of the converted model:
    the answer is ``sat`` with a witness in the box whose replay meets the
    bound. Catches the wrapper not applied to a graph declared without a
    batch dimension (every batched forward then fails) and a wrapper whose
    outputs do not belong to their samples."""
    onnx_p = _unbatched_net(tmp_path / "u.onnx")
    spec = _spec_v1(tmp_path / "s.vnnlib", [([0.0] * 3, [1.0] * 3, None)],
                    ["(>= Y_0 3.0)"])
    _config(monkeypatch, n_rand=200)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert res["result"] == R.RESULT_SAT
    x, y = _witness_v1(res["counterexample"])
    assert all(0.0 <= v <= 1.0 for v in x) and len(x) == 3
    replay = replay_onnx(onnx_p, [x], (3,))[0]
    assert y == replay.tolist() and replay[0] >= 3.0


@pytest.mark.parametrize("op,expected", [("<", R.RESULT_UNKNOWN),
                                         ("<=", R.RESULT_SAT)],
                         ids=["strict", "closed"])
def test_a_strict_comparison_met_only_with_equality_is_no_counterexample(
        tmp_path, monkeypatch, op, expected):
    """VNN-LIB 2.0 with x pinned to 0.5 and y = x - 0.5, which is 0 exactly
    (in float32 too): ``Y < 0`` has no counterexample, ``Y <= 0`` has one at
    the only point. The parser lowers the strict comparison to the closed
    row and keeps the spec's assertions, and the falsify stage must decide on
    them: ``unknown`` for the strict spec, ``sat`` with the replayed point
    for the closed one. Catches the sat gate handed the region instead of
    the parsed spec, so that the kept assertions are not found and the
    closed row admits the boundary point."""
    onnx_p = _affine_net(tmp_path / "n.onnx", [[1.0]], [-0.5])
    spec = tmp_path / "s.vnnlib"
    spec.write_text(
        "(vnnlib-version <2.0>)\n(declare-network N\n"
        "    (declare-input X float32 [1, 1])\n"
        "    (declare-output Y float32 [1, 1])\n)\n"
        "(assert (>= X[0,0] 0.5))\n(assert (<= X[0,0] 0.5))\n"
        f"(assert ({op} Y[0,0] 0.0))\n")
    _config(monkeypatch)
    res = R.verify_instance(onnx_p, str(spec), "falsify_demo", workers=1)
    assert res["result"] == expected
    if expected == R.RESULT_SAT:
        lines = res["counterexample"].splitlines()
        assert lines == ["X float32 [1,1]", "0.5", "Y float32 [1,1]", "0.0"]


def test_the_per_sample_wrapper_runs_each_sample_on_its_own(tmp_path):
    """``_falsify_model`` on the unbatched net: a batch-1 net (its batch
    dimension stripped) is handed back unchanged; the net declared on X [3]
    is wrapped, and the wrapper's outputs on a batch of five different dyadic
    samples are exactly their onnxruntime replays, in order, with gradients
    flowing back to the input (the gradient attacks need them). Catches
    outputs that do not belong to their samples and a wrapper that cuts the
    gradient."""
    batched = R.load_onnx(_affine_net(tmp_path / "b.onnx", [[1.0, 1.0]],
                                      [0.0]))
    assert R._falsify_model(batched, (2,), (1, 2), np.zeros(2)) is batched
    onnx_p = _unbatched_net(tmp_path / "u.onnx")
    wrapped = R._falsify_model(R.load_onnx(onnx_p), (3,), (3,),
                               np.full(3, 0.5))
    assert isinstance(wrapped, R._PerSampleModel)
    xs = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25], [0.125, 1.0, 0.0],
                   [0.75, 0.75, 0.75], [0.0, 0.0, 1.0]])
    x = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    out = wrapped(x)
    assert out.shape == (5, 1)
    assert out.detach().numpy().reshape(-1).tolist() == [
        float(y[0]) for y in replay_onnx(onnx_p, list(xs), (3,))]
    out.sum().backward()
    assert x.grad.tolist() == [[1.0, 1.0, 1.0]] * 5


def test_a_falsifier_error_on_one_region_does_not_end_the_search(
        tmp_path, monkeypatch):
    """Two regions with unsafe ``Y_0 >= 2.5``: region 1 ``[0,1]^2`` (y <= 2,
    no counterexample) and region 2 ``[2,3] x [0,1]`` (unsafe on 7/8 of the
    box). ``falsify`` raises on region 1 and runs for real on region 2: the
    answer is ``sat`` with a witness in region 2 whose replay meets the
    bound. Catches a falsifier error that ends the search, so the
    counterexample of a later region is lost (the reach cannot prove the
    instance, so it is then written unknown)."""
    onnx_p = _affine_net(tmp_path / "n.onnx", [[1.0, 1.0]], [0.0])
    spec = _spec_v1(tmp_path / "s.vnnlib",
                    [([0.0, 0.0], [1.0, 1.0], None),
                     ([2.0, 0.0], [3.0, 1.0], None)], ["(>= Y_0 2.5)"])
    _config(monkeypatch, n_rand=200)
    real, seen = R.falsify, []

    def first_fails(model, lb, ub, prop, **kwargs):
        seen.append(np.asarray(lb, dtype=np.float64).tolist())
        if len(seen) == 1:
            raise RuntimeError("the attack failed on this region")
        return real(model, lb, ub, prop, **kwargs)
    monkeypatch.setattr(R, "falsify", first_fails)
    res = R.verify_instance(onnx_p, spec, "falsify_demo", workers=1)
    assert seen[0] == [0.0, 0.0]
    assert res["result"] == R.RESULT_SAT
    x, y = _witness_v1(res["counterexample"])
    assert 2.0 <= x[0] <= 3.0 and 0.0 <= x[1] <= 1.0
    replay = replay_onnx(onnx_p, [x], (1, 2))[0]
    assert y == replay.tolist() and replay[0] >= 2.5
