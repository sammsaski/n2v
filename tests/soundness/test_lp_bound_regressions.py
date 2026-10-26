"""Regressions for LP optima that are used as bounds.

An LP optimum is exact only to the solver's tolerance. Where one is used as a
bound (the operand ranges of an element-wise Mul/Div of two sets, the
per-neuron box a Sigmoid, Tanh, Softmax, Pow, Sin, rounding or set-by-set
MatMul relaxation is built on, the ``lp_cpu`` bounds of the refine engine), it
must be

  * solved with HiGHS, whatever solver the caller or the global setting asks
    for: SCS and OSQP report 'optimal' maxima up to ~4e-5 below the true max;
  * padded outward by the HiGHS tolerance: HiGHS reported maxima 5e-10 below
    the true max on thin wedges.

These ops took the raw optimum of the caller's solver. With SCS the product
of two computed values came out a false UNSAT, and the refine ``lp_cpu``
bound and the Sigmoid output left out the attained maximum. The phase LPs of
ReLU, LeakyReLU and Sign are covered in test_neuron_phase_regressions.py.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.config import config
from n2v.nn.layer_ops.bilinear_matmul_reach import bilinear_matmul_star
from n2v.nn.layer_ops.pow_reach import pow_star
from n2v.nn.layer_ops.round_reach import round_star
from n2v.nn.layer_ops.sigmoid_reach import sigmoid_star_approx
from n2v.nn.layer_ops.softmax_attention_reach import softmax_attn_star
from n2v.nn.layer_ops.softmax_reach import softmax_star
from n2v.nn.layer_ops.tanh_reach import tanh_star_approx
from n2v.nn.layer_ops.trig_reach import trig_star
from n2v.nn.reach import _div_sets, _mul_sets
from n2v.refine.tighten import _lp_ranges
from n2v.sets import HalfSpace, Star
from n2v.utils.lp_solver_enum import LPSolver, resolve
from n2v.utils.verify_specification import verify_specification


def _line_star(center, radius):
    """x = center + radius * a, a in [-1, 1], one row per entry, all driven
    by one predicate variable (so the LP ranges are the exact endpoints)."""
    center = np.asarray(center, dtype=np.float64).reshape(-1, 1)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1, 1)
    return Star(np.hstack([center, radius]), np.zeros((0, 1)),
                np.zeros((0, 1)), -np.ones((1, 1)), np.ones((1, 1)))


def _box_star(center, radius):
    """x = center + diag(radius) a, a in [-1, 1]^n (independent entries)."""
    center = np.asarray(center, dtype=np.float64).reshape(-1, 1)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    n = center.size
    return Star(np.hstack([center, np.diag(radius)]), np.zeros((0, n)),
                np.zeros((0, 1)), -np.ones((n, 1)), np.ones((n, 1)))


_FLOOR = SimpleNamespace(round_function=torch.floor)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Each case: a function ``run(lp_solver)`` returning the output stars of one
# op on x = 0.25 + 0.75 a (range [-0.5, 1.0]) or, for the two-logit softmax
# ops, on x0 in [-0.5, 1.0], x1 in [-0.6, 0.4]; and the true range of output
# 0 over that input.
_X = _line_star([0.25], [0.75])
_X2 = _box_star([0.25, -0.1], [0.75, 0.5])
_CASES = {
    "sigmoid": (lambda s: sigmoid_star_approx([_X], lp_solver=s),
                (_sigmoid(-0.5), _sigmoid(1.0))),
    "tanh": (lambda s: tanh_star_approx([_X], lp_solver=s),
             (np.tanh(-0.5), np.tanh(1.0))),
    "pow3": (lambda s: pow_star([_X], 3, lp_solver=s),
             (-0.125, 1.0)),
    "sin": (lambda s: trig_star([_X], "sin", lp_solver=s),
            (np.sin(-0.5), np.sin(1.0))),
    # floor of a range whose max is exactly the integer 1
    "floor": (lambda s: round_star(_FLOOR, [_X], lp_solver=s),
              (-1.0, 1.0)),
    "softmax": (lambda s: softmax_star(nn.Softmax(dim=-1), [_X2],
                                       lp_solver=s),
                (_sigmoid(-0.5 - 0.4), _sigmoid(1.0 + 0.6))),
    "softmax_attn": (lambda s: softmax_attn_star([_X2], (1, 2), -1,
                                                 lp_solver=s, bounds="lp"),
                     (_sigmoid(-0.5 - 0.4), _sigmoid(1.0 + 0.6))),
    # 3 * x as a set-by-set MatMul with a constant right operand
    "bilinear": (lambda s: bilinear_matmul_star(
        [_X], [_line_star([3.0], [0.0])], (1, 1), (1, 1),
        lp_solver=s, bounds="lp"),
        (-1.5, 3.0)),
}


# ---------------------------------------------------------------------------
# The bound LPs run on HiGHS whatever solver is asked for
# ---------------------------------------------------------------------------

def _spy_solvers(monkeypatch):
    """Record the solver of every LP a Star solves over its predicates."""
    seen = []
    orig = Star._solve_lp_batch

    def spy(self, objectives, minimize_flags, lp_solver=LPSolver.DEFAULT):
        seen.append(lp_solver)
        return orig(self, objectives, minimize_flags, lp_solver)

    monkeypatch.setattr(Star, "_solve_lp_batch", spy)
    return seen


def _all_highs(solvers):
    return all(resolve(s, allow_sentinel=False).is_scipy() for s in solvers)


@pytest.mark.parametrize("op", sorted(_CASES))
def test_relaxation_box_lps_use_highs_whatever_the_requested_solver(
        monkeypatch, op):
    seen = _spy_solvers(monkeypatch)
    _CASES[op][0]("SCS")
    assert seen, "the op solved no LP"
    assert _all_highs(seen), seen


@pytest.mark.parametrize("op", ["mul", "div"])
def test_mccormick_operand_lps_use_highs_whatever_the_requested_solver(
        monkeypatch, op):
    seen = _spy_solvers(monkeypatch)
    y = _line_star([2.0], [0.5])                   # denominator in [1.5, 2.5]
    (_mul_sets if op == "mul" else _div_sets)([_X], [y], "SCS")
    assert seen and _all_highs(seen), seen


def test_refine_lp_cpu_bounds_use_highs_whatever_the_global_solver(
        monkeypatch):
    monkeypatch.setattr(config, "_default_lp_solver", LPSolver.SCS)
    seen = _spy_solvers(monkeypatch)
    _lp_ranges(_X2, "lp_cpu", np.arange(2))
    assert seen and _all_highs(seen), seen


# ---------------------------------------------------------------------------
# The relaxation boxes are padded outward
# ---------------------------------------------------------------------------

_UNDERSHOOT = 5e-10     # measured HiGHS max undershoot on thin wedges


@pytest.mark.parametrize("op", sorted(_CASES))
def test_relaxation_box_survives_lp_undershoot(monkeypatch, op):
    """Every LP optimum is moved 5e-10 inside the true range. The output
    enclosure must still contain the true range of output 0 (floor: the
    unpadded max 1 - 5e-10 rounded down to 0 and dropped floor(1) = 1)."""
    orig = Star._solve_lp_batch

    def undershooting(self, objectives, minimize_flags,
                      lp_solver=LPSolver.DEFAULT):
        vals = orig(self, objectives, minimize_flags, lp_solver)
        return [v if v is None or not np.any(np.asarray(f) != 0)
                else v + _UNDERSHOOT if mn else v - _UNDERSHOOT
                for v, f, mn in zip(vals, objectives, minimize_flags)]

    monkeypatch.setattr(Star, "_solve_lp_batch", undershooting)
    run, (lo, hi) = _CASES[op]
    outs = run("linprog")
    monkeypatch.undo()
    out_lo = min(float(o.estimate_ranges()[0][0, 0]) for o in outs)
    out_hi = max(float(o.estimate_ranges()[1][0, 0]) for o in outs)
    assert out_lo <= lo and out_hi >= hi, (out_lo, lo, out_hi, hi)


def test_constant_rows_stay_exact():
    # A row with no generator is a constant: it is not padded, so an op can
    # still map it exactly (sigmoid of the constant 0 is 0.5).
    S = Star(np.array([[0.0, 0.0], [0.25, 0.75]]), np.zeros((0, 1)),
             np.zeros((0, 1)), -np.ones((1, 1)), np.ones((1, 1)))
    out = sigmoid_star_approx([S])[0]
    lo, hi = out.estimate_ranges()
    assert lo[0, 0] == hi[0, 0] == 0.5


# ---------------------------------------------------------------------------
# End to end with SCS: a product of two computed values
# ---------------------------------------------------------------------------

def test_mccormick_with_an_inexact_solver_is_not_a_false_unsat(inexact_scs):
    """z = x * y with x a coordinate of a constrained star whose true max is
    half the inexact solver's shortfall and y in [1, 1.001]. The point
    x = max, y = 1 is reachable and lies in z >= max / 2. Operand ranges taken
    from the inexact optimum put the product box below it: a false UNSAT. The
    McCormick ranges must come from HiGHS (``_bound_lp_solver``) whatever
    solver the caller names.

    The shortfall is injected (``inexact_scs``): the earlier version searched
    twenty random stars for a real SCS shortfall and skipped on hosts where
    SCS happened to be accurate."""
    rng = np.random.default_rng(0)
    n = 6
    S = Star.from_bounds(-np.ones((n, 1)), np.ones((n, 1)))
    S = S.affine_map(rng.normal(size=(8, n)), rng.normal(size=(8, 1)))
    C = rng.normal(size=(5, S.nVar))
    d = rng.uniform(0.2, 1.0, size=(5, 1))
    S = Star(S.V, np.vstack([S.C, C]), np.vstack([S.d, d]),
             S.predicate_lb, S.predicate_ub)
    i = 0
    M = S.get_max(i, "highs")
    V = S.V.copy()
    V[i, 0] -= M - inexact_scs / 2                     # true max: delta / 2 > 0
    P = np.zeros((1, S.dim))
    P[0, i] = 1.0
    x = Star(V, S.C, S.d, S.predicate_lb, S.predicate_ub).affine_map(P)
    true_max = x.get_max(0, "highs")
    assert true_max == pytest.approx(inexact_scs / 2, abs=1e-8)
    assert x.get_max(0, "SCS") < 0                     # the inexact solver's range excludes it
    y = Star.from_bounds(np.array([[1.0]]), np.array([[1.001]]))
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-true_max / 2]]))
    Z = _mul_sets([x], [y], "SCS")
    assert verify_specification(Z, unsafe).verdict != "UNSAT"
