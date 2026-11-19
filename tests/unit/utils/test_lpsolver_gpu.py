"""
Unit tests for n2v.utils.lpsolver_gpu that run without CUDA.

  * Profiling never touches CUDA: with profiling active, a CPU-torch solve
    must not call ``torch.cuda.Event`` or ``torch.cuda.synchronize`` (the
    removed GPU-timing hook did, and crashed on CPU-only hosts).
  * Device resolution: ``None`` picks CUDA only when a CUDA device is usable;
    an explicit device is kept.
  * The unit promises of the certified GPU LP entries, on CPU torch: refusals,
    the default flags, warm starts on a prefix of the rows, the returned duals
    being the certified ones, the best iterate never worse than the start,
    and the primal heuristic's points inside their boxes. The soundness of
    the bounds themselves is in tests/soundness/test_soundness_ns_certificate.py.
"""

from fractions import Fraction as F

import numpy as np
import pytest
import torch

import n2v.profiling
import n2v.utils.lpsolver_gpu as lpg
from n2v.utils.lpsolver_gpu import (
    solve_lp_batch_gpu, solve_lp_population_gpu, solve_lp_primal_population_gpu)
from tests._oracles.rational import vertices
from tests.known_bugs import known_bug


def _cuda_timing_used(*args, **kwargs):
    raise AssertionError("CUDA timing called on a CPU solve")


def test_profiled_cpu_solve_never_calls_cuda_timing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _cuda_timing_used)
    monkeypatch.setattr(torch.cuda, "synchronize", _cuda_timing_used)
    with n2v.profiling.profile():
        bounds = solve_lp_population_gpu(
            A=[[[1.0, 1.0]]], b=[[1.0]],
            lb=np.zeros((1, 2)), ub=np.ones((1, 2)),
            C=np.eye(2)[None], minimize_flags=[False, False],
            max_iters=50, device="cpu")
    # certified upper bounds on max x_i over {x0 + x1 <= 1, [0, 1]^2}: the
    # true maximum is 1 for each i
    assert bounds.shape == (1, 2)
    assert np.all(bounds >= 1.0 - 1e-12)


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(lpg, "gpu_available", lambda: False)
    assert lpg._resolve_device(None) == "cpu"
    assert lpg._resolve_device("cuda") == "cuda"


# --------------------------------------------------------------------------- #
# Device resolution, refusals,                                                  #
# defaults, warm starts, returned duals, best iterate, the primal heuristic.     #
# Every solve runs on CPU torch (device='cpu').                                 #
# --------------------------------------------------------------------------- #
def _q(x):
    return F(float(x))


def _exact_opt(c, A, b, lb, ub, minimize):
    """Exact LP optimum over {A x <= b, lb <= x <= ub} from the vertices."""
    n = len(c)
    rows = [[_q(v) for v in r] for r in (A if A is not None else [])]
    rhs = [_q(v) for v in (b if b is not None else [])]
    for j in range(n):
        e = [F(0)] * n
        e[j] = F(1)
        rows += [e, [-v for v in e]]
        rhs += [_q(ub[j]), -_q(lb[j])]
    vals = [sum(_q(ci) * vi for ci, vi in zip(c, v)) for v in vertices(rows, rhs)]
    return min(vals) if minimize else max(vals)


def _ns_exact(c, y, A, b, lb, ub, minimize):
    """Exact Neumaier-Shcherbina value of the multipliers y (clamped >= 0):
    b'y + sum_j max(r_j lb_j, r_j ub_j), r = s c - A'y, s = -1 for a
    minimisation; returned in the sense of the objective (a lower bound of the
    minimum is minus the value)."""
    s = -1 if minimize else 1
    yq = [max(_q(v), F(0)) for v in np.ravel(y)]
    r = [s * _q(c[j]) - sum(_q(A[i][j]) * yq[i] for i in range(len(yq)))
         for j in range(len(c))]
    v = (sum(_q(bi) * yi for bi, yi in zip(b, yq))
         + sum(max(rj * _q(l), rj * _q(u)) for rj, l, u in zip(r, lb, ub)))
    return s * v


def _f32_tolerance(c, y, A, b, lb, ub):
    """A bound on how far the float32 evaluation of the dual function (which
    picks the best iterate) can be from the exact one, doubled for the
    comparison of two iterates, plus the float64 certificate's own
    allowance: (m + n + 3) roundings of 2^-23 relative on the magnitude of
    every term."""
    m, n = len(b), len(c)
    ya = [abs(float(v)) for v in np.ravel(y)]
    mag = sum(abs(bi) * yi for bi, yi in zip(b, ya))
    for j in range(n):
        rj = abs(c[j]) + sum(abs(A[i][j]) * ya[i] for i in range(m))
        mag += rj * max(abs(lb[j]), abs(ub[j]))
    return 2 * (m + n + 3) * 2.0 ** -23 * mag + 1e-12 * (1 + mag)


# max x1 + x2 s.t. x1 + x2 <= 1.5, x1 - x2 <= 1 over [0, 1]^2: optimum 1.5,
# certified exactly by the multipliers (1, 0); min x1 + x2: optimum 0,
# certified by (0, 0).
_A = np.array([[1.0, 1.0], [1.0, -1.0]])
_B = np.array([1.5, 1.0])
_LB = np.zeros(2)
_UB = np.ones(2)
_C = np.array([1.0, 1.0])


def test_resolve_device_uses_cuda_only_when_it_is_usable(monkeypatch):
    """``device=None`` resolves to 'cuda' when a CUDA device is usable and to
    'cpu' when not; an explicit device is kept either way (a solve never
    moves to a device other than the one named)."""
    monkeypatch.setattr(lpg, "gpu_available", lambda: True)
    assert lpg._resolve_device(None) == "cuda"
    assert lpg._resolve_device("cpu") == "cpu"
    monkeypatch.setattr(lpg, "gpu_available", lambda: False)
    assert lpg._resolve_device("cpu") == "cpu"


def test_gpu_available_is_false_when_the_cuda_query_fails(monkeypatch):
    """``gpu_available`` answers False, not an exception, when asking torch
    for CUDA raises (a broken driver), so ``device=None`` falls back to
    CPU torch."""
    def broken():
        raise RuntimeError("CUDA driver initialisation failed")
    monkeypatch.setattr(torch.cuda, "is_available", broken)
    assert lpg.gpu_available() is False
    assert lpg._resolve_device(None) == "cpu"


def test_batch_refuses_objectives_of_different_lengths_and_a_wrong_flag_count():
    """Inputs that cannot be read are refused with ``ValueError``: objectives
    of different lengths, and a ``minimize_flags`` list whose length is not
    the number of objectives (each flag belongs to one objective)."""
    with pytest.raises(ValueError):
        solve_lp_batch_gpu([np.ones(2), np.ones(3)], lb=np.zeros(3),
                           ub=np.ones(3), device="cpu")
    with pytest.raises(ValueError):
        solve_lp_batch_gpu([_C, _C], A=_A, b=_B, lb=_LB, ub=_UB,
                           minimize_flags=[True], device="cpu")


def test_batch_of_no_objectives_returns_no_bounds():
    """No objectives give an empty list of bounds."""
    assert solve_lp_batch_gpu([], device="cpu") == []


@known_bug(17, "solve_lp_batch_gpu with no objectives returns [] even when "
               "return_duals=True, where the docstring promises the pair "
               "(bounds, duals); a caller that unpacks the pair fails")
def test_batch_of_no_objectives_with_duals_returns_the_pair():
    """``return_duals=True`` returns ``(bounds, duals)``, also for an empty
    batch (both empty)."""
    out = solve_lp_batch_gpu([], A=_A, b=_B, lb=_LB, ub=_UB,
                             return_duals=True, device="cpu")
    assert isinstance(out, tuple) and len(out) == 2
    assert list(out[0]) == [] and list(out[1]) == []


@pytest.mark.parametrize("rows", [True, False])
def test_batch_flags_default_to_minimisation(rows):
    """``minimize_flags=None`` means every objective is minimised (matching
    ``solve_lp_batch``): each value is a lower bound of the exact minimum,
    on objectives whose minimum and maximum differ."""
    objs = [_C, np.array([-0.5, 2.0])]
    A, b = (_A, _B) if rows else (None, None)
    out = solve_lp_batch_gpu(objs, A=A, b=b, lb=_LB, ub=_UB, max_iters=20,
                             device="cpu")
    assert len(out) == 2 and all(isinstance(v, float) for v in out)
    for o, v in zip(objs, out):
        assert _q(v) <= _exact_opt(o, A, b, _LB, _UB, True), (o, v)


def test_batch_warm_duals_cover_a_prefix_of_the_rows_and_are_certified():
    """``warm_duals`` cover a PREFIX of the rows, zero-padded to the row
    count; ``None`` entries start cold; negative and non-finite entries
    start at 0; entries beyond the row count are dropped. With no descent
    step (``max_iters=0``) the returned duals are those start points, and
    each returned bound is the certificate of its returned duals: at or
    beyond their exact Neumaier-Shcherbina value, within its rounding
    allowance."""
    A = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
    b = np.array([1.0, 0.5, 0.25])
    lb = np.array([-2.0, -1.0])
    ub = np.array([1.0, 3.0])
    objs = [np.array([1.0, 2.0])] * 4
    flags = [False, True, False, True]
    warm = [np.array([0.75]), np.array([0.5, 0.25, 9.0, 7.0]), None,
            np.array([np.nan, np.inf, -3.0])]
    expect = [[0.75, 0.0, 0.0], [0.5, 0.25, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    out, duals = solve_lp_batch_gpu(objs, A=A, b=b, lb=lb, ub=ub,
                                    minimize_flags=flags, max_iters=0,
                                    warm_duals=warm, return_duals=True,
                                    device="cpu")
    assert [np.asarray(d, dtype=np.float64).tolist() for d in duals] == expect
    for o, f, d, v in zip(objs, flags, duals, out):
        exact = _ns_exact(o, d, A, b, lb, ub, f)
        allowance = 1e-12 * (1 + abs(float(exact)))
        if f:
            assert _q(v) <= exact and float(exact) - v <= allowance, (v, float(exact))
        else:
            assert _q(v) >= exact and v - float(exact) <= allowance, (v, float(exact))


def test_population_warm_start_reaches_the_descent():
    """``solve_lp_population_gpu`` hands ``Y0`` to the descent: with no
    descent step the returned duals are ``Y0`` clamped to >= 0, and each
    bound is the certificate of those duals (at or beyond their exact value,
    within its rounding allowance)."""
    A = np.stack([_A, _A])
    b = np.stack([_B, np.array([2.0, 0.5])])
    lb = np.stack([_LB, np.array([-1.0, 0.0])])
    ub = np.stack([_UB, np.array([1.0, 2.0])])
    C = np.stack([np.stack([_C, _C], axis=1)] * 2)
    flags = [False, True]
    Y0 = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.5, -2.0], [0.25, 3.0]]])
    bounds, Y = solve_lp_population_gpu(A, b, lb, ub, C, flags, max_iters=0,
                                        Y0=Y0, return_duals=True, device="cpu")
    assert np.asarray(Y, dtype=np.float64).tolist() == np.clip(Y0, 0.0, None).tolist()
    for s in range(2):
        for j, f in enumerate(flags):
            exact = _ns_exact(C[s, :, j], Y0[s, :, j], A[s], b[s], lb[s], ub[s], f)
            v = float(bounds[s, j])
            allowance = 1e-12 * (1 + abs(float(exact)))
            assert (_q(v) <= exact) if f else (_q(v) >= exact), (s, j, v)
            assert abs(v - float(exact)) <= allowance, (s, j, v, float(exact))


# Warm starts at the multipliers that certify the optimum exactly, for a
# maximisation and for a minimisation of the same objective.
_WARM_CASES = [
    # (c, A, b, lb, ub, minimize, warm multipliers)
    (_C, _A, _B, _LB, _UB, False, np.array([1.0, 0.0])),
    (_C, _A, _B, _LB, _UB, True, np.array([0.0, 0.0])),
    (np.array([3.0, -1.0]), np.array([[3.0, -1.0], [0.0, -1.0]]),
     np.array([0.75, 2.5]), np.array([-4.0, -3.0]), np.array([6.0, 5.0]), False,
     np.array([1.0, 0.0])),
    (np.array([-3.0, 1.0]), np.array([[3.0, -1.0], [0.0, -1.0]]),
     np.array([0.75, 2.5]), np.array([-4.0, -3.0]), np.array([6.0, 5.0]), True,
     np.array([1.0, 0.0])),
]


@pytest.mark.parametrize("entry", ["batch", "population"])
def test_descent_never_returns_a_bound_worse_than_its_warm_start(entry):
    """The descent keeps the best iterate and the warm start is an iterate,
    so the certified bound is never worse than the warm start's own
    certificate (up to the float32 evaluation that compares iterates): from
    the multipliers that certify the optimum, 40 steps of a large learning
    rate must not leave a worse bound. Catches the last iterate returned
    instead of the best one, a descent on the objective of the wrong sense,
    and duals dropped on the way out."""
    for c, A, b, lb, ub, minimize, y0 in _WARM_CASES:
        start = _ns_exact(c, y0, A, b, lb, ub, minimize)
        assert start == _exact_opt(c, A, b, lb, ub, minimize)
        if entry == "batch":
            v = solve_lp_batch_gpu([c], A=A, b=b, lb=lb, ub=ub,
                                   minimize_flags=[minimize], max_iters=40,
                                   warm_duals=[y0], device="cpu")[0]
        else:
            v = float(solve_lp_population_gpu(
                A[None], b[None], lb[None], ub[None], c.reshape(1, -1, 1),
                [minimize], max_iters=40, Y0=y0.reshape(1, -1, 1),
                device="cpu")[0, 0])
        tol = _f32_tolerance(c, y0, A, b, lb, ub)
        worse = (float(start) - v) if minimize else (v - float(start))
        assert worse <= tol, (c.tolist(), minimize, v, float(start))


def test_cold_descent_never_returns_a_bound_worse_than_the_box_bound():
    """A cold start is the zero multiplier vector, whose certificate is the
    box bound; the best iterate can only improve on it. Both senses, on rows
    that cut the box."""
    for c, A, b, lb, ub, minimize, _ in _WARM_CASES:
        box = _ns_exact(c, np.zeros(len(b)), A, b, lb, ub, minimize)
        v = solve_lp_batch_gpu([c], A=A, b=b, lb=lb, ub=ub,
                               minimize_flags=[minimize], max_iters=40,
                               device="cpu")[0]
        tol = _f32_tolerance(c, np.zeros(len(b)), A, b, lb, ub)
        worse = (float(box) - v) if minimize else (v - float(box))
        assert worse <= tol, (c.tolist(), minimize, v, float(box))


def test_primal_population_points_lie_in_their_boxes():
    """``solve_lp_primal_population_gpu`` returns float64 points of shape
    (B, n), each inside its own system's box (clamped every iteration), also
    when the objective pushes the iterate out of the box, on a box that does
    not hold 0, and on a zero-width padded variable (pinned to 0)."""
    Ap = np.array([[[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
                   [[0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    bp = np.array([[1.5, 1.0], [6.0, 0.0]])
    lbp = np.array([[0.0, 0.0, 0.0], [2.0, -3.5, 0.0]])
    ubp = np.array([[1.0, 1.0, 0.0], [5.0, -1.25, 0.0]])
    fp = np.array([[1.0, 1.0, 0.0], [-4.0, 3.0, 0.0]])
    X = solve_lp_primal_population_gpu(Ap, bp, lbp, ubp, fp, max_iters=60,
                                       device="cpu")
    assert X.shape == (2, 3) and X.dtype == np.float64
    assert np.all(X >= lbp) and np.all(X <= ubp), X
    assert np.all(X[:, 2] == 0.0)
