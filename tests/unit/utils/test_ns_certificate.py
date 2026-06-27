"""Tests for the Neumaier-Shcherbina sound LP bound certificate.

The certificate turns an *approximate* dual ``y`` (from any solver, e.g. a
first-order GPU PDLP) into a *guaranteed* bound on the LP optimum by weak
duality. Soundness is independent of ``y``'s accuracy: for any ``y >= 0`` the
returned value over-estimates ``max c^T x`` (resp. under-estimates ``min``).
A sloppy dual only loosens the bound; it never makes it unsound.

These tests pin:
  * soundness for arbitrary ``y >= 0`` (the core guarantee),
  * tightness when ``y`` is the optimal dual (hand LPs + solver duals),
  * the box-only / no-constraint reductions,
  * negative-dual clamping,
  * batch == per-objective,
  * strict outward enclosure of the exact optimum on many random LPs
    (the directed-rounding / FP-soundness guard).
"""

import numpy as np
import pytest

from n2v.utils.lpsolver import solve_lp
from n2v.utils.ns_certificate import ns_bound, ns_bounds_batch, ns_bounds_population


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _exact_opt(c, A, b, lb, ub, minimize):
    """Ground-truth optimum via HiGHS (the solver we must enclose)."""
    _, fval, status, _ = solve_lp(
        f=c, A=A, b=b, lb=lb, ub=ub, minimize=minimize,
    )
    assert status == "optimal", f"setup LP not optimal: {status}"
    return fval


def _random_lp(rng, n, m):
    """A random feasible bounded LP over a finite box (the n2v regime)."""
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    # RHS generous enough that the box-bounded problem stays feasible.
    b = rng.uniform(1.0, 3.0, size=m) + np.abs(A).sum(axis=1)
    lb = rng.uniform(-2.0, 0.0, size=n)
    ub = lb + rng.uniform(0.5, 3.0, size=n)
    return c, A, b, lb, ub


# --------------------------------------------------------------------------- #
# Core soundness: any y >= 0 yields a sound bound
# --------------------------------------------------------------------------- #
class TestNSSoundnessArbitraryDual:
    def test_random_nonneg_dual_encloses_optimum(self):
        """For *any* y >= 0 the NS upper bound >= true max, lower bound <= true min."""
        rng = np.random.default_rng(47)
        for _ in range(200):
            n, m = rng.integers(1, 6), rng.integers(1, 6)
            c, A, b, lb, ub = _random_lp(rng, n, m)
            y = rng.uniform(0.0, 2.0, size=m)  # arbitrary nonneg dual

            true_max = _exact_opt(c, A, b, lb, ub, minimize=False)
            true_min = _exact_opt(c, A, b, lb, ub, minimize=True)

            ub_bound = ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=False)
            lb_bound = ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=True)

            # Sound: never tighter than the true optimum.
            assert ub_bound >= true_max - 1e-9
            assert lb_bound <= true_min + 1e-9

    def test_zero_dual_reduces_to_box_bound(self):
        """y = 0 ignores the general constraints -> the (sound) box bound."""
        c = np.array([1.0, -2.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([0.5])
        lb = np.array([-1.0, -1.0])
        ub = np.array([2.0, 3.0])
        y = np.zeros(1)

        # Box max of c^T x: 1*2 + (-2)*(-1) = 4 ; box min: 1*(-1)+(-2)*3 = -7.
        assert ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=False) == pytest.approx(4.0, abs=1e-12)
        assert ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=True) == pytest.approx(-7.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Tightness when the dual is optimal
# --------------------------------------------------------------------------- #
class TestNSTightness:
    def test_hand_lp_binding_constraint(self):
        """max x1+x2 s.t. x1+x2<=1, 0<=x<=1. opt=1, optimal dual y=1 -> NS exact."""
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        y = np.array([1.0])  # exact optimal dual
        assert ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=False) == pytest.approx(1.0, abs=1e-12)

    def test_hand_lp_box_binds_slack_constraint(self):
        """max x s.t. x<=5, 0<=x<=2. Box binds; optimal dual y=0 -> NS = 2 (exact)."""
        c = np.array([1.0])
        A = np.array([[1.0]])
        b = np.array([5.0])
        lb = np.array([0.0])
        ub = np.array([2.0])
        assert ns_bound(c, np.array([0.0]), A=A, b=b, lb=lb, ub=ub, minimize=False) == pytest.approx(2.0, abs=1e-12)

    def test_optimal_solver_dual_is_tight(self):
        """Feeding a solver's optimal dual makes NS within ~tol of the true optimum.

        Orientation-agnostic: any y>=0 is sound, so we take the tighter of the
        two sign orientations of the extracted dual and assert it matches opt.
        """
        rng = np.random.default_rng(11)
        for _ in range(50):
            n, m = rng.integers(1, 5), rng.integers(1, 5)
            c, A, b, lb, ub = _random_lp(rng, n, m)
            true_max = _exact_opt(c, A, b, lb, ub, minimize=False)
            row_dual = _highs_row_dual(c, A, b, lb, ub, minimize=False)
            cand = [np.clip(row_dual, 0, None), np.clip(-row_dual, 0, None)]
            bounds = [ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=False) for y in cand]
            # Both sound:
            for bd in bounds:
                assert bd >= true_max - 1e-7
            # The correctly-oriented one is tight:
            assert min(bounds) == pytest.approx(true_max, abs=1e-6)


# --------------------------------------------------------------------------- #
# Reductions / edge cases
# --------------------------------------------------------------------------- #
class TestNSEdgeCases:
    def test_no_general_constraints_is_exact_box_bound(self):
        c = np.array([2.0, -1.0, 0.5])
        lb = np.array([-1.0, -2.0, 0.0])
        ub = np.array([1.0, 0.0, 4.0])
        # max: 2*1 + (-1)*(-2) + 0.5*4 = 2+2+2 = 6 ; min: 2*-1 + -1*0 + 0.5*0 = -2.
        assert ns_bound(c, np.zeros(0), A=None, b=None, lb=lb, ub=ub, minimize=False) == pytest.approx(6.0, abs=1e-12)
        assert ns_bound(c, np.zeros(0), A=None, b=None, lb=lb, ub=ub, minimize=True) == pytest.approx(-2.0, abs=1e-12)

    def test_negative_dual_entries_are_clamped(self):
        """Negative y is clamped to 0 (weak duality requires y>=0); still sound."""
        rng = np.random.default_rng(3)
        c, A, b, lb, ub = _random_lp(rng, 4, 4)
        true_max = _exact_opt(c, A, b, lb, ub, minimize=False)
        y_neg = rng.uniform(-2.0, 2.0, size=4)
        bd = ns_bound(c, y_neg, A=A, b=b, lb=lb, ub=ub, minimize=False)
        bd_clamped = ns_bound(c, np.clip(y_neg, 0, None), A=A, b=b, lb=lb, ub=ub, minimize=False)
        assert bd == pytest.approx(bd_clamped, rel=0, abs=0)  # identical: clamp is internal
        assert bd >= true_max - 1e-9

    def test_string_or_column_shapes_accepted(self):
        """(n,1) / (m,1) column inputs are flattened like the rest of the LP stack."""
        c = np.array([[1.0], [1.0]])
        A = np.array([[1.0, 1.0]])
        b = np.array([[1.0]])
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        y = np.array([[1.0]])
        assert ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=False) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Batch == per-objective
# --------------------------------------------------------------------------- #
class TestNSBatch:
    def test_batch_matches_scalar(self):
        rng = np.random.default_rng(99)
        n, m = 4, 3
        c0, A, b, lb, ub = _random_lp(rng, n, m)
        objectives = [c0, -c0, rng.standard_normal(n), rng.standard_normal(n)]
        minimize_flags = [False, True, False, True]
        duals = [rng.uniform(0, 1, size=m) for _ in objectives]

        batched = ns_bounds_batch(objectives, duals, A=A, b=b, lb=lb, ub=ub,
                                  minimize_flags=minimize_flags)
        for i, (c, y, mn) in enumerate(zip(objectives, duals, minimize_flags)):
            scalar = ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=mn)
            assert batched[i] == pytest.approx(scalar, rel=0, abs=0)

    def test_population_matches_scalar(self):
        """Vectorized population NS == scalar ns_bound per (star, objective)."""
        rng = np.random.default_rng(2025)
        B, n, m, k = 5, 4, 3, 6
        minimize_flags = [j % 2 == 1 for j in range(k)]
        # Per-star constraints/box; shared shapes (padding-free case).
        A = rng.standard_normal((B, m, n))
        b = rng.uniform(1.0, 3.0, size=(B, m))
        lb = rng.uniform(-2.0, 0.0, size=(B, n))
        ub = lb + rng.uniform(0.5, 3.0, size=(B, n))
        C = rng.standard_normal((B, n, k))
        Y = rng.uniform(0.0, 1.5, size=(B, m, k))

        pop = ns_bounds_population(C, Y, A, b, lb, ub, minimize_flags)
        for s in range(B):
            for j in range(k):
                scalar = ns_bound(C[s, :, j], Y[s, :, j], A=A[s], b=b[s],
                                  lb=lb[s], ub=ub[s], minimize=minimize_flags[j])
                assert pop[s, j] == pytest.approx(scalar, rel=0, abs=1e-12)

    def test_population_zero_pad_rows_inert(self):
        """All-zero padded constraint rows (A row 0, b 0) don't change the bound."""
        rng = np.random.default_rng(5)
        B, n, m, k = 3, 4, 2, 4
        flags = [False, True, False, True]
        A = rng.standard_normal((B, m, n))
        b = rng.uniform(1.0, 3.0, size=(B, m))
        lb = rng.uniform(-1.0, 0.0, size=(B, n))
        ub = lb + rng.uniform(0.5, 2.0, size=(B, n))
        C = rng.standard_normal((B, n, k))
        Y = rng.uniform(0.0, 1.0, size=(B, m, k))
        base = ns_bounds_population(C, Y, A, b, lb, ub, flags)
        # Append two inert padded rows + arbitrary duals on them.
        Apad = np.concatenate([A, np.zeros((B, 2, n))], axis=1)
        bpad = np.concatenate([b, np.zeros((B, 2))], axis=1)
        Ypad = np.concatenate([Y, rng.uniform(0, 1, size=(B, 2, k))], axis=1)
        padded = ns_bounds_population(C, Ypad, Apad, bpad, lb, ub, flags)
        assert np.allclose(base, padded, atol=1e-9)


# --------------------------------------------------------------------------- #
# Strict outward enclosure (directed-rounding / FP-soundness guard)
# --------------------------------------------------------------------------- #
class TestNSDirectedRounding:
    def test_strict_enclosure_with_optimal_dual_many_lps(self):
        """The certified bound must enclose the exact optimum even when the dual
        is (near-)optimal and the bound is razor-tight -- this is where FP error
        in the certificate could flip it inward. The directed inflation must
        keep it outward."""
        rng = np.random.default_rng(2024)
        viol = 0
        for _ in range(300):
            n, m = rng.integers(1, 6), rng.integers(1, 6)
            c, A, b, lb, ub = _random_lp(rng, n, m)
            for minimize in (False, True):
                true_opt = _exact_opt(c, A, b, lb, ub, minimize=minimize)
                row_dual = _highs_row_dual(c, A, b, lb, ub, minimize=minimize)
                for orient in (row_dual, -row_dual):
                    y = np.clip(orient, 0, None)
                    bd = ns_bound(c, y, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
                    if minimize:
                        # lower bound must be <= true min (allow only solver tol)
                        if bd > true_opt + 1e-7:
                            viol += 1
                    else:
                        if bd < true_opt - 1e-7:
                            viol += 1
        assert viol == 0, f"{viol} inward (unsound) certified bounds"


# --------------------------------------------------------------------------- #
# Dual extraction helper (test-only)
# --------------------------------------------------------------------------- #
def _highs_row_dual(c, A, b, lb, ub, minimize):
    """Extract row duals from HiGHS for a max/min over Ax<=b, lb<=x<=ub.

    Sign convention is irrelevant to callers here: tests clamp both
    orientations and rely on NS soundness for any y>=0.
    """
    import highspy

    c = np.asarray(c, dtype=np.float64).flatten()
    n = c.shape[0]
    h = highspy.Highs()
    h.silent()
    inf = highspy.kHighsInf
    col_lb = np.asarray(lb, dtype=np.float64).flatten()
    col_ub = np.asarray(ub, dtype=np.float64).flatten()
    h.addVars(n, col_lb, col_ub)
    h.changeObjectiveSense(
        highspy.ObjSense.kMinimize if minimize else highspy.ObjSense.kMaximize
    )
    h.changeColsCost(n, np.arange(n, dtype=np.int32), c)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    for i in range(A.shape[0]):
        row = A[i, :]
        nz = np.nonzero(row)[0]
        h.addRow(-inf, b[i], len(nz), nz.astype(np.int32), row[nz])
    h.run()
    return np.array(h.getSolution().row_dual, dtype=np.float64)
