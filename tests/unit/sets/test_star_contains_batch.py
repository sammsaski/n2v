"""Tests for Star.contains single/batch + LP/algebraic parity."""

from fractions import Fraction

import numpy as np
import pytest

from n2v.sets.star import Star
from tests._oracles import rational
from tests.known_bugs import known_bug


def _identity_star(dim, lb=-1.0, ub=1.0, C=None, d=None):
    """Star whose image is the hyperbox [lb, ub]^dim (identity basis)."""
    V = np.zeros((dim, dim + 1))
    V[:, 1:] = np.eye(dim)
    plb = np.full(dim, lb)
    pub = np.full(dim, ub)
    return Star(V=V, C=C, d=d, pred_lb=plb, pred_ub=pub)


class TestSinglePoint:
    def test_interior_point_is_contained(self):
        s = _identity_star(2)
        assert s.contains(np.array([0.0, 0.0])) is True

    def test_boundary_point_is_contained(self):
        s = _identity_star(2)
        assert s.contains(np.array([1.0, 0.0])) is True

    def test_faraway_point_is_not_contained(self):
        s = _identity_star(2)
        assert s.contains(np.array([5.0, 0.0])) is False

    def test_column_point_shape_accepted(self):
        s = _identity_star(2)
        assert s.contains(np.array([[0.0], [0.0]])) is True

    @pytest.mark.parametrize("point", ["finite-far", pytest.param("non-finite", marks=known_bug(
        8, "Star.contains with the default LP method answers True for a point with a NaN or an infinite "
           "entry: the point is the right-hand side of the equality rows of a feasibility check whose "
           "answer True means only 'not shown empty', and contains reads that answer as 'a member' "
           "(the algebraic method answers False; registry row SET-50, the Hexatope and Octatope twin has "
           "a row of its own)"))])
    def test_point_far_outside_or_not_finite_is_not_contained(self, point):
        """A point far outside the box [-1, 3] x [2, 5], and points with a NaN or an infinite entry, are not members of a bounded set: ``contains`` answers False, or refuses a point that is not finite with ``ValueError``. The finite point is the passing neighbour. Catches a membership test that a NaN passes."""
        s = Star.from_bounds(np.array([-1.0, 2.0]), np.array([3.0, 5.0]))
        points = ([[100.0, 3.0]] if point == "finite-far"
                  else [[np.nan, 3.0], [1.0, np.nan], [np.inf, 3.0], [-np.inf, 3.0]])
        for p in points:
            try:
                answer = s.contains(np.array(p))
            except ValueError:
                assert point == "non-finite", "a finite point was refused"
                continue
            assert not answer, f"Star.contains({p}) is True"


class TestBatch:
    def test_batch_shape_output(self):
        s = _identity_star(3)
        X = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        r = s.contains(X)
        assert isinstance(r, np.ndarray)
        assert r.dtype == bool
        assert r.shape == (3,)
        np.testing.assert_array_equal(r, np.array([True, False, True]))

    def test_batch_vs_single_on_identity(self):
        rng = np.random.default_rng(0)
        s = _identity_star(3)
        X = rng.uniform(-1.5, 1.5, size=(200, 3))
        single = np.array([s.contains(X[i]) for i in range(X.shape[0])])
        batch = s.contains(X)
        np.testing.assert_array_equal(single, batch)
        # Independent truth: the star is the box [-1, 1]^3.
        np.testing.assert_array_equal(batch, np.all(np.abs(X) <= 1, axis=1))

    def test_batch_vs_single_on_cd_star(self):
        """Predicate [-1,1]^3 intersected with {alpha[0] + alpha[1] <= 0.5}."""
        rng = np.random.default_rng(1)
        s = _identity_star(3)
        s.C = np.array([[1.0, 1.0, 0.0]])
        s.d = np.array([0.5])
        X = rng.uniform(-1.5, 1.5, size=(200, 3))
        single = np.array([s.contains(X[i]) for i in range(X.shape[0])])
        batch = s.contains(X)
        np.testing.assert_array_equal(single, batch)
        # Independent truth: the box [-1, 1]^3 with x0 + x1 <= 1/2, decided
        # in rationals (the random points are never within rounding of the
        # boundary).
        truth = [all(abs(Fraction(v)) <= 1 for v in x)
                 and Fraction(x[0]) + Fraction(x[1]) <= Fraction(1, 2)
                 for x in X]
        np.testing.assert_array_equal(batch, truth)

    def test_batch_wrong_second_dim_raises(self):
        s = _identity_star(2)
        with pytest.raises(ValueError):
            s.contains(np.array([[0.0, 0.0, 0.0]]))

    def test_input_of_rank_three_raises(self):
        """A point array of three axes cannot be read as a point or a batch:
        ``ValueError``, not a guess."""
        s = _identity_star(2)
        with pytest.raises(ValueError):
            s.contains(np.zeros((1, 2, 1)))

    def test_unknown_method_raises(self):
        """A containment method other than 'lp' and 'algebraic' is refused
        with ``ValueError`` (a misspelt method must not fall back to one of
        them silently)."""
        s = _identity_star(2)
        with pytest.raises(ValueError):
            s.contains(np.zeros(2), method="exact")


class TestAlgebraicParity:
    def test_algebraic_matches_lp_random_full_rank(self):
        """Generate full-rank Stars (square basis), random C/d, random points;
        LP and algebraic must agree everywhere."""
        rng = np.random.default_rng(42)
        dim = 3
        for trial in range(10):
            basis = rng.normal(size=(dim, dim))
            # guarantee full rank
            while abs(np.linalg.det(basis)) < 0.1:
                basis = rng.normal(size=(dim, dim))
            V = np.zeros((dim, dim + 1))
            V[:, 0] = rng.normal(size=dim)
            V[:, 1:] = basis
            plb = rng.uniform(-1.0, 0.0, size=dim)
            pub = rng.uniform(0.0, 1.0, size=dim)
            # Random C/d that is satisfiable at alpha=0.
            nC = rng.integers(0, 3)
            if nC > 0:
                C = rng.normal(size=(nC, dim))
                d = np.abs(rng.normal(size=nC)) + 0.5  # keep alpha=0 feasible
            else:
                C = None; d = None
            s = Star(V=V, C=C, d=d, pred_lb=plb, pred_ub=pub)
            X = rng.uniform(-3.0, 3.0, size=(50, dim))
            r_lp = s.contains(X, method='lp')
            r_alg = s.contains(X, method='algebraic')
            np.testing.assert_array_equal(
                r_lp, r_alg,
                err_msg=f'trial={trial}: LP {r_lp.sum()} vs alg {r_alg.sum()}',
            )
            # Independent truth: exact membership in rationals of the same
            # arrays (no n2v code), so a defect both paths share is seen.
            member = rational.Membership(rational.star(s))
            truth = np.array([member(x) for x in X])
            np.testing.assert_array_equal(
                r_lp, truth, err_msg=f'trial={trial}: LP vs exact')

    def test_algebraic_raises_on_wide_basis(self):
        """Wide basis (more predicate vars than output dim) -> algebraic not
        applicable, must raise ValueError."""
        dim, nVar = 2, 3
        V = np.zeros((dim, nVar + 1))
        V[:, 1:] = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])  # 2x3 wide
        plb = np.full(nVar, -1.0)
        pub = np.full(nVar,  1.0)
        s = Star(V=V, C=None, d=None, pred_lb=plb, pred_ub=pub)
        with pytest.raises(ValueError):
            s.contains(np.array([0.0, 0.0]), method='algebraic')

    def test_algebraic_raises_on_rank_deficient_basis(self):
        """A basis whose columns are dependent has no unique predicate point
        for a point, so the algebraic path must refuse with ``ValueError``
        (tall: 3 entries, 2 parallel generators; square: 2 by 2 of rank 1),
        and never answer from an arbitrary least-squares solution."""
        V = np.zeros((3, 3))
        V[:, 1] = [1.0, 2.0, 0.5]
        V[:, 2] = [2.0, 4.0, 1.0]
        tall = Star(V=V, C=None, d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
        with pytest.raises(ValueError):
            tall.contains(np.array([3.0, 6.0, 1.5]), method='algebraic')
        W = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 1.0]])
        square = Star(V=W, C=None, d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
        with pytest.raises(ValueError):
            square.contains(np.array([1.0, 0.5]), method='algebraic')


class TestEmptyStar:
    def test_infeasible_predicate_rejects_all(self):
        """plb > pub -> empty feasible set -> contains False for every point."""
        dim = 2
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        # infeasible: lb above ub
        plb = np.array([1.0, 1.0])
        pub = np.array([-1.0, -1.0])
        s = Star(V=V, C=None, d=None, pred_lb=plb, pred_ub=pub)
        X = np.array([[0.0, 0.0], [0.5, 0.5]])
        r = s.contains(X)
        assert not r.any()
