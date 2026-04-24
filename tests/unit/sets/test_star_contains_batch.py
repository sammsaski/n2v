"""Tests for Star.contains single/batch + LP/algebraic parity."""

import numpy as np
import pytest

from n2v.sets.star import Star


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

    def test_batch_wrong_second_dim_raises(self):
        s = _identity_star(2)
        with pytest.raises(ValueError):
            s.contains(np.array([[0.0, 0.0, 0.0]]))


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
