"""Tests for :mod:`n2v.utils.lp_solver_enum`.

Covers:
- string mixin back-compat (``LPSolver.LINPROG == 'linprog'``)
- :func:`resolve` string handling (case-insensitive, None, enum passthrough,
  hex_oct alias, unknown raises)
- backend/scipy_method/cvxpy_name property correctness for every member
- pickle round-trip (needed for ProcessPoolExecutor usage in parallel code)
"""

from __future__ import annotations

import pickle

import pytest

from n2v.utils.lp_solver_enum import Backend, LPSolver, resolve


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------

class TestResolve:
    def test_known_string(self):
        assert resolve('linprog') is LPSolver.LINPROG
        assert resolve('highs') is LPSolver.HIGHS
        assert resolve('highs-ds') is LPSolver.HIGHS_DS
        assert resolve('GUROBI') is LPSolver.GUROBI

    def test_case_insensitive(self):
        assert resolve('gurobi') is LPSolver.GUROBI
        assert resolve('Gurobi') is LPSolver.GUROBI
        assert resolve('GUROBI') is LPSolver.GUROBI
        assert resolve('LINPROG') is LPSolver.LINPROG
        assert resolve('linprog') is LPSolver.LINPROG

    def test_none_maps_to_default(self):
        assert resolve(None) is LPSolver.DEFAULT

    def test_enum_passthrough(self):
        assert resolve(LPSolver.HIGHS) is LPSolver.HIGHS
        assert resolve(LPSolver.CLARABEL) is LPSolver.CLARABEL
        assert resolve(LPSolver.DEFAULT) is LPSolver.DEFAULT

    def test_hex_oct_alias(self):
        assert resolve('lp', hex_oct_alias=True) is LPSolver.DEFAULT
        # Without the flag, 'lp' is not a valid name.
        with pytest.raises(ValueError):
            resolve('lp')

    def test_hex_oct_alias_does_not_affect_other_names(self):
        assert resolve('linprog', hex_oct_alias=True) is LPSolver.LINPROG

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError) as exc_info:
            resolve('bogus')
        msg = str(exc_info.value)
        assert 'bogus' in msg
        # The error message should list valid names to help the user.
        assert 'linprog' in msg

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve(42)

    def test_allow_sentinel_false_resolves_through_config(self):
        """With allow_sentinel=False, DEFAULT resolves via global config."""
        from n2v.config import config
        original = config.lp_solver
        try:
            config.lp_solver = 'highs'
            assert resolve('default', allow_sentinel=False) is LPSolver.HIGHS
            assert resolve(None, allow_sentinel=False) is LPSolver.HIGHS
            assert resolve(LPSolver.DEFAULT, allow_sentinel=False) is LPSolver.HIGHS
        finally:
            config.lp_solver = original


# ---------------------------------------------------------------------------
# String mixin back-compat
# ---------------------------------------------------------------------------

class TestStringMixin:
    def test_linprog_equals_string(self):
        assert LPSolver.LINPROG == 'linprog'

    def test_gurobi_equals_string(self):
        assert LPSolver.GUROBI == 'GUROBI'

    def test_highs_ds_equals_string(self):
        # Ensures hyphenated names round-trip cleanly.
        assert LPSolver.HIGHS_DS == 'highs-ds'

    def test_value_is_str(self):
        assert isinstance(LPSolver.LINPROG.value, str)
        # LPSolver members are themselves str instances.
        assert isinstance(LPSolver.LINPROG, str)


# ---------------------------------------------------------------------------
# Taxonomy properties
# ---------------------------------------------------------------------------

# (member, expected backend, expected scipy_method, expected cvxpy_name)
_EXPECTED = [
    (LPSolver.DEFAULT, Backend.SENTINEL, None, None),
    (LPSolver.LINPROG, Backend.SCIPY, 'highs', None),
    (LPSolver.HIGHS, Backend.SCIPY, 'highs', None),
    (LPSolver.HIGHS_DS, Backend.SCIPY, 'highs-ds', None),
    (LPSolver.HIGHS_IPM, Backend.SCIPY, 'highs-ipm', None),
    (LPSolver.CLARABEL, Backend.CVXPY, None, 'CLARABEL'),
    (LPSolver.SCS, Backend.CVXPY, None, 'SCS'),
    (LPSolver.ECOS, Backend.CVXPY, None, 'ECOS'),
    (LPSolver.OSQP, Backend.CVXPY, None, 'OSQP'),
    (LPSolver.GUROBI, Backend.CVXPY, None, 'GUROBI'),
    (LPSolver.MOSEK, Backend.CVXPY, None, 'MOSEK'),
    (LPSolver.CBC, Backend.CVXPY, None, 'CBC'),
    (LPSolver.GLPK, Backend.CVXPY, None, 'GLPK'),
    (LPSolver.GLPK_MI, Backend.CVXPY, None, 'GLPK_MI'),
    (LPSolver.CPLEX, Backend.CVXPY, None, 'CPLEX'),
    (LPSolver.COPT, Backend.CVXPY, None, 'COPT'),
    (LPSolver.XPRESS, Backend.CVXPY, None, 'XPRESS'),
    (LPSolver.PIQP, Backend.CVXPY, None, 'PIQP'),
    (LPSolver.PROXQP, Backend.CVXPY, None, 'PROXQP'),
    (LPSolver.SCIP, Backend.CVXPY, None, 'SCIP'),
    (LPSolver.NAG, Backend.CVXPY, None, 'NAG'),
    (LPSolver.CUOPT, Backend.CVXPY, None, 'CUOPT'),
    (LPSolver.DAQP, Backend.CVXPY, None, 'DAQP'),
    (LPSolver.SDPA, Backend.CVXPY, None, 'SDPA'),
]


class TestProperties:
    @pytest.mark.parametrize('member,backend,scipy_method,cvxpy_name', _EXPECTED)
    def test_backend(self, member, backend, scipy_method, cvxpy_name):
        assert member.backend is backend

    @pytest.mark.parametrize('member,backend,scipy_method,cvxpy_name', _EXPECTED)
    def test_scipy_method(self, member, backend, scipy_method, cvxpy_name):
        assert member.scipy_method == scipy_method

    @pytest.mark.parametrize('member,backend,scipy_method,cvxpy_name', _EXPECTED)
    def test_cvxpy_name(self, member, backend, scipy_method, cvxpy_name):
        assert member.cvxpy_name == cvxpy_name

    def test_is_scipy(self):
        assert LPSolver.LINPROG.is_scipy()
        assert LPSolver.HIGHS.is_scipy()
        assert LPSolver.HIGHS_DS.is_scipy()
        assert LPSolver.HIGHS_IPM.is_scipy()
        assert not LPSolver.GUROBI.is_scipy()
        assert not LPSolver.DEFAULT.is_scipy()

    def test_is_cvxpy(self):
        assert LPSolver.GUROBI.is_cvxpy()
        assert LPSolver.CLARABEL.is_cvxpy()
        assert not LPSolver.LINPROG.is_cvxpy()
        assert not LPSolver.DEFAULT.is_cvxpy()

    def test_is_highspy_batch_eligible(self):
        # Only LINPROG and HIGHS are eligible for the batch hot path.
        assert LPSolver.LINPROG.is_highspy_batch_eligible()
        assert LPSolver.HIGHS.is_highspy_batch_eligible()
        assert not LPSolver.HIGHS_DS.is_highspy_batch_eligible()
        assert not LPSolver.HIGHS_IPM.is_highspy_batch_eligible()
        assert not LPSolver.GUROBI.is_highspy_batch_eligible()
        assert not LPSolver.DEFAULT.is_highspy_batch_eligible()

    def test_is_sentinel(self):
        assert LPSolver.DEFAULT.is_sentinel()
        assert not LPSolver.LINPROG.is_sentinel()
        assert not LPSolver.GUROBI.is_sentinel()


# ---------------------------------------------------------------------------
# Pickle round-trip (critical for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

class TestPickle:
    @pytest.mark.parametrize('member', list(LPSolver))
    def test_roundtrip(self, member):
        pickled = pickle.dumps(member)
        restored = pickle.loads(pickled)
        assert restored is member
        # And equality against the legacy string still holds after unpickling.
        assert restored == member.value
