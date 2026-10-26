"""Fixtures shared by the soundness tests."""

import pytest

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver, resolve


@pytest.fixture
def inexact_scs(monkeypatch):
    """Make every LP that names the SCS backend return its optimum ``delta``
    inward (a maximum too small, a minimum too large) with status optimal,
    while HiGHS stays exact.

    This is what SCS does at its default tolerance (``_bound_lp_solver``:
    maxima up to 3.6e-5 below the true value, reported as optimal), made
    deterministic: the shortfall is the same on every host and no SCS install
    is needed, so a test of how n2v guards against an inexact solver cannot
    skip or pass by luck of the solver being accurate today. Returns ``delta``.
    """
    delta = 2e-5
    orig_single, orig_batch = Star._solve_lp, Star._solve_lp_batch

    def is_scs(lp_solver):
        return resolve(lp_solver, allow_sentinel=False) is LPSolver.SCS

    def single(self, f, minimize=True, lp_solver=LPSolver.DEFAULT):
        if not is_scs(lp_solver):
            return orig_single(self, f, minimize, lp_solver)
        v = orig_single(self, f, minimize, "highs")
        return None if v is None else (v + delta if minimize else v - delta)

    def batch(self, objectives, minimize_flags, lp_solver=LPSolver.DEFAULT):
        if not is_scs(lp_solver):
            return orig_batch(self, objectives, minimize_flags, lp_solver)
        vals = orig_batch(self, objectives, minimize_flags, "highs")
        return [None if v is None else (v + delta if m else v - delta)
                for v, m in zip(vals, minimize_flags)]

    monkeypatch.setattr(Star, "_solve_lp", single)
    monkeypatch.setattr(Star, "_solve_lp_batch", batch)
    return delta
