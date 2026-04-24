"""Tests for the HalfSpace → scenario-verify bridge helpers."""

import numpy as np
import pytest


def test_spec_summary_single_row_halfspace():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    hs = HalfSpace(np.array([[1.0, 0.0, -1.0]]), np.array([[2.5]]))
    s = spec_summary(hs)
    assert isinstance(s, str)
    assert '1 constraint' in s or '1 halfspace' in s.lower()
    assert 'dim=3' in s


def test_spec_summary_and_halfspace():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    # 3 rows = AND of 3 halfspaces
    hs = HalfSpace(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        np.array([[1.0], [1.0], [0.0]]),
    )
    s = spec_summary(hs)
    assert '3 constraint' in s or '3 halfspace' in s.lower()


def test_spec_summary_list_of_halfspaces():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    a = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1.0]]))
    b = HalfSpace(np.array([[0.0, 1.0]]), np.array([[1.0]]))
    s = spec_summary([a, b])
    assert 'OR' in s
    assert '2' in s


import math
import torch

from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow


def _train_small_flow(dim: int = 2, seed: int = 0):
    """Train a quick flow on a 2D Gaussian for use in spec tests."""
    torch.manual_seed(seed)
    y_train = torch.randn(1000, dim)  # N(0, I)
    vf = VelocityField(dim=dim, hidden=32, n_layers=2, activation='silu')
    vf, _ = train_flow(
        vf, y_train, n_epochs=100, batch_size=128, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=10,
        use_ema=True, standardize_outputs=False,
    )
    return FlowODE(vf.eval())


def test_verify_spec_returns_required_keys():
    from examples.FlowConformal.benchmarks._spec import verify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace

    flow = _train_small_flow(dim=2, seed=0)
    # A very loose spec that any reasonable flow reachset should satisfy.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))
    result = verify_spec_on_flow(
        flow_ode=flow,
        threshold_q=5.0,
        spec=hs,
        input_lb=np.array([-1.0, -1.0]),
        input_ub=np.array([1.0, 1.0]),
        network=None,                # preimage search disabled
        alpha=0.01,
        delta_1=0.997,
        beta_2=0.001,
        n_samples=2000,
    )
    # Shape of the returned dict:
    assert set(result.keys()) >= {
        'verdict', 'epsilon_2', 'delta_2', 'n_samples_used',
        'counterexample', 'per_constraint_results',
    }
    assert result['verdict'] in ('SAT', 'UNSAT', 'UNKNOWN')


def test_verify_spec_rejects_or_of_ands():
    """Lists of HalfSpaces should raise NotImplementedError in Phase 2."""
    from examples.FlowConformal.benchmarks._spec import verify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace

    flow = _train_small_flow(dim=2, seed=0)
    a = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1.0]]))
    b = HalfSpace(np.array([[0.0, 1.0]]), np.array([[1.0]]))
    with pytest.raises(NotImplementedError):
        verify_spec_on_flow(
            flow_ode=flow, threshold_q=1.0,
            spec=[a, b],  # OR-of-ANDs
            input_lb=np.array([-1.0, -1.0]),
            input_ub=np.array([1.0, 1.0]),
            network=None,
            alpha=0.01, delta_1=0.997, beta_2=0.001, n_samples=500,
        )


def test_verify_spec_and_halfspace_loops_over_rows():
    """A 3-row HalfSpace (AND of 3 constraints) should produce 3 per-
    constraint results and a unified verdict.

    VNN-LIB convention: ``G y <= g`` defines the UNSAFE region. Here
    the rows encode an empty intersection (``y_0 <= -100 AND y_0 >= 100``
    plus a third constraint), so the unsafe region is empty and every
    reach set is trivially outside it — expect UNSAT.
    """
    from examples.FlowConformal.benchmarks._spec import verify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace

    flow = _train_small_flow(dim=2, seed=0)
    # Rows: y_0 <= -100, -y_0 <= -100 (i.e. y_0 >= 100), y_1 <= -100.
    # The AND is empty → unsafe region is empty → UNSAT for any reach set.
    hs = HalfSpace(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        np.array([[-100.0], [-100.0], [-100.0]]),
    )
    result = verify_spec_on_flow(
        flow_ode=flow, threshold_q=5.0, spec=hs,
        input_lb=np.array([-1.0, -1.0]),
        input_ub=np.array([1.0, 1.0]),
        network=None,
        alpha=0.01, delta_1=0.997, beta_2=0.001, n_samples=2000,
    )
    assert len(result['per_constraint_results']) == 3
    # Unsafe region empty → no reach set hits it → UNSAT (verified).
    assert result['verdict'] == 'UNSAT'
