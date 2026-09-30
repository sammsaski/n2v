"""Unit tests for GCN forward eval, GraphStar reach, and ReLU bridging."""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gcn_reach import (
    gcn_evaluate,
    gcn_graph_star,
    gcn_stack_evaluate,
    gcn_stack_reach,
    relu_graph_star,
)
from n2v.utils import load_gnn_mat
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


IEEE24_GCN_PF = checkpoint("gcn")
NEEDS_IEEE24_GCN_PF = pytest.mark.skipif(
    not IEEE24_GCN_PF.exists(),
    reason=f"requires trained checkpoint at {IEEE24_GCN_PF}",
)


# =============================================================== Forward eval


def test_gcn_evaluate_matches_manual_formula():
    rng = np.random.default_rng(0)
    N, F_in, F_out = 5, 3, 4
    X = rng.normal(size=(N, F_in))
    A = rng.normal(size=(N, N))
    W = rng.normal(size=(F_in, F_out))
    b = rng.normal(size=(F_out,))
    Y = gcn_evaluate(X, A, W, b)
    np.testing.assert_allclose(Y, A @ X @ W + b[None, :])


def test_gcn_evaluate_no_bias():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 2))
    A = rng.normal(size=(4, 4))
    W = rng.normal(size=(2, 3))
    Y = gcn_evaluate(X, A, W, b=None)
    np.testing.assert_allclose(Y, A @ X @ W)


def test_gcn_evaluate_validates_shapes():
    X = np.zeros((3, 4))
    A = np.zeros((4, 4))                     # wrong N
    W = np.zeros((4, 2))
    with pytest.raises(ValueError):
        gcn_evaluate(X, A, W)


# ======================================================== GraphStar reach


def test_gcn_graph_star_center_matches_evaluate():
    rng = np.random.default_rng(2)
    N, F_in, F_out = 4, 3, 5
    X = rng.normal(size=(N, F_in))
    A = rng.normal(size=(N, N))
    W = rng.normal(size=(F_in, F_out))
    b = rng.normal(size=(F_out,))

    gs = GraphStar.from_bounds(X - 0.05, X + 0.05, adjacency=A)
    out = gcn_graph_star(gs, W, b)
    np.testing.assert_allclose(out.center(), gcn_evaluate(X, A, W, b), atol=1e-10)


def test_gcn_graph_star_uses_attached_adjacency():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3))
    X = np.zeros((3, 2))
    W = rng.normal(size=(2, 2))
    gs = GraphStar.from_bounds(X, X + 0.1, adjacency=A)
    out = gcn_graph_star(gs, W)
    # attached adjacency should propagate to the result for downstream layers
    np.testing.assert_array_equal(out.adjacency, A)


def test_gcn_graph_star_soundness_against_samples():
    """Every sampled forward output must be inside the reach output's box."""
    rng = np.random.default_rng(4)
    N, F_in, F_out = 4, 3, 5
    A = rng.normal(size=(N, N)) * 0.3
    X = rng.normal(size=(N, F_in))
    eps = 0.1
    W = rng.normal(size=(F_in, F_out))
    b = rng.normal(size=(F_out,))

    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
    out = gcn_graph_star(gs, W, b)
    lb, ub = out.get_ranges()

    for _ in range(64):
        delta = rng.uniform(-eps, eps, size=X.shape)
        Y = gcn_evaluate(X + delta, A, W, b)
        assert np.all(Y >= lb - 1e-9)
        assert np.all(Y <= ub + 1e-9)


# ====================================================================== ReLU


def test_relu_graph_star_exact_zero_perturbation_returns_relu_of_center():
    X = np.array([[1.0, -2.0], [3.0, -4.0]])
    gs = GraphStar.from_bounds(X, X)
    outs = relu_graph_star(gs, method="exact")
    # No perturbation -> single output set whose center is ReLU(X).
    assert len(outs) == 1
    np.testing.assert_allclose(outs[0].center(), np.maximum(X, 0))


def test_relu_graph_star_approx_is_sound():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, 4))
    eps = 0.5  # large enough that some neurons cross zero
    gs = GraphStar.from_bounds(X - eps, X + eps)
    outs = relu_graph_star(gs, method="approx")
    assert len(outs) == 1
    out = outs[0]
    lb, ub = out.get_ranges()

    for _ in range(64):
        delta = rng.uniform(-eps, eps, size=X.shape)
        Y = np.maximum(X + delta, 0.0)
        assert np.all(Y >= lb - 1e-9)
        assert np.all(Y <= ub + 1e-9)


# ============================================== End-to-end on IEEE24 checkpoint


@NEEDS_IEEE24_GCN_PF
def test_ieee24_pf_evaluate_matches_python_predictions():
    model = load_gnn_mat(IEEE24_GCN_PF)
    assert model.num_layers == 3
    assert model.has_relu is True
    assert model.A_norm.shape == (24, 24)
    assert model.num_test_instances >= 5

    worst = 0.0
    for i in range(5):
        Y = gcn_stack_evaluate(model.X_test[i], model.A_norm, model.gcn_layers, model.has_relu)
        worst = max(worst, float(np.max(np.abs(Y - model.python_predictions[i]))))
    # PyTorch float32 precision is the floor; expect <= ~1e-5.
    assert worst < 1e-5, f"max abs diff {worst:.3e} exceeds tolerance"


@NEEDS_IEEE24_GCN_PF
def test_ieee24_pf_reach_zero_perturbation_recovers_prediction():
    """Reach with epsilon=0 collapses to the deterministic prediction."""
    model = load_gnn_mat(IEEE24_GCN_PF)
    X = model.X_test[0]
    expected = model.python_predictions[0]

    gs_in = GraphStar.from_bounds(X, X, adjacency=model.A_norm)
    out_sets = gcn_stack_reach(gs_in, model.gcn_layers, has_relu=model.has_relu)
    assert len(out_sets) == 1
    np.testing.assert_allclose(out_sets[0].center(), expected, atol=1e-5)


@NEEDS_IEEE24_GCN_PF
def test_ieee24_pf_reach_soundness_small_perturbation():
    """Sampled outputs under a small perturbation must lie inside the reach box."""
    model = load_gnn_mat(IEEE24_GCN_PF)
    X = model.X_test[0]
    eps = 0.01
    gs_in = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.A_norm)
    out_sets = gcn_stack_reach(gs_in, model.gcn_layers, has_relu=model.has_relu)
    assert len(out_sets) == 1
    lb, ub = out_sets[0].get_ranges()

    rng = np.random.default_rng(123)
    for _ in range(16):
        delta = rng.uniform(-eps, eps, size=X.shape)
        Y = gcn_stack_evaluate(X + delta, model.A_norm, model.gcn_layers, model.has_relu)
        # Float32 reference + LP solver tolerance
        assert np.all(Y >= lb - 1e-5)
        assert np.all(Y <= ub + 1e-5)


@NEEDS_IEEE24_GCN_PF
def test_ieee24_pf_loader_metadata():
    model = load_gnn_mat(IEEE24_GCN_PF)
    assert model.gcn_layers[0].W.shape == (4, 32)
    assert model.gcn_layers[1].W.shape == (32, 32)
    assert model.gcn_layers[2].W.shape == (32, 4)
    assert model.norm_stats.X_max is not None
    assert model.norm_stats.Y_max is not None
    assert model.norm_stats.X_max.shape == (4,)
    assert model.norm_stats.Y_max.shape == (4,)
