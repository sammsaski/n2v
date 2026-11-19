"""Unit tests for SAGEConv forward eval, GraphStar reach, and the loader."""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn.layer_ops.sage_reach import (
    sage_evaluate,
    sage_graph_star,
    sage_stack_evaluate,
    sage_stack_reach,
)
from n2v.utils import load_gnn_mat, SAGELayerSpec
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


IEEE24_SAGE_PF = checkpoint("sage")
NEEDS_IEEE24_SAGE_PF = pytest.mark.skipif(
    not IEEE24_SAGE_PF.exists(),
    reason=f"requires trained checkpoint at {IEEE24_SAGE_PF}",
)


# =============================================================== Forward eval


def test_sage_evaluate_matches_manual_formula():
    rng = np.random.default_rng(0)
    N, F_in, F_out = 5, 3, 4
    X = rng.normal(size=(N, F_in))
    A = rng.normal(size=(N, N))
    Wn = rng.normal(size=(F_in, F_out))
    We = rng.normal(size=(F_in, F_out))
    b = rng.normal(size=(F_out,))
    Y = sage_evaluate(X, A, Wn, We, b)
    np.testing.assert_allclose(Y, X @ Wn + (A @ X) @ We + b[None, :])


def test_sage_evaluate_no_bias():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 2))
    A = rng.normal(size=(4, 4))
    Wn = rng.normal(size=(2, 3))
    We = rng.normal(size=(2, 3))
    Y = sage_evaluate(X, A, Wn, We, b=None)
    np.testing.assert_allclose(Y, X @ Wn + (A @ X) @ We)


def test_sage_evaluate_validates_shapes():
    X = np.zeros((3, 4))
    A = np.zeros((4, 4))                     # wrong N
    Wn = np.zeros((4, 2))
    We = np.zeros((4, 2))
    with pytest.raises(ValueError):
        sage_evaluate(X, A, Wn, We)


# ================================================================= Reach (exact)


def test_sage_graph_star_is_exact_affine():
    """SAGEConv is linear in X, so the reach center equals forward(center)."""
    rng = np.random.default_rng(2)
    N, F_in, F_out = 6, 3, 5
    A = (rng.uniform(size=(N, N)) > 0.5).astype(float)
    Wn = rng.normal(size=(F_in, F_out))
    We = rng.normal(size=(F_in, F_out))
    b = rng.normal(size=(F_out,))
    base = rng.normal(size=(N, F_in))
    gs = GraphStar.from_bounds(base - 0.1, base + 0.1, adjacency=A)
    out = sage_graph_star(gs, Wn, We, b, A)
    np.testing.assert_allclose(out.center(), sage_evaluate(gs.center(), A, Wn, We, b), atol=1e-10)
    assert out.nVar == gs.nVar          # no new predicates introduced
    assert out.V.shape == (N, F_out, gs.nVar + 1)


def test_sage_graph_star_uses_attached_adjacency():
    rng = np.random.default_rng(3)
    A = (rng.uniform(size=(4, 4)) > 0.5).astype(float)
    base = rng.normal(size=(4, 2))
    gs = GraphStar.from_bounds(base - 0.05, base + 0.05, adjacency=A)
    Wn = rng.normal(size=(2, 3))
    We = rng.normal(size=(2, 3))
    out = sage_graph_star(gs, Wn, We, None)             # A taken from set
    np.testing.assert_allclose(out.center(), sage_evaluate(gs.center(), A, Wn, We, None), atol=1e-10)


def test_sage_stack_reach_contains_sampled_forward():
    rng = np.random.default_rng(4)
    N, F = 5, 3
    A = (rng.uniform(size=(N, N)) > 0.4).astype(float)
    layers = [
        SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)), rng.normal(size=(F,))),
        SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)), rng.normal(size=(F,))),
    ]
    base = rng.normal(size=(N, F))
    eps = 0.02
    gs = GraphStar.from_bounds(base - eps, base + eps, adjacency=A)
    out = sage_stack_reach(gs, layers, has_relu=True, A=A)[0]
    olb, oub = out.get_ranges()
    for _ in range(100):
        Xi = base + rng.uniform(-eps, eps, size=base.shape)
        Yi = sage_stack_evaluate(Xi, A, layers, has_relu=True)
        assert np.all(Yi >= olb - 1e-7) and np.all(Yi <= oub + 1e-7)


# ===================================================================== Loader


@NEEDS_IEEE24_SAGE_PF
def test_load_sage_checkpoint_shapes_and_parity():
    model = load_gnn_mat(IEEE24_SAGE_PF)
    assert model.model_type == "sage"
    assert model.A_adj.shape == (24, 24)
    assert model.adjacency is model.A_adj
    assert all(isinstance(l, SAGELayerSpec) for l in model.layers)
    assert model.layers[0].W_node.shape[0] == 4
    worst = 0.0
    for i in range(min(5, model.num_test_instances)):
        Y = sage_stack_evaluate(model.X_test[i], model.A_adj, model.layers, model.has_relu)
        worst = max(worst, float(np.max(np.abs(Y - model.python_predictions[i]))))
    assert worst < 1e-5
