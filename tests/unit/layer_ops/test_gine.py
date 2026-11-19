"""Unit tests for GINE forward eval, GraphStar reach, and the loader.

Covers both architecture variants: 'hugine' (Hu et al., gine_pretrain) and
'pyg' (PyTorch-Geometric GINEConv, gine_conv).
"""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gine_reach import (
    gine_evaluate,
    gine_graph_star,
    gine_stack_evaluate,
    gine_stack_reach,
)
from n2v.utils import load_gnn_mat, GINELayerSpec
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


IEEE24_GINE_PF = checkpoint("gine_pretrain")
NEEDS_IEEE24_GINE_PF = pytest.mark.skipif(
    not IEEE24_GINE_PF.exists(),
    reason=f"requires trained checkpoint at {IEEE24_GINE_PF}",
)


def _make_layer(rng, F_in, hidden, F_out, E_in, eps=0.0):
    return GINELayerSpec(
        W1=rng.normal(size=(F_in, hidden)), b1=rng.normal(size=(hidden,)),
        W2=rng.normal(size=(hidden, F_out)), b2=rng.normal(size=(F_out,)),
        W_edge=rng.normal(size=(E_in, F_in)), b_edge=rng.normal(size=(F_in,)),
        eps=eps,
    )


def _random_graph(rng, N, m, with_self_loops=True):
    src = rng.integers(0, N, size=m)
    dst = rng.integers(0, N, size=m)
    if with_self_loops:
        loops = np.arange(N)
        src = np.concatenate([src, loops])
        dst = np.concatenate([dst, loops])
    return np.vstack([src, dst])


# =============================================================== Forward eval


def test_gine_evaluate_hugine_matches_manual_formula():
    rng = np.random.default_rng(0)
    N, F_in, hidden, F_out, E_in = 6, 3, 5, 4, 2
    edge_index = _random_graph(rng, N, 8)
    m = edge_index.shape[1]
    X = rng.normal(size=(N, F_in))
    E = rng.normal(size=(m, E_in))
    L = _make_layer(rng, F_in, hidden, F_out, E_in)
    src, dst = edge_index
    e_proj = E @ L.W_edge + L.b_edge[None, :]
    agg = np.zeros((N, F_in))
    np.add.at(agg, dst, X[src] + e_proj)         # no message ReLU, self in edges
    expected = np.maximum(agg @ L.W1 + L.b1[None, :], 0.0) @ L.W2 + L.b2[None, :]
    got = gine_evaluate(X, E, edge_index, L, variant="hugine")
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_gine_evaluate_pyg_has_message_relu_and_eps():
    rng = np.random.default_rng(1)
    N, F_in, hidden, F_out, E_in = 5, 3, 4, 3, 2
    edge_index = _random_graph(rng, N, 6, with_self_loops=False)
    m = edge_index.shape[1]
    X = rng.normal(size=(N, F_in))
    E = rng.normal(size=(m, E_in))
    L = _make_layer(rng, F_in, hidden, F_out, E_in, eps=0.3)
    src, dst = edge_index
    msg = np.maximum(X[src] + E @ L.W_edge + L.b_edge[None, :], 0.0)   # message ReLU
    agg = np.zeros((N, F_in))
    np.add.at(agg, dst, msg)
    combined = (1.0 + L.eps) * X + agg
    expected = np.maximum(combined @ L.W1 + L.b1[None, :], 0.0) @ L.W2 + L.b2[None, :]
    got = gine_evaluate(X, E, edge_index, L, variant="pyg")
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_gine_output_relu_clamps_negatives():
    rng = np.random.default_rng(2)
    N, F_in, hidden, F_out, E_in = 4, 2, 3, 2, 2
    edge_index = _random_graph(rng, N, 5)
    m = edge_index.shape[1]
    X = rng.normal(size=(N, F_in))
    E = rng.normal(size=(m, E_in))
    L = _make_layer(rng, F_in, hidden, F_out, E_in)
    out = gine_evaluate(X, E, edge_index, L, variant="hugine", apply_output_relu=True)
    assert np.all(out >= 0.0)


# ===================================================================== Reach


@pytest.mark.parametrize("variant", ["hugine", "pyg"])
def test_gine_reach_contains_forward_center(variant):
    rng = np.random.default_rng(3)
    N, F_in, hidden, F_out, E_in = 6, 3, 5, 3, 2
    edge_index = _random_graph(rng, N, 10)
    m = edge_index.shape[1]
    E = rng.normal(size=(m, E_in))
    L = _make_layer(rng, F_in, hidden, F_out, E_in, eps=0.1 if variant == "pyg" else 0.0)
    base = rng.normal(size=(N, F_in))
    gs = GraphStar.from_bounds(base - 0.01, base + 0.01)
    out = gine_graph_star(gs, E, edge_index, L, variant=variant)
    Yc = gine_evaluate(gs.center(), E, edge_index, L, variant=variant)
    assert out.contains(Yc)
    assert out.F == F_out and out.N == N


@pytest.mark.parametrize("variant", ["hugine", "pyg"])
def test_gine_edge_perturbation_combines_predicate_spaces(variant):
    """E as a GraphStar -> output predicate space is node preds + edge preds."""
    rng = np.random.default_rng(5)
    N, F_in, hidden, F_out, E_in = 6, 3, 5, 3, 2
    edge_index = _random_graph(rng, N, 8)
    m = edge_index.shape[1]
    layer = _make_layer(rng, F_in, hidden, F_out, E_in,
                        eps=0.2 if variant == "pyg" else 0.0)
    Xb = rng.normal(size=(N, F_in))
    Eb = rng.normal(size=(m, E_in))
    node_gs = GraphStar.from_bounds(Xb - 0.01, Xb + 0.01)
    edge_gs = GraphStar.from_bounds(Eb - 0.02, Eb + 0.02)   # uncertain edges
    out = gine_graph_star(node_gs, edge_gs, edge_index, layer, variant=variant)
    assert out.N == N and out.F == F_out
    # node + edge predicates both feed the output (before any ReLU growth)
    assert out.nVar >= node_gs.nVar + edge_gs.nVar


@pytest.mark.parametrize("variant", ["hugine", "pyg"])
def test_gine_edge_perturbation_is_sound(variant):
    rng = np.random.default_rng(6)
    N, F_in, hidden, F_out, E_in = 5, 3, 4, 3, 2
    edge_index = _random_graph(rng, N, 7)
    m = edge_index.shape[1]
    layer = _make_layer(rng, F_in, hidden, F_out, E_in,
                        eps=0.1 if variant == "pyg" else 0.0)
    Xb = rng.normal(size=(N, F_in))
    Eb = rng.normal(size=(m, E_in))
    en, ee = 0.01, 0.02
    node_gs = GraphStar.from_bounds(Xb - en, Xb + en)
    edge_gs = GraphStar.from_bounds(Eb - ee, Eb + ee)
    out = gine_graph_star(node_gs, edge_gs, edge_index, layer, variant=variant)
    olb, oub = out.get_ranges()
    for _ in range(150):
        Xi = Xb + rng.uniform(-en, en, size=Xb.shape)
        Ei = Eb + rng.uniform(-ee, ee, size=Eb.shape)   # edges perturbed too
        Yi = gine_evaluate(Xi, Ei, edge_index, layer, variant=variant)
        assert np.all(Yi >= olb - 1e-6) and np.all(Yi <= oub + 1e-6)


@pytest.mark.parametrize("variant", ["hugine", "pyg"])
def test_gine_stack_reach_contains_sampled_forward(variant):
    rng = np.random.default_rng(4)
    N, F, hidden, E_in = 5, 3, 4, 2
    edge_index = _random_graph(rng, N, 8)
    m = edge_index.shape[1]
    E = rng.normal(size=(m, E_in))
    layers = [
        _make_layer(rng, F, hidden, F, E_in, eps=0.0),
        _make_layer(rng, F, hidden, F, E_in, eps=0.0),
    ]
    base = rng.normal(size=(N, F))
    eps = 0.01
    gs = GraphStar.from_bounds(base - eps, base + eps)
    out = gine_stack_reach(gs, E, edge_index, layers, variant=variant)[0]
    olb, oub = out.get_ranges()
    for _ in range(80):
        Xi = base + rng.uniform(-eps, eps, size=base.shape)
        Yi = gine_stack_evaluate(Xi, E, edge_index, layers, variant=variant)
        assert np.all(Yi >= olb - 1e-6) and np.all(Yi <= oub + 1e-6)


# ===================================================================== Loader


@NEEDS_IEEE24_GINE_PF
def test_load_gine_pretrain_checkpoint_shapes_and_parity():
    model = load_gnn_mat(IEEE24_GINE_PF)
    assert model.model_type == "gine_pretrain"
    assert model.gine_variant == "hugine"
    assert model.has_relu is False
    assert model.edge_index.shape[0] == 2
    assert model.E is not None and model.E.shape[0] == model.edge_index.shape[1]
    assert all(isinstance(l, GINELayerSpec) for l in model.layers)
    assert model.layers[0].eps == 0.0          # hugine has no eps
    worst = 0.0
    for i in range(min(5, model.num_test_instances)):
        Y = gine_stack_evaluate(model.X_test[i], model.E, model.edge_index,
                                model.layers, variant="hugine",
                                edge_weights=model.edge_weights)
        worst = max(worst, float(np.max(np.abs(Y - model.python_predictions[i]))))
    assert worst < 1e-5
