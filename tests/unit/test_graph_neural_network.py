"""Unit tests for the GraphNeuralNetwork wrapper (dispatch + reach parity)."""

import numpy as np
import pytest

from n2v.nn import GraphNeuralNetwork
from n2v.sets import GraphStar
from n2v.utils import load_gnn_mat, GCNLayerSpec, SAGELayerSpec, GINELayerSpec
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


CHECKPOINTS = {
    "gcn": checkpoint("gcn"),
    "sage": checkpoint("sage"),
    "gine_pretrain": checkpoint("gine_pretrain"),
}


def test_rejects_mixed_layer_types():
    with pytest.raises(ValueError):
        GraphNeuralNetwork(
            [GCNLayerSpec(np.eye(2), None), SAGELayerSpec(np.eye(2), np.eye(2), None)],
            adjacency=np.eye(3),
        )


def test_requires_graph_structure():
    with pytest.raises(ValueError):
        GraphNeuralNetwork([GCNLayerSpec(np.eye(2), None)])          # no adjacency
    with pytest.raises(ValueError):
        GraphNeuralNetwork([GINELayerSpec(np.eye(2), None, np.eye(2), None,
                                          np.eye(2), None, 0.0)])    # no edge_index/E


def test_synthetic_sage_wrapper_matches_layer_ops():
    rng = np.random.default_rng(0)
    N, F = 5, 3
    A = (rng.uniform(size=(N, N)) > 0.4).astype(float)
    layers = [SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)),
                            rng.normal(size=(F,))) for _ in range(2)]
    net = GraphNeuralNetwork(layers, adjacency=A, has_relu=True)
    base = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(base - 0.02, base + 0.02, adjacency=A)
    out = net.reach(gs)[0]
    olb, oub = out.get_ranges()
    for _ in range(60):
        Xi = base + rng.uniform(-0.02, 0.02, size=base.shape)
        assert np.all(net.evaluate(Xi) >= olb - 1e-7)
        assert np.all(net.evaluate(Xi) <= oub + 1e-7)


def test_reach_rejects_unknown_method():
    net = GraphNeuralNetwork([SAGELayerSpec(np.eye(2), np.eye(2), None)],
                             adjacency=np.eye(3))
    gs = GraphStar.from_bounds(np.zeros((3, 2)), np.ones((3, 2)), adjacency=np.eye(3))
    with pytest.raises(ValueError):
        net.reach(gs, method="abstract")


@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_wrapper_forward_parity_real_checkpoints(name):
    path = CHECKPOINTS[name]
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    worst = max(
        float(np.max(np.abs(net.evaluate(model.X_test[i]) - model.python_predictions[i])))
        for i in range(min(5, model.num_test_instances))
    )
    assert worst < 1e-5


@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_wrapper_reach_is_sound_real_checkpoints(name):
    path = CHECKPOINTS[name]
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    rng = np.random.default_rng(7)
    X = model.X_test[0]
    eps = 1e-3
    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.adjacency)
    out = net.reach(gs)[0]
    olb, oub = out.get_ranges()
    for _ in range(40):
        Xi = X + rng.uniform(-eps, eps, size=X.shape)
        Yi = net.evaluate(Xi)
        assert np.all(Yi >= olb - 1e-6) and np.all(Yi <= oub + 1e-6)
