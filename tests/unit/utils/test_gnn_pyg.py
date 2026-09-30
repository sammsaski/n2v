"""GraphNeuralNetwork.from_pyg: parity and reach soundness against real PyG layers.

The reference forward pass is PyTorch Geometric itself (float64), so these
tests check n2v's GNN semantics independently of n2v's own evaluate functions.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import torch.nn as tnn  # noqa: E402
from torch_geometric.nn import GCNConv, SAGEConv, GINEConv  # noqa: E402

from n2v.nn import GraphNeuralNetwork  # noqa: E402
from n2v.sets import GraphStar  # noqa: E402


pytestmark = pytest.mark.unit

N, F, H = 6, 3, 4
# directed edges (src -> dst); node 5 has no incoming edges (isolated for aggregation)
EDGES = np.array([[0, 1, 1, 2, 3, 2, 4, 0, 3],
                  [1, 0, 2, 1, 2, 3, 3, 4, 4]])
EDGE_DIM = 2


class PygNet(tnn.Module):
    """Conv stack with ReLU between layers (optionally after the last)."""

    def __init__(self, convs, relu_output=False, edge_arg=None):
        super().__init__()
        self.convs = tnn.ModuleList(convs)
        self.relu_output = relu_output
        self.edge_arg = edge_arg

    def forward(self, x, edge_index, edge=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge) if edge is not None else conv(x, edge_index)
            if i < len(self.convs) - 1 or self.relu_output:
                x = torch.relu(x)
        return x


def _mlp(fin, fout):
    return tnn.Sequential(tnn.Linear(fin, H), tnn.ReLU(), tnn.Linear(H, fout))


def _build(case):
    torch.manual_seed(0)
    edge = None
    if case == "gcn":
        convs = [GCNConv(F, F), GCNConv(F, 2)]
    elif case == "gcn_no_self_loops":
        convs = [GCNConv(F, F, add_self_loops=False), GCNConv(F, 2, add_self_loops=False)]
    elif case == "gcn_improved":
        convs = [GCNConv(F, F, improved=True), GCNConv(F, 2, improved=True)]
    elif case == "gcn_unnormalized":
        convs = [GCNConv(F, F, normalize=False), GCNConv(F, 2, normalize=False)]
    elif case == "gcn_edge_weight":
        convs = [GCNConv(F, F), GCNConv(F, 2)]
        edge = torch.rand(EDGES.shape[1], dtype=torch.float64) + 0.5
    elif case == "sage_mean":
        convs = [SAGEConv(F, F), SAGEConv(F, 2)]
    elif case == "sage_add":
        convs = [SAGEConv(F, F, aggr="add"), SAGEConv(F, 2, aggr="add")]
    elif case == "sage_no_root":
        convs = [SAGEConv(F, F, root_weight=False), SAGEConv(F, 2, root_weight=False, bias=False)]
    elif case == "gine_edge_dim":
        convs = [GINEConv(_mlp(F, F), edge_dim=EDGE_DIM, train_eps=True),
                 GINEConv(_mlp(F, 2), edge_dim=EDGE_DIM, train_eps=True)]
        with torch.no_grad():
            convs[0].eps.fill_(0.3)
        edge = torch.randn(EDGES.shape[1], EDGE_DIM, dtype=torch.float64)
    elif case == "gine_no_edge_dim":
        convs = [GINEConv(_mlp(F, F)), GINEConv(_mlp(F, F))]
        edge = torch.randn(EDGES.shape[1], F, dtype=torch.float64)
    else:
        raise ValueError(case)
    return PygNet(convs).double().eval(), edge


def _n2v_net(case, model, edge, relu_output=False):
    kwargs = {}
    if case.startswith("gine"):
        kwargs["edge_attr"] = edge
    elif edge is not None:
        kwargs["edge_weight"] = edge
    return GraphNeuralNetwork.from_pyg(model, torch.as_tensor(EDGES), N,
                                       relu_output=relu_output, **kwargs)


def _pyg_forward(model, X, edge):
    with torch.no_grad():
        return model(torch.as_tensor(X), torch.as_tensor(EDGES), edge).numpy()


CASES = ["gcn", "gcn_no_self_loops", "gcn_improved", "gcn_unnormalized", "gcn_edge_weight",
         "sage_mean", "sage_add", "sage_no_root", "gine_edge_dim", "gine_no_edge_dim"]


@pytest.mark.parametrize("case", CASES)
def test_forward_parity_with_pyg(case):
    model, edge = _build(case)
    net = _n2v_net(case, model, edge)
    rng = np.random.default_rng(1)
    for _ in range(5):
        X = rng.normal(size=(N, F))
        np.testing.assert_allclose(net.evaluate(X), _pyg_forward(model, X, edge), atol=1e-10)


def test_relu_output_flag_matches_model():
    model, edge = _build("sage_mean")
    model.relu_output = True
    net = _n2v_net("sage_mean", model, edge, relu_output=True)
    X = np.random.default_rng(2).normal(size=(N, F))
    np.testing.assert_allclose(net.evaluate(X), _pyg_forward(model, X, edge), atol=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_reach_contains_pyg_outputs(case):
    model, edge = _build(case)
    net = _n2v_net(case, model, edge)
    rng = np.random.default_rng(3)
    base, eps = rng.normal(size=(N, F)), 0.1
    out = net.reach(GraphStar.from_bounds(base - eps, base + eps))
    lb, ub = out[0].get_ranges()
    samples = [base + rng.uniform(-eps, eps, size=base.shape) for _ in range(150)]
    samples += [base + eps * rng.choice([-1.0, 1.0], size=base.shape) for _ in range(150)]
    for X in samples:
        Y = _pyg_forward(model, X, edge)
        assert np.all(Y >= lb - 1e-7) and np.all(Y <= ub + 1e-7)


@pytest.mark.parametrize("case", ["gcn", "sage_mean"])
def test_exact_reach_contains_pyg_outputs(case):
    model, edge = _build(case)
    net = _n2v_net(case, model, edge)
    rng = np.random.default_rng(4)
    base, eps = rng.normal(size=(N, F)), 0.05
    boxes = [s.get_ranges() for s in net.reach(GraphStar.from_bounds(base - eps, base + eps),
                                               method="exact")]
    for _ in range(200):
        Y = _pyg_forward(model, base + eps * rng.choice([-1.0, 1.0], size=base.shape), edge)
        assert any(np.all(Y >= lb - 1e-7) and np.all(Y <= ub + 1e-7) for lb, ub in boxes)


def test_accepts_single_conv_and_list():
    torch.manual_seed(0)
    conv = GCNConv(F, 2).double()
    X = np.random.default_rng(5).normal(size=(N, F))
    ref = conv(torch.as_tensor(X), torch.as_tensor(EDGES)).detach().numpy()
    for model in (conv, [conv]):
        net = GraphNeuralNetwork.from_pyg(model, EDGES, N)
        np.testing.assert_allclose(net.evaluate(X), ref, atol=1e-10)


def test_rejects_unsupported_configurations():
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg([SAGEConv(F, 2, project=True)], EDGES, N)
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg([SAGEConv(F, 2, aggr="max")], EDGES, N)
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg([GCNConv(F, F), SAGEConv(F, 2)], EDGES, N)
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg([GCNConv(F, F), GCNConv(F, 2, improved=True)], EDGES, N)
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg([GINEConv(_mlp(F, 2), edge_dim=EDGE_DIM)], EDGES, N)
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg(
            [GINEConv(tnn.Sequential(tnn.Linear(F, 2)), edge_dim=EDGE_DIM)], EDGES, N,
            edge_attr=np.zeros((EDGES.shape[1], EDGE_DIM)))
    with pytest.raises(ValueError):
        GraphNeuralNetwork.from_pyg(tnn.Linear(F, 2), EDGES, N)
