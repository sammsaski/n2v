"""
Verify a PyTorch Geometric GNN directly with GraphNeuralNetwork.from_pyg.

Builds a small (untrained) two-layer GCN on a ring graph, converts it, and
compares approx-star and exact reach under an L_inf perturbation of the node
features, then checks node-level local robustness with verify-or-falsify.

Requires torch_geometric (pip install n2v[gnn]).

Usage:
    python examples/GNN/verify_pyg.py
"""

import numpy as np
import torch
from torch_geometric.nn import GCNConv

from n2v.nn import GraphNeuralNetwork
from n2v.sets import GraphStar


class GCN(torch.nn.Module):
    def __init__(self, f_in, hidden, f_out):
        super().__init__()
        self.conv1 = GCNConv(f_in, hidden)
        self.conv2 = GCNConv(hidden, f_out)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))   # ReLU between layers only
        return self.conv2(x, edge_index)


def main() -> None:
    torch.manual_seed(0)
    N, F = 8, 3
    ring = [(i, (i + 1) % N) for i in range(N)]
    edge_index = torch.tensor(ring + [(j, i) for i, j in ring]).T   # undirected
    model = GCN(F, 8, 2).double().eval()

    # relu_hidden/relu_output describe the activations in GCN.forward
    net = GraphNeuralNetwork.from_pyg(model, edge_index, num_nodes=N,
                                      relu_hidden=True, relu_output=False)
    X = np.random.default_rng(0).normal(size=(N, F))
    with torch.no_grad():
        Y = model(torch.as_tensor(X), edge_index).numpy()
    print(f"{net}; forward parity vs PyG: {np.max(np.abs(net.evaluate(X) - Y)):.1e}")

    eps = 0.05
    gs = GraphStar.from_bounds(X - eps, X + eps)
    for method in ("approx", "exact"):
        sets = net.reach(gs, method=method)
        ranges = [s.get_ranges() for s in sets]
        lb = np.min([r[0] for r in ranges], axis=0)
        ub = np.max([r[1] for r in ranges], axis=0)
        print(f"{method:6s} reach: {len(sets):3d} set(s), max output width {np.max(ub - lb):.3f}")

    print(f"\nnode outputs within +/- delta of f(X) under eps = {eps}:")
    for delta in (0.15, 0.10):
        res = net.verify(gs, Y - delta, Y + delta)
        msg = res.status
        if res.status == "falsified":
            msg += f" (counterexample violates the spec by {res.counterexample.margin:.3f})"
        print(f"  delta = {delta:.2f}: {msg}")


if __name__ == "__main__":
    main()
