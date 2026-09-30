"""
Soundness tests for the GCN(+ReLU) reachability pipeline.

These tests assert the over-approximation property: every output produced
by the deterministic forward pass on a perturbed input must lie inside the
reach output's box.  Two scales are covered:

  * Synthetic small graphs with hand-controlled weights (fast).
  * The IEEE-24 power-flow checkpoint bundled in tests/fixtures/gnn.

NNV's MATLAB GNN soundness check uses the same sample-and-contain
methodology; we replicate it with a tighter sample budget where speed allows.
"""

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
from n2v.nn.layer_ops.graph_pool_reach import (
    mean_pool_evaluate,
    mean_pool_graph_star,
    sum_pool_evaluate,
    sum_pool_graph_star,
)
from n2v.utils import GCNLayerSpec, load_gnn_mat
from tests.fixtures.gnn import checkpoint


IEEE24_GCN_PF = checkpoint("gcn")
NEEDS_IEEE24_GCN_PF = pytest.mark.skipif(
    not IEEE24_GCN_PF.exists(),
    reason=f"requires {IEEE24_GCN_PF}",
)


def _sample_perturbations(rng, X, eps, k):
    """Yield k uniform samples drawn from the L_inf box around X."""
    for _ in range(k):
        yield X + rng.uniform(-eps, eps, size=X.shape)


# ============================================================ Single GCN layer


class TestSingleGCNLayerSoundness:
    """A single A_norm @ X @ W + b layer should bound every sampled output."""

    def test_random_small_graph(self):
        rng = np.random.default_rng(0)
        N, F_in, F_out = 4, 3, 5
        A = rng.normal(size=(N, N)) * 0.4
        X = rng.normal(size=(N, F_in))
        W = rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))
        eps = 0.1

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        out = gcn_graph_star(gs, W, b)
        lb, ub = out.get_ranges()

        for sample in _sample_perturbations(rng, X, eps, 200):
            Y = gcn_evaluate(sample, A, W, b)
            assert np.all(Y >= lb - 1e-9)
            assert np.all(Y <= ub + 1e-9)

    def test_disconnected_graph(self):
        """Identity adjacency: every node decoupled, bounds should be tight per node."""
        rng = np.random.default_rng(1)
        N, F_in, F_out = 3, 2, 4
        A = np.eye(N)
        X = rng.normal(size=(N, F_in))
        W = rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))
        eps = 0.05

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        out = gcn_graph_star(gs, W, b)
        lb, ub = out.get_ranges()

        # With identity adjacency, each output row is independent.  For each row,
        # the per-feature bound is sum_j |W[j, k]| * eps, centered at X[i] @ W + b.
        center = X @ W + b
        per_node_radius = np.abs(W).sum(axis=0) * eps
        np.testing.assert_allclose(lb, center - per_node_radius, atol=1e-9)
        np.testing.assert_allclose(ub, center + per_node_radius, atol=1e-9)


# ====================================================== GCN + ReLU composition


class TestReluOnGCNSoundness:
    """ReLU after a GCN layer must keep all sampled outputs in the reach box."""

    @pytest.mark.parametrize("eps", [0.01, 0.1, 0.5])
    def test_random_eps(self, eps):
        rng = np.random.default_rng(2)
        N, F_in, F_out = 4, 3, 6
        A = rng.normal(size=(N, N)) * 0.3
        X = rng.normal(size=(N, F_in))
        W = rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        post_gcn = gcn_graph_star(gs, W, b)
        post_relu = relu_graph_star(post_gcn, method="approx")
        assert len(post_relu) == 1
        lb, ub = post_relu[0].get_ranges()

        for sample in _sample_perturbations(rng, X, eps, 100):
            Y = np.maximum(gcn_evaluate(sample, A, W, b), 0.0)
            assert np.all(Y >= lb - 1e-9)
            assert np.all(Y <= ub + 1e-9)


# ============================================================ Pooling soundness


class TestPoolSoundness:
    """Sum and mean pool are linear; reach bounds are exact up to LP tolerance."""

    def test_sum_pool_after_gcn(self):
        rng = np.random.default_rng(3)
        N, F_in, F_out = 5, 3, 4
        A = rng.normal(size=(N, N)) * 0.3
        X = rng.normal(size=(N, F_in))
        W = rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))
        eps = 0.05

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        out = sum_pool_graph_star(gcn_graph_star(gs, W, b))
        lb, ub = out.get_ranges()

        for sample in _sample_perturbations(rng, X, eps, 100):
            Y = sum_pool_evaluate(gcn_evaluate(sample, A, W, b))
            assert np.all(Y >= lb - 1e-9)
            assert np.all(Y <= ub + 1e-9)

    def test_mean_pool_after_gcn(self):
        rng = np.random.default_rng(4)
        N, F_in, F_out = 6, 3, 4
        A = rng.normal(size=(N, N)) * 0.3
        X = rng.normal(size=(N, F_in))
        W = rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))
        eps = 0.05

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        out = mean_pool_graph_star(gcn_graph_star(gs, W, b))
        lb, ub = out.get_ranges()

        for sample in _sample_perturbations(rng, X, eps, 100):
            Y = mean_pool_evaluate(gcn_evaluate(sample, A, W, b))
            assert np.all(Y >= lb - 1e-9)
            assert np.all(Y <= ub + 1e-9)


# ================================================== Multi-layer GCN+ReLU stack


class TestStackSoundness:
    """A small synthetic 3-layer GCN+ReLU stack should be sound."""

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.1])
    def test_stack(self, eps):
        rng = np.random.default_rng(5)
        N = 5
        sizes = [4, 8, 8, 4]  # 4 -> 8 -> 8 -> 4
        A = rng.normal(size=(N, N)) * 0.3
        X = rng.normal(size=(N, sizes[0]))
        layers = [
            GCNLayerSpec(W=rng.normal(size=(sizes[i], sizes[i + 1])),
                         b=rng.normal(size=(sizes[i + 1],)))
            for i in range(3)
        ]

        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=A)
        outs = gcn_stack_reach(gs, layers, has_relu=True)
        assert len(outs) == 1
        lb, ub = outs[0].get_ranges()

        for sample in _sample_perturbations(rng, X, eps, 50):
            Y = gcn_stack_evaluate(sample, A, layers, has_relu=True)
            assert np.all(Y >= lb - 1e-9)
            assert np.all(Y <= ub + 1e-9)


# ==================================================== IEEE-24 power-flow checkpoint


@NEEDS_IEEE24_GCN_PF
class TestIEEE24Soundness:
    """Real-world soundness check on the trained IEEE-24 PF GCN."""

    @pytest.mark.parametrize("eps", [0.001, 0.01, 0.05])
    def test_first_test_instance(self, eps):
        model = load_gnn_mat(IEEE24_GCN_PF)
        X = model.X_test[0]
        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.A_norm)
        outs = gcn_stack_reach(gs, model.gcn_layers, has_relu=model.has_relu)
        assert len(outs) == 1
        lb, ub = outs[0].get_ranges()

        rng = np.random.default_rng(eps_to_seed(eps))
        for sample in _sample_perturbations(rng, X, eps, 32):
            Y = gcn_stack_evaluate(sample, model.A_norm, model.gcn_layers,
                                   has_relu=model.has_relu)
            # Tolerance accounts for float32 PyTorch reference + LP solver noise.
            assert np.all(Y >= lb - 1e-5)
            assert np.all(Y <= ub + 1e-5)


def eps_to_seed(eps):
    return int(round(eps * 1e6))
