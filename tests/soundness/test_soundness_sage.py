"""
Soundness tests for the SAGEConv(+ReLU) reachability pipeline.

Asserts the over-approximation property: every deterministic forward output on
a perturbed input lies inside the reach output's box.  SAGEConv is linear in X,
so a single layer is exact; multi-layer stacks introduce ReLU relaxation.

Synthetic small graphs plus the bundled IEEE-24 power-flow checkpoint
(tests/fixtures/gnn).
"""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn.layer_ops.sage_reach import (
    sage_evaluate,
    sage_graph_star,
    sage_stack_evaluate,
    sage_stack_reach,
)
from n2v.utils import SAGELayerSpec, load_gnn_mat
from tests.fixtures.gnn import checkpoint


IEEE24_SAGE_PF = checkpoint("sage")
NEEDS_IEEE24_SAGE_PF = pytest.mark.skipif(
    not IEEE24_SAGE_PF.exists(), reason=f"requires {IEEE24_SAGE_PF}",
)


def _samples(rng, X, eps, k):
    for _ in range(k):
        yield X + rng.uniform(-eps, eps, size=X.shape)


class TestSingleSAGELayerSoundness:
    def test_random_small_graph_is_exact(self):
        rng = np.random.default_rng(0)
        N, F_in, F_out = 6, 4, 3
        A = (rng.uniform(size=(N, N)) > 0.5).astype(float)
        Wn, We = rng.normal(size=(F_in, F_out)), rng.normal(size=(F_in, F_out))
        b = rng.normal(size=(F_out,))
        base = rng.normal(size=(N, F_in))
        eps = 0.05
        gs = GraphStar.from_bounds(base - eps, base + eps, adjacency=A)
        out = sage_graph_star(gs, Wn, We, b, A)
        lb, ub = out.get_ranges()
        for X in _samples(rng, base, eps, 200):
            Y = sage_evaluate(X, A, Wn, We, b)
            assert np.all(Y >= lb - 1e-9) and np.all(Y <= ub + 1e-9)


class TestSAGEStackSoundness:
    @pytest.mark.parametrize("eps", [1e-3, 1e-2, 5e-2])
    def test_two_layer_relu_stack(self, eps):
        rng = np.random.default_rng(1)
        N, F = 5, 4
        A = (rng.uniform(size=(N, N)) > 0.4).astype(float)
        layers = [
            SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)), rng.normal(size=(F,))),
            SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)), rng.normal(size=(F,))),
        ]
        base = rng.normal(size=(N, F))
        gs = GraphStar.from_bounds(base - eps, base + eps, adjacency=A)
        out = sage_stack_reach(gs, layers, has_relu=True, A=A)[0]
        lb, ub = out.get_ranges()
        for X in _samples(rng, base, eps, 150):
            Y = sage_stack_evaluate(X, A, layers, has_relu=True)
            assert np.all(Y >= lb - 1e-7) and np.all(Y <= ub + 1e-7)


@NEEDS_IEEE24_SAGE_PF
class TestSAGEIEEE24Soundness:
    @pytest.mark.parametrize("eps", [1e-3, 1e-2])
    def test_checkpoint_reach_contains_samples(self, eps):
        rng = np.random.default_rng(2)
        model = load_gnn_mat(IEEE24_SAGE_PF)
        X = model.X_test[0]
        gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.A_adj)
        out = sage_stack_reach(gs, model.layers, has_relu=model.has_relu, A=model.A_adj)[0]
        lb, ub = out.get_ranges()
        for Xi in _samples(rng, X, eps, 60):
            Y = sage_stack_evaluate(Xi, model.A_adj, model.layers, model.has_relu)
            assert np.all(Y >= lb - 1e-6) and np.all(Y <= ub + 1e-6)
