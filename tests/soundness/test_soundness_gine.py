"""
Soundness tests for the GINE reachability pipeline (both variants).

Asserts the over-approximation property: every deterministic forward output on
a perturbed input lies inside the reach output's box.  Both 'hugine'
(gine_pretrain) and 'pyg' (gine_conv) message-passing variants are exercised.

Synthetic small graphs plus the bundled IEEE-24 power-flow checkpoint
(gine_pretrain, tests/fixtures/gnn).
"""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gine_reach import (
    gine_evaluate,
    gine_stack_evaluate,
    gine_stack_reach,
)
from n2v.utils import GINELayerSpec, load_gnn_mat
from tests.fixtures.gnn import checkpoint


IEEE24_GINE_PF = checkpoint("gine_pretrain")
NEEDS_IEEE24_GINE_PF = pytest.mark.skipif(
    not IEEE24_GINE_PF.exists(), reason=f"requires {IEEE24_GINE_PF}",
)


def _samples(rng, X, eps, k):
    for _ in range(k):
        yield X + rng.uniform(-eps, eps, size=X.shape)


def _make_layer(rng, F_in, hidden, F_out, E_in, eps=0.0):
    return GINELayerSpec(
        W1=rng.normal(size=(F_in, hidden)), b1=rng.normal(size=(hidden,)),
        W2=rng.normal(size=(hidden, F_out)), b2=rng.normal(size=(F_out,)),
        W_edge=rng.normal(size=(E_in, F_in)), b_edge=rng.normal(size=(F_in,)),
        eps=eps,
    )


def _graph_with_self_loops(rng, N, m):
    src = np.concatenate([rng.integers(0, N, size=m), np.arange(N)])
    dst = np.concatenate([rng.integers(0, N, size=m), np.arange(N)])
    return np.vstack([src, dst])


class TestGINESingleLayerSoundness:
    @pytest.mark.parametrize("variant", ["hugine", "pyg"])
    @pytest.mark.parametrize("eps", [1e-3, 1e-2])
    def test_single_layer(self, variant, eps):
        rng = np.random.default_rng(0)
        N, F_in, hidden, F_out, E_in = 6, 3, 6, 3, 2
        edge_index = _graph_with_self_loops(rng, N, 10)
        m = edge_index.shape[1]
        E = rng.normal(size=(m, E_in))
        layer = _make_layer(rng, F_in, hidden, F_out, E_in,
                            eps=0.2 if variant == "pyg" else 0.0)
        base = rng.normal(size=(N, F_in))
        gs = GraphStar.from_bounds(base - eps, base + eps)
        out = gine_stack_reach(gs, E, edge_index, [layer], variant=variant)[0]
        lb, ub = out.get_ranges()
        for X in _samples(rng, base, eps, 120):
            Y = gine_stack_evaluate(X, E, edge_index, [layer], variant=variant)
            assert np.all(Y >= lb - 1e-6) and np.all(Y <= ub + 1e-6)


class TestGINEStackSoundness:
    @pytest.mark.parametrize("variant", ["hugine", "pyg"])
    def test_three_layer_stack(self, variant):
        rng = np.random.default_rng(1)
        N, F, hidden, E_in = 5, 3, 5, 2
        edge_index = _graph_with_self_loops(rng, N, 8)
        m = edge_index.shape[1]
        E = rng.normal(size=(m, E_in))
        layers = [_make_layer(rng, F, hidden, F, E_in,
                              eps=0.1 if variant == "pyg" else 0.0) for _ in range(3)]
        base = rng.normal(size=(N, F))
        eps = 5e-3
        gs = GraphStar.from_bounds(base - eps, base + eps)
        out = gine_stack_reach(gs, E, edge_index, layers, variant=variant)[0]
        lb, ub = out.get_ranges()
        for X in _samples(rng, base, eps, 100):
            Y = gine_stack_evaluate(X, E, edge_index, layers, variant=variant)
            assert np.all(Y >= lb - 1e-6) and np.all(Y <= ub + 1e-6)


class TestGINEEdgePerturbationSoundness:
    """Uncertain node AND edge features: combined predicate space stays sound."""

    @pytest.mark.parametrize("variant", ["hugine", "pyg"])
    def test_two_layer_stack(self, variant):
        rng = np.random.default_rng(3)
        N, F, hidden, E_in = 5, 3, 5, 2
        edge_index = _graph_with_self_loops(rng, N, 7)
        m = edge_index.shape[1]
        layers = [_make_layer(rng, F, hidden, F, E_in,
                              eps=0.1 if variant == "pyg" else 0.0) for _ in range(2)]
        Xb = rng.normal(size=(N, F))
        Eb = rng.normal(size=(m, E_in))
        en, ee = 5e-3, 1e-2
        node_gs = GraphStar.from_bounds(Xb - en, Xb + en)
        edge_gs = GraphStar.from_bounds(Eb - ee, Eb + ee)
        out = gine_stack_reach(node_gs, edge_gs, edge_index, layers, variant=variant)[0]
        lb, ub = out.get_ranges()
        for _ in range(100):
            Xi = Xb + rng.uniform(-en, en, size=Xb.shape)
            Ei = Eb + rng.uniform(-ee, ee, size=Eb.shape)
            Y = gine_stack_evaluate(Xi, Ei, edge_index, layers, variant=variant)
            assert np.all(Y >= lb - 1e-6) and np.all(Y <= ub + 1e-6)


@NEEDS_IEEE24_GINE_PF
class TestGINEIEEE24Soundness:
    @pytest.mark.parametrize("eps", [1e-3, 1e-2])
    def test_checkpoint_reach_contains_samples(self, eps):
        rng = np.random.default_rng(2)
        model = load_gnn_mat(IEEE24_GINE_PF)
        v = model.gine_variant
        X = model.X_test[0]
        gs = GraphStar.from_bounds(X - eps, X + eps)
        out = gine_stack_reach(gs, model.E, model.edge_index, model.layers,
                               variant=v, edge_weights=model.edge_weights)[0]
        lb, ub = out.get_ranges()
        for Xi in _samples(rng, X, eps, 50):
            Y = gine_stack_evaluate(Xi, model.E, model.edge_index, model.layers,
                                    variant=v, edge_weights=model.edge_weights)
            assert np.all(Y >= lb - 1e-6) and np.all(Y <= ub + 1e-6)
