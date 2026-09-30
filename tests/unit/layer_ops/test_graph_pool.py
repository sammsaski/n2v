"""Unit tests for graph-level sum/mean pooling on GraphStar."""

import numpy as np
import pytest

from n2v.sets import GraphStar, Star
from n2v.nn.layer_ops.graph_pool_reach import (
    graph_star_to_star,
    mean_pool_evaluate,
    mean_pool_graph_star,
    sum_pool_evaluate,
    sum_pool_graph_star,
)


pytestmark = pytest.mark.unit


# ===================================================================== Forward eval


def test_sum_pool_evaluate_known_value():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(sum_pool_evaluate(X), np.array([[9.0, 12.0]]))


def test_mean_pool_evaluate_known_value():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(mean_pool_evaluate(X), np.array([[3.0, 4.0]]))


def test_pool_evaluate_rejects_1d():
    with pytest.raises(ValueError):
        sum_pool_evaluate(np.array([1.0, 2.0]))


# ============================================================ Reachability shape


def test_sum_pool_collapses_node_axis():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    gs = GraphStar.from_bounds(X - 0.1, X + 0.1)
    out = sum_pool_graph_star(gs)
    assert out.N == 1
    assert out.F == 3
    assert out.V.shape == (1, 3, gs.nVar + 1)


def test_mean_pool_collapses_node_axis():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 6))
    gs = GraphStar.from_bounds(X - 0.05, X + 0.05)
    out = mean_pool_graph_star(gs)
    assert out.N == 1
    assert out.F == 6


# ================================================== Center matches forward pass


def test_sum_pool_center_matches_evaluate():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 3))
    gs = GraphStar.from_bounds(X - 0.05, X + 0.05)
    out = sum_pool_graph_star(gs)
    np.testing.assert_allclose(out.center(), sum_pool_evaluate(X))


def test_mean_pool_center_matches_evaluate():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 2))
    gs = GraphStar.from_bounds(X - 0.05, X + 0.05)
    out = mean_pool_graph_star(gs)
    np.testing.assert_allclose(out.center(), mean_pool_evaluate(X))


# ==================================== Pooling preserves predicate constraints


def test_pool_preserves_predicate_count(simple_graph_star):
    out_sum = sum_pool_graph_star(simple_graph_star)
    out_mean = mean_pool_graph_star(simple_graph_star)
    assert out_sum.nVar == simple_graph_star.nVar
    assert out_mean.nVar == simple_graph_star.nVar
    np.testing.assert_array_equal(out_sum.C, simple_graph_star.C)
    np.testing.assert_array_equal(out_sum.d, simple_graph_star.d)


# ====================================================== Soundness vs sampling


def test_sum_pool_is_exact_under_sampling():
    """Sum pool on a box-derived GraphStar gives exact bounds: lb_pool = sum(lb)."""
    rng = np.random.default_rng(4)
    N, F = 5, 3
    eps = 0.1
    X = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(X - eps, X + eps)

    out = sum_pool_graph_star(gs)
    lb, ub = out.get_ranges()
    expected_lb = (X - eps).sum(axis=0, keepdims=True)
    expected_ub = (X + eps).sum(axis=0, keepdims=True)
    np.testing.assert_allclose(lb, expected_lb, atol=1e-9)
    np.testing.assert_allclose(ub, expected_ub, atol=1e-9)


def test_mean_pool_is_exact_under_sampling():
    """Mean pool: bounds equal mean of element-wise input bounds."""
    rng = np.random.default_rng(5)
    N, F = 6, 4
    eps = 0.05
    X = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(X - eps, X + eps)

    out = mean_pool_graph_star(gs)
    lb, ub = out.get_ranges()
    expected_lb = (X - eps).mean(axis=0, keepdims=True)
    expected_ub = (X + eps).mean(axis=0, keepdims=True)
    np.testing.assert_allclose(lb, expected_lb, atol=1e-9)
    np.testing.assert_allclose(ub, expected_ub, atol=1e-9)


def test_sum_pool_contains_all_samples():
    rng = np.random.default_rng(6)
    N, F = 4, 3
    eps = 0.1
    X = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(X - eps, X + eps)
    out = sum_pool_graph_star(gs)
    lb, ub = out.get_ranges()

    for _ in range(64):
        delta = rng.uniform(-eps, eps, size=X.shape)
        Y = sum_pool_evaluate(X + delta)
        assert np.all(Y >= lb - 1e-9)
        assert np.all(Y <= ub + 1e-9)


# ============================================================= Star bridge


def test_graph_star_to_star_dim_matches_F():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3, 5))
    pooled = sum_pool_graph_star(GraphStar.from_bounds(X - 0.1, X + 0.1))
    star = graph_star_to_star(pooled)
    assert isinstance(star, Star)
    assert star.dim == 5


def test_graph_star_to_star_rejects_unpooled(simple_graph_star):
    with pytest.raises(ValueError):
        graph_star_to_star(simple_graph_star)
