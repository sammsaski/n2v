"""Unit tests for GraphStar set representation."""

import numpy as np
import pytest

from n2v.sets import GraphStar, Star


pytestmark = pytest.mark.unit


# ============================================================ Construction & shape


def test_construction_shapes(simple_graph_star):
    gs = simple_graph_star
    assert gs.N == 4
    assert gs.F == 3
    assert gs.dim == 12
    assert gs.V.shape == (4, 3, gs.nVar + 1)
    assert gs.C.shape == (2 * gs.nVar, gs.nVar)
    assert gs.d.shape == (2 * gs.nVar, 1)
    assert gs.predicate_lb.shape == (gs.nVar, 1)
    assert gs.predicate_ub.shape == (gs.nVar, 1)


def test_from_bounds_collapses_zero_width():
    lb = np.array([[0.0, 1.0], [2.0, 3.0]])
    ub = np.array([[0.0, 1.5], [2.0, 3.0]])
    gs = GraphStar.from_bounds(lb, ub)
    # Only one entry has lb < ub, so nVar == 1.
    assert gs.nVar == 1
    np.testing.assert_allclose(gs.center(), 0.5 * (lb + ub))


def test_from_bounds_rejects_inverted():
    lb = np.array([[1.0, 0.0]])
    ub = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError):
        GraphStar.from_bounds(lb, ub)


def test_invalid_V_dim():
    with pytest.raises(ValueError):
        GraphStar(np.zeros((4, 5)))  # 2D, should fail


# ===================================================== Flatten / unflatten roundtrip


def test_flatten_then_unflatten_is_identity(simple_graph_star):
    gs = simple_graph_star
    star = gs.to_star()
    assert isinstance(star, Star)
    assert star.dim == gs.N * gs.F
    gs2 = GraphStar.from_star(star, gs.N, gs.F, adjacency=gs.adjacency)
    np.testing.assert_allclose(gs2.V, gs.V)
    np.testing.assert_allclose(gs2.C, gs.C)
    np.testing.assert_allclose(gs2.d, gs.d)


def test_flatten_preserves_membership(simple_graph_star):
    gs = simple_graph_star
    center = gs.center()
    assert gs.contains(center)
    star = gs.to_star()
    assert star.contains(center.reshape(-1, 1))


# ============================================================== Range queries


def test_estimate_ranges_matches_box_construction(simple_graph_star):
    gs = simple_graph_star
    lb_e, ub_e = gs.estimate_ranges()
    # GraphStar was built from a box; estimated ranges must equal that box exactly.
    center = gs.center()
    np.testing.assert_allclose(0.5 * (lb_e + ub_e), center, atol=1e-12)


def test_get_ranges_recovers_input_bounds():
    rng = np.random.default_rng(7)
    center = rng.uniform(-1.0, 1.0, size=(3, 2))
    eps = 0.05
    gs = GraphStar.from_bounds(center - eps, center + eps)
    lb, ub = gs.get_ranges()
    np.testing.assert_allclose(lb, center - eps, atol=1e-9)
    np.testing.assert_allclose(ub, center + eps, atol=1e-9)


def test_zero_perturbation_collapses_to_point():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    gs = GraphStar.from_bounds(X, X)
    assert gs.nVar == 0
    lb, ub = gs.estimate_ranges()
    np.testing.assert_allclose(lb, X)
    np.testing.assert_allclose(ub, X)


# ============================================================== Affine transforms


def test_affine_map_right_matches_explicit_evaluation():
    rng = np.random.default_rng(1)
    center = rng.uniform(-1.0, 1.0, size=(4, 3))
    gs = GraphStar.from_bounds(center - 0.1, center + 0.1)
    W = rng.normal(size=(3, 5))
    b = rng.normal(size=(5,))

    out = gs.affine_map_right(W, b)

    # Check shape and that the center transforms correctly.
    assert out.V.shape == (4, 5, gs.nVar + 1)
    np.testing.assert_allclose(out.center(), gs.center() @ W + b)

    # Verify against a sampled predicate.
    alpha = np.clip(rng.normal(size=gs.nVar), -1.0, 1.0)
    expected = gs.evaluate(alpha) @ W + b
    actual = out.evaluate(alpha)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_affine_map_left_matches_explicit_evaluation():
    rng = np.random.default_rng(2)
    center = rng.uniform(-1.0, 1.0, size=(4, 3))
    gs = GraphStar.from_bounds(center - 0.1, center + 0.1)
    A = rng.normal(size=(4, 4))

    out = gs.affine_map_left(A)

    assert out.V.shape == (4, 3, gs.nVar + 1)
    np.testing.assert_allclose(out.center(), A @ gs.center())

    alpha = np.clip(rng.normal(size=gs.nVar), -1.0, 1.0)
    expected = A @ gs.evaluate(alpha)
    actual = out.evaluate(alpha)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_compose_left_then_right_models_gcn_layer():
    """A_norm @ X @ W + b should be the composition of left-map then right-map."""
    rng = np.random.default_rng(3)
    N, F, K = 5, 4, 6
    A = rng.normal(size=(N, N))
    W = rng.normal(size=(F, K))
    b = rng.normal(size=(K,))
    center = rng.uniform(-1.0, 1.0, size=(N, F))
    gs = GraphStar.from_bounds(center - 0.05, center + 0.05)

    out = gs.affine_map_left(A).affine_map_right(W, b)
    np.testing.assert_allclose(out.center(), A @ center @ W + b)

    # Random sample check.
    alpha = np.clip(rng.normal(size=gs.nVar), -1.0, 1.0)
    expected = A @ gs.evaluate(alpha) @ W + b
    np.testing.assert_allclose(out.evaluate(alpha), expected, atol=1e-9)


# =========================================================== Membership / sampling


def test_center_is_contained(simple_graph_star):
    assert simple_graph_star.contains(simple_graph_star.center())


def test_obvious_outside_point_rejected(simple_graph_star):
    far = simple_graph_star.center() + 1e3
    assert not simple_graph_star.contains(far)


def test_sample_returns_contained_points():
    rng = np.random.default_rng(11)
    center = rng.uniform(-1.0, 1.0, size=(3, 3))
    gs = GraphStar.from_bounds(center - 0.05, center + 0.05)
    samples = gs.sample(8)
    assert samples.shape[1:] == (3, 3)
    # Every sample lies inside the element-wise box, hence inside the GraphStar.
    lb, ub = gs.get_ranges()
    for k in range(samples.shape[0]):
        x = samples[k]
        assert np.all(x >= lb - 1e-9)
        assert np.all(x <= ub + 1e-9)


def test_evaluate_at_corner_matches_bound():
    center = np.array([[1.0]])
    eps = 0.5
    gs = GraphStar.from_bounds(center - eps, center + eps)
    # alpha = +1 should give the upper bound; alpha = -1 the lower bound.
    np.testing.assert_allclose(gs.evaluate(np.array([1.0])), center + eps)
    np.testing.assert_allclose(gs.evaluate(np.array([-1.0])), center - eps)


# ==================================================================== Edge cases


def test_repr_contains_shape(simple_graph_star):
    s = repr(simple_graph_star)
    assert "GraphStar" in s
    assert "N=4" in s
    assert "F=3" in s


def test_adjacency_validation():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        GraphStar.from_bounds(X, X + 0.1, adjacency=np.eye(4))


def test_adjacency_passes_through_with_state():
    X = np.zeros((3, 2))
    A = np.eye(3)
    gs = GraphStar.from_bounds(X, X + 0.1, adjacency=A)
    out = gs.affine_map_left(A)
    np.testing.assert_array_equal(out.adjacency, A)
