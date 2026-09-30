"""Unit tests for the GNN verify-spec API (gnn_verify)."""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.utils import (
    verify_graph_output,
    verify_node_bounds,
    box_violation_halfspaces,
    graph_output_index,
)


pytestmark = pytest.mark.unit


def _box_graph_star(lb, ub):
    """A GraphStar that is exactly the box [lb, ub] over (N, F)."""
    return GraphStar.from_bounds(np.asarray(lb, float), np.asarray(ub, float))


def test_graph_output_index_row_major():
    assert graph_output_index(0, 0, 4) == 0
    assert graph_output_index(0, 3, 4) == 3
    assert graph_output_index(2, 1, 4) == 9


def test_box_violation_halfspaces_count_and_dim():
    hs = box_violation_halfspaces(N=3, F=2, lb=np.zeros((3, 2)), ub=np.ones((3, 2)))
    assert len(hs) == 3 * 2 * 2                  # two halfspaces per output entry
    assert all(h.G.shape == (1, 6) for h in hs)  # flattened dim = N*F


def test_verify_node_bounds_holds_for_containing_box():
    gs = _box_graph_star(np.zeros((3, 2)), np.ones((3, 2)))
    # box widened by an absolute margin must contain the set -> verified
    res = verify_node_bounds([gs], np.full((3, 2), -0.5), np.full((3, 2), 1.5))
    assert res == "verified"


def test_verify_node_bounds_unknown_when_box_cuts_set():
    gs = _box_graph_star(np.zeros((3, 2)), np.ones((3, 2)))
    # tightened box slices through the set -> over-approx cannot certify
    res = verify_node_bounds([gs], np.full((3, 2), 0.25), np.full((3, 2), 0.75))
    assert res == "unknown"


def test_verify_node_bounds_target_subset():
    gs = _box_graph_star(np.zeros((4, 2)), np.ones((4, 2)))
    # only constrain node 0; bound holds there even though others are unconstrained
    res = verify_node_bounds([gs], np.full((4, 2), -0.5), np.full((4, 2), 1.5),
                             target_nodes=[0])
    assert res == "verified"


def test_verify_graph_output_rejects_non_graphstar():
    with pytest.raises(TypeError):
        verify_graph_output([object()], box_violation_halfspaces(1, 1, [[0.0]], [[1.0]]))
