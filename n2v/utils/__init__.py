"""
Utility functions for n2v.

This module provides helper functions for model loading, LP solving,
conversions, verification, falsification, and other utilities.
"""

from n2v.utils.lp_solver_enum import Backend, LPSolver, resolve as resolve_lp_solver
from n2v.utils.lpsolver import solve_lp, solve_lp_batch
from n2v.utils.model_loader import load_onnx, load_pytorch
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.falsify import falsify
from n2v.utils.model_preprocessing import fuse_batchnorm, strip_final_softmax

# NOTE: ``verify_specification`` (and ``spec_summary``) are intentionally
# NOT re-exported here. ``verify_specification.py`` imports from
# ``n2v.sets`` at module level, and ``n2v.sets.star`` imports from
# ``n2v.utils.lpsolver``. Re-exporting through this ``__init__`` triggers
# a circular import during ``n2v.sets`` initialisation. Callers should
# use the explicit ``from n2v.utils.verify_specification import ...`` path
# (which is what every existing caller in the tree does anyway).

from n2v.utils.gnn_loader import (
    load_gnn_mat, GNNModel, GCNModel,
    GCNLayerSpec, SAGELayerSpec, GINELayerSpec, NormStats,
)
from n2v.utils.gnn_verify import (
    verify_graph_output, verify_node_bounds,
    box_violation_halfspaces, graph_output_index,
)
from n2v.utils.subgraph import khop_subgraph_matrix, khop_subgraph_edges
from n2v.utils.gnn_pyg import convert_pyg
from n2v.utils.gnn_falsify import falsify_node_bounds, FalsifyResult

__all__ = [
    "solve_lp",
    "solve_lp_batch",
    "load_onnx",
    "load_pytorch",
    "load_vnnlib",
    "falsify",
    "fuse_batchnorm",
    "strip_final_softmax",
    "Backend",
    "LPSolver",
    "resolve_lp_solver",
    "load_gnn_mat",
    "GNNModel",
    "GCNModel",
    "GCNLayerSpec",
    "SAGELayerSpec",
    "GINELayerSpec",
    "NormStats",
    "verify_graph_output",
    "verify_node_bounds",
    "box_violation_halfspaces",
    "graph_output_index",
    "khop_subgraph_matrix",
    "khop_subgraph_edges",
    "convert_pyg",
    "falsify_node_bounds",
    "FalsifyResult",
]
