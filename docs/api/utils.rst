Utilities
=========

Utility functions for LP solving, model loading, falsification, and
preprocessing.

LP Solver
---------

.. autofunction:: n2v.utils.solve_lp

Model Loading
-------------

.. autofunction:: n2v.utils.load_onnx

.. autofunction:: n2v.utils.load_pytorch

VNNLIB Parsing
--------------

.. autofunction:: n2v.utils.load_vnnlib

Falsification
-------------

.. autofunction:: n2v.utils.falsify

Preprocessing
-------------

.. autofunction:: n2v.utils.fuse_batchnorm

Graph Neural Networks
---------------------

.. autofunction:: n2v.utils.load_gnn_mat

.. automodule:: n2v.utils.gnn_loader
   :no-members:

.. autofunction:: n2v.utils.convert_pyg

.. autofunction:: n2v.utils.verify_node_bounds

.. autofunction:: n2v.utils.verify_graph_output

.. autofunction:: n2v.utils.falsify_node_bounds

.. autofunction:: n2v.utils.khop_subgraph_matrix

.. autofunction:: n2v.utils.khop_subgraph_edges
