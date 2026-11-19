Layer Operations
================

Layer-level reachability operations. These are used internally by
:class:`~n2v.nn.NeuralNetwork` but can be called directly for advanced use
cases.

Dispatcher
----------

.. autofunction:: n2v.nn.layer_ops.dispatcher.reach_layer

Linear Layers
-------------

.. automodule:: n2v.nn.layer_ops.linear_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.conv2d_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.conv1d_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.batchnorm_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.flatten_reach
   :members:
   :undoc-members: False

Nonlinear Layers
----------------

.. automodule:: n2v.nn.layer_ops.relu_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.leakyrelu_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.sigmoid_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.tanh_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.sign_reach
   :members:
   :undoc-members: False

Pooling Layers
--------------

.. automodule:: n2v.nn.layer_ops.maxpool2d_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.avgpool2d_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.global_avgpool_reach
   :members:
   :undoc-members: False

Graph Layers
------------

Used by :class:`~n2v.nn.GraphNeuralNetwork`; operate on
:class:`~n2v.sets.GraphStar` sets.

.. automodule:: n2v.nn.layer_ops.gcn_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.sage_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.gine_reach
   :members:
   :undoc-members: False

.. automodule:: n2v.nn.layer_ops.graph_pool_reach
   :members:
   :undoc-members: False
