Basic Verification
==================

This example walks through ``examples/simple_verification.py``, demonstrating
the core verification workflow.

Example 1: Basic Reachability Analysis
--------------------------------------

Define a simple feedforward network and compute the reachable output set:

.. code-block:: python

   import torch
   import torch.nn as nn
   import numpy as np
   import n2v as nnv
   from n2v.sets import Box

   # Define a simple neural network
   class SimpleNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(2, 4)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(4, 2)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x

   model = SimpleNet()
   model.eval()

   # Define input region: perturbation around [0.5, 0.5]
   center = np.array([0.5, 0.5])
   epsilon = 0.1
   lb = center - epsilon
   ub = center + epsilon

   # Create input set and verify
   input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))
   nn_verifier = nnv.NeuralNetwork(model, input_size=(2,))
   output_sets = nn_verifier.reach(input_box, method='approx')

   # Extract output bounds
   for output_set in output_sets:
       if isinstance(output_set, Box):
           print(f"Dim 1: [{output_set.lb[0, 0]:.4f}, {output_set.ub[0, 0]:.4f}]")
           print(f"Dim 2: [{output_set.lb[1, 0]:.4f}, {output_set.ub[1, 0]:.4f}]")

Example 2: Safety Property Verification
----------------------------------------

Check whether a network's output satisfies a safety property:

.. code-block:: python

   # Larger input region
   input_box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

   nn_verifier = nnv.NeuralNetwork(model)
   output_sets = nn_verifier.reach(input_box, method='approx')

   # Check: is output dimension 0 always <= 2.0?
   safety_bound = 2.0
   is_safe = True
   for output_set in output_sets:
       if hasattr(output_set, 'get_box'):
           box = output_set.get_box()
       else:
           box = output_set
       if box.ub[0, 0] > safety_bound:
           is_safe = False

   print(f"Property y[0] <= {safety_bound}: {'SAFE' if is_safe else 'UNKNOWN'}")

Example 3: Set Operations
--------------------------

Demonstrate set operations independent of neural network verification:

.. code-block:: python

   from n2v.sets import Box

   # Create two boxes
   box1 = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
   box2 = Box(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

   # Convert to zonotopes for richer operations
   zono1 = box1.to_zono()
   zono2 = box2.to_zono()

   # Minkowski sum
   zono_sum = zono1.minkowski_sum(zono2)
   box_sum = zono_sum.get_box()
   print(f"Minkowski sum bounds: [{box_sum.lb.flatten()}, {box_sum.ub.flatten()}]")

   # Convex hull
   zono_hull = zono1.convex_hull(zono2)
   box_hull = zono_hull.get_box()
   print(f"Convex hull bounds: [{box_hull.lb.flatten()}, {box_hull.ub.flatten()}]")

   # Affine transformation: W*x + b
   W = np.array([[2.0, 0.0], [0.0, 0.5]])
   b = np.array([[1.0], [0.5]])
   zono_transformed = zono1.affine_map(W, b)

Running the Example
-------------------

.. code-block:: bash

   python examples/simple_verification.py
