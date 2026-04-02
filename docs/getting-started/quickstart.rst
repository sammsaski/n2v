Quick Start
===========

This guide walks you through your first neural network verification with n2v.

Your First Verification
-----------------------

Create a simple feedforward network and verify it using exact reachability analysis:

.. code-block:: python

   import torch.nn as nn
   import numpy as np
   import n2v
   from n2v.sets import Star

   # Define a PyTorch model
   model = nn.Sequential(
       nn.Linear(3, 10),
       nn.ReLU(),
       nn.Linear(10, 2)
   )
   model.eval()

   # Create input set: L-inf ball around a center point
   center = np.array([0.5, 0.5, 0.5])
   epsilon = 0.1
   input_star = Star.from_bounds(center - epsilon, center + epsilon)

   # Compute reachable output set
   net = n2v.NeuralNetwork(model)
   output_stars = net.reach(input_star, method='exact')

   # Extract output bounds
   for star in output_stars:
       lb, ub = star.get_ranges()
       print(f"Output bounds: [{lb.flatten()}, {ub.flatten()}]")

The ``method='exact'`` approach splits at ReLU boundaries, producing multiple
output Star sets whose union is the exact reachable set.

Approximate Reachability with Box
---------------------------------

For faster (but over-approximate) results, use Box sets with approximate
reachability:

.. code-block:: python

   from n2v.sets import Box

   # Create input Box
   lb = np.array([0.4, 0.4, 0.4])
   ub = np.array([0.6, 0.6, 0.6])
   input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))

   # Approximate reachability (no splitting, single output set)
   net = n2v.NeuralNetwork(model)
   output_sets = net.reach(input_box, method='approx')

   for s in output_sets:
       out_lb, out_ub = s.get_ranges()
       print(f"Output bounds: [{out_lb.flatten()}, {out_ub.flatten()}]")

Safety Property Checking
------------------------

Check whether a network's output stays within safe bounds:

.. code-block:: python

   # Perform reachability
   output_sets = net.reach(input_box, method='approx')

   # Check: is output dimension 0 always <= 2.0?
   safety_bound = 2.0
   is_safe = True
   for output_set in output_sets:
       out_lb, out_ub = output_set.get_ranges()
       if out_ub[0, 0] > safety_bound:
           is_safe = False
           break

   print(f"Property y[0] <= {safety_bound}: {'SAFE' if is_safe else 'UNKNOWN'}")

Loading an ONNX Model
----------------------

n2v can load and verify ONNX models directly:

.. code-block:: python

   from n2v.utils import load_onnx

   # Load an ONNX model (converted to PyTorch internally)
   model = load_onnx("path/to/model.onnx")

   # Verify as usual
   net = n2v.NeuralNetwork(model)
   output_sets = net.reach(input_star, method='approx')

Next Steps
----------

* :doc:`/user-guide/set-representations` -- Learn about all available set types
* :doc:`/user-guide/verification-methods` -- Compare exact, approximate, and probabilistic methods
* :doc:`/examples/basic-verification` -- Detailed walkthrough of the verification examples
* :doc:`/examples/mnist-tutorial` -- Train and verify MNIST classifiers
