Verification Methods
====================

n2v supports four verification methods, each offering different trade-offs
between precision, completeness, and scalability.

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Method
     - Guarantee
     - Speed
     - Description
   * - ``exact``
     - Sound and complete
     - Slowest
     - Star splitting at nonlinear layers
   * - ``approx``
     - Sound (over-approximate)
     - Fast
     - Triangle/S-curve relaxation, no splitting
   * - ``probabilistic``
     - Coverage + confidence
     - Constant-time
     - Conformal inference, model-agnostic
   * - ``hybrid``
     - Mixed
     - Adaptive
     - Exact until threshold, then probabilistic

Exact Reachability
------------------

Exact reachability propagates Star sets through the network layer by layer. At
each nonlinear layer (e.g., ReLU), the input star is split into sub-stars based
on activation patterns:

.. code-block:: python

   net = n2v.NeuralNetwork(model)
   output_stars = net.reach(input_star, method='exact')

   # Multiple output stars -- their union is the exact reachable set
   print(f"Number of output stars: {len(output_stars)}")

**When to use:** Small networks (< ~100 neurons) where you need a complete
answer (provably safe or provably unsafe).

**Trade-off:** The number of stars can grow exponentially with the number of
unstable ReLU neurons.

Approximate Reachability
------------------------

Approximate reachability relaxes nonlinear layers using convex approximations:

* **ReLU / LeakyReLU**: Triangle relaxation (linear upper bound, piecewise lower bound)
* **Sigmoid / Tanh**: S-curve tangent/secant relaxation
* **MaxPool2D**: Bounds-based over-approximation

.. code-block:: python

   output_sets = net.reach(input_star, method='approx')

   # Always returns a single output set (no splitting)
   print(f"Number of output sets: {len(output_sets)}")  # 1

**Parameters:**

* ``relax_factor`` (float, 0.0-1.0): Controls the tightness of the ReLU relaxation.
  Default is adaptive based on the input star's dimension.

**When to use:** Medium-sized networks where exact verification is too expensive.
The result is always an over-approximation: if verification says "safe", the
network is safe. If it says "unsafe", the network may or may not be unsafe.

Probabilistic Verification
--------------------------

Probabilistic verification treats the network as a black box and uses conformal
inference to compute output bounds with formal statistical guarantees:

.. code-block:: python

   from n2v.probabilistic import verify

   prob_box = verify(
       model,
       input_box,
       m=8000,        # calibration samples
       epsilon=0.001,  # miscoverage level
   )

   print(f"Coverage: {prob_box.guarantee.coverage}")
   print(f"Confidence: {prob_box.guarantee.confidence}")
   print(f"Bounds: [{prob_box.lb.flatten()}, {prob_box.ub.flatten()}]")

**When to use:** Large networks where deterministic methods are infeasible.
Provides formal probabilistic guarantees regardless of network size.

See :doc:`/theory/probabilistic-verification` for the mathematical foundations.

Hybrid Verification
-------------------

Hybrid verification starts with exact reachability and switches to probabilistic
verification when the number of stars exceeds a threshold:

.. code-block:: python

   output_sets = net.reach(
       input_star,
       method='hybrid',
       max_stars=1000,  # switch to probabilistic after this many stars
   )

**When to use:** When you want exact results where possible but need to bound
computation time.

Choosing a Method
-----------------

.. code-block:: text

   Is the network small (< ~100 ReLU neurons)?
   ├── Yes → Use 'exact' for sound + complete results
   └── No
       ├── Is sound over-approximation sufficient?
       │   ├── Yes → Use 'approx'
       │   └── No → Use 'probabilistic' or 'hybrid'
       └── Is the network very large (> 10K neurons)?
           └── Yes → Use 'probabilistic'
