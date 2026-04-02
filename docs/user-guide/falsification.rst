Falsification
=============

Falsification searches for counterexamples that violate a safety property.
It is much faster than full reachability analysis and is useful for quickly
identifying property violations before running expensive verification.

Usage
-----

.. code-block:: python

   from n2v.utils import falsify
   import numpy as np

   # Define input bounds
   lb = np.array([0.0, 0.0])
   ub = np.array([1.0, 1.0])

   # Define safety property as a HalfSpace: G*y <= g
   # Example: y[0] <= 2.0
   G = np.array([[1.0, 0.0]])
   g = np.array([[2.0]])
   property_spec = (G, g)

   # Search for counterexamples
   result, counterexample = falsify(
       model, lb, ub, property_spec,
       method='random+pgd'
   )

   if result == 'sat':
       print(f"Found counterexample: {counterexample}")
   else:
       print("No counterexample found (property may still be violated)")

Methods
-------

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``random``
     - Uniform random sampling from the input bounds
   * - ``pgd``
     - Projected Gradient Descent -- gradient-based adversarial search
   * - ``random+pgd``
     - Combined: random sampling followed by PGD refinement

Limitations
-----------

* Input must be a hyperbox (lower/upper bounds), not a general polytope
* Finding no counterexample does **not** prove safety -- use reachability
  analysis for formal guarantees
