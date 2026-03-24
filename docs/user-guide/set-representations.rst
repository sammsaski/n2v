Set Representations
===================

n2v uses set-based methods to propagate input regions through neural networks.
Different set representations offer different trade-offs between precision and
computational cost.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 15 40 20 25

   * - Set
     - Representation
     - Precision
     - Cost
   * - **Star**
     - ``x = c + V*a, C*a <= d``
     - Exact (polytope)
     - LP per bound query
   * - **Zono**
     - ``x = c + V*a, a in [-1,1]``
     - Over-approximate
     - No LP needed
   * - **Box**
     - ``lb <= x <= ub``
     - Coarse
     - O(1) per bound
   * - **ImageStar**
     - 4D Star for CNNs
     - Exact (polytope)
     - LP per bound query
   * - **ImageZono**
     - 4D Zonotope for CNNs
     - Over-approximate
     - No LP needed
   * - **Hexatope**
     - DCS-constrained zonotope
     - Tighter than Zono
     - Min-cost flow
   * - **Octatope**
     - UTVPI-constrained zonotope
     - Tighter than Zono
     - Strongly polynomial
   * - **ProbabilisticBox**
     - Box + coverage guarantee
     - Statistical
     - Sampling-based

Star
----

The primary set representation. A Star set is defined as:

``S = { x in R^n | x = c + V*a, C*a <= d }``

where ``c`` is the center, ``V`` is the basis matrix, and ``C*a <= d`` are the
predicate constraints.

.. code-block:: python

   from n2v.sets import Star
   import numpy as np

   # From bounds (L-inf ball)
   lb = np.array([0.0, 0.0])
   ub = np.array([1.0, 1.0])
   star = Star.from_bounds(lb, ub)

   # Get tight bounds (uses LP)
   lower, upper = star.get_ranges()

   # Check containment
   point = np.array([[0.5], [0.5]])
   print(star.contains(point))  # True

Stars support exact reachability through ReLU layers via case splitting, making
them the most precise representation available.

Zonotope (Zono)
---------------

A zonotope is a centrally-symmetric polytope:

``Z = { x in R^n | x = c + V*a, -1 <= a_i <= 1 }``

.. code-block:: python

   from n2v.sets import Zono

   zono = Zono.from_bounds(lb, ub)

   # Bounds are computed without LP (fast)
   lower, upper = zono.get_ranges()

   # Set operations
   zono2 = Zono.from_bounds(lb + 0.5, ub + 0.5)
   zono_sum = zono.minkowski_sum(zono2)
   zono_hull = zono.convex_hull(zono2)

Zonotopes are efficient for approximate reachability but lose precision at
nonlinear layers due to symmetric over-approximation.

Box
---

An axis-aligned hyperrectangle defined by lower and upper bounds:

.. code-block:: python

   from n2v.sets import Box

   box = Box(
       lb=np.array([[0.0], [0.0]]),
       ub=np.array([[1.0], [1.0]])
   )

   # Conversion to other sets
   star = box.to_star()
   zono = box.to_zono()

Boxes are the fastest representation but also the least precise. They are useful
for quick approximate analyses and as starting points for constructing other sets.

ImageStar and ImageZono
-----------------------

4D variants of Star and Zonotope designed for convolutional neural networks.
They preserve the spatial structure ``(H, W, C)`` of image data, enabling direct
4D convolution and pooling operations without flattening.

.. code-block:: python

   from n2v.sets import ImageStar

   # Create from image bounds (H, W, C)
   img_lb = np.zeros((28, 28, 1))
   img_ub = np.ones((28, 28, 1))
   istar = ImageStar.from_bounds(img_lb, img_ub)

Hexatope and Octatope
---------------------

Constrained zonotopes that provide tighter approximations than standard
zonotopes:

* **Hexatope**: DCS (Difference Constraint System) constrained zonotope, solved
  via minimum-cost flow optimization
* **Octatope**: UTVPI (Unit Two Variable Per Inequality) constrained zonotope,
  with strongly polynomial-time optimization

These representations sit between zonotopes and stars in the precision-cost
trade-off.

HalfSpace
---------

A linear constraint ``G*x <= g`` used to define safety properties and output
specifications:

.. code-block:: python

   from n2v.sets import HalfSpace

   # Define constraint: x[0] - x[1] <= 0 (i.e., x[0] <= x[1])
   G = np.array([[1.0, -1.0]])
   g = np.array([[0.0]])
   hs = HalfSpace(G, g)

ProbabilisticBox
----------------

Output of probabilistic verification -- a Box with associated coverage and
confidence guarantees:

.. code-block:: python

   from n2v.probabilistic import verify

   prob_box = verify(model, input_box, m=8000, epsilon=0.001)
   print(prob_box.guarantee)  # ConformalGuarantee with coverage and confidence

Common Operations
-----------------

All set types support a common interface:

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``from_bounds(lb, ub)``
     - Construct from lower/upper bounds
   * - ``affine_map(W, b)``
     - Apply affine transformation ``W*x + b``
   * - ``get_ranges()``
     - Compute tight lower/upper bounds
   * - ``estimate_ranges()``
     - Fast (possibly loose) bound estimation
   * - ``contains(point)``
     - Check if a point is inside the set
   * - ``is_empty_set()``
     - Check if the set is empty
   * - ``to_star()``
     - Convert to Star representation
   * - ``sample(n)``
     - Sample ``n`` random points from the set
