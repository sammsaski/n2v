User Guide
==========

.. rst-class:: lead

   In-depth guides for using n2v's features, from set representations to
   ONNX model verification.

----

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Set Representations
      :link: set-representations
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Star, Zonotope, Box, ImageStar, Hexatope, Octatope -- learn when and
      how to use each set type.

   .. grid-item-card:: Verification Methods
      :link: verification-methods
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Exact, approximate, probabilistic, and hybrid methods -- choose the
      right approach for your problem.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Parallel LP solving, solver selection, and global settings.

   .. grid-item-card:: LP Solvers
      :link: lp-solvers
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Detailed comparison of LP solver backends and performance tuning.

   .. grid-item-card:: Falsification
      :link: falsification
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Fast counterexample search via random sampling and PGD.

   .. grid-item-card:: ONNX Support
      :link: onnx-support
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Load ONNX models, parse VNNLIB specs, and run VNN-COMP benchmarks.

.. toctree::
   :maxdepth: 2
   :hidden:

   set-representations
   verification-methods
   configuration
   lp-solvers
   falsification
   onnx-support
