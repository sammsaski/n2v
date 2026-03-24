n2v: Neural Network Verification Toolbox
========================================

.. rst-class:: lead

   **n2v** is a Python toolbox for neural network verification using sound set-based
   reachability analysis and probabilistic verification via conformal inference.
   It supports PyTorch models and ONNX networks.

.. rst-class:: lead

   Translated from the MATLAB `NNV <https://github.com/verivital/nnv>`_ tool by the
   `VeriVITAL <https://www.verivital.com/>`_ research group.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Set-Based Reachability
      :class-card: sd-border-0 sd-shadow-sm
      :text-align: center

      Star, Zonotope, Box, Hexatope, Octatope --
      propagate sets through neural networks with formal guarantees.

   .. grid-item-card:: 20+ Layer Types
      :class-card: sd-border-0 sd-shadow-sm
      :text-align: center

      ReLU, Conv2D, MaxPool2D, Sigmoid, Tanh, BatchNorm,
      and more -- with exact and approximate methods.

   .. grid-item-card:: Probabilistic Verification
      :class-card: sd-border-0 sd-shadow-sm
      :text-align: center

      Conformal inference for model-agnostic verification
      with coverage and confidence guarantees.

   .. grid-item-card:: ONNX & VNN-COMP
      :class-card: sd-border-0 sd-shadow-sm
      :text-align: center

      Load ONNX models, parse VNNLIB specifications, and
      run VNN-COMP benchmarks out of the box.

----

Quick Example
-------------

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

   # Create input set (L-inf ball)
   center = np.array([0.5, 0.5, 0.5])
   epsilon = 0.1
   input_star = Star.from_bounds(center - epsilon, center + epsilon)

   # Compute reachable output set
   net = n2v.NeuralNetwork(model)
   output_stars = net.reach(input_star, method='exact')

   # Extract bounds
   for star in output_stars:
       lb, ub = star.get_ranges()
       print(f"Output bounds: [{lb.flatten()}, {ub.flatten()}]")

----

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started/index
   user-guide/index
   theory/index
   api/index
   examples/index

Getting Started
---------------

.. rst-class:: lead

   Install n2v and run your first neural network verification in minutes.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: getting-started/installation
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Set up n2v with all dependencies, including PyTorch and the onnx2torch
      submodule.

   .. grid-item-card:: Quick Start
      :link: getting-started/quickstart
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Create your first input set, run reachability analysis, and check
      safety properties.

----

User Guide
----------

.. rst-class:: lead

   In-depth guides for using n2v's features, from set representations to
   ONNX model verification.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Set Representations
      :link: user-guide/set-representations
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Star, Zonotope, Box, ImageStar, Hexatope, Octatope -- learn when and
      how to use each set type.

   .. grid-item-card:: Verification Methods
      :link: user-guide/verification-methods
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Exact, approximate, probabilistic, and hybrid methods -- choose the
      right approach for your problem.

   .. grid-item-card:: Configuration
      :link: user-guide/configuration
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Parallel LP solving, solver selection, and global settings.

   .. grid-item-card:: LP Solvers
      :link: user-guide/lp-solvers
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Detailed comparison of LP solver backends and performance tuning.

   .. grid-item-card:: Falsification
      :link: user-guide/falsification
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Fast counterexample search via random sampling and PGD.

   .. grid-item-card:: ONNX Support
      :link: user-guide/onnx-support
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Load ONNX models, parse VNNLIB specs, and run VNN-COMP benchmarks.

----

Theoretical Foundations
-----------------------

.. rst-class:: lead

   Mathematical details behind n2v's verification algorithms, set
   representations, and probabilistic guarantees.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Mathematical Foundations
      :link: theory/theoretical-foundations
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Set representations, exact and approximate layer operations,
      verification algorithms, and optimization techniques.

   .. grid-item-card:: Probabilistic Verification Theory
      :link: theory/probabilistic-verification
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Conformal inference, nonconformity scores, surrogate models,
      and the coverage-confidence guarantee framework.

----

API Reference
-------------

.. rst-class:: lead

   Complete reference for the n2v public API.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Sets
      :link: api/sets
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope,
      HalfSpace, ProbabilisticBox

   .. grid-item-card:: Neural Network
      :link: api/neural-network
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``NeuralNetwork`` class for reachability analysis

   .. grid-item-card:: Layer Operations
      :link: api/layer-ops
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Layer-level reachability dispatch and operations

   .. grid-item-card:: Probabilistic
      :link: api/probabilistic
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      ``verify()``, conformal inference functions

   .. grid-item-card:: Utilities
      :link: api/utils
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      LP solvers, model loading, falsification, preprocessing

   .. grid-item-card:: Configuration
      :link: api/config
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Global settings for parallelization and solver selection

----

Examples
--------

.. rst-class:: lead

   Worked examples demonstrating n2v's verification capabilities, from simple
   feedforward networks to real-world benchmarks.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 1. Basic Verification
      :link: examples/basic-verification
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Understand set operations and basic reachability analysis on a simple
      feedforward network.

   .. grid-item-card:: 2. MNIST Tutorial
      :link: examples/mnist-tutorial
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Train and verify MNIST classifiers -- both fully-connected and
      convolutional architectures.

   .. grid-item-card:: 3. ACAS Xu Benchmark
      :link: examples/acasxu
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Real-world collision avoidance benchmark with falsification and
      multi-stage verification.

   .. grid-item-card:: 4. Probabilistic Verification
      :link: examples/probabilistic-verification
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Model-agnostic verification via conformal inference with formal
      coverage and confidence guarantees.

----

References
----------

The methods implemented in n2v are based upon or used in the following publications:

.. admonition:: Key Publications

   D.\  Manzanas Lopez, S.W. Choi, H.-D. Tran, T.T. Johnson,
   "NNV 2.0: The Neural Network Verification Tool,"
   in *Computer Aided Verification (CAV)*, Springer, 2023.
   `DOI: 10.1007/978-3-031-37703-7_19 <https://doi.org/10.1007/978-3-031-37703-7_19>`__

   N.\  Hashemi, S. Sasaki, I. Oguz, M. Ma, T.T. Johnson,
   "Scaling Data-Driven Probabilistic Robustness Analysis for Semantic Segmentation Neural Networks,"
   in *38th Conference on Neural Information Processing Systems (NeurIPS)*, 2025.

   S.\  Sasaki, D. Manzanas Lopez, P.K. Robinette, T.T. Johnson,
   "Robustness Verification of Video Classification Neural Networks,"
   in *IEEE/ACM 13th International Conference on Formal Methods in Software Engineering (FormaliSE)*, 2025.
   `DOI: 10.1109/FormaliSE66629.2025.00009 <https://doi.org/10.1109/FormaliSE66629.2025.00009>`__

   H.-D.\  Tran, P. Musau, D. Manzanas Lopez, X. Yang, L.V. Nguyen, W. Xiang, T.T. Johnson,
   "Star-Based Reachability Analysis for Deep Neural Networks,"
   in *23rd International Symposium on Formal Methods (FM)*, 2019.

   H.-D.\  Tran, S. Bak, W. Xiang, T.T. Johnson,
   "Towards Verification of Large Convolutional Neural Networks Using ImageStars,"
   in *32nd International Conference on Computer-Aided Verification (CAV)*, 2020.

Acknowledgements
----------------

This work is supported in part by AFOSR, DARPA, NSF.
