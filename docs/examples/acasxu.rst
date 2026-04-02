ACAS Xu Benchmark
=================

ACAS Xu (Airborne Collision Avoidance System for Unmanned Aircraft) is a
standard benchmark for neural network verification, consisting of 45 networks
and 10 safety properties.

The benchmark files are in ``examples/ACASXu/``.

Background
----------

Each ACAS Xu network takes 5 inputs (relative position, heading, speed) and
outputs 5 advisory scores (clear-of-conflict, weak left, weak right, strong
left, strong right). The safety properties specify input-output constraints
that must hold for the network to be safe.

Running Verification
--------------------

**Single instance:**

.. code-block:: bash

   cd examples/ACASXu
   python verify_acasxu.py

**Full benchmark (186 instances):**

.. code-block:: bash

   bash run_benchmark.sh

The benchmark uses VNNLIB format for property specifications and ONNX format
for the network models.

Verification Pipeline
---------------------

The ACAS Xu verification follows a three-stage approach:

1. **Falsification**: Quick counterexample search via random sampling + PGD
2. **Approximate reachability**: Fast sound over-approximation
3. **Exact reachability**: Complete verification for remaining instances

.. code-block:: python

   from n2v.utils import load_onnx, load_vnnlib, falsify

   # Load model and property
   model = load_onnx("networks/N_1_1.onnx")
   input_specs, output_specs = load_vnnlib("properties/prop_1.vnnlib",
                                            num_inputs=5, num_outputs=5)

   # Stage 1: Try falsification first
   result, cex = falsify(model, lb, ub, output_specs, method='random+pgd')

   # Stage 2: If no counterexample, verify with reachability
   if result != 0:
       net = n2v.NeuralNetwork(model)
       output_sets = net.reach(input_star, method='exact')
