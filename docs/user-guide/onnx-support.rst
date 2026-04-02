ONNX Support
============

n2v can load and verify ONNX neural network models, enabling verification of
models trained in any framework.

Loading ONNX Models
-------------------

.. code-block:: python

   from n2v.utils import load_onnx

   # Load and convert ONNX model to PyTorch
   model = load_onnx("path/to/model.onnx")

   # Use as usual
   net = n2v.NeuralNetwork(model)
   output_sets = net.reach(input_star, method='approx')

Internally, ``load_onnx`` uses the `onnx2torch <https://github.com/sammsaski/onnx2torch>`_
library (included as a git submodule) to convert the ONNX model to a PyTorch
``GraphModule``. The graph execution engine then handles ONNX-specific operations.

Supported ONNX Operations
--------------------------

n2v supports a wide range of ONNX operations through its graph execution engine:

* **Arithmetic**: Add, Sub, Mul, Div, MatMul, Neg
* **Shape**: Reshape, Transpose, Flatten, Concat, Slice, Split
* **Neural network**: Conv, BatchNormalization, Relu, Sigmoid, MaxPool, AveragePool
* **Utility**: Cast, Resize, Pad, Upsample

VNNLIB Specifications
---------------------

n2v can parse `VNNLIB <https://www.vnnlib.org/>`_ format specifications, the
standard for neural network verification benchmarks:

.. code-block:: python

   from n2v.utils import load_vnnlib

   # Parse a VNNLIB specification file
   input_specs, output_specs = load_vnnlib("property.vnnlib", num_inputs=5, num_outputs=5)

BatchNorm Fusion
----------------

For models with BatchNorm layers, fusing them into preceding Linear/Conv layers
can improve verification performance:

.. code-block:: python

   from n2v.utils import fuse_batchnorm, has_batchnorm

   if has_batchnorm(model):
       model = fuse_batchnorm(model)

VNN-COMP
--------

n2v includes infrastructure for running `VNN-COMP <https://sites.google.com/view/vnn2025>`_
benchmarks. See the ``examples/VNN-COMP/`` directory for the complete
competition setup.
