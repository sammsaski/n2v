Installation
============

Prerequisites
-------------

* Python >= 3.8
* PyTorch >= 2.0.0
* Git (for submodule support)

Install from Source
-------------------

**Step 1: Clone with submodules**

.. code-block:: bash

   git clone --recurse-submodules https://github.com/sammsaski/n2v.git

   # If already cloned without submodules:
   git submodule update --init --recursive

**Step 2: Install dependencies and package**

.. code-block:: bash

   cd n2v
   pip install -r requirements.txt
   pip install -e third_party/onnx2torch
   pip install -e .

Alternatively, use the install script:

.. code-block:: bash

   bash install.sh

Dependencies
------------

**Core:**

.. list-table::
   :header-rows: 1

   * - Package
     - Version
   * - numpy
     - >= 1.20.0
   * - scipy
     - >= 1.7.0
   * - torch
     - >= 2.0.0
   * - cvxpy
     - >= 1.2.0
   * - onnx2torch
     - (from submodule)

**Optional:**

* ``matplotlib`` -- Plotting for examples and tutorials
* ``torchvision`` -- MNIST dataset for tutorials
* ``jupyter`` -- Running example notebooks
* ``onnx``, ``onnxruntime`` -- ONNX model support
* ``gurobipy`` -- Gurobi LP solver (requires license)

Verify Installation
-------------------

.. code-block:: bash

   python -c "import n2v; print(n2v.__version__)"

This should print the current version (e.g., ``0.1.0``).

Troubleshooting
---------------

**Missing submodule (onnx2torch)**

If you see ``ModuleNotFoundError: No module named 'onnx2torch'``:

.. code-block:: bash

   git submodule update --init --recursive
   pip install -e third_party/onnx2torch

**PyTorch version mismatch**

n2v requires PyTorch >= 2.0.0 for ``torch.fx`` tracing support. Check your
version with:

.. code-block:: bash

   python -c "import torch; print(torch.__version__)"
