Probabilistic Verification
==========================

Probabilistic verification uses conformal inference to compute output bounds
with formal statistical guarantees. It treats the network as a black box,
making it scalable to arbitrarily large models.

The tutorial files are in ``examples/ProbVer/``.

Basic Usage
-----------

.. code-block:: python

   import torch.nn as nn
   import numpy as np
   from n2v.sets import Box
   from n2v.probabilistic import verify

   # Any callable model
   model = nn.Sequential(
       nn.Linear(5, 100),
       nn.ReLU(),
       nn.Linear(100, 3)
   )
   model.eval()

   # Define input region
   lb = np.zeros(5)
   ub = np.ones(5)
   input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))

   # Probabilistic verification
   prob_box = verify(
       model,
       input_box,
       m=8000,          # calibration samples
       epsilon=0.001,   # miscoverage level
   )

   # Results
   print(f"Output bounds: [{prob_box.lb.flatten()}, {prob_box.ub.flatten()}]")
   print(f"Coverage: {prob_box.guarantee.coverage}")
   print(f"Confidence: {prob_box.guarantee.confidence}")

Understanding the Guarantees
----------------------------

The probabilistic verification provides an ``(epsilon, ell, m)`` guarantee:

* **Coverage** (1 - epsilon): The probability that a random output from the
  input set falls inside the computed bounds
* **Confidence**: The probability that the coverage guarantee holds
* **m**: Number of calibration samples used

For example, with ``m=8000`` and ``epsilon=0.001``:

* Coverage >= 99.9%
* Confidence is computed from the Beta distribution CDF

Surrogate Models
----------------

The verification uses a surrogate model to predict "typical" outputs, then
measures how far actual outputs deviate:

* **clipping_block** (default): Projects outputs onto the L-inf ball defined
  by the training set, using LP-based clipping
* **naive**: Uses the center of training outputs as the prediction

.. code-block:: python

   # Use naive surrogate (faster but may produce looser bounds)
   prob_box = verify(model, input_box, m=8000, surrogate='naive')

PCA Dimensionality Reduction
-----------------------------

For high-dimensional outputs (e.g., image segmentation), PCA can reduce the
output dimension before computing bounds:

.. code-block:: python

   prob_box = verify(
       model,
       input_box,
       m=8000,
       pca_components=50,  # reduce output to 50 dimensions
   )
