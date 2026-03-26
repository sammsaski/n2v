MNIST Tutorial
==============

This tutorial demonstrates training and verifying MNIST digit classifiers,
covering both fully-connected and convolutional architectures.

The tutorial files are in ``examples/Tutorial/``.

Step 1: Train a Fully-Connected Network
----------------------------------------

.. code-block:: bash

   cd examples/Tutorial
   python train_fc.py

This trains a simple fully-connected network on MNIST and saves the model to
``models/fc_mnist.pth``.

Step 2: Verify the FC Network
------------------------------

.. code-block:: bash

   python verify_fc.py

The verification script:

1. Loads a test image from MNIST
2. Creates a Star input set representing an L-inf perturbation around the image
3. Runs exact reachability analysis through the network
4. Checks whether the perturbed outputs still classify correctly

.. code-block:: python

   import n2v
   from n2v.sets import Star

   # Load the trained model
   model = torch.load("models/fc_mnist.pth")
   model.eval()

   # Create input perturbation (epsilon-ball around test image)
   epsilon = 0.02
   input_star = Star.from_bounds(image - epsilon, image + epsilon)

   # Verify
   net = n2v.NeuralNetwork(model)
   output_stars = net.reach(input_star, method='exact')

   # Check if the correct class always has the highest output
   for star in output_stars:
       lb, ub = star.get_ranges()
       # Verify classification is robust

Step 3: Train a CNN
--------------------

.. code-block:: bash

   python train_cnn.py

This trains a CNN with ``AvgPool2d`` (instead of ``MaxPool2d``) for efficient
verification with ImageStar sets.

Step 4: Verify the CNN
-----------------------

.. code-block:: bash

   python verify_cnn.py

CNN verification uses ``ImageStar`` sets that preserve the spatial structure
of the input image:

.. code-block:: python

   from n2v.sets import ImageStar

   # Create 4D input perturbation (H, W, C)
   img_lb = image - epsilon
   img_ub = image + epsilon
   input_istar = ImageStar.from_bounds(img_lb, img_ub)

   # Verify with approximate reachability
   net = n2v.NeuralNetwork(model)
   output_sets = net.reach(input_istar, method='approx')
