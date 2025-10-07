# n2v Tutorial: MNIST Verification

A step-by-step tutorial demonstrating neural network verification with n2v
using MNIST digit classifiers.

## Prerequisites

- n2v installed (`pip install -e .` from repo root)
- torchvision (for MNIST dataset download)
- matplotlib (optional, for plots)

## Usage

Run the scripts in order:

```bash
cd examples/Tutorial

# Step 1: Train a fully connected classifier
python train_fc.py

# Step 2: Verify the FC classifier with exact Star reachability
python verify_fc.py

# Step 3: Train a CNN classifier (with AvgPool2d)
python train_cnn.py

# Step 4: Verify the CNN classifier with exact ImageStar reachability
python verify_cnn.py
```

## Scripts

### train_fc.py

Trains a fully connected MNIST classifier (784 -> 128 -> 64 -> 10) for 10
epochs. Saves the model to `models/mnist_fc_classifier.pth`.

### verify_fc.py

Loads the trained FC model and performs exact reachability analysis using Star
sets. Creates L-inf perturbation regions around a test image and verifies local
robustness at multiple epsilon values.

### train_cnn.py

Trains a CNN MNIST classifier using AvgPool2d (instead of MaxPool2d) for
efficient verification. AvgPool2d is a linear operation that does not cause star
splitting, resulting in 10-100x faster verification. Includes a comparison of
AvgPool vs MaxPool accuracy. Saves the model to `models/mnist_cnn_classifier.pth`.

### verify_cnn.py

Loads the trained CNN model and performs exact reachability analysis using
ImageStar sets, which preserve spatial structure for convolutional layers.
Includes layer-by-layer analysis showing where star splitting occurs and a
comparison of AvgPool vs MaxPool verification speed.
