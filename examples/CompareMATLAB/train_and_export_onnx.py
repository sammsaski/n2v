#!/usr/bin/env python3
"""
Train FC MNIST Classifier and Export to ONNX

This script trains a simple fully connected network and exports it to ONNX format
for comparison between NNV-Python and MATLAB NNV.

Outputs:
1. fc_mnist.onnx - Model in ONNX format (usable in both Python and MATLAB)
2. test_sample.npy - Single test image for verification
3. test_sample.mat - Test sample in MATLAB format
4. test_sample_info.txt - Metadata about the test sample
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import onnx
import onnxruntime as ort
from scipy.io import savemat

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')  # Use CPU for consistency
print(f"Using device: {device}")

# =============================================================================
# Define Simple FC Network
# =============================================================================

class SimpleFCMnist(nn.Module):
    """Simple FC network for ONNX export and MATLAB comparison."""

    def __init__(self):
        super(SimpleFCMnist, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


model = SimpleFCMnist().to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# Load MNIST Data
# =============================================================================

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# =============================================================================
# Training
# =============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

print("\nTraining...\n")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    test_acc = 100. * test_correct / test_total
    print(f"Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

print("\nTraining complete!")

# =============================================================================
# Select and Save Test Sample
# =============================================================================

# Find a correctly classified sample
model.eval()
test_sample_idx = None

for i in range(len(test_dataset)):
    img, label = test_dataset[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = output.argmax(dim=1).item()
    if pred == label:
        test_sample_idx = i
        break

# Get the sample
test_image, test_label = test_dataset[test_sample_idx]
test_image_np = test_image.squeeze().numpy()

# Get network output (logits)
with torch.no_grad():
    test_output = model(test_image.unsqueeze(0)).squeeze().numpy()
    test_pred = test_output.argmax()

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.imshow(test_image_np, cmap='gray')
ax1.set_title(f"Test Sample {test_sample_idx}\nTrue Label: {test_label}")
ax1.axis('off')

ax2.bar(range(10), test_output)
ax2.set_xlabel('Class')
ax2.set_ylabel('Logit Value')
ax2.set_title(f"Network Output (Logits)\nPredicted: {test_pred}")
ax2.set_xticks(range(10))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_sample_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSelected Test Sample:")
print(f"  Index: {test_sample_idx}")
print(f"  True Label: {test_label}")
print(f"  Predicted: {test_pred}")
print(f"  Image shape: {test_image_np.shape}")
print(f"  Pixel range: [{test_image_np.min():.3f}, {test_image_np.max():.3f}]")

# =============================================================================
# Print Network Output
# =============================================================================

print("="*60)
print("NETWORK OUTPUT (PYTORCH - BEFORE ONNX EXPORT)")
print("="*60)
print(f"Input: Test sample {test_sample_idx}, label {test_label}")
print(f"\nOutput logits (10 values):")
for i, val in enumerate(test_output):
    marker = " <-- PREDICTED" if i == test_pred else ""
    print(f"  Class {i}: {val:12.6f}{marker}")
print(f"\nPredicted class: {test_pred}")
print("="*60)

# =============================================================================
# Save Test Sample to File
# =============================================================================

Path('outputs').mkdir(exist_ok=True)

# Save as NumPy array
np.save('outputs/test_sample.npy', test_image_np)
print(f"\nSaved test sample to: outputs/test_sample.npy")

# Save as MAT file (MATLAB compatible)
savemat('outputs/test_sample.mat', {
    'image': test_image_np,
    'label': test_label,
    'sample_idx': test_sample_idx,
    'predicted': test_pred,
    'logits': test_output
})
print(f"Saved test sample to: outputs/test_sample.mat")

# Save metadata
with open('outputs/test_sample_info.txt', 'w') as f:
    f.write(f"Test Sample Information\n")
    f.write(f"=" * 60 + "\n")
    f.write(f"Sample Index: {test_sample_idx}\n")
    f.write(f"True Label: {test_label}\n")
    f.write(f"Predicted Label: {test_pred}\n")
    f.write(f"Image Shape: {test_image_np.shape}\n")
    f.write(f"Pixel Range: [{test_image_np.min():.6f}, {test_image_np.max():.6f}]\n")
    f.write(f"\nNetwork Output (Logits):\n")
    for i, val in enumerate(test_output):
        f.write(f"  Class {i}: {val:12.6f}\n")

print(f"Saved metadata to: outputs/test_sample_info.txt")

# =============================================================================
# Export Model to ONNX
# =============================================================================

model.eval()

# Create dummy input
dummy_input = torch.randn(1, 1, 28, 28, device=device)

# Export to ONNX
onnx_path = 'outputs/fc_mnist.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"\nModel exported to: {onnx_path}")
print(f"ONNX opset version: 11")
print(f"Input name: 'input'")
print(f"Output name: 'output'")

# =============================================================================
# Verify ONNX Export
# =============================================================================

# Load ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("\nONNX model is valid!")

# Create ONNX Runtime session
ort_session = ort.InferenceSession(onnx_path)

# Run inference with ONNX
test_input_onnx = test_image.unsqueeze(0).numpy()
onnx_output = ort_session.run(
    None,
    {'input': test_input_onnx}
)[0].squeeze()

print("\n" + "="*60)
print("NETWORK OUTPUT (ONNX RUNTIME)")
print("="*60)
print(f"Input: Test sample {test_sample_idx}, label {test_label}")
print(f"\nOutput logits (10 values):")
for i, val in enumerate(onnx_output):
    marker = " <-- PREDICTED" if i == onnx_output.argmax() else ""
    print(f"  Class {i}: {val:12.6f}{marker}")
print(f"\nPredicted class: {onnx_output.argmax()}")
print("="*60)

# Compare outputs
print("\n" + "="*60)
print("COMPARISON: PyTorch vs ONNX")
print("="*60)
max_diff = np.abs(test_output - onnx_output).max()
print(f"Maximum difference: {max_diff:.2e}")

if max_diff < 1e-5:
    print("✅ ONNX export verified! Outputs match PyTorch.")
else:
    print("⚠️  Warning: Outputs differ slightly (but may be acceptable)")

print("\nPer-class differences:")
for i in range(10):
    diff = test_output[i] - onnx_output[i]
    print(f"  Class {i}: {diff:+.2e}")
print("="*60)

# =============================================================================
# Save PyTorch Model
# =============================================================================

torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_acc,
    'architecture': 'FC: 784-50-20-10',
    'test_sample_idx': test_sample_idx,
    'test_label': test_label
}, 'outputs/fc_mnist_pytorch.pth')

print(f"\nPyTorch model saved to: outputs/fc_mnist_pytorch.pth")

# =============================================================================
# Create Summary Document
# =============================================================================

with open('outputs/export_summary.txt', 'w') as f:
    f.write("MNIST FC Model Export Summary\n")
    f.write("="*60 + "\n\n")

    f.write("Model Architecture:\n")
    f.write("  Input: 28x28 image (flattened to 784)\n")
    f.write("  Layer 1: Linear(784, 50) + ReLU\n")
    f.write("  Layer 2: Linear(50, 20) + ReLU\n")
    f.write("  Layer 3: Linear(20, 10)\n")
    f.write(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")

    f.write("Training:\n")
    f.write(f"  Epochs: {num_epochs}\n")
    f.write(f"  Test Accuracy: {test_acc:.2f}%\n\n")

    f.write("Test Sample:\n")
    f.write(f"  Index: {test_sample_idx}\n")
    f.write(f"  True Label: {test_label}\n")
    f.write(f"  Predicted: {test_pred}\n\n")

    f.write("Exported Files:\n")
    f.write("  - fc_mnist.onnx (ONNX model)\n")
    f.write("  - test_sample.npy (NumPy format)\n")
    f.write("  - test_sample.mat (MATLAB format)\n")
    f.write("  - test_sample_info.txt (metadata)\n")
    f.write("  - fc_mnist_pytorch.pth (PyTorch checkpoint)\n\n")

    f.write("ONNX Model Details:\n")
    f.write(f"  Opset Version: 11\n")
    f.write(f"  Input Name: 'input'\n")
    f.write(f"  Output Name: 'output'\n")
    f.write(f"  Input Shape: [batch, 1, 28, 28]\n")
    f.write(f"  Output Shape: [batch, 10]\n\n")

    f.write("Verification Status:\n")
    f.write(f"  PyTorch vs ONNX max diff: {max_diff:.2e}\n")
    f.write(f"  Status: {'✅ Verified' if max_diff < 1e-5 else '⚠️ Check outputs'}\n")

print("Summary saved to: outputs/export_summary.txt")
print("\n✅ All files exported successfully!")
