#!/usr/bin/env python3
"""
Train all models and export to ONNX format.

This script:
1. Trains all MNIST models (FC and CNN)
2. Creates toy models with random weights
3. Exports all models to ONNX format
4. Saves test samples for verification

Usage:
    python train_all.py                  # Train all models
    python train_all.py --model fc_mnist # Train specific model
    python train_all.py --epochs 10      # Custom epochs
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.io import savemat

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.architectures import MODEL_REGISTRY, get_model, get_model_info


def get_mnist_loaders(batch_size: int = 64, data_dir: str = '../data'):
    """Get MNIST train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def train_mnist_model(model_name: str, epochs: int = 5, device: str = 'cpu'):
    """Train an MNIST model."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model_info = get_model_info(model_name)
    model = get_model(model_name).to(device)

    train_loader, test_loader = get_mnist_loaders(data_dir='../data')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    print(f"Final test accuracy: {test_acc:.2f}%")
    return model, test_loader


def create_toy_model(model_name: str, device: str = 'cpu'):
    """Create a toy model with random weights."""
    print(f"\n{'='*60}")
    print(f"Creating toy model: {model_name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = get_model(model_name).to(device)
    print(f"Created {model_name} with random weights")
    return model


def export_to_onnx(model, model_name: str, output_dir: Path, device: str = 'cpu'):
    """Export model to ONNX format."""
    model_info = get_model_info(model_name)
    input_shape = model_info['input_shape']

    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    onnx_path = output_dir / f"{model_name}.onnx"
    pth_path = output_dir / f"{model_name}.pth"

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes=None,  # Fixed batch size for verification
    )
    print(f"Exported ONNX: {onnx_path}")

    # Save PyTorch checkpoint
    torch.save(model.state_dict(), pth_path)
    print(f"Saved PyTorch: {pth_path}")

    # Verify ONNX export
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()

    diff = np.abs(ort_output - torch_output).max()
    print(f"ONNX verification: max diff = {diff:.2e}")
    if diff > 1e-5:
        print("WARNING: ONNX output differs significantly from PyTorch!")

    return onnx_path


def save_mnist_test_sample(test_loader, model, model_name: str, sample_dir: Path, device: str = 'cpu'):
    """Save a test sample for verification."""
    model.eval()

    # Find a correctly classified sample
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)

        # Find first correctly classified sample
        for i in range(len(pred)):
            if pred[i] == target[i]:
                sample_image = data[i].cpu().numpy()
                sample_label = target[i].cpu().item()
                sample_output = output[i].cpu().numpy()
                sample_pred = pred[i].cpu().item()

                # Save sample
                sample_path = sample_dir / f"{model_name}_sample.mat"
                savemat(str(sample_path), {
                    'image': sample_image.squeeze(),  # Remove batch/channel dims for FC, keep for CNN
                    'label': sample_label,
                    'predicted': sample_pred,
                    'logits': sample_output,
                    'image_full': sample_image,  # Full tensor with all dims
                })
                print(f"Saved test sample: {sample_path}")
                print(f"  Label: {sample_label}, Predicted: {sample_pred}")
                return sample_path

    raise RuntimeError("Could not find a correctly classified sample")


def save_toy_test_sample(model, model_name: str, sample_dir: Path, device: str = 'cpu'):
    """Save a test sample for toy model verification."""
    model_info = get_model_info(model_name)
    input_dim = model_info['input_shape'][-1]

    # Create a simple test input
    torch.manual_seed(42)
    sample_input = torch.randn(1, input_dim).to(device)

    with torch.no_grad():
        sample_output = model(sample_input).cpu().numpy()

    sample_path = sample_dir / f"{model_name}_sample.mat"
    savemat(str(sample_path), {
        'input': sample_input.cpu().numpy().flatten(),
        'output': sample_output.flatten(),
        'input_dim': input_dim,
        'output_dim': model_info['num_classes'],
    })
    print(f"Saved toy test sample: {sample_path}")
    return sample_path


def main():
    parser = argparse.ArgumentParser(description='Train and export models for comparison')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (default: cpu)')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Get base directory
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)

    # Determine which models to train
    if args.model:
        model_names = [args.model]
    else:
        model_names = list(MODEL_REGISTRY.keys())

    print(f"Models to process: {model_names}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    for model_name in model_names:
        model_info = get_model_info(model_name)
        model_dir = base_dir / 'models' / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if model_info['dataset'] == 'mnist':
            # Train MNIST model
            model, test_loader = train_mnist_model(model_name, epochs=args.epochs, device=args.device)
            export_to_onnx(model, model_name, model_dir, device=args.device)
            save_mnist_test_sample(test_loader, model, model_name, sample_dir, device=args.device)

        elif model_info['dataset'] == 'toy':
            # Create toy model
            model = create_toy_model(model_name, device=args.device)
            export_to_onnx(model, model_name, model_dir, device=args.device)
            save_toy_test_sample(model, model_name, sample_dir, device=args.device)

    print(f"\n{'='*60}")
    print("All models processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
