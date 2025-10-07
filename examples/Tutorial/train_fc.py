"""Train a fully connected MNIST classifier.

Trains a simple fully connected neural network on MNIST digits.
Architecture: 784 -> 30 -> 30 -> 10

The trained model is saved to models/mnist_fc_classifier.pth for use
by verify_fc.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


class FCMnistClassifier(nn.Module):
    """Fully connected MNIST classifier."""

    def __init__(self):
        super(FCMnistClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )

    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_epoch(model, loader, device):
    """Evaluate on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load MNIST Dataset ---
    print("\nLoading MNIST dataset...")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # --- Define and Train FC Model ---
    model = FCMnistClassifier().to(device)
    print(f"\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    print("\nTraining...\n")

    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_acc = test_epoch(model, test_loader, device)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train: {train_acc:.2f}%  "
              f"Test: {test_acc:.2f}%")

    print(f"\nFinal test accuracy: {test_accuracies[-1]:.2f}%")

    # --- Save Model ---
    Path('models').mkdir(exist_ok=True)

    model_path = 'models/mnist_fc_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracies[-1],
        'architecture': 'FC: 784-30-30-10'
    }, model_path)

    print(f"Model saved to: {model_path}")
    print("\nNext: Verify this model using n2v by running verify_fc.py")
