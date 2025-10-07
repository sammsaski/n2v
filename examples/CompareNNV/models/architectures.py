"""
Model architectures for MATLAB NNV vs n2v comparison experiments.

This module defines all neural network architectures used in the comparison:
- FC models: Fully-connected networks for MNIST
- CNN models: Convolutional networks with various pooling layers
- Toy models: Small networks for Zono/Box set testing
"""

import torch
import torch.nn as nn


# =============================================================================
# Fully-Connected Models (MNIST)
# =============================================================================

class FCMnist(nn.Module):
    """
    Standard FC network for MNIST classification.
    Architecture: Flatten -> 784 -> 50 -> 20 -> 10

    This is the reference model matching the original CompareMATLAB setup.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class FCMnistSmall(nn.Module):
    """
    Smaller FC network for faster testing.
    Architecture: Flatten -> 784 -> 32 -> 16 -> 10
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# CNN Models (MNIST)
# =============================================================================

class CNNConvRelu(nn.Module):
    """
    Simple CNN with Conv2D + ReLU + FC.
    Architecture: Conv2d(1->4, k=5, s=2) -> ReLU -> Flatten -> 576 -> 32 -> 10

    Input: 28x28 -> Conv -> 12x12 (4 channels) -> Flatten -> 576 -> 32 -> 10
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 12 * 12, 32)  # 576 -> 32
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNAvgPool(nn.Module):
    """
    CNN with Conv2D + ReLU + AvgPool2D + FC.
    Architecture: Conv2d(1->4, k=3, s=1, p=1) -> ReLU -> AvgPool(k=4, s=4) -> Flatten -> 196 -> 32 -> 10

    Input: 28x28 -> Conv -> 28x28 -> AvgPool -> 7x7 (4 channels) -> Flatten -> 196 -> 32 -> 10
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 7 * 7, 32)  # 196 -> 32
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNMaxPool(nn.Module):
    """
    CNN with Conv2D + ReLU + MaxPool2D + FC.
    Architecture: Conv2d(1->4, k=3, s=1, p=1) -> ReLU -> MaxPool(k=4, s=4) -> Flatten -> 196 -> 32 -> 10

    Input: 28x28 -> Conv -> 28x28 -> MaxPool -> 7x7 (4 channels) -> Flatten -> 196 -> 32 -> 10
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 7 * 7, 32)  # 196 -> 32
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Toy Models (for Zono/Box testing)
# =============================================================================

class ToyFC_4_3_2(nn.Module):
    """
    Very small FC network for testing Zono/Box set representations.
    Architecture: 4 -> 3 -> 2
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ToyFC_8_4_2(nn.Module):
    """
    Small FC network for testing Zono/Box set representations.
    Architecture: 8 -> 4 -> 2
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    # FC Models (MNIST)
    'fc_mnist': {
        'class': FCMnist,
        'input_shape': (1, 1, 28, 28),
        'num_classes': 10,
        'dataset': 'mnist',
        'description': 'Standard FC network: 784->50->20->10',
    },
    'fc_mnist_small': {
        'class': FCMnistSmall,
        'input_shape': (1, 1, 28, 28),
        'num_classes': 10,
        'dataset': 'mnist',
        'description': 'Small FC network: 784->32->16->10',
    },

    # CNN Models (MNIST)
    'cnn_conv_relu': {
        'class': CNNConvRelu,
        'input_shape': (1, 1, 28, 28),
        'num_classes': 10,
        'dataset': 'mnist',
        'description': 'CNN with Conv2D+ReLU+FC',
    },
    'cnn_avgpool': {
        'class': CNNAvgPool,
        'input_shape': (1, 1, 28, 28),
        'num_classes': 10,
        'dataset': 'mnist',
        'description': 'CNN with Conv2D+ReLU+AvgPool2D+FC',
    },
    'cnn_maxpool': {
        'class': CNNMaxPool,
        'input_shape': (1, 1, 28, 28),
        'num_classes': 10,
        'dataset': 'mnist',
        'description': 'CNN with Conv2D+ReLU+MaxPool2D+FC',
    },

    # Toy Models
    'toy_fc_4_3_2': {
        'class': ToyFC_4_3_2,
        'input_shape': (1, 4),
        'num_classes': 2,
        'dataset': 'toy',
        'description': 'Toy network: 4->3->2',
    },
    'toy_fc_8_4_2': {
        'class': ToyFC_8_4_2,
        'input_shape': (1, 8),
        'num_classes': 2,
        'dataset': 'toy',
        'description': 'Toy network: 8->4->2',
    },
}


def get_model(name: str) -> nn.Module:
    """Get a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]['class']()


def get_model_info(name: str) -> dict:
    """Get model metadata by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def list_models() -> list:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())
