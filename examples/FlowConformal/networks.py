"""
Synthetic neural networks for flow-conformal reachability experiments.

These networks are designed so that their output distributions under
uniform input perturbation have strong inter-dimensional correlations
and nonlinear structure, making the hyperrectangle demonstrably wasteful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatedBananaNet(nn.Module):
    """
    Network whose output under uniform [0,1]^2 input is a banana shape.

    Approximates the mapping:
        y1 = x1
        y2 = x1^2 + 0.3*x2

    The output traces a parabolic strip. The hyperrectangle bounding
    this strip has large empty corners where the parabola doesn't reach.

    Weights are trained at construction time (brief optimization).
    """

    def __init__(self, n_train_steps: int = 2000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self._init_banana_weights(n_train_steps)

    def _init_banana_weights(self, n_steps: int):
        """Train weights to approximate (x1, x2) -> (x1, x1^2 + 0.3*x2)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for _ in range(n_steps):
            x = torch.rand(1000, 2)
            y_target = torch.stack(
                [x[:, 0], x[:, 0] ** 2 + 0.3 * x[:, 1]], dim=1
            )
            loss = F.mse_loss(self.net(x), y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThreeBlobClassifier(nn.Module):
    """
    Small MLP classifier: R^2 -> R^3 (logits for 3 classes).

    Two hidden layers of 32 units each with ReLU activations. Trained
    on synthetic 3-Gaussian-blob data at construction time.

    The three blobs are centered at:
      class 0: (-2, 0)
      class 1: (2, 0)
      class 2: (0, 2)

    Each blob has std 0.7. This gives a classifier with a well-defined
    decision boundary and enough output-space structure to be
    meaningful as a verification target.
    """

    def __init__(self, n_train_steps: int = 2000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self._centers = torch.tensor([
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ])
        self._train_weights(n_train_steps)

    def _train_weights(self, n_steps: int):
        """Train on synthetic Gaussian blobs until accurate."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        torch.manual_seed(0)  # deterministic initialization
        for _ in range(n_steps):
            x, y = self._sample_batch(256)
            logits = self.net(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.eval()

    def _sample_batch(self, batch_size: int):
        """Sample a batch of (x, y) from the three blobs."""
        labels = torch.randint(0, 3, (batch_size,))
        centers = self._centers[labels]
        noise = torch.randn(batch_size, 2) * 0.7
        x = centers + noise
        return x, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample_data(self, n_samples: int, seed: int = None):
        """Sample labeled data for external use (test point selection)."""
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            labels = torch.randint(0, 3, (n_samples,), generator=gen)
            noise = torch.randn(n_samples, 2, generator=gen) * 0.7
        else:
            labels = torch.randint(0, 3, (n_samples,))
            noise = torch.randn(n_samples, 2) * 0.7
        centers = self._centers[labels]
        x = centers + noise
        return x, labels


class ThreeBlobClassifier3D(nn.Module):
    """
    Small MLP classifier: R^3 -> R^3 (logits for 3 classes).

    Two hidden layers of 16 units each with ReLU activations. Trained
    on synthetic uniform data labeled by nearest blob center at
    construction time.

    The three blobs are centered at:
      class 0: (1, 0, 0)
      class 1: (0, 1, 0)
      class 2: (0, 0, 1)

    Inputs are sampled uniformly from [-1, 1]^3 and labeled by the
    nearest blob center under Euclidean distance. Because the input
    space is genuinely 3D, the exact reach set (under a 3D input box)
    is a union of 3D polytopes with positive volume, enabling
    flow-volume vs. exact-volume comparisons (unlike the 2D version
    whose reach set is a measure-zero manifold in 3D).
    """

    def __init__(self, n_train_steps: int = 2000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        self.blob_centers = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self._train_weights(n_train_steps)

    def _train_weights(self, n_steps: int):
        """Train on synthetic uniform data with nearest-center labels."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        torch.manual_seed(0)  # deterministic initialization
        for _ in range(n_steps):
            x, y = self._sample_batch(256)
            logits = self.net(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.eval()

    def _sample_batch(self, batch_size: int):
        """Sample a batch of (x, y) uniformly in [-1, 1]^3 with nearest-center labels."""
        x = torch.rand(batch_size, 3) * 2.0 - 1.0
        # Pairwise squared distances to each blob center: (batch, 3)
        dists = torch.cdist(x, self.blob_centers)
        labels = dists.argmin(dim=1)
        return x, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample_data(self, n_samples: int, seed: int = None):
        """Sample labeled data for external use (test point selection)."""
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            x = torch.rand(n_samples, 3, generator=gen) * 2.0 - 1.0
        else:
            x = torch.rand(n_samples, 3) * 2.0 - 1.0
        dists = torch.cdist(x, self.blob_centers)
        labels = dists.argmin(dim=1)
        return x, labels
