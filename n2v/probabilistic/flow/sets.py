"""
Probabilistic set representation for flow-based conformal reachability.

Represents the implicit set {y : score(y) <= threshold} with a
probabilistic coverage guarantee from conformal inference.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional

from n2v.probabilistic.flow.scores import NonconformityScore
from n2v.probabilistic.flow.calibrate import compute_guarantee


class ProbabilisticSet:
    """
    Implicit set {y : score(y) <= threshold} with probabilistic guarantee.

    Works with any NonconformityScore, allowing direct comparison of
    hyperrectangular, ellipsoidal, ball, and flow-based reach sets.

    Guarantee: Pr[Pr[f(x) in this set] > 1-epsilon] > confidence

    Args:
        score_fn: NonconformityScore instance.
        threshold: Calibrated threshold q.
        m: Calibration set size.
        ell: Rank parameter.
        epsilon: Miscoverage level.
        dim: Output dimensionality.
    """

    def __init__(
        self,
        score_fn: NonconformityScore,
        threshold: float,
        m: int,
        ell: int,
        epsilon: float,
        dim: int,
    ):
        self.score_fn = score_fn
        self.threshold = threshold
        self.m = m
        self.ell = ell
        self.epsilon = epsilon
        self.dim = dim

        coverage, confidence = compute_guarantee(m, ell, epsilon)
        self.coverage = coverage
        self.confidence = confidence

    def contains(self, y: torch.Tensor) -> torch.Tensor:
        """
        Check membership: is score(y) <= threshold?

        Args:
            y: (batch, d) tensor of points.

        Returns:
            (batch,) boolean tensor.
        """
        scores = self.score_fn(y)
        return scores <= self.threshold

    def estimate_volume(
        self,
        n_samples: int = 1_000_000,
        bounding_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[float, float]:
        """
        Estimate volume of the set via Monte Carlo sampling.

        Args:
            n_samples: Number of MC samples.
            bounding_box: (low, high) tensors defining the sampling region.
                If None, uses a heuristic based on the threshold.

        Returns:
            (volume_estimate, standard_error)
        """
        if bounding_box is not None:
            low, high = bounding_box
            samples = (
                torch.rand(n_samples, self.dim) * (high - low) + low
            )
            sampling_volume = (high - low).prod().item()
        else:
            radius = self.threshold * 3.0
            samples = (
                torch.rand(n_samples, self.dim) * 2 * radius - radius
            )
            sampling_volume = (2 * radius) ** self.dim

        with torch.no_grad():
            inside = self.contains(samples).float()

        frac = inside.mean().item()
        volume = frac * sampling_volume
        std_err = inside.std().item() / np.sqrt(n_samples) * sampling_volume

        return volume, std_err

    def boundary_2d(
        self,
        resolution: int = 200,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[np.ndarray]:
        """
        Extract the 2D boundary contour of the set.

        Evaluates the score on a grid and extracts the contour
        at the threshold level.

        Args:
            resolution: Grid resolution per dimension.
            bounds: (low, high) tensors for the grid extent.
                If None, uses a heuristic.

        Returns:
            List of (N, 2) numpy arrays, each a contour path.

        Raises:
            ValueError: If dim != 2.
        """
        if self.dim != 2:
            raise ValueError(
                f"boundary_2d only works for dim=2, got dim={self.dim}"
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if bounds is not None:
            low, high = bounds
            x_min, y_min = low[0].item(), low[1].item()
            x_max, y_max = high[0].item(), high[1].item()
        else:
            r = self.threshold * 3.0
            x_min, y_min = -r, -r
            x_max, y_max = r, r

        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(xs, ys)
        grid_points = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1),
            dtype=torch.float32,
        )

        with torch.no_grad():
            scores = self.score_fn(grid_points).numpy()

        zz = scores.reshape(resolution, resolution)

        # Use matplotlib contour to extract paths
        fig, ax = plt.subplots()
        cs = ax.contour(xx, yy, zz, levels=[self.threshold])

        paths = []
        for seg_list in cs.allsegs:
            for seg in seg_list:
                arr = np.asarray(seg)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    paths.append(arr)

        plt.close(fig)

        return paths

    def get_guarantee(self) -> Tuple[float, float]:
        """
        Get the probabilistic guarantee.

        Returns:
            (coverage, confidence) where coverage = 1-epsilon.
        """
        return (self.coverage, self.confidence)

    def __repr__(self) -> str:
        return (
            f"ProbabilisticSet(dim={self.dim}, "
            f"threshold={self.threshold:.4f}, "
            f"coverage={self.coverage:.4f}, "
            f"confidence={self.confidence:.4f})"
        )
