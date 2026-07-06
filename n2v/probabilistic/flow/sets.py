"""
Probabilistic set representation for flow-based conformal reachability.

Represents the implicit set ``{y : score(y) <= threshold}`` with a
probabilistic coverage guarantee from conformal inference.

Coordinate-frame convention
---------------------------

Flow-matching reachability trains the flow on *whitened* outputs to keep
the source-to-target bridge well-conditioned (see the discussion in
``n2v.probabilistic.flow.reach`` for why this matters numerically).
Consequently the calibrated level set ``{y_white : score(y_white) <= q}``
is defined in whitened coordinates, not the raw output coordinates that
specs are written in.

To bridge the two frames, :class:`ProbabilisticSet` optionally carries
an :class:`AffineTransform` describing the per-dimension whitening:
``y_white = (y_raw - mean) / std``. Callers that work in raw output
coordinates (e.g. the spec-disjointness check in
:func:`n2v.utils.verify_specification.verify_specification`) use the
transform to map raw inputs into the frame the set lives in. When
``affine_transform`` is ``None`` the set is already in raw coordinates;
this is the case for synthetic-network examples where no whitening was
applied during training.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch

from n2v.probabilistic.flow.scores import NonconformityScore
from n2v.probabilistic.flow.calibrate import compute_guarantee
from n2v.sets.halfspace import HalfSpace


@dataclass(frozen=True)
class AffineTransform:
    """Per-dimension shift+scale transform for output-space whitening.

    Encodes the coordinate transform ``y_white = (y_raw - mean) / std``
    used by :class:`ProbabilisticSet` to bridge raw network outputs and
    the whitened frame the flow was trained in. Both arrays are flat 1-D
    of length equal to the network output dimensionality.

    Attributes:
        mean: Per-dimension shift; the empirical mean of training outputs.
            Shape ``(d,)``.
        std: Per-dimension scale; the empirical std of training outputs.
            Shape ``(d,)``. All entries strictly positive (the constructor
            rejects zero/negative entries — degenerate dimensions should
            be flattened upstream, not silently divided by epsilon).

    Example:
        >>> at = AffineTransform(mean=np.zeros(3), std=np.ones(3))
        >>> at.whiten(np.array([1., 2., 3.]))      # identity
        array([1., 2., 3.])
        >>> at = AffineTransform(mean=np.array([0., 1., 0.]),
        ...                      std=np.array([1., 2., 1.]))
        >>> at.unwhiten(at.whiten(np.array([5., 7., 9.])))   # round-trip
        array([5., 7., 9.])
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64).flatten()
        std = np.asarray(self.std, dtype=np.float64).flatten()
        if mean.shape != std.shape:
            raise ValueError(
                f"AffineTransform: mean.shape ({mean.shape}) and "
                f"std.shape ({std.shape}) must match"
            )
        if mean.ndim != 1:
            raise ValueError(
                f"AffineTransform: mean and std must be 1-D, got shape "
                f"{mean.shape}"
            )
        if not np.all(std > 0):
            raise ValueError(
                "AffineTransform: std must be strictly positive in every "
                "dimension (degenerate dims should be flattened upstream)"
            )
        # Replace with the normalised float64 1-D versions.
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'std', std)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def whiten(self, y_raw: np.ndarray) -> np.ndarray:
        """Map raw outputs to whitened coordinates: ``(y_raw - mean) / std``.

        Accepts either a single point ``(d,)`` or a batch ``(N, d)``.
        Returns the same shape.
        """
        return (np.asarray(y_raw) - self.mean) / self.std

    def unwhiten(self, y_white: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`whiten`: ``y_white * std + mean``."""
        return np.asarray(y_white) * self.std + self.mean

    def transform_halfspace(self, hs: HalfSpace) -> HalfSpace:
        """Express a halfspace ``G·y_raw <= g`` in whitened coordinates.

        Substituting ``y_raw = mean + std * y_white`` into ``G·y_raw <= g``
        gives ``(G * std)·y_white <= g - G·mean``. Returns a new
        :class:`HalfSpace` carrying the transformed ``(G, g)`` pair.

        The transformed halfspace describes the *same physical set* of
        points as the input — just in the coordinates the flow operates
        in. Disjointness with the calibrated conformal level set is
        equivalent in either frame.

        Soundness: this is an exact linear coordinate change, no
        approximation. The whitening matrix ``diag(std)`` is full rank
        (``std > 0`` by ``__post_init__``).
        """
        sigma = self.std
        mu = self.mean
        G_white = hs.G * sigma[None, :]                 # row-wise scale
        g_white = hs.g.flatten() - hs.G @ mu             # shape (n_rows,)
        return HalfSpace(G_white, g_white.reshape(-1, 1))


@dataclass(frozen=True)
class PCATransform:
    """Whiten-then-project transform for PCA-reduced flow reachability.

    Encodes the map from raw network outputs to the reduced coordinate frame the
    flow was trained in, composing per-dimension whitening with an orthonormal
    PCA projection::

        white = (y_raw - mean) / std                 # per-dim whitening
        z     = (white - pca_mean) @ components.T    # project to k dims

    Drop-in replacement for :class:`AffineTransform` in
    :class:`ProbabilisticSet` (same ``whiten`` / ``unwhiten`` /
    ``transform_halfspace`` surface), used when ``flow_reach`` is run with PCA on
    the output. Because the projection is lossy, ``transform_halfspace`` adds a
    **residual inflation** so that spec-disjointness certified in the reduced
    frame remains sound for the true full-dimensional output.

    Attributes:
        mean: Per-dim whitening shift (train-output mean), shape ``(D,)``.
        std: Per-dim whitening scale (train-output std, > 0), shape ``(D,)``.
        components: Orthonormal PCA directions (rows), shape ``(k, D)``.
        pca_mean: PCA mean in whitened space, shape ``(D,)``.
        residual_bound: Calibrated upper bound (whitened **ℓ∞** — max component)
            on the dropped PCA residual ``white - pca_mean - components.T @ z``.
            ``0.0`` gives the exact linear map (no inflation); ``flow_reach``
            sets it from a conformal quantile of the calibration residuals so
            disjointness is sound. Paired with an ℓ1 row-norm in
            :meth:`transform_halfspace` (Hölder ℓ1–ℓ∞). Default ``0.0``.
    """

    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray
    pca_mean: np.ndarray
    residual_bound: float = 0.0

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64).flatten()
        std = np.asarray(self.std, dtype=np.float64).flatten()
        comp = np.asarray(self.components, dtype=np.float64)
        pca_mean = np.asarray(self.pca_mean, dtype=np.float64).flatten()
        if comp.ndim != 2:
            raise ValueError(
                f"PCATransform: components must be 2-D (k, D), got {comp.shape}"
            )
        D = comp.shape[1]
        if not (mean.shape == std.shape == pca_mean.shape == (D,)):
            raise ValueError(
                "PCATransform: mean/std/pca_mean must all be shape (D,) matching "
                f"components' second axis ({D}); got {mean.shape}, {std.shape}, "
                f"{pca_mean.shape}"
            )
        if not np.all(std > 0):
            raise ValueError("PCATransform: std must be strictly positive")
        if self.residual_bound < 0:
            raise ValueError("PCATransform: residual_bound must be >= 0")
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'std', std)
        object.__setattr__(self, 'components', comp)
        object.__setattr__(self, 'pca_mean', pca_mean)

    @property
    def dim(self) -> int:
        """Reduced (latent) dimensionality k."""
        return int(self.components.shape[0])

    @property
    def full_dim(self) -> int:
        """Full output dimensionality D."""
        return int(self.components.shape[1])

    def whiten(self, y_raw: np.ndarray) -> np.ndarray:
        """Map raw outputs ``(…, D)`` to reduced coordinates ``(…, k)``."""
        white = (np.asarray(y_raw) - self.mean) / self.std
        return (white - self.pca_mean) @ self.components.T

    def unwhiten(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct raw outputs ``(…, D)`` from reduced ``(…, k)`` (lossy)."""
        white = np.asarray(z) @ self.components + self.pca_mean
        return white * self.std + self.mean

    def residual_norm(self, y_raw: np.ndarray) -> np.ndarray:
        """Whitened ℓ∞ reconstruction residual per sample, shape ``(…,)``.

        ``max_i |white_i - (pca_mean + componentsᵀ z)_i|`` — the quantity the
        calibrated :attr:`residual_bound` upper-bounds. Useful for validating
        the joint (projection ∈ set AND residual ≤ bound) coverage of the
        full-output reach set.
        """
        white = (np.asarray(y_raw) - self.mean) / self.std
        z = (white - self.pca_mean) @ self.components.T
        recon = z @ self.components + self.pca_mean
        return np.abs(white - recon).max(axis=-1)

    def transform_halfspace(self, hs: HalfSpace) -> HalfSpace:
        """Express raw-space ``G·y_raw <= g`` in the reduced frame, soundly.

        Substituting ``y_raw = mean + std ⊙ (pca_mean + componentsᵀ z + r)``
        with whitened residual ``‖r‖_∞ <= residual_bound`` into ``G·y_raw <= g``:

            (G ⊙ std) componentsᵀ · z  <=  g − G·(mean + std ⊙ pca_mean)
                                              + residual_bound · ‖G ⊙ std‖₁  (row-wise)

        The residual term bounds ``|(G⊙std)·r| ≤ ‖G⊙std‖₁ · ‖r‖_∞`` (Hölder), and
        **enlarges** the unsafe region in the reduced frame by that worst-case
        residual contribution, so certifying the reach set disjoint from this
        inflated halfspace implies disjointness from the true full-dimensional
        unsafe set (conservative → sound). With ``residual_bound = 0`` this is
        the exact linear coordinate change.
        """
        G = np.asarray(hs.G, dtype=np.float64)              # (n_rows, D)
        g = np.asarray(hs.g, dtype=np.float64).flatten()    # (n_rows,)
        Gstd = G * self.std[None, :]                        # (n_rows, D)
        G_red = Gstd @ self.components.T                    # (n_rows, k)
        offset = G @ (self.mean + self.std * self.pca_mean)  # (n_rows,)
        inflate = self.residual_bound * np.abs(Gstd).sum(axis=1)  # ℓ1 row-norm
        g_red = g - offset + inflate
        return HalfSpace(G_red, g_red.reshape(-1, 1))


class ProbabilisticSet:
    """Implicit set ``{y : score(y) <= threshold}`` with a probabilistic guarantee.

    Works with any :class:`NonconformityScore`, allowing direct comparison
    of hyperrectangular, ellipsoidal, ball, and flow-based reach sets.

    Coverage guarantee (conformal inference):
        ``Pr[ Pr[f(x) in this set] > 1 - epsilon ] > confidence``

    Coordinate-frame contract:
        The set is defined in the coordinates ``score_fn`` operates in.
        For flow-based sets that's the *whitened* output space (see the
        module-level ``Coordinate-frame convention`` docstring above).
        Raw-coordinate inputs (e.g. user-provided VNN-LIB spec
        halfspaces) must be transformed via :attr:`affine_transform`
        before membership / disjointness tests.

        If ``affine_transform`` is ``None`` the set is in raw
        coordinates and no transformation is needed (typical for
        synthetic test networks that don't need whitening).

    Args:
        score_fn: :class:`NonconformityScore` instance defining the
            implicit set's score function.
        threshold: Calibrated threshold ``q`` from conformal inference.
        m: Calibration set size.
        ell: Rank parameter (``ell``-th smallest score is used as ``q``).
        epsilon: Miscoverage level — the guarantee is
            ``Pr[f(x) in set] > 1 - epsilon`` with confidence below.
        dim: Output dimensionality.
        affine_transform: Optional :class:`AffineTransform` describing
            the per-dim shift+scale relating raw network outputs to the
            coordinate frame the set lives in. ``None`` means the set
            is already in raw coordinates. Default ``None`` for backwards
            compatibility with callers that don't whiten.
    """

    def __init__(
        self,
        score_fn: NonconformityScore,
        threshold: float,
        m: int,
        ell: int,
        epsilon: float,
        dim: int,
        affine_transform: Optional[AffineTransform] = None,
    ):
        self.score_fn = score_fn
        self.threshold = threshold
        self.m = m
        self.ell = ell
        self.epsilon = epsilon
        self.dim = dim
        self.affine_transform = affine_transform

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

    def estimate_coverage(
        self,
        model,
        input_box,
        n_test: int = 2000,
        seed: Optional[int] = None,
    ) -> float:
        """Empirical coverage on uniform samples from ``input_box``.

        Samples ``n_test`` points uniformly from ``input_box``, forwards
        them through ``model`` to get outputs, applies the set's
        ``affine_transform`` (when present) to bring outputs into the
        set's coordinate frame, and returns the fraction inside the set
        (i.e. the fraction of test samples ``y`` satisfying
        ``score(y_whitened) <= threshold``).

        This is a *diagnostic* — it does not produce the conformal
        guarantee (which is fixed by the calibration parameters
        ``(m, ell, epsilon)``). It's useful for sanity-checking that
        the calibrated set behaves as expected on a held-out sample.

        Args:
            model: A numpy callable (``y_np = model(x_np)``, any
                framework) or a PyTorch ``nn.Module`` (device-aware
                forward).
            input_box: :class:`n2v.sets.Box` to sample inputs from
                uniformly.
            n_test: Number of test samples. Default ``2000``.
            seed: RNG seed for the sample. Default ``None`` (uses 0).

        Returns:
            Fraction (in ``[0, 1]``) of test samples whose output lies
            in the probabilistic set.
        """
        import torch.nn as nn
        from n2v.probabilistic.flow.sampling import sample_box

        # Sample uniformly from input_box. ``Box`` stores lb/ub as
        # column vectors; flatten to 1-D for ``sample_box``.
        lb_arr = np.asarray(input_box.lb).reshape(-1).astype(np.float32)
        ub_arr = np.asarray(input_box.ub).reshape(-1).astype(np.float32)
        lb_t = torch.as_tensor(lb_arr, dtype=torch.float32)
        ub_t = torch.as_tensor(ub_arr, dtype=torch.float32)
        x_te = sample_box(lb_t, ub_t, n_samples=n_test,
                          seed=0 if seed is None else seed)

        # Forward through model (device-aware for nn.Module, numpy
        # round-trip for arbitrary callable).
        if isinstance(model, nn.Module):
            with torch.no_grad():
                try:
                    target_device = next(model.parameters()).device
                except StopIteration:
                    try:
                        target_device = next(model.buffers()).device
                    except StopIteration:
                        target_device = torch.device('cpu')
                y_te = model(x_te.to(target_device))
        else:
            y_np = model(x_te.detach().cpu().numpy())
            y_te = torch.as_tensor(y_np, dtype=torch.float32)

        # Bring into the set's coordinate frame. Use the transform's own
        # ``whiten`` so this works for both the diagonal AffineTransform and the
        # whiten-then-project PCATransform (which maps to the reduced frame).
        if self.affine_transform is not None:
            z_np = self.affine_transform.whiten(y_te.detach().cpu().numpy())
            y_te = torch.as_tensor(np.asarray(z_np), dtype=torch.float32)

        return self.contains(y_te).float().mean().item()

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
