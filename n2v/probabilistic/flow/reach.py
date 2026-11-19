"""Probabilistic reachability via flow-matching with conformal calibration.

Public API:
    flow_reach(model, input_box, config) -> ProbabilisticSet

The model is invoked as ``y = model(x)`` for sampled inputs. Both numpy
callables (any framework) and PyTorch ``nn.Module``s are accepted;
``nn.Module`` inputs are forwarded with device-awareness so GPU acceleration
is preserved.

This module is the *spec-agnostic* half of the flow-matching verification
pipeline: it trains a flow on whitened outputs and calibrates a conformal
threshold, returning a :class:`ProbabilisticSet`. Spec disjointness is
checked separately by
:func:`n2v.utils.verify_specification.verify_specification`.

Coordinate frame:
    The returned :class:`ProbabilisticSet` carries the per-dimension
    whitening (``AffineTransform`` with the empirical mean / std of
    training outputs) as ``affine_transform``. Downstream consumers
    (e.g. the spec-disjointness dispatch) use this transform to map
    raw-coordinate specs into the set's frame. See
    :mod:`n2v.probabilistic.flow.sets` for the convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.sampling import sample_box
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.sets import (
    AffineTransform, PCATransform, ProbabilisticSet,
)
from n2v.probabilistic.flow.train import _train_flow, _train_flow_tight
from n2v.sets import Box


def _make_pca(backend: str, n_components: int, seed: int):
    """Construct a PCA reducer (shared ``fit_transform`` / ``transform`` /
    ``inverse_transform`` / ``components_`` / ``mean_`` interface)."""
    if backend == 'randomized':
        from n2v.probabilistic.dimensionality import RandomizedSVDPCA
        return RandomizedSVDPCA(n_components=n_components, random_state=seed)
    if backend == 'deflation':
        from n2v.probabilistic.dimensionality import DeflationPCA
        return DeflationPCA(n_components=n_components)
    raise ValueError(
        f"pca_backend must be 'randomized' or 'deflation', got {backend!r}"
    )


@dataclass(frozen=True)
class FlowReachConfig:
    """Configuration for flow-matching probabilistic reachability.

    All fields have defaults that match the legacy
    ``_calibrate_flow_for_spec`` pipeline so seeded runs are bit-identical
    to pre-refactor results.

    Attributes:
        epsilon: Conformal-layer miscoverage level (a.k.a. α). The
            returned :class:`ProbabilisticSet` carries the guarantee
            ``Pr[f(x) in set] > 1 - epsilon`` with confidence determined
            by ``(m, ell)``. Must be in ``(0, 1)``. Default ``0.001``.

            *Not* the same as
            :attr:`ProbVerifyConfig.amls_bounded_eps_2_target` (the
            *verification-layer* η used by AMLS-bounded / raw-MC). The
            joint guarantee is
            ``epsilon_total = 1 - (1 - epsilon)(1 - eps_2_target)``;
            ``FlowReachConfig.epsilon`` controls only the conformal layer
            (the calibrated reach set), and
            ``ProbVerifyConfig.amls_bounded_eps_2_target`` controls only
            the verification layer (the spec-disjointness test).
        m: Calibration set size; larger ``m`` yields tighter confidence.
            Default ``8000``.
        ell: Rank parameter; the ``ell``-th smallest score is the
            calibrated threshold ``q``. ``None`` defaults to ``m - 1``.
            Must satisfy ``1 <= ell <= m``.

        n_train: Number of training samples drawn uniformly from the
            input box for flow training. Default ``10_000``.
        flow_epochs: Number of training epochs. Default ``5000``.
        flow_config: Training preset — ``'tight'`` (high-capacity, suits
            multimodal output distributions) or ``'base'`` (lighter).
            Default ``'tight'``.
        flow_coupling: OT coupling — ``'sinkhorn'`` (default),
            ``'hungarian'``, or ``'none'``.
        flow_use_ema: Use exponential moving average of the flow weights
            in the returned model. Default ``True``.

        infer_solver: ODE solver for inference (``'rk4'`` by default).
        infer_steps: Number of ODE steps for inference. Default ``30``.

        seed: Master seed; controls flow training and calibration
            sampling unless ``flow_seed`` / ``cal_seed`` override.
            Default ``0``.
        flow_seed: Optional override for the flow-training seed.
        cal_seed: Optional override for the calibration-sampling seed.

    Example:
        >>> # Default config (parity with legacy ACAS-Xu sweep):
        >>> cfg = FlowReachConfig()
        >>> # Custom: tighter coverage, larger calibration set:
        >>> cfg = FlowReachConfig(epsilon=0.0001, m=20_000)
    """

    # Conformal calibration
    epsilon: float = 0.001
    m: int = 8000
    ell: Optional[int] = None

    # Flow training
    n_train: int = 10_000
    flow_epochs: int = 5000
    flow_config: str = 'tight'
    flow_coupling: str = 'sinkhorn'
    flow_use_ema: bool = True

    # ODE inference
    infer_solver: str = 'rk4'
    infer_steps: int = 30

    # Output dimensionality reduction (for high-dim outputs, e.g. dense depth /
    # segmentation maps that far exceed the ~10s of dims the flow handles well).
    # When set, the flow is trained on the PCA-reduced output and the returned
    # ProbabilisticSet carries a PCATransform. Coverage is calibrated in the
    # reduced frame; a calibrated residual bound (see ``residual_bound`` on the
    # PCATransform) keeps downstream halfspace spec-disjointness sound for the
    # full-dimensional output. ``None`` disables PCA (original behaviour).
    pca_components: Optional[int] = None
    pca_backend: str = 'randomized'   # 'randomized' (SVD) or 'deflation'

    # Reproducibility
    seed: int = 0
    flow_seed: Optional[int] = None
    cal_seed: Optional[int] = None

    def __post_init__(self):
        # Default ell to m-1 if not provided.
        if self.ell is None:
            object.__setattr__(self, 'ell', self.m - 1)

        if not 0.0 < self.epsilon < 1.0:
            raise ValueError(
                f"FlowReachConfig.epsilon must be in (0, 1), got {self.epsilon}"
            )
        if self.m < 1:
            raise ValueError(f"FlowReachConfig.m must be >= 1, got {self.m}")
        if not 1 <= self.ell <= self.m:
            raise ValueError(
                f"FlowReachConfig.ell must be in [1, {self.m}], got {self.ell}"
            )
        if self.flow_config not in ('tight', 'base'):
            raise ValueError(
                f"FlowReachConfig.flow_config must be 'tight' or 'base', "
                f"got {self.flow_config!r}"
            )
        if self.flow_coupling not in ('sinkhorn', 'hungarian', 'none'):
            raise ValueError(
                f"FlowReachConfig.flow_coupling must be 'sinkhorn', "
                f"'hungarian', or 'none', got {self.flow_coupling!r}"
            )
        if self.n_train < 1:
            raise ValueError(
                f"FlowReachConfig.n_train must be >= 1, got {self.n_train}"
            )
        if self.flow_epochs < 1:
            raise ValueError(
                f"FlowReachConfig.flow_epochs must be >= 1, got {self.flow_epochs}"
            )
        if self.infer_steps < 1:
            raise ValueError(
                f"FlowReachConfig.infer_steps must be >= 1, got {self.infer_steps}"
            )
        if self.pca_components is not None and self.pca_components < 1:
            raise ValueError(
                f"FlowReachConfig.pca_components must be >= 1 or None, "
                f"got {self.pca_components}"
            )
        if self.pca_backend not in ('randomized', 'deflation'):
            raise ValueError(
                f"FlowReachConfig.pca_backend must be 'randomized' or "
                f"'deflation', got {self.pca_backend!r}"
            )


def _forward_for_pipeline(
    model: Union[Callable[[np.ndarray], np.ndarray], nn.Module],
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward ``x`` through ``model`` and return a torch tensor.

    For ``nn.Module``: device-aware (matches the model's device) so GPU
    acceleration on the forward pass is preserved.

    For arbitrary callables: numpy round-trip (``x`` is converted to
    numpy, the callable's output is converted back to a float32 tensor
    on CPU). No device acceleration; users that want GPU should provide
    an ``nn.Module``.
    """
    if isinstance(model, nn.Module):
        with torch.no_grad():
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                try:
                    target_device = next(model.buffers()).device
                except StopIteration:
                    target_device = torch.device('cpu')
            return model(x.to(target_device))
    # Arbitrary callable: numpy round-trip.
    x_np = x.detach().cpu().numpy()
    y_np = model(x_np)
    return torch.as_tensor(y_np, dtype=torch.float32)


def flow_reach(
    model: Union[Callable[[np.ndarray], np.ndarray], nn.Module],
    input_box: Box,
    config: Optional[FlowReachConfig] = None,
    **kwargs,
) -> ProbabilisticSet:
    """Probabilistic reachability via flow-matching + conformal calibration.

    Trains a flow-matching model on outputs drawn from ``input_box``,
    calibrates a conformal threshold, and returns the implicit
    probabilistic reach set ``{y_white : score(y_white) <= q}`` defined
    in whitened output coordinates.

    The returned :class:`ProbabilisticSet` carries the whitening
    transform as ``affine_transform`` so callers (e.g.
    :func:`n2v.utils.verify_specification.verify_specification`) can map
    raw-coordinate specs into the set's frame for disjointness checks.

    Args:
        model: Either a numpy callable (``y_np = model(x_np)`` — any
            framework) or a PyTorch ``nn.Module``. ``nn.Module`` keeps
            its device for the forward pass.
        input_box: :class:`n2v.sets.Box` defining the input region; the
            flow is trained on outputs of samples drawn uniformly from
            this box.
        config: :class:`FlowReachConfig` with hyperparameters. ``None``
            (the default) builds a config from any ``**kwargs`` passed,
            falling back to all-defaults when no kwargs are given. The
            two styles are mutually exclusive — passing both ``config=``
            and overlapping kwargs raises ``TypeError``.
        **kwargs: Equivalent to constructing ``config=FlowReachConfig(**kwargs)``.
            Allows the call-style ``flow_reach(model, box, epsilon=1e-4,
            seed=42)`` without an explicit config object.

    Returns:
        :class:`ProbabilisticSet` with:
        - ``score_fn``: calibrated flow score function.
        - ``threshold``: calibrated conformal threshold ``q``.
        - ``affine_transform``: per-dim whitening
          (``y_white = (y_raw - y_mean) / y_std``).
        - ``m``, ``ell``, ``epsilon``, ``dim``: calibration parameters.

    Reproducibility:
        Identical seeds + identical config produce bit-identical reach
        sets (modulo non-determinism in third-party libraries). The seed
        derivation matches the legacy ``_calibrate_flow_for_spec``:
        - flow training uses ``flow_seed`` (default ``seed``).
        - calibration sampling uses ``cal_seed + 1_000_000`` (default
          ``seed + 1_000_000``).
        - test sampling (diagnostic) uses ``cal_seed + 2_000_000``.

    Example:
        >>> from n2v.sets import Box
        >>> import numpy as np
        >>> box = Box(np.zeros(5), np.ones(5))
        >>> # With an nn.Module:
        >>> prob_set = flow_reach(my_torch_model, box,
        ...                       FlowReachConfig(seed=47))
        >>> # With any callable (numpy-in, numpy-out):
        >>> prob_set = flow_reach(lambda x: my_jax_apply(params, x), box,
        ...                       FlowReachConfig(seed=47))
        >>> # Bare kwargs (no explicit config object):
        >>> prob_set = flow_reach(my_torch_model, box, seed=47, epsilon=1e-4)

    See Also:
        :meth:`n2v.nn.NeuralNetwork.reach` — the OO entry point that
        wraps this function. Use it when you already have a
        ``NeuralNetwork`` instance and want to share the dispatch
        surface with sound reach methods:

            net = NeuralNetwork(model)
            prob_set = net.reach(box, method='flow_matching',
                                 config=FlowReachConfig(seed=47))

        The two surfaces are equivalent and produce identical results
        on the same ``(model, box, config)``; choose based on whether
        your callsite is OO-style or free-function-style.
    """
    if config is not None and kwargs:
        raise TypeError(
            "pass either config= or FlowReachConfig kwargs, not both "
            f"(got config={type(config).__name__} and kwargs={list(kwargs)})"
        )
    if config is None:
        config = FlowReachConfig(**kwargs)

    flow_seed = (config.flow_seed
                 if config.flow_seed is not None else config.seed)
    cal_seed = (config.cal_seed
                if config.cal_seed is not None else config.seed)

    # Sample input box. Seed structure matches _calibrate_flow_for_spec
    # exactly for parity. ``Box`` stores ``lb``/``ub`` as column vectors
    # ``(dim, 1)``; flatten to ``(dim,)`` for ``sample_box``.
    lb_arr = np.asarray(input_box.lb).reshape(-1).astype(np.float32)
    ub_arr = np.asarray(input_box.ub).reshape(-1).astype(np.float32)
    lb_t = torch.as_tensor(lb_arr, dtype=torch.float32)
    ub_t = torch.as_tensor(ub_arr, dtype=torch.float32)
    x_tr = sample_box(lb_t, ub_t, n_samples=config.n_train, seed=flow_seed)
    x_ca = sample_box(lb_t, ub_t, n_samples=config.m,
                      seed=cal_seed + 1_000_000)
    x_te = sample_box(lb_t, ub_t, n_samples=2_000,
                      seed=cal_seed + 2_000_000)

    y_tr = _forward_for_pipeline(model, x_tr)
    y_ca = _forward_for_pipeline(model, x_ca)
    y_te = _forward_for_pipeline(model, x_te)

    # Per-dim whitening from training outputs (matches legacy).
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std

    raw_output_dim = y_tr_w.shape[1]

    # Optional PCA reduction of the (whitened) output. When enabled, the flow is
    # trained on the reduced coordinates z = (y_white - pca_mean) @ componentsᵀ.
    # Coverage is calibrated in this reduced frame; we additionally calibrate a
    # conformal upper bound on the dropped whitened residual norm and stash it in
    # the PCATransform, so that halfspace spec-disjointness done in the reduced
    # frame (with the transform's residual inflation) stays sound for the full
    # output. See PCATransform / verify_specification.
    use_pca = (config.pca_components is not None
               and config.pca_components < raw_output_dim)
    pca = None
    residual_bound = 0.0
    if use_pca:
        ytrw_np = y_tr_w.detach().cpu().numpy()
        ycaw_np = y_ca_w.detach().cpu().numpy()
        pca = _make_pca(config.pca_backend, config.pca_components, flow_seed)
        z_tr_np = pca.fit_transform(ytrw_np)              # (n_train, k)
        z_ca_np = pca.transform(ycaw_np)                  # (m, k)
        # Whitened reconstruction residual on calibration -> conformal bound.
        # We bound the ℓ∞ (max whitened component) residual: paired with an ℓ1
        # row-norm in PCATransform.transform_halfspace (Hölder), this is much
        # tighter than an ℓ2 bound for the sparse (single-cell) halfspaces that
        # spatial per-region specs use, while staying sound for any halfspace.
        recon_ca = pca.inverse_transform(z_ca_np)         # (m, D)
        resid_inf = np.abs(ycaw_np - recon_ca).max(axis=1)  # (m,) whitened ℓ∞
        residual_bound = float(
            calibrate(torch.as_tensor(resid_inf, dtype=torch.float32),
                      config.ell).item()
        )
        y_tr_w = torch.as_tensor(z_tr_np, dtype=torch.float32)
        y_ca_w = torch.as_tensor(z_ca_np, dtype=torch.float32)

    output_dim = y_tr_w.shape[1]

    # Train the flow. ``internal_standardize=False`` because we pre-whiten.
    if config.flow_config == 'base':
        flow, _losses = _train_flow(
            y_tr_w, output_dim, config.flow_epochs, flow_seed,
            internal_standardize=False,
            return_losses=True,
            coupling=config.flow_coupling,
            use_ema=config.flow_use_ema,
        )
    else:  # 'tight'
        flow, _losses = _train_flow_tight(
            y_tr_w, output_dim, config.flow_epochs, flow_seed,
            internal_standardize=False,
            return_losses=True,
            coupling=config.flow_coupling,
            use_ema=config.flow_use_ema,
        )
    flow = flow.to('cpu').eval()

    base_score_fn = FlowScore(
        flow, t=1.0, n_steps=config.infer_steps,
        method=config.infer_solver, batch_size=65536,
    )

    # Conformal calibration.
    calib_scores = base_score_fn(y_ca_w)
    q = calibrate(calib_scores, config.ell).item()

    # Coordinate transform consumed by verify_specification's probabilistic
    # dispatch to map raw-coordinate specs into the set's frame. Diagonal
    # whitening without PCA; a whiten-then-project PCATransform (carrying the
    # calibrated residual bound for sound spec inflation) with PCA.
    mean_np = y_mean.detach().cpu().numpy()
    std_np = y_std.detach().cpu().numpy()
    if use_pca:
        transform = PCATransform(
            mean=mean_np,
            std=std_np,
            components=pca.components_,
            pca_mean=pca.mean_,
            residual_bound=residual_bound,
        )
    else:
        transform = AffineTransform(mean=mean_np, std=std_np)

    # Quiet "y_te unused" — kept for parity with the legacy pipeline so
    # the calibration RNG sequence advances by exactly the same number of
    # draws. The diagnostic coverage_empirical that the legacy pipeline
    # computed from y_te can be reproduced by the caller via
    # ``ProbabilisticSet.contains`` if needed.
    _ = y_te

    return ProbabilisticSet(
        score_fn=base_score_fn,
        threshold=q,
        m=config.m,
        ell=config.ell,
        epsilon=config.epsilon,
        dim=output_dim,
        affine_transform=transform,
    )
