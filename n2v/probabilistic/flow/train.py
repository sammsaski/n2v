"""
OT-CFM training loop for flow matching.

Trains a velocity field to transport data toward a standard Gaussian.
Pairs (x0 ~ N(0,I), x1 = data), interpolates x_t = (1-t)*x0 + t*x1,
and regresses v_theta(t, x_t) against the target velocity x1 - x0.

Supports two OT coupling methods:
  - Hungarian (exact, O(n^3), CPU-only via scipy)
  - Sinkhorn (approximate, GPU-friendly, pure tensor ops)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union, Callable

from n2v.probabilistic.flow.model import VelocityField


def compute_adaptive_sinkhorn_reg(
    training_outputs: torch.Tensor,
    alpha: float = 0.1,
    n_probe: int = 256,
    floor: float = 1e-6,
) -> float:
    """
    Compute an adaptive Sinkhorn regularization strength based on data scale.

    The Sinkhorn kernel is K_ij = exp(-||x0_i - x1_j||^2 / reg). For numerical
    stability in float32 we need cost/reg to stay below roughly 50-80, so K
    doesn't underflow. This helper probes the actual cost by sampling a batch
    of Gaussian noise and computing pairwise squared distances against the
    training data — this matches the cost Sinkhorn actually sees during
    training, making the resulting reg numerically stable at any data scale.

    Args:
        training_outputs: (n, d) tensor of training samples.
        alpha: Proportionality constant. Smaller alpha = sharper coupling
            but closer to numerical underflow. Default 0.1 (tuned via the
            alpha sweep in docs/audits/2026-04-14-adaptive-sinkhorn-alpha-sweep.md).
        n_probe: Number of probe samples for the distance estimation.
        floor: Minimum reg value to prevent returning zero on degenerate data.

    Returns:
        A scalar float suitable for passing as `reg` to sinkhorn_coupling.
    """
    with torch.no_grad():
        n = training_outputs.shape[0]
        probe_data = training_outputs[: min(n_probe, n)]
        probe_noise = torch.randn_like(probe_data)
        probe_cost_sq = (
            torch.cdist(probe_noise, probe_data, p=2) ** 2
        ).median().item()
    return max(floor, probe_cost_sq * alpha)


def ot_coupling(
    x0: torch.Tensor, x1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Hungarian algorithm.

    Finds the permutation of (x0, x1) that minimizes total L2 cost.
    Requires CPU (uses scipy). O(n^3) per batch.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.

    Returns:
        (x0_permuted, x1_permuted) with optimal coupling.
    """
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x0, x1, p=2).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return x0[row_ind], x1[col_ind]


def sinkhorn_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float,
    max_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Sinkhorn algorithm.

    Approximates optimal transport via entropic regularization.
    Pure tensor ops — works on CPU and GPU. O(n^2) per iteration.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.
        reg: Entropic regularization strength (smaller = closer to exact OT).
        max_iters: Number of Sinkhorn iterations.

    Returns:
        (x0_permuted, x1_permuted) with approximate optimal coupling.
    """
    with torch.no_grad():
        cost = torch.cdist(x0, x1, p=2)
        K = torch.exp(-cost / reg)

        # Sinkhorn iterations (row/column normalization)
        u = torch.ones(x0.shape[0], device=x0.device)
        for _ in range(max_iters):
            v = 1.0 / (K.T @ u + 1e-8)
            u = 1.0 / (K @ v + 1e-8)

        # Transport plan
        plan = torch.diag(u) @ K @ torch.diag(v)

        # Extract hard assignment via argmax per row
        col_ind = plan.argmax(dim=1)
        row_ind = torch.arange(x0.shape[0], device=x0.device)

    return x0[row_ind], x1[col_ind]


def train_flow(
    velocity_field: VelocityField,
    training_outputs: torch.Tensor,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    coupling: str = 'hungarian',
    sinkhorn_reg: Union[float, str] = 'auto',
    sinkhorn_iters: int = 50,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    optimizer: str = 'adam',
    weight_decay: float = 0.0,
    lr_warmup_frac: float = 0.0,
    grad_clip: Optional[float] = None,
    time_sampling: str = 'uniform',
    coupling_batch_size: Optional[int] = None,
    standardize_outputs: bool = False,
    refresh_data_each_epoch: Optional[Callable[[], torch.Tensor]] = None,
    fixed_noise: Optional[torch.Tensor] = None,
    compile: bool = False,
) -> Tuple[VelocityField, List[float]]:
    """
    Train the velocity field using OT-CFM.

    Args:
        velocity_field: VelocityField module to train.
        training_outputs: (n, d) tensor of training data points.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        coupling: OT coupling method. One of:
            'none' — random pairing (no OT)
            'hungarian' — exact OT via Hungarian algorithm (CPU-only)
            'sinkhorn' — approximate OT via Sinkhorn (GPU-friendly)
        sinkhorn_reg: Entropic regularization strength for Sinkhorn coupling.
            If ``'auto'`` (the default), it is computed adaptively from the
            training data via :func:`compute_adaptive_sinkhorn_reg`, which
            scales ``reg`` with the data's typical cost magnitude so the
            Sinkhorn kernel stays numerically stable regardless of output
            scale. Pass a numeric value to override (e.g. ``0.05`` to
            reproduce the pre-2026-04-14 hardcoded behavior). Smaller values
            are closer to exact OT but numerically less stable. Ignored for
            non-sinkhorn couplings.
        sinkhorn_iters: Number of Sinkhorn iterations when
            ``coupling='sinkhorn'``. Ignored for other coupling methods.
            Defaults to 50 (the previously hardcoded value).
        use_ema: If True, maintain an exponential moving average of the
            model parameters during training and copy the EMA weights into
            the returned model at the end. Has no effect on the in-loop
            training dynamics — only the final returned weights change.
            Defaults to False (byte-identical to pre-change behavior).
        ema_decay: EMA decay factor. Each step,
            ``ema = ema_decay * ema + (1 - ema_decay) * current``. Ignored
            when ``use_ema=False``. Defaults to 0.999.
        optimizer: Optimizer to use. One of ``'adam'`` (default) or
            ``'adamw'``. Defaults to ``'adam'`` (byte-identical to
            pre-change behavior).
        weight_decay: Weight decay coefficient passed to ``AdamW`` when
            ``optimizer='adamw'``. Ignored for ``'adam'``. Defaults to 0.0.
        lr_warmup_frac: Fraction of ``n_epochs`` used as a linear LR warmup
            prefix. When ``> 0``, the LR linearly ramps from near-0 to ``lr``
            over the first ``lr_warmup_frac * n_epochs`` epochs, then cosine
            decays from ``lr`` to 0 over the remainder, via
            ``SequentialLR(LinearLR, CosineAnnealingLR)``. When ``0.0`` (the
            default), the existing pure cosine schedule is preserved
            byte-identically.
        grad_clip: If not None, clip the gradient norm of the velocity
            field parameters to this maximum value after ``loss.backward()``
            and before ``opt.step()`` via
            ``torch.nn.utils.clip_grad_norm_``. When ``None`` (the default),
            no clipping is applied and behavior is byte-identical to the
            pre-change path.
        time_sampling: How to sample the interpolation time ``t`` for each
            batch. One of ``'uniform'`` (default; ``t ~ U(0, 1)``) or
            ``'logit_normal'`` (``u ~ N(0, 1)``, ``t = sigmoid(u)``, which
            concentrates ``t`` around ``0.5`` with tails toward ``0`` and
            ``1``). The default preserves byte-identical behavior with the
            pre-change path.
        coupling_batch_size: If not None, the DataLoader is configured with
            this batch size and each large coupled batch is sliced into
            ``batch_size``-sized minibatches for gradient steps. OT coupling
            is computed once per large batch (not per optimizer step), which
            yields higher-quality coupling for the same number of optimizer
            updates. Must be a multiple of ``batch_size``. When ``None`` (the
            default), coupling runs on each optimizer batch and behavior is
            byte-identical to the pre-change path.
        standardize_outputs: If True, compute per-dimension mean/std of
            ``training_outputs`` once, whiten the training data in-place to
            unit variance, train on the whitened data, and then register
            ``y_mean`` / ``y_std`` as buffers on ``velocity_field`` so that
            ``forward(t, y)`` whitens incoming ``y`` and de-whitens its
            output velocity. This keeps inference-time callers in the
            original output space. The buffers are deliberately set
            **after** training (not before), otherwise the forward pass
            during training would double-whiten the already-whitened data.
            When ``False`` (the default), no buffers are registered and
            behavior is byte-identical to the pre-change path.
        refresh_data_each_epoch: Optional zero-arg callable returning a
            fresh ``(n, d)`` training tensor. When provided, the callable is
            invoked at the start of every epoch (including epoch 0) and the
            DataLoader is rebuilt from the returned tensor, enabling
            "fresh pushforward samples per epoch" workflows. The initial
            ``training_outputs`` passed to this function is ignored for
            training in that case (but is still used to compute the
            whitening statistics when ``standardize_outputs=True``). Fresh
            data is whitened using the *initial* statistics — they are not
            recomputed per epoch. When ``None`` (the default), the
            DataLoader is built once before the loop and behavior is
            byte-identical to the pre-change path.
        fixed_noise: Optional ``(n, d)`` tensor of pre-coupled Gaussian
            source samples paired row-wise with ``training_outputs``. When
            provided, the training loop uses ``fixed_noise[i]`` as the
            ``x_0`` paired with ``training_outputs[i]`` (rather than
            sampling fresh ``randn_like`` noise each batch). This is the
            entry point for ReFlow: the caller integrates the ODE to obtain
            matched ``(x_0, x_1)`` pairs and passes them in so training
            respects the coupling. Requires ``coupling='none'`` (since any
            other coupling would re-permute the pairs and discard the
            provided coupling) and ``fixed_noise.shape ==
            training_outputs.shape``. When ``None`` (the default), the
            pre-change behavior is preserved byte-identically.
        compile: If True, wrap ``velocity_field`` with
            ``torch.compile(..., mode='reduce-overhead')`` for the hot
            forward pass in the training loop. The compiled callable is
            used for ``pred_v`` inside the inner loop only; EMA updates and
            the final EMA copy-back still operate on the original module's
            parameters (which are shared with the compiled wrapper). Pays
            a ~15-30s one-time compile cost — worth it for runs ≥ 2000
            epochs, usually a wash for shorter ones. Defaults to False so
            behavior is unchanged unless explicitly requested.

    Returns:
        (velocity_field, losses) — the trained model and per-epoch losses.

    Raises:
        ValueError: If coupling is not one of the valid options.
    """
    valid_couplings = ('none', 'hungarian', 'sinkhorn')
    if coupling not in valid_couplings:
        raise ValueError(
            f"coupling must be one of {valid_couplings}, got '{coupling}'"
        )

    valid_ts = ('uniform', 'logit_normal')
    if time_sampling not in valid_ts:
        raise ValueError(
            f"time_sampling must be one of {valid_ts}, got '{time_sampling}'"
        )

    if coupling_batch_size is not None:
        if coupling_batch_size % batch_size != 0:
            raise ValueError(
                f"coupling_batch_size ({coupling_batch_size}) must be a "
                f"multiple of batch_size ({batch_size})"
            )

    if fixed_noise is not None:
        if coupling != 'none':
            raise ValueError(
                "fixed_noise requires coupling='none' (pairs are already "
                f"coupled by the caller); got coupling={coupling!r}"
            )
        if fixed_noise.shape != training_outputs.shape:
            raise ValueError(
                f"fixed_noise.shape ({tuple(fixed_noise.shape)}) must match "
                f"training_outputs.shape ({tuple(training_outputs.shape)})"
            )

    # Resolve adaptive reg if requested
    if coupling == 'sinkhorn' and sinkhorn_reg == 'auto':
        sinkhorn_reg = compute_adaptive_sinkhorn_reg(training_outputs)

    valid_opt = ('adam', 'adamw')
    if optimizer not in valid_opt:
        raise ValueError(
            f"optimizer must be one of {valid_opt}, got '{optimizer}'"
        )
    if optimizer == 'adam':
        opt = torch.optim.Adam(velocity_field.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(
            velocity_field.parameters(), lr=lr, weight_decay=weight_decay
        )
    if lr_warmup_frac > 0.0:
        warmup_epochs = max(1, int(round(lr_warmup_frac * n_epochs)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, n_epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs
        )

    # Optional torch.compile wrapper for the forward pass. We only use the
    # compiled callable inside the hot loop for pred_v; all parameter-touching
    # code (optimizer, EMA, final copy-back) keeps using the uncompiled module
    # since compiled wrappers proxy to the same parameters.
    forward_vf = torch.compile(
        velocity_field, mode='reduce-overhead'
    ) if compile else velocity_field

    ema_state = None
    ema_param_list = None  # live parameter views, for _foreach ops
    ema_state_list = None  # matching EMA tensors, same order
    if use_ema:
        ema_state = {
            name: p.detach().clone()
            for name, p in velocity_field.named_parameters()
        }
        # Lists in a single order so EMA step can use fused _foreach kernels.
        ema_param_list = list(velocity_field.parameters())
        ema_state_list = [ema_state[n]
                          for n, _ in velocity_field.named_parameters()]

    if coupling_batch_size is None:
        loader_bs = batch_size
        n_inner = 1
    else:
        loader_bs = coupling_batch_size
        n_inner = coupling_batch_size // batch_size

    # Output standardization: whiten training data BEFORE training. We
    # deliberately do NOT set the buffers on velocity_field yet — doing so
    # would cause forward() to whiten again during training and the model
    # would see doubly-whitened data. Buffers are set after the training
    # loop (and after EMA copy-back) so inference-time callers see
    # original-space y in and original-space velocity out.
    y_mean_for_later = None
    y_std_for_later = None
    if standardize_outputs:
        y_mean_for_later = training_outputs.mean(dim=0)
        y_std_for_later = training_outputs.std(dim=0).clamp_min(1e-8)
        training_outputs = (
            training_outputs - y_mean_for_later
        ) / y_std_for_later

    # Fast path: when the data fits in a single tensor on the training
    # device (the common case), bypass DataLoader entirely — it's pure
    # Python overhead and profiling shows it dominates for small batches
    # on a warm GPU. Shuffle indices each epoch on the same device.
    fast_path = (refresh_data_each_epoch is None)

    if not fast_path:
        # Fallback to DataLoader when the caller requests fresh data per
        # epoch (ReFlow's refresh_data_each_epoch hook).
        if fixed_noise is not None:
            dataset = torch.utils.data.TensorDataset(
                training_outputs, fixed_noise
            )
        else:
            dataset = torch.utils.data.TensorDataset(training_outputs)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=loader_bs,
            shuffle=True,
            drop_last=(coupling_batch_size is not None),
        )

    losses = []
    data_device = training_outputs.device
    n_total = training_outputs.shape[0]

    for epoch in range(n_epochs):
        if refresh_data_each_epoch is not None:
            current_outputs = refresh_data_each_epoch()
            if standardize_outputs and y_mean_for_later is not None:
                current_outputs = (
                    current_outputs - y_mean_for_later
                ) / y_std_for_later
            if fixed_noise is not None:
                dataset = torch.utils.data.TensorDataset(
                    current_outputs, fixed_noise
                )
            else:
                dataset = torch.utils.data.TensorDataset(current_outputs)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=loader_bs,
                shuffle=True,
                drop_last=(coupling_batch_size is not None),
            )

        # Accumulate loss on-device to avoid per-batch .item() sync; we
        # only read the scalar at the end of the epoch.
        epoch_loss_t = None
        n_batches = 0

        # Build per-epoch iterator. Fast path keeps everything on the
        # training device; slow path goes through DataLoader.
        if fast_path:
            perm = torch.randperm(n_total, device=data_device)
            if coupling_batch_size is not None:
                n_full = (n_total // loader_bs) * loader_bs
                perm = perm[:n_full]
            def _iter():
                for start in range(0, perm.shape[0], loader_bs):
                    idx = perm[start:start + loader_bs]
                    x1 = training_outputs.index_select(0, idx)
                    if fixed_noise is not None:
                        yield (x1, fixed_noise.index_select(0, idx))
                    else:
                        yield (x1,)
            batch_iter = _iter()
        else:
            batch_iter = loader

        for batch in batch_iter:
            if fixed_noise is not None:
                # Caller-provided pre-coupled (x_1, x_0) pairs — DataLoader
                # shuffles the rows in lockstep, so pair correspondence is
                # preserved. No further coupling is applied (validation
                # already enforced coupling == 'none').
                x1_large, x0_large = batch
            else:
                (x1_large,) = batch
                # Sample source noise
                x0_large = torch.randn_like(x1_large)

            # OT coupling runs ONCE on the large batch
            if coupling == 'hungarian':
                x0_large, x1_large = ot_coupling(x0_large, x1_large)
            elif coupling == 'sinkhorn':
                x0_large, x1_large = sinkhorn_coupling(
                    x0_large, x1_large, reg=sinkhorn_reg, max_iters=sinkhorn_iters
                )

            # Slice into optimizer-sized minibatches for gradient steps
            for i in range(n_inner):
                sl = slice(i * batch_size, (i + 1) * batch_size)
                x0_batch = x0_large[sl]
                x1_batch = x1_large[sl]

                # Sample time
                if time_sampling == 'uniform':
                    t = torch.rand(x1_batch.shape[0], device=x1_batch.device)
                else:
                    u = torch.randn(x1_batch.shape[0], device=x1_batch.device)
                    t = torch.sigmoid(u)

                # Interpolate
                x_t = (1 - t.unsqueeze(1)) * x0_batch + t.unsqueeze(1) * x1_batch

                # Target velocity
                target_v = x1_batch - x0_batch

                # Predicted velocity (compiled wrapper when compile=True)
                pred_v = forward_vf(t, x_t)

                loss = F.mse_loss(pred_v, target_v)

                opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        velocity_field.parameters(), max_norm=grad_clip
                    )
                opt.step()

                if use_ema:
                    with torch.no_grad():
                        # Fused EMA step: state = decay*state + (1-decay)*param
                        torch._foreach_mul_(ema_state_list, ema_decay)
                        torch._foreach_add_(
                            ema_state_list, ema_param_list,
                            alpha=1 - ema_decay,
                        )

                # Accumulate as GPU scalar to avoid per-iter sync.
                if epoch_loss_t is None:
                    epoch_loss_t = loss.detach()
                else:
                    epoch_loss_t = epoch_loss_t + loss.detach()
                n_batches += 1

        scheduler.step()
        epoch_loss_val = (
            0.0 if epoch_loss_t is None
            else (epoch_loss_t / max(n_batches, 1)).item()
        )
        losses.append(epoch_loss_val)

    if use_ema:
        with torch.no_grad():
            for name, p in velocity_field.named_parameters():
                p.copy_(ema_state[name])

    # Set standardization buffers AFTER training (and after EMA copy-back),
    # so that the forward pass during training treats the whitened data as
    # ordinary input. From here on, velocity_field.forward() will whiten
    # incoming y and de-whiten outgoing velocity.
    if standardize_outputs:
        with torch.no_grad():
            device = next(velocity_field.parameters()).device
            velocity_field.y_mean = y_mean_for_later.to(device)
            velocity_field.y_std = y_std_for_later.to(device)

    return velocity_field, losses


def generate_reflow_pairs(
    flow_ode: 'FlowODE',
    n_pairs: int,
    dim: int,
    device: Union[str, torch.device] = 'cpu',
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample z_0 ~ N(0, I) and forward-integrate through flow_ode to
    produce coupled (z_0, x_1) pairs for ReFlow.

    Samples noise on the given device, calls
    ``flow_ode.inverse(z_0, t=1.0, n_steps=n_steps)`` to get matching
    x_1 points, and returns both tensors detached from any autograd
    graph. The pairs are coupled by construction: x_1[i] is the image
    of z_0[i] under the trained flow's Gaussian -> data direction.

    Args:
        flow_ode: trained ``FlowODE`` wrapping a ``VelocityField``.
        n_pairs: number of (z_0, x_1) pairs to generate.
        dim: data dimensionality (must match the velocity field's dim).
        device: target device for both tensors.
        n_steps: number of ODE integration steps.
        seed: if provided, seeds a local generator so output is
            reproducible without touching the global RNG state.

    Returns:
        (z_0, x_1): tensors of shape ``(n_pairs, dim)`` on ``device``.
    """
    device = torch.device(device)
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        z0 = torch.randn(n_pairs, dim, generator=gen, device=device)
    else:
        z0 = torch.randn(n_pairs, dim, device=device)

    with torch.no_grad():
        x1 = flow_ode.inverse(z0, t=1.0, n_steps=n_steps)

    return z0.detach(), x1.detach()
