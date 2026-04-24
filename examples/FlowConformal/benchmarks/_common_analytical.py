"""Shared runner for Phase-2 golden-path end-to-end benchmarks.

Runs the full flow-conformal pipeline on a given ``(network, x_center,
radius)`` triple and prints a comparison table of conformal set volumes
vs. an analytical reach-set reference. Score families compared:

* hyperrect score (Hashemi baseline),
* Euclidean ball score,
* flow-matching score (via a short flow training run).

Because the reach set on these benchmarks is closed form (identity
network -> input cube; rotated linear -> rotated cube), we cross-check
MC volumes against an analytical reference; any discrepancy is a pipeline
bug, not a method-tightness artifact.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sampling import sample_l_inf_ball
from n2v.probabilistic.flow.scores import BallScore, FlowScore, HyperrectScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.train import train_flow


@dataclass
class MethodResult:
    name: str
    threshold: float
    volume: float
    volume_se: float
    empirical_coverage: float
    fit_time_s: float


def _forward(net, x):
    with torch.no_grad():
        return net(torch.as_tensor(x, dtype=torch.float32))


def _pick_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _train_flow_small(y_train: torch.Tensor, dim: int, n_epochs: int,
                      seed: int) -> FlowODE:
    """Small-scale flow training for fast iteration; matches the prior
    golden-path default. See ``_train_flow_production`` for the config
    used by the baseline tightness runs (§9 of the project description).

    Runs on GPU when available; the model stays on its training device so
    downstream MC volume estimation uses the same accelerator.
    """
    torch.manual_seed(seed)
    device = _pick_device()
    vf = VelocityField(dim=dim, hidden=64, n_layers=3, activation='silu').to(device)
    y_train = y_train.to(device)
    vf, _ = train_flow(
        vf, y_train, n_epochs=n_epochs, batch_size=512, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=20,
        use_ema=True, standardize_outputs=True,
    )
    vf.eval()
    return FlowODE(vf)


def _train_flow_production(y_train: torch.Tensor, dim: int, n_epochs: int,
                           seed: int) -> FlowODE:
    """Production-grade OT-CFM training: larger network, standardized
    outputs, Sinkhorn coupling, GPU-end-to-end.

    Kept on GPU after training so MC volume estimation uses the same
    accelerator (previously we moved back to CPU and MC volume became the
    dominant cost).
    """
    torch.manual_seed(seed)
    device = _pick_device()
    vf = VelocityField(dim=dim, hidden=128, n_layers=4, activation='silu').to(device)
    y_train = y_train.to(device)
    vf, _ = train_flow(
        vf, y_train, n_epochs=n_epochs, batch_size=2048, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=20,
        use_ema=True, standardize_outputs=True,
    )
    vf.eval()
    return FlowODE(vf)


def run_pipeline(
    net,
    x_center: np.ndarray,
    radius: float,
    output_dim: int,
    exact_volume: float,
    n_train: int = 4000,
    n_calib: int = 2000,
    n_test: int = 2000,
    alpha: float = 0.01,
    seed: int = 0,
    flow_epochs: int = 200,
    flow_config: str = 'small',
    n_mc_volume: int = 200_000,
) -> dict:
    """Run the full flow-conformal pipeline for a given network.

    Returns a dict with: 'results' (list[MethodResult]), 'bbox' (tuple),
    'y_train', 'y_calib', 'y_test'.
    """
    import math
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))
    torch.manual_seed(seed)
    dim_in = x_center.shape[0]
    x_center_t = torch.as_tensor(x_center, dtype=torch.float32)

    x_tr = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_train, seed=seed, dim=dim_in,
    )
    x_ca = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_calib,
        seed=seed + 1_000_000, dim=dim_in,
    )
    x_te = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_test,
        seed=seed + 2_000_000, dim=dim_in,
    )
    y_tr = _forward(net, x_tr)
    y_ca = _forward(net, x_ca)
    y_te = _forward(net, x_te)

    y_all = torch.cat([y_tr, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)

    results: list[MethodResult] = []
    for name, builder in (
        ('hyperrect', lambda: HyperrectScore(
            center=y_ca.mean(dim=0),
            scales=y_ca.std(dim=0).clamp(min=1e-8),
        )),
        ('ball', lambda: BallScore(center=y_ca.mean(dim=0))),
    ):
        t0 = time.time()
        score_fn = builder()
        thresh = calibrate(score_fn(y_ca), ell).item()
        s = ProbabilisticSet(
            score_fn=score_fn, threshold=thresh,
            m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
        )
        vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
        cov = s.contains(y_te).float().mean().item()
        results.append(MethodResult(
            name=name, threshold=thresh, volume=vol, volume_se=se,
            empirical_coverage=cov, fit_time_s=time.time() - t0,
        ))

    t0 = time.time()
    if flow_config == 'small':
        flow = _train_flow_small(y_tr, output_dim, flow_epochs, seed)
    elif flow_config == 'production':
        flow = _train_flow_production(y_tr, output_dim, flow_epochs, seed)
    else:
        raise ValueError(f"unknown flow_config {flow_config!r}")
    train_time = time.time() - t0

    # Inference: rk4 with 30 steps is >3x faster than default adaptive
    # dopri5 at atol=rtol=1e-5 and is plenty accurate once the flow has
    # converged. Batch size caps the ODE solve's peak memory on GPU.
    score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4', batch_size=65536)
    t1 = time.time()
    calib_scores = score_fn(y_ca)
    thresh = calibrate(calib_scores, ell).item()
    s = ProbabilisticSet(
        score_fn=score_fn, threshold=thresh,
        m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
    cov = s.contains(y_te).float().mean().item()
    infer_time = time.time() - t1
    results.append(MethodResult(
        name='flow', threshold=thresh, volume=vol, volume_se=se,
        empirical_coverage=cov, fit_time_s=train_time + infer_time,
    ))
    bundle_extra = {'flow_train_time_s': train_time, 'flow_infer_time_s': infer_time}

    return {
        'results': results,
        'bbox': bbox,
        'y_train': y_tr, 'y_calib': y_ca, 'y_test': y_te,
        'exact_volume': exact_volume,
        'alpha': alpha,
        **bundle_extra,
    }


def print_report(bundle: dict):
    results = bundle['results']
    exact = bundle['exact_volume']
    alpha = bundle['alpha']
    print(f"\n  analytical exact reach-set volume = {exact:.4f}")
    print(f"  alpha = {alpha}  coverage floor = {1 - alpha}")
    print(f"  {'method':<10} {'vol':>10} {'+/-SE':>10} {'vol/exact':>10} "
          f"{'cov':>8} {'fit(s)':>8}")
    for r in results:
        ratio = r.volume / exact if exact > 0 else float('nan')
        print(f"  {r.name:<10} {r.volume:>10.4f} {r.volume_se:>10.4f} "
              f"{ratio:>10.3f} {r.empirical_coverage:>8.4f} {r.fit_time_s:>8.1f}")
    if 'flow_train_time_s' in bundle:
        print(f"  (flow: train {bundle['flow_train_time_s']:.1f}s, "
              f"infer {bundle['flow_infer_time_s']:.1f}s)")
