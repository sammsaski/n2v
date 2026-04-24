"""Enhanced ThreeBlobClassifier3D sweep: (c)+(d)+(e)+(f).

Compares tightness of the flow-conformal reachset under increasingly
heavy training configurations, and contrasts rk4 (fast fixed-step) vs
dopri5 (adaptive tight-tol) inference solvers. Uses parallel MC volume
(n_workers=-1) for the ground-truth Star-union and the flow's
ProbabilisticSet estimate_volume.

(c) bigger net + more data + longer training
(d) dopri5 w/ atol=1e-4 at inference instead of rk4/30
(e) sinusoidal time embedding + logit-normal time sampling
(f) sinkhorn_iters=10 (torch.compile didn't help on this scale)
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.networks import ThreeBlobClassifier3D
from examples.FlowConformal.utils import compute_exact_reach
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sampling import sample_l_inf_ball
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.train import train_flow
from n2v.sets.volume import star_union_volume_mc


@dataclass
class Config:
    name: str
    # Training
    hidden: int = 128
    n_layers: int = 4
    time_embed: str = 'concat'
    time_sampling: str = 'uniform'
    n_epochs: int = 2000
    n_train: int = 10_000
    batch_size: int = 2048
    sinkhorn_iters: int = 10
    # Inference
    infer_solver: str = 'rk4'
    infer_atol: float = 1e-5
    infer_rtol: float = 1e-5
    infer_steps: int = 30


def run_config(cfg: Config, net, x_center_t, radius, y_ca, y_te, bbox,
               alpha, ell, n_mc_volume, seed, output_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fresh sample of training data to match requested n_train.
    x_tr = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=cfg.n_train,
        seed=seed, dim=output_dim,
    )
    with torch.no_grad():
        y_tr = net(x_tr)

    torch.manual_seed(seed)
    vf = VelocityField(
        dim=output_dim, hidden=cfg.hidden, n_layers=cfg.n_layers,
        activation='silu', time_embed=cfg.time_embed,
    ).to(device)

    t0 = time.time()
    vf, _ = train_flow(
        vf, y_tr.to(device),
        n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto',
        sinkhorn_iters=cfg.sinkhorn_iters,
        use_ema=True, standardize_outputs=True,
        time_sampling=cfg.time_sampling,
    )
    train_time = time.time() - t0
    flow = FlowODE(vf.eval())

    score_fn = FlowScore(
        flow, t=1.0, n_steps=cfg.infer_steps, method=cfg.infer_solver,
        batch_size=65536, atol=cfg.infer_atol, rtol=cfg.infer_rtol,
    )
    t1 = time.time()
    thresh = calibrate(score_fn(y_ca), ell).item()
    s = ProbabilisticSet(
        score_fn=score_fn, threshold=thresh, m=y_ca.shape[0],
        ell=ell, epsilon=alpha, dim=output_dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
    cov = s.contains(y_te).float().mean().item()
    infer_time = time.time() - t1
    return {
        'name': cfg.name, 'vol': vol, 'se': se, 'cov': cov,
        'train_s': train_time, 'infer_s': infer_time,
    }


def main():
    alpha = 0.01
    seed = 0
    n_calib = 2_000
    n_test = 2_000
    n_mc_volume = 400_000

    torch.manual_seed(seed)
    net = ThreeBlobClassifier3D().eval()

    print('Computing Star-union ground truth (parallel MC)...')
    t0 = time.time()
    reach = compute_exact_reach(net, np.zeros(3), 1.0, output_dim=3)
    stars = reach['stars']
    star_vol = star_union_volume_mc(
        stars, n_samples=500_000, batch_size=25_000, seed=42,
        contains_method='algebraic', n_workers=16,
    ).mean
    floor = (1 - alpha) * star_vol
    print(f'  n_stars = {len(stars)}  Star-union = {star_vol:.2f}  '
          f'floor = {floor:.2f}  ({time.time()-t0:.1f}s)')

    # Shared calibration / test sample
    x_center_t = torch.zeros(3)
    x_ca = sample_l_inf_ball(x_center_t, 1.0, n_calib,
                                 seed + 1_000_000, dim=3)
    x_te = sample_l_inf_ball(x_center_t, 1.0, n_test,
                                 seed + 2_000_000, dim=3)
    with torch.no_grad():
        y_ca = net(x_ca)
        y_te = net(x_te)

    # Pre-sample 10k training points to compute a stable bbox that covers
    # the y-distribution; we'll reuse this bbox for all configs.
    x_tr_bbox = sample_l_inf_ball(x_center_t, 1.0, 10_000, seed, dim=3)
    with torch.no_grad():
        y_tr_bbox = net(x_tr_bbox)
    y_all = torch.cat([y_tr_bbox, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))

    configs = [
        # Baseline reference (current production config in existing_benchmarks).
        Config(name='baseline (h128/L4/concat/uniform, 10k, 2000ep, rk4)'),
        # Isolate (e) — sinusoidal time embedding only, uniform time sampling.
        # (logit-normal time sampling was tested above and HURT this benchmark,
        # so we keep uniform. Finding documented in the sweep log.)
        Config(name='+sin (h128/L4/sin/uniform, 10k, 2000ep, rk4)',
               time_embed='sinusoidal'),
        # (c): bigger network, 5000 epochs.
        Config(name='tight (h256/L6/sin/uniform, 10k, 5000ep, rk4)',
               hidden=256, n_layers=6, time_embed='sinusoidal',
               n_epochs=5000),
        # (c) + more data.
        Config(name='tight+32k (h256/L6/sin/uniform, 32k, 5000ep, rk4)',
               hidden=256, n_layers=6, time_embed='sinusoidal',
               n_epochs=5000, n_train=32_000),
        # (d): same as above but dopri5 inference solver at atol=1e-4.
        Config(name='tight+32k+dopri5 (h256/L6, 32k, 5000ep, dopri5 1e-4)',
               hidden=256, n_layers=6, time_embed='sinusoidal',
               n_epochs=5000, n_train=32_000,
               infer_solver='dopri5', infer_atol=1e-4, infer_rtol=1e-4,
               infer_steps=50),
        # Much longer training to find the quality plateau.
        Config(name='long (h256/L6/sin/uniform, 32k, 10k ep, dopri5 1e-4)',
               hidden=256, n_layers=6, time_embed='sinusoidal',
               n_epochs=10_000, n_train=32_000,
               infer_solver='dopri5', infer_atol=1e-4, infer_rtol=1e-4,
               infer_steps=50),
    ]

    results = []
    for cfg in configs:
        print(f'\n--- running {cfg.name} ---')
        r = run_config(cfg, net, x_center_t, 1.0, y_ca, y_te, bbox,
                       alpha, ell, n_mc_volume, seed, output_dim=3)
        r['ratio'] = r['vol'] / floor
        results.append(r)
        print(f'    vol={r["vol"]:.2f}  ratio={r["ratio"]:.2f}x  '
              f'cov={r["cov"]:.4f}  train={r["train_s"]:.1f}s  infer={r["infer_s"]:.1f}s')

    print('\n' + '=' * 88)
    print(f'ThreeBlobClassifier3D: enhanced sweep (floor = {floor:.2f})')
    print('=' * 88)
    header = f'{"config":<70}{"vol":>8} {"ratio":>7} {"cov":>7} {"train(s)":>9} {"inf(s)":>7}'
    print(header)
    for r in results:
        print(f'{r["name"]:<70}{r["vol"]:>8.2f} {r["ratio"]:>6.2f}x '
              f'{r["cov"]:>7.4f} {r["train_s"]:>9.1f} {r["infer_s"]:>7.1f}')


if __name__ == '__main__':
    main()
