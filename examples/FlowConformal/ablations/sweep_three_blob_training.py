"""Training-quality sweep on ThreeBlobClassifier3D.

Runs multiple flow-training configs against the same Star-union ground
truth and prints a single comparison table. Answers:

- does sinusoidal time embedding help on a 3D curved output distribution?
- does a bigger network / more layers tighten the flow ratio?
- does much-longer training close the gap?
- does ReFlow after initial training help?

Each config reuses the same sampled (x_train, x_calib, x_test) tensors
so differences are due to training only (not sampling noise).
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
from n2v.probabilistic.flow.train import train_flow, generate_reflow_pairs
from n2v.sets.volume import star_union_volume_mc


@dataclass
class Config:
    name: str
    hidden: int = 128
    n_layers: int = 4
    time_embed: str = 'concat'
    n_epochs: int = 2000
    batch_size: int = 2048
    sinkhorn_iters: int = 20
    n_reflow_rounds: int = 0


def run_config(cfg: Config, y_train, y_calib, y_test, bbox, alpha, ell,
               n_mc_volume, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    vf = VelocityField(
        dim=3, hidden=cfg.hidden, n_layers=cfg.n_layers,
        activation='silu', time_embed=cfg.time_embed,
    ).to(device)
    y_train_dev = y_train.to(device)

    t0 = time.time()
    vf, _ = train_flow(
        vf, y_train_dev,
        n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto',
        sinkhorn_iters=cfg.sinkhorn_iters,
        use_ema=True, standardize_outputs=True,
    )
    flow_ode = FlowODE(vf.eval())

    # ReFlow rounds: generate (z0, x1) pairs via the current flow, then
    # retrain on that coupling with coupling='none'. Straightens
    # trajectories. Each round is ~n_epochs/2 since it's fine-tuning.
    for r in range(cfg.n_reflow_rounds):
        z0, x1 = generate_reflow_pairs(
            flow_ode, n_pairs=y_train.shape[0], dim=3, device=device,
            n_steps=50, seed=seed + 100 + r,
        )
        vf, _ = train_flow(
            vf, x1,
            n_epochs=cfg.n_epochs // 2, batch_size=cfg.batch_size, lr=5e-4,
            coupling='none', use_ema=True, standardize_outputs=True,
            fixed_noise=z0,
        )
        flow_ode = FlowODE(vf.eval())
    train_time = time.time() - t0

    score_fn = FlowScore(flow_ode, t=1.0, n_steps=30, method='rk4',
                         batch_size=65536)
    t1 = time.time()
    thresh = calibrate(score_fn(y_calib), ell).item()
    s = ProbabilisticSet(score_fn=score_fn, threshold=thresh,
                         m=y_calib.shape[0], ell=ell, epsilon=alpha, dim=3)
    vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
    cov = s.contains(y_test).float().mean().item()
    infer_time = time.time() - t1

    return {
        'name': cfg.name, 'vol': vol, 'se': se, 'cov': cov,
        'train_s': train_time, 'infer_s': infer_time,
    }


def main():
    alpha = 0.01
    seed = 0
    n_train = 10_000
    n_calib = 2_000
    n_test = 2_000
    n_mc_volume = 400_000

    torch.manual_seed(seed)
    net = ThreeBlobClassifier3D().eval()

    print('Computing Star-union ground truth...')
    reach = compute_exact_reach(net, np.zeros(3), 1.0, output_dim=3)
    stars = reach['stars']
    star_vol = star_union_volume_mc(
        stars, n_samples=500_000, batch_size=25_000, seed=42,
        contains_method='algebraic',
    ).mean
    floor = (1 - alpha) * star_vol
    print(f'  n_stars = {len(stars)}  Star-union = {star_vol:.2f}  '
          f'floor (1-alpha)*union = {floor:.2f}')

    # Shared sample tensors
    x_center_t = torch.zeros(3)
    x_tr = sample_l_inf_ball(x_center_t, 1.0, n_train, seed, dim=3)
    x_ca = sample_l_inf_ball(x_center_t, 1.0, n_calib, seed + 1_000_000, dim=3)
    x_te = sample_l_inf_ball(x_center_t, 1.0, n_test, seed + 2_000_000, dim=3)
    with torch.no_grad():
        y_tr = net(x_tr)
        y_ca = net(x_ca)
        y_te = net(x_te)
    y_all = torch.cat([y_tr, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))

    configs = [
        Config(name='baseline          (hidden=128, L=4, concat, 2000ep)'),
        Config(name='sinusoidal-t     (hidden=128, L=4, sin,    2000ep)',
               time_embed='sinusoidal'),
        Config(name='bigger-net       (hidden=256, L=6, concat, 2000ep)',
               hidden=256, n_layers=6),
        Config(name='long-train       (hidden=128, L=4, concat, 8000ep)',
               n_epochs=8000),
        Config(name='big+long+sinus   (hidden=256, L=6, sin,    5000ep)',
               hidden=256, n_layers=6, time_embed='sinusoidal', n_epochs=5000),
        Config(name='reflow x1        (hidden=128, L=4, sin,    2000+1k)',
               time_embed='sinusoidal', n_reflow_rounds=1),
    ]

    results = []
    for cfg in configs:
        print(f'\n--- running {cfg.name} ---')
        r = run_config(cfg, y_tr, y_ca, y_te, bbox, alpha, ell, n_mc_volume, seed)
        r['ratio'] = r['vol'] / floor
        results.append(r)
        print(f'    vol={r["vol"]:.2f}  ratio={r["ratio"]:.2f}x  '
              f'cov={r["cov"]:.4f}  train={r["train_s"]:.1f}s')

    print('\n' + '=' * 78)
    print(f'ThreeBlobClassifier3D: flow tightness sweep (floor={floor:.2f})')
    print('=' * 78)
    print(f'{"config":<55} {"vol":>8} {"ratio":>8} {"cov":>7} {"train(s)":>9}')
    for r in results:
        print(f'{r["name"]:<55} {r["vol"]:>8.2f} {r["ratio"]:>7.2f}x '
              f'{r["cov"]:>7.4f} {r["train_s"]:>9.1f}')


if __name__ == '__main__':
    main()
