"""Multi-seed diagnostic for the 'more data / longer training hurts' finding.

Runs 4 configs × 3 seeds. Records:
- final volume / ratio / coverage
- final training loss + downsampled loss trajectory (to see divergence or plateaus)
- training time

Analyses we'll look for in the output:
- Seed variance within each config (tests hypothesis A: seed noise)
- Monotonicity of loss over training (tests D: loss divergence)
- Whether low final_loss correlates with low ratio (tests E: score miscal)

Saves a JSON artifact alongside this file with all raw numbers.
"""
from __future__ import annotations

import json
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
    hidden: int = 128
    n_layers: int = 4
    time_embed: str = 'concat'
    n_epochs: int = 2000
    n_train: int = 10_000
    batch_size: int = 2048
    sinkhorn_iters: int = 10


def _downsample_losses(losses, n_keep=50):
    """Downsample a loss trajectory to ~n_keep evenly-spaced points for
    compact reporting. Always includes the final epoch."""
    n = len(losses)
    if n <= n_keep:
        return [(i, float(losses[i])) for i in range(n)]
    step = max(1, n // n_keep)
    pts = [(i, float(losses[i])) for i in range(0, n, step)]
    if pts[-1][0] != n - 1:
        pts.append((n - 1, float(losses[-1])))
    return pts


def run_once(cfg, seed, net, x_center_t, radius, y_ca, y_te, bbox,
             alpha, ell, n_mc_volume, output_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    vf, losses = train_flow(
        vf, y_tr.to(device),
        n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto',
        sinkhorn_iters=cfg.sinkhorn_iters,
        use_ema=True, standardize_outputs=True,
    )
    train_time = time.time() - t0
    flow = FlowODE(vf.eval())

    score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4', batch_size=65536)
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
        'vol': float(vol), 'se': float(se), 'cov': float(cov),
        'thresh': float(thresh),
        'train_s': float(train_time), 'infer_s': float(infer_time),
        'final_loss': float(losses[-1]),
        'min_loss': float(min(losses)),
        'min_loss_epoch': int(np.argmin(losses)),
        'losses_sampled': _downsample_losses(losses, n_keep=50),
    }


def main():
    alpha = 0.01
    n_calib = 2_000
    n_test = 2_000
    n_mc_volume = 400_000

    torch.manual_seed(0)
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
    print(f'  n_stars={len(stars)}  Star-union={star_vol:.2f}  floor={floor:.2f}  ({time.time()-t0:.1f}s)')

    x_center_t = torch.zeros(3)
    x_ca = sample_l_inf_ball(x_center_t, 1.0, n_calib, 1_000_000, dim=3)
    x_te = sample_l_inf_ball(x_center_t, 1.0, n_test, 2_000_000, dim=3)
    with torch.no_grad():
        y_ca = net(x_ca)
        y_te = net(x_te)

    # Stable bbox: always use 10k training-sample envelope so bbox doesn't
    # shift between configs (would confound volume comparisons).
    x_tr_bbox = sample_l_inf_ball(x_center_t, 1.0, 10_000, 0, dim=3)
    with torch.no_grad():
        y_tr_bbox = net(x_tr_bbox)
    y_all = torch.cat([y_tr_bbox, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))

    # Step count rationale (batch_size=2048):
    #   10k / 2048 = 5 batches/epoch  → n_epochs × 5  total steps
    #   32k / 2048 = 16 batches/epoch → n_epochs × 16 total steps
    #
    # Step-matched pairs:
    #   tight-10k-5000ep         = 25 000 steps   (10k × 5 × 5000)
    #   step-matched-32k-1562ep  = 25 000 steps   (32k × 16 × 1562)  — same training, more data
    #
    #   tight-32k-5000ep         = 80 000 steps   (32k × 16 × 5000)
    #   step-matched-10k-16000ep = 80 000 steps   (10k × 5 × 16000)  — same training, less data
    #
    # If "more data at same STEPS" helps, step-matched-32k-1562ep beats tight-10k-5000ep.
    # If "more training hurts" is the real issue, step-matched-10k-16000ep also blows up.
    configs = [
        Config(name='baseline-10k-2000ep',       hidden=128, n_layers=4,
               time_embed='concat',    n_epochs=2000,   n_train=10_000),
        Config(name='tight-10k-5000ep',          hidden=256, n_layers=6,
               time_embed='sinusoidal', n_epochs=5000,   n_train=10_000),
        Config(name='tight-32k-5000ep',          hidden=256, n_layers=6,
               time_embed='sinusoidal', n_epochs=5000,   n_train=32_000),
        Config(name='long-32k-10000ep',          hidden=256, n_layers=6,
               time_embed='sinusoidal', n_epochs=10_000, n_train=32_000),
        # Step-matched vs tight-10k-5000ep — isolates "more data" from "more steps"
        Config(name='step-matched-32k-1562ep',   hidden=256, n_layers=6,
               time_embed='sinusoidal', n_epochs=1562,   n_train=32_000),
        # Step-matched vs tight-32k-5000ep — isolates "more training" from "more data"
        Config(name='step-matched-10k-16000ep',  hidden=256, n_layers=6,
               time_embed='sinusoidal', n_epochs=16_000, n_train=10_000),
    ]

    seeds = [0, 1, 2]
    all_results: dict[str, dict[int, dict]] = {}

    for cfg in configs:
        all_results[cfg.name] = {}
        for seed in seeds:
            print(f'\n=== {cfg.name}, seed={seed} ===', flush=True)
            r = run_once(cfg, seed, net, x_center_t, 1.0, y_ca, y_te, bbox,
                         alpha, ell, n_mc_volume, output_dim=3)
            r['ratio'] = r['vol'] / floor
            all_results[cfg.name][seed] = r
            print(f'    vol={r["vol"]:.2f}  ratio={r["ratio"]:.2f}x  '
                  f'cov={r["cov"]:.4f}  train={r["train_s"]:.1f}s  '
                  f'final_loss={r["final_loss"]:.5f}  '
                  f'min_loss={r["min_loss"]:.5f}@ep{r["min_loss_epoch"]}',
                  flush=True)

    # ------- Summary table -------
    print('\n' + '=' * 92)
    print(f'Multi-seed sweep (floor={floor:.2f}, 3 seeds each)')
    print('=' * 92)
    print(f'{"config":<22} {"seed":>4} {"ratio":>7} {"cov":>7} '
          f'{"final_loss":>11} {"min_loss":>10} {"min_ep":>7} {"train(s)":>9}')
    for cfg in configs:
        ratios = [all_results[cfg.name][s]['ratio'] for s in seeds]
        for seed in seeds:
            r = all_results[cfg.name][seed]
            print(f'{cfg.name:<22} {seed:>4} {r["ratio"]:>6.2f}x '
                  f'{r["cov"]:>7.4f} {r["final_loss"]:>11.5f} '
                  f'{r["min_loss"]:>10.5f} {r["min_loss_epoch"]:>7d} '
                  f'{r["train_s"]:>9.1f}')
        mean_ratio = sum(ratios) / len(ratios)
        spread = max(ratios) / min(ratios)
        print(f'  -> mean ratio = {mean_ratio:.2f}x,  spread = {spread:.2f}x,'
              f'  range = [{min(ratios):.2f}, {max(ratios):.2f}]')

    # ------- Cross-config comparisons that test each hypothesis -------
    print('\n' + '=' * 92)
    print('Hypothesis tests')
    print('=' * 92)
    # Seed variance: is any config\'s spread > 1.3x?
    any_high_variance = False
    for cfg in configs:
        ratios = [all_results[cfg.name][s]['ratio'] for s in seeds]
        spread = max(ratios) / min(ratios)
        if spread > 1.3:
            any_high_variance = True
            print(f'  HIGH variance in {cfg.name}: spread {spread:.2f}x '
                  f'(ratios {sorted(ratios)})')

    # Do the mean ratios confirm "more data / more epochs hurts"?
    mean_by = {cfg.name: sum(all_results[cfg.name][s]['ratio'] for s in seeds) / len(seeds)
               for cfg in configs}
    print('\n  mean ratios per config:')
    for name, m in mean_by.items():
        print(f'    {name:<30} {m:.3f}x')

    print('\n  Data vs Training (epochs-at-fixed-n_train):')
    print(f'    10k/5000ep   -> 32k/5000ep    delta = {mean_by["tight-32k-5000ep"] / mean_by["tight-10k-5000ep"]:.2f}x  (more data, same epochs)')
    print(f'    32k/5000ep   -> 32k/10000ep   delta = {mean_by["long-32k-10000ep"] / mean_by["tight-32k-5000ep"]:.2f}x  (more epochs, same data)')

    print('\n  Step-matched pairs (same total optimizer steps):')
    print(f'    10k/5000ep (25k steps)   vs 32k/1562ep (25k steps)   '
          f'  ratio_32k/ratio_10k = {mean_by["step-matched-32k-1562ep"] / mean_by["tight-10k-5000ep"]:.2f}x')
    print(f'    -> INTERPRETATION: ratio < 1.0 means more data helps at fixed training.')
    print(f'    -> ratio close to 1.0 means data is saturated on this target.')
    print()
    print(f'    32k/5000ep (80k steps)   vs 10k/16000ep (80k steps)  '
          f'  ratio_10k/ratio_32k = {mean_by["step-matched-10k-16000ep"] / mean_by["tight-32k-5000ep"]:.2f}x')
    print(f'    -> INTERPRETATION: ratio ~1 with both blown up means "too much training" is the cause.')
    print(f'    -> ratio < 1 means 10k saturates and 32k does not; ratio > 1 means data helps even under heavy training.')

    # Does final_loss correlate with ratio?
    print('\n  For each config, is the LOW-loss seed also LOW-ratio?')
    for cfg in configs:
        by_loss = sorted(seeds, key=lambda s: all_results[cfg.name][s]['final_loss'])
        by_ratio = sorted(seeds, key=lambda s: all_results[cfg.name][s]['ratio'])
        agree = (by_loss == by_ratio)
        losses = [all_results[cfg.name][s]['final_loss'] for s in seeds]
        ratios = [all_results[cfg.name][s]['ratio'] for s in seeds]
        print(f'    {cfg.name:<22} '
              f'loss_order={by_loss} ratio_order={by_ratio}  '
              f'agree={agree}')

    if not any_high_variance:
        print('\n  CONCLUSION: low within-config variance; effects look systematic, not seed noise.')
    else:
        print('\n  NOTE: at least one config has >1.3x seed spread. Interpret cautiously.')

    # ------- Loss curves (seed 0 for each config) -------
    print('\n' + '=' * 92)
    print('Loss trajectories (seed 0 per config)')
    print('=' * 92)
    for cfg in configs:
        losses_s = all_results[cfg.name][0]['losses_sampled']
        print(f'\n{cfg.name} (n_epochs={cfg.n_epochs}):')
        n = len(losses_s)
        # Print 7 representative points across the trajectory.
        idxs = sorted({0, n // 8, n // 4, n // 2,
                       3 * n // 4, 7 * n // 8, n - 1})
        for idx in idxs:
            ep, loss = losses_s[idx]
            frac = ep / max(cfg.n_epochs - 1, 1) if cfg.n_epochs > 1 else 0.0
            print(f'   epoch {ep:>6} ({frac*100:>5.1f}%): loss={loss:.6f}')

    # Save raw JSON
    out = Path(__file__).parent / 'multi_seed_results.json'
    with open(out, 'w') as f:
        json.dump({
            'star_vol': float(star_vol),
            'floor': float(floor),
            'results': {
                cfg_name: {str(seed): r for seed, r in sd.items()}
                for cfg_name, sd in all_results.items()
            },
        }, f, indent=2)
    print(f'\nFull raw results -> {out}')


if __name__ == '__main__':
    main()
