"""Head-to-head: naive FlowScore vs LogDetFlowScore across 4 PoC benchmarks.

Each benchmark runs at 3 seeds × 2 score classes. hyperrect and ball
baselines run once per seed (shared with the flow runs through
run_pipeline). The sweep reuses the current production flow training
config (h256/L6/sinusoidal, 10k data, 5000ep, batch 2048, sinkhorn-10).

Outputs a table of mean±spread ratio, coverage, and timing, plus a
JSON file with all raw seed-level numbers.
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

from examples.FlowConformal.networks import (
    RotatedBananaNet, ThreeBlobClassifier3D,
)
from examples.FlowConformal.benchmarks._common import (
    _forward, exact_star_union_volume, run_pipeline,
)
from examples.FlowConformal.benchmarks.test_rotated_linear import (
    RotatedLinear, _rot_2d, _rot_3d,
)
from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore
from n2v.probabilistic.flow.scores import FlowScore


@dataclass
class Benchmark:
    name: str
    net_factory: callable
    x_center: np.ndarray
    radius: float
    output_dim: int
    # One of:
    #   'analytical' -> exact = (2*radius)^d * (1-alpha)
    #   'star_union' -> compute via exact_star_union_volume
    ground_truth_kind: str


def _make_rotated_linear_net(R_fn):
    def factory():
        R = R_fn()
        return RotatedLinear(R).eval()
    return factory


def _analytical_exact(bm: Benchmark, alpha: float) -> tuple[float, list]:
    """Analytical floor for rotated-linear: volume of the rotated cube."""
    import math as _m
    vol = (2.0 * bm.radius) ** bm.output_dim
    floor = (1 - alpha) * vol
    # Return (star_union_volume_for_harness, stars=[]). The harness treats
    # `star_union_volume` as "the reference the ratio is divided by times
    # (1-alpha)", which for analytical cases is just the cube volume.
    return vol, []


def main():
    import math
    alpha = 0.01
    seeds = (0, 1, 2)

    benchmarks = [
        Benchmark(
            name='rotated-linear-2D',
            net_factory=_make_rotated_linear_net(lambda: _rot_2d(math.pi / 6)),
            x_center=np.zeros(2), radius=1.0, output_dim=2,
            ground_truth_kind='analytical',
        ),
        Benchmark(
            name='rotated-linear-3D',
            net_factory=_make_rotated_linear_net(_rot_3d),
            x_center=np.zeros(3), radius=1.0, output_dim=3,
            ground_truth_kind='analytical',
        ),
        Benchmark(
            name='banana-2D',
            net_factory=lambda: RotatedBananaNet().eval(),
            x_center=np.array([0.5, 0.5]), radius=0.5, output_dim=2,
            ground_truth_kind='star_union',
        ),
        Benchmark(
            name='three-blob-3D',
            net_factory=lambda: ThreeBlobClassifier3D().eval(),
            x_center=np.zeros(3), radius=1.0, output_dim=3,
            ground_truth_kind='star_union',
        ),
    ]

    score_classes = [
        ('naive', FlowScore),
        ('log-density', LogDetFlowScore),
    ]

    # Log-density MUST use an adaptive solver: rk4 on a 30-step grid under-
    # resolves the divergence integral and biases the score. dopri5 with a
    # modest atol/rtol is the right default here.
    infer_kwargs_by_score = {
        'naive':       dict(infer_solver='rk4',    infer_steps=30,
                            infer_atol=1e-5, infer_rtol=1e-5),
        'log-density': dict(infer_solver='dopri5', infer_steps=30,
                            infer_atol=1e-4, infer_rtol=1e-4),
    }

    # 1. Compute ground truth per benchmark (once — shared across seeds + scores).
    gt_cache = {}
    for bm in benchmarks:
        torch.manual_seed(0)
        net = bm.net_factory()
        if bm.ground_truth_kind == 'analytical':
            vol, stars = _analytical_exact(bm, alpha)
            print(f'[{bm.name}] analytical reference volume = {vol:.4f}  '
                  f'(floor {(1-alpha)*vol:.4f})')
        else:
            print(f'[{bm.name}] computing Star-union ground truth...')
            t0 = time.time()
            vol, stars = exact_star_union_volume(
                net, x_center=bm.x_center, radius=bm.radius,
                output_dim=bm.output_dim, n_mc=300_000,
            )
            print(f'  n_stars={len(stars)}  vol={vol:.4f}  '
                  f'floor={(1-alpha)*vol:.4f}  ({time.time()-t0:.1f}s)')
        gt_cache[bm.name] = vol

    # 2. Run the sweep.
    all_rows = []

    for bm in benchmarks:
        for seed in seeds:
            for score_name, score_cls in score_classes:
                print(f'\n=== {bm.name} | {score_name} | seed={seed} ===')
                torch.manual_seed(seed)
                net = bm.net_factory()
                bundle = run_pipeline(
                    net,
                    x_center=bm.x_center, radius=bm.radius,
                    output_dim=bm.output_dim,
                    star_union_volume=gt_cache[bm.name],
                    alpha=alpha, seed=seed,
                    n_train=10_000, flow_epochs=5000,
                    flow_config='tight',  # h256/L6/sinusoidal, sinkhorn-10
                    flow_score_class=score_cls,
                    **infer_kwargs_by_score[score_name],
                )
                for r in bundle['results']:
                    if r.name == 'flow':
                        # Rename the 'flow' row to reflect which score was used.
                        row_name = f'flow-{score_name}'
                    else:
                        row_name = r.name
                    all_rows.append({
                        'benchmark': bm.name,
                        'seed': seed,
                        'score': (score_name if r.name == 'flow' else r.name),
                        'row_label': row_name,
                        'vol': r.volume,
                        'se': r.volume_se,
                        'ratio': r.volume / ((1 - alpha) * gt_cache[bm.name]),
                        'cov': r.empirical_coverage,
                        'fit_time_s': r.fit_time_s,
                    })
                print(f'    flow-{score_name}: vol={bundle["results"][-1].volume:.2f}  '
                      f'train={bundle["flow_train_time_s"]:.1f}s  '
                      f'infer={bundle["flow_infer_time_s"]:.1f}s')

    # 3. Summary table.
    print('\n' + '=' * 100)
    print('Head-to-head: naive vs log-density (3 seeds each, ratio vs (1-alpha)*ground-truth)')
    print('=' * 100)
    print(f'{"benchmark":<22} {"method":<16} '
          f'{"mean ratio":>11} {"spread":>8} {"cov":>8} {"infer(s)":>10}')
    # Deduplicate: hyperrect/ball appear twice per seed (one per score class).
    # Collapse those rows to one per (benchmark, method, seed) for clarity.
    seen = set()
    compact = []
    for row in all_rows:
        key = (row['benchmark'], row['row_label'], row['seed'])
        if key in seen:
            continue
        seen.add(key)
        compact.append(row)

    for bm in benchmarks:
        for label in ('hyperrect', 'ball', 'flow-naive', 'flow-log-density'):
            rows = [r for r in compact
                    if r['benchmark'] == bm.name and r['row_label'] == label]
            if not rows:
                continue
            ratios = [r['ratio'] for r in rows]
            covs = [r['cov'] for r in rows]
            fit_times = [r['fit_time_s'] for r in rows]
            m = sum(ratios) / len(ratios)
            s = max(ratios) / max(min(ratios), 1e-12)
            print(f'{bm.name:<22} {label:<16} '
                  f'{m:>10.3f}x {s:>7.2f}x {sum(covs)/len(covs):>7.4f} '
                  f'{sum(fit_times)/len(fit_times):>10.1f}')

    # 4. Save raw JSON.
    out = Path(__file__).parent / 'logdensity_vs_naive_results.json'
    with open(out, 'w') as f:
        json.dump({
            'alpha': alpha,
            'ground_truth': gt_cache,
            'rows': all_rows,
        }, f, indent=2, default=str)
    print(f'\nRaw results -> {out}')


if __name__ == '__main__':
    main()
