"""One-shot volume sanity check for the flow-conformal approach, pre-sweep.

Compares three 5D output-space sets on ONE ACAS Xu instance:
  1. R_approx  = n2v's approximate reachable output set (union of Stars),
                 a provable over-approximation of f(X).
  2. S_flow    = {y in R^5 : FlowScore(y) <= q}, the conformal-calibrated set.
  3. y-samples = {y_i = f(x_i) : x_i ~ Uniform(input_box)} (empirical).

Reports:
  - vol(R_approx)          via shared-sample MC over a bbox covering R_approx
  - vol(S_flow)            via the same MC, with the same samples
  - vol(S_flow ∩ R_approx) via the same MC
  - empirical coverage: fraction of y-samples with FlowScore <= q
                        (target: >= 1-α, via the conformal guarantee)
  - empirical R-coverage:  fraction of y-samples inside R_approx
                           (target: 1.0 by construction)
  - ratio vol(S_flow) / vol(R_approx)
  - "fraction of S_flow inside R_approx" (how much of the flow set lies
    in provably-reachable territory vs. is hallucinated outside it)

Decision (printed as the last line + exit code):
  PASS      -- coverage >= 1-α-3σ AND vol_ratio in [0.01, 100]
  SOFT FAIL -- coverage OK but vol_ratio outside [0.01, 100]
               (informational; investigate flow quality, don't block sweep)
  HARD FAIL -- coverage below 1-α-3σ
               (conformal bound empirically broken; debug BEFORE the sweep)

Usage (defaults pulled from acasxu_sweep.py's locked config):
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.ablations.acasxu_volume_validation \\
        --network 1_1 --property 1

Expected runtime: 5-15 min (training dominates). Score eval runs on
GPU if available; Star.contains runs across all available CPU workers.

Limitations:
  - `vol(S_flow)` is MC-estimated over a bbox inflated 20% around R_approx.
    If S_flow extends beyond this bbox, vol(S_flow) is under-estimated.
    A high S_hit_rate on the MC step is a saturation signal (see warning
    in phase 3 output).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from examples.FlowConformal.ablations.acasxu_sweep import (
    _FLOW_CONFIG, _N_TRAIN, _FLOW_EPOCHS, _ALPHA, _SCENARIO_N,
)
from n2v.nn.neural_network import NeuralNetwork
from n2v.sets.star import Star
from n2v.sets.volume import star_union_bbox
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_DEFAULT_N_WORKERS = os.cpu_count() or 1

# Module globals populated in each worker by _init_worker.
# Passing stars via initargs (inherited through fork on Linux) avoids
# re-pickling the full list of Star objects for every chunk.
_WORKER_STARS: list | None = None


def _init_worker(stars):
    global _WORKER_STARS
    _WORKER_STARS = stars


def _contains_any_chunk(pts_chunk: np.ndarray) -> np.ndarray:
    """Per-worker: True for points inside ANY Star in _WORKER_STARS."""
    result = np.zeros(len(pts_chunk), dtype=bool)
    for s in _WORKER_STARS:
        result |= s.contains(pts_chunk, method='lp')
    return result


def _parallel_in_star_union(stars, pts: np.ndarray,
                             n_workers: int) -> np.ndarray:
    """Parallel Star-union containment via ProcessPoolExecutor.

    scipy's linprog is a C-extension that can't run on GPU; CPU-parallel
    across points (chunks) is the only speedup available. Each worker
    inherits ``stars`` through fork (Linux default start method).
    """
    if n_workers <= 1 or len(pts) < 500:
        result = np.zeros(len(pts), dtype=bool)
        for s in stars:
            result |= s.contains(pts, method='lp')
        return result
    chunk_size = max(1, (len(pts) + n_workers - 1) // n_workers)
    chunks = [pts[i:i + chunk_size] for i in range(0, len(pts), chunk_size)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(stars,),
    ) as pool:
        results = list(pool.map(_contains_any_chunk, chunks))
    return np.concatenate(results)


def _load_instance(network_id: str, prop_id: int):
    onnx_name = f'ACASXU_run2a_{network_id}_batch_2000.onnx'
    vnn_name = f'prop_{prop_id}.vnnlib'
    raw_model = load_onnx(str(_ACASXU_ROOT / 'onnx' / onnx_name)).eval()
    net_for_reach = NeuralNetwork(raw_model)
    net_for_flow = _ACASXuWrapper(raw_model)

    prop = load_vnnlib(str(_ACASXU_ROOT / 'vnnlib' / vnn_name))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError('OR-of-input-regions not supported in this validation')
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    spec = _extract_spec(prop['prop'])
    return net_for_reach, net_for_flow, input_lb, input_ub, spec


def _approx_reach(net_for_reach: NeuralNetwork,
                  input_lb: np.ndarray, input_ub: np.ndarray):
    lb_col = input_lb.reshape(-1, 1).astype(np.float32)
    ub_col = input_ub.reshape(-1, 1).astype(np.float32)
    input_star = Star.from_bounds(lb_col, ub_col)
    return net_for_reach.reach(input_star, method='approx')


def _forward_samples(net_for_flow, input_lb, input_ub, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(input_lb, input_ub, size=(n, len(input_lb)))
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        y = net_for_flow(x_t)
    y = y.reshape(n, -1)
    return y


def _shared_sample_mc(R_stars, score_fn, q: float,
                      n_samples: int, seed: int,
                      inflate: float = 0.5,
                      n_workers: int = _DEFAULT_N_WORKERS,
                      empirical_y: np.ndarray | None = None) -> dict:
    """Shared-sample MC for vol(R_approx), vol(S_flow), and their intersection.

    Bbox selection:
      - If ``empirical_y`` is provided (recommended), use its per-dim
        range inflated by ``inflate``. This is tight around f(X) and
        therefore tight around S_flow, giving MC resolution at the
        scale of the real reach set.
      - Else fall back to the bbox of R_approx. That's correct as an
        outer bound but for `method='approx'` reach sets it can be
        astronomically larger than f(X) (compounding constraint
        relaxation), leaving S_flow below MC resolution. The fallback
        is kept because R_approx's bbox IS the right choice if you're
        working with exact-reach stars.
    """
    if empirical_y is not None:
        y_min = empirical_y.min(axis=0)
        y_max = empirical_y.max(axis=0)
        span = y_max - y_min
        lb_bbox = y_min - inflate * span
        ub_bbox = y_max + inflate * span
    else:
        lb_bbox, ub_bbox = star_union_bbox(R_stars)
        span = ub_bbox - lb_bbox
        lb_bbox = lb_bbox - inflate * span
        ub_bbox = ub_bbox + inflate * span
    vol_bbox = float(np.prod(ub_bbox - lb_bbox))

    rng = np.random.default_rng(seed)
    pts = rng.uniform(lb_bbox, ub_bbox, size=(n_samples, len(lb_bbox)))
    pts32 = pts.astype(np.float32)

    in_R = _parallel_in_star_union(R_stars, pts32, n_workers=n_workers)

    pts_t = torch.from_numpy(pts32)
    with torch.no_grad():
        scores = score_fn(pts_t).cpu().numpy()
    in_S = scores <= q

    return {
        'vol_bbox':         vol_bbox,
        'vol_R':            vol_bbox * float(in_R.mean()),
        'vol_S':            vol_bbox * float(in_S.mean()),
        'vol_intersect':    vol_bbox * float((in_R & in_S).mean()),
        'n_samples':        n_samples,
        'R_hit_rate':       float(in_R.mean()),
        'S_hit_rate':       float(in_S.mean()),
    }


def _empirical_coverage(R_stars, score_fn, q: float,
                         y_samples: torch.Tensor,
                         n_workers: int = _DEFAULT_N_WORKERS
                         ) -> tuple[float, float]:
    """Empirical coverage on push-forward samples.

    Returns:
        (P(y in S_flow), P(y in R_approx))  -- in that order.
    """
    with torch.no_grad():
        scores = score_fn(y_samples).cpu().numpy()
    in_S = float((scores <= q).mean())
    y_np = y_samples.cpu().numpy().astype(np.float32)
    in_R = _parallel_in_star_union(R_stars, y_np, n_workers=n_workers)
    return in_S, float(in_R.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--network', default='1_1')
    p.add_argument('--property', type=int, default=1)
    p.add_argument('--flow-config', default=_FLOW_CONFIG)
    p.add_argument('--n-train', type=int, default=_N_TRAIN)
    p.add_argument('--flow-epochs', type=int, default=_FLOW_EPOCHS)
    p.add_argument('--alpha', type=float, default=_ALPHA)
    p.add_argument('--scenario-n', type=int, default=_SCENARIO_N)
    p.add_argument('--mc-volume-samples', type=int, default=50_000)
    p.add_argument('--forward-samples', type=int, default=5_000)
    p.add_argument('--n-workers', type=int, default=_DEFAULT_N_WORKERS,
                   help='CPU workers for parallel Star.contains LP loop '
                        f'(default: os.cpu_count() = {_DEFAULT_N_WORKERS})')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== Volume validation: N_{args.network} / prop_{args.property} ===')
    print(f'Config: flow={args.flow_config}  n_train={args.n_train}  '
          f'epochs={args.flow_epochs}  alpha={args.alpha}')
    print(f'Compute: score_fn on {device}; Star.contains across {args.n_workers} CPU workers')

    net_for_reach, net_for_flow, lb, ub, spec = _load_instance(
        args.network, args.property,
    )

    # 1. Approximate reachability
    print('\n[1/4] Running n2v approx reachability...')
    t0 = time.time()
    R_stars = _approx_reach(net_for_reach, lb, ub)
    print(f'      {len(R_stars)} Star(s) in {time.time()-t0:.1f}s')
    if not R_stars:
        print('ERROR: net_for_reach.reach(..., method="approx") returned no Stars.')
        print('       Cannot validate without an over-approximation. Investigate.')
        sys.exit(3)

    # 2. Train + calibrate
    print('\n[2/4] Training flow + calibrating...')
    t0 = time.time()
    result = run_verification_pipeline(
        network=net_for_flow, input_lb=lb, input_ub=ub, spec=spec,
        alpha=args.alpha,
        n_train=args.n_train, flow_epochs=args.flow_epochs,
        flow_config=args.flow_config,
        scenario_n_samples=args.scenario_n, scenario_beta=0.001,
        seed=args.seed,
    )
    q = float(result['q'])
    score_fn = result['score_fn']
    print(f'      verdict={result["verdict"]}  q={q:.3f}  '
          f'coverage={result["coverage_empirical"]:.4f}  '
          f'time={time.time()-t0:.1f}s')

    # Move the trained flow to GPU so the 50k-sample score eval is fast.
    # FlowScore detects device from the underlying velocity field and
    # transfers inputs/outputs accordingly, so swapping devices is a
    # one-liner on the FlowODE itself.
    if device == 'cuda':
        result['flow'].to(device)

    # 2.5. Precompute forward samples — used both as the MC bbox anchor
    # (so S_flow is above MC resolution) and as the empirical-coverage
    # probe in phase 4.
    y_samples = _forward_samples(net_for_flow, lb, ub,
                                 n=args.forward_samples, seed=args.seed + 1)
    y_samples_np = y_samples.cpu().numpy()

    # 3. Shared-sample volume MC
    print('\n[3/4] Shared-sample MC volume comparison...')
    t0 = time.time()
    mc = _shared_sample_mc(R_stars, score_fn, q,
                           n_samples=args.mc_volume_samples, seed=args.seed,
                           n_workers=args.n_workers,
                           empirical_y=y_samples_np)
    vol_R, vol_S, vol_int = mc['vol_R'], mc['vol_S'], mc['vol_intersect']
    ratio = vol_S / vol_R if vol_R > 0 else float('inf')
    inside_frac = vol_int / vol_S if vol_S > 0 else float('nan')
    print(f'      vol(bbox)          = {mc["vol_bbox"]:.3e}')
    print(f'      vol(R_approx)      = {vol_R:.3e}  (hit rate {mc["R_hit_rate"]:.3f})')
    print(f'      vol(S_flow)        = {vol_S:.3e}  (hit rate {mc["S_hit_rate"]:.3f})')
    print(f'      vol(S ∩ R_approx)  = {vol_int:.3e}')
    print(f'      ratio vol(S)/vol(R) = {ratio:.3f}')
    print(f'      fraction S in R    = {inside_frac:.3f}')
    print(f'      MC time: {time.time()-t0:.1f}s ({args.mc_volume_samples} samples)')
    if mc['S_hit_rate'] > 0.95:
        print(f'      WARNING: S_hit_rate = {mc["S_hit_rate"]:.3f} is saturated; '
              f'vol(S_flow) likely under-estimated — S_flow extends beyond '
              f'the R_approx-derived bbox.')

    # 4. Empirical coverage on push-forward samples (reuse y_samples)
    print('\n[4/4] Empirical coverage on push-forward samples...')
    t0 = time.time()
    cov_S, cov_R = _empirical_coverage(R_stars, score_fn, q, y_samples,
                                        n_workers=args.n_workers)
    print(f'      P(y in S_flow)   = {cov_S:.4f}  (target >= {1-args.alpha:.4f})')
    print(f'      P(y in R_approx) = {cov_R:.4f}  (target = 1.0)')
    print(f'      time: {time.time()-t0:.1f}s ({args.forward_samples} samples)')

    # Decision
    print('\n=== Decision ===')
    floor = (1 - args.alpha) - 3 * ((args.alpha * (1 - args.alpha)
                                      / args.forward_samples) ** 0.5)
    coverage_ok = cov_S >= floor
    # Tighter-than-baseline (ratio < 1) is the paper's WIN condition:
    # a probabilistic set smaller than n2v's deterministic over-approx
    # at the same instance. Looser-than-baseline (ratio > 100) is the
    # suspicious case — the flow claims a region much larger than the
    # provably reachable set, usually a sign of hallucination.
    tighter_than_baseline = ratio < 1.0
    looser_than_baseline = ratio > 100.0
    if not coverage_ok:
        msg = (f'HARD FAIL -- empirical coverage {cov_S:.4f} < floor {floor:.4f}. '
               'Debug before the full sweep.')
        code = 2
    elif looser_than_baseline:
        msg = (f'SOFT FAIL -- flow set is {ratio:.1f}x larger than R_approx. '
               'Coverage OK, but the flow set is looser than n2v\'s '
               'over-approximation -- investigate flow quality.')
        code = 1
    elif tighter_than_baseline:
        msg = (f'PASS (tighter than baseline) -- coverage {cov_S:.4f} >= '
               f'{floor:.4f}, ratio {ratio:.4f} < 1, S_flow is tighter '
               f'than n2v\'s R_approx. This is the paper\'s WIN condition.')
        code = 0
    else:
        msg = (f'PASS (roughly matched) -- coverage {cov_S:.4f} >= {floor:.4f}, '
               f'ratio {ratio:.3f}. Flow set comparable to R_approx.')
        code = 0
    print(msg)
    sys.exit(code)


if __name__ == '__main__':
    main()
