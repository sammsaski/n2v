"""
Tightness Ablation Experiment.

Sweeps each pipeline knob one-at-a-time on a fixed banana network and
measures how it affects the tightness of the flow-conformal reach set.
The goal is to identify which knobs matter (worth tuning end-to-end) and
which don't.

Two blocks of knobs:
  Block A — flow architecture / training: t, n_layers, hidden, n_train,
            epochs, coupling, empirical-latent sigma.
  Block B — conformal guarantee: epsilon_1, ell, n_calib.

Per-config metrics: reach-set volume (2D MC), empirical coverage on a
held-out test set, calibration threshold q, conformal confidence delta_1,
flow training time.

Output: examples/FlowConformal/outputs/exp_tightness_ablation.csv
"""

import os
import sys
import time
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from n2v.probabilistic.flow import (
    FlowScore,
    VelocityField,
    FlowODE,
    train_flow,
    calibrate,
    compute_guarantee,
    ProbabilisticSet,
)
from FlowConformal.networks import RotatedBananaNet


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_tightness_ablation.csv')


# ----- Default config (matches coverage validation) -----

@dataclass
class Config:
    t: float = 1.0
    n_layers: int = 4
    hidden: int = 64
    activation: str = 'silu'
    n_train: int = 2000
    n_calib: int = 8000
    n_test: int = 5000
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    coupling: str = 'sinkhorn'
    epsilon_1: float = 0.001
    ell_offset: int = 1   # ell = n_calib - ell_offset; default 1 = m-1
    n_volume_mc: int = 50_000


N_SEEDS = 3
SEEDS = [0, 1, 2]


# ----- Sweeps -----

SWEEPS = [
    # Block A
    ('A', 't',          [0.1, 0.25, 0.5, 0.75, 1.0]),
    ('A', 'n_layers',   [2, 3, 4, 5, 6]),
    ('A', 'hidden',     [16, 32, 64, 128, 256]),
    ('A', 'n_train',    [500, 1000, 2000, 4000, 8000]),
    ('A', 'epochs',     [25, 50, 100, 200, 400]),
    ('A', 'coupling',   ['sinkhorn', 'hungarian']),
    # Block B
    ('B', 'epsilon_1',  [0.0001, 0.001, 0.01, 0.05, 0.1]),
    ('B', 'ell_offset', [1, 5, 10, 25, 50]),
    ('B', 'n_calib',    [1000, 2000, 4000, 8000, 16000]),
]


# ----- Fixed input set -----

# RotatedBananaNet takes inputs in [0, 1]^2; that's our perturbation set.
INPUT_LOW = torch.tensor([0.0, 0.0])
INPUT_HIGH = torch.tensor([1.0, 1.0])


def make_seed(knob: str, value_index: int, seed_index: int) -> int:
    return abs(hash((knob, value_index, seed_index))) % (2 ** 31)


def sample_inputs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g) * (INPUT_HIGH - INPUT_LOW) + INPUT_LOW


def compute_bbox(net) -> tuple:
    """Fixed bounding box for volume MC, computed once from the full input set."""
    with torch.no_grad():
        x = torch.rand(20000, 2) * (INPUT_HIGH - INPUT_LOW) + INPUT_LOW
        y = net(x)
    pad = 0.5
    low = torch.tensor([y[:, 0].min().item() - pad, y[:, 1].min().item() - pad])
    high = torch.tensor([y[:, 0].max().item() + pad, y[:, 1].max().item() + pad])
    return (low, high)


def run_one(net, cfg: Config, seed: int, bbox: tuple) -> Dict[str, Any]:
    """Run a single (config, seed) configuration. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Sample inputs and compute outputs
    x_train = sample_inputs(cfg.n_train, seed=seed)
    x_calib = sample_inputs(cfg.n_calib, seed=seed + 100_000)
    x_test = sample_inputs(cfg.n_test, seed=seed + 200_000)

    with torch.no_grad():
        y_train = net(x_train)
        y_calib = net(x_calib)
        y_test = net(x_test)

    center = y_train.mean(dim=0)
    y_train_c = y_train - center
    y_calib_c = y_calib - center
    y_test_c = y_test - center

    # 2. Train flow
    t0 = time.time()
    vf = VelocityField(
        dim=2,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        activation=cfg.activation,
    )
    flow_ode = FlowODE(vf)
    train_flow(
        vf, y_train_c,
        n_epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        coupling=cfg.coupling,
    )
    flow_training_time = time.time() - t0

    # 3. Build score and calibrate
    flow_score_raw = FlowScore(flow_ode, t=cfg.t)

    class CenteredFlowScore:
        def __init__(self, fs, c):
            self.fs = fs
            self.c = c

        def __call__(self, y):
            return self.fs(y - self.c)

    flow_score = CenteredFlowScore(flow_score_raw, center)

    t0 = time.time()
    ell = max(1, cfg.n_calib - cfg.ell_offset)
    calib_scores = flow_score(y_calib)
    threshold_q = calibrate(calib_scores, ell).item()
    calibration_time = time.time() - t0

    coverage_target, delta_1 = compute_guarantee(cfg.n_calib, ell, cfg.epsilon_1)

    # 4. Build ProbabilisticSet, measure volume + empirical coverage
    pset = ProbabilisticSet(
        flow_score, threshold_q, cfg.n_calib, ell, cfg.epsilon_1, dim=2
    )

    t0 = time.time()
    volume, vol_se = pset.estimate_volume(
        n_samples=cfg.n_volume_mc, bounding_box=bbox
    )
    volume_estimation_time = time.time() - t0

    with torch.no_grad():
        empirical_coverage = pset.contains(y_test).float().mean().item()

    return {
        'volume': volume,
        'volume_se': vol_se,
        'empirical_coverage': empirical_coverage,
        'threshold_q': threshold_q,
        'delta_1': delta_1,
        'ell': ell,
        'flow_training_time': flow_training_time,
        'calibration_time': calibration_time,
        'volume_estimation_time': volume_estimation_time,
    }


def main():
    print(f"=== Tightness Ablation Experiment ===")
    print(f"Output: {CSV_PATH}")

    print("\nBuilding RotatedBananaNet (fixed across all runs)...")
    torch.manual_seed(42)
    net = RotatedBananaNet()

    print("Computing fixed bounding box for volume MC...")
    bbox = compute_bbox(net)
    print(f"  bbox low={bbox[0].tolist()}, high={bbox[1].tolist()}")

    n_total = sum(len(values) * N_SEEDS for _, _, values in SWEEPS)
    print(f"\nTotal runs: {n_total}")

    rows: List[Dict[str, Any]] = []
    run_idx = 0
    overall_t0 = time.time()

    for block, knob, values in SWEEPS:
        print(f"\n--- Block {block}: sweeping {knob} over {values} ---")
        for value_index, value in enumerate(values):
            for seed_index in range(N_SEEDS):
                run_idx += 1
                seed = make_seed(knob, value_index, seed_index)

                cfg = Config()
                setattr(cfg, knob, value)

                t_run = time.time()
                metrics = run_one(net, cfg, seed, bbox)
                run_time = time.time() - t_run

                row = {
                    'block': block,
                    'knob': knob,
                    'value': value,
                    'seed_index': seed_index,
                    'seed': seed,
                    **metrics,
                    'run_time': run_time,
                    **{f'cfg_{k}': v for k, v in asdict(cfg).items()},
                }
                rows.append(row)

                elapsed = time.time() - overall_t0
                print(
                    f"  [{run_idx}/{n_total}] {knob}={value} seed_idx={seed_index} "
                    f"vol={metrics['volume']:.4f} cov={metrics['empirical_coverage']:.4f} "
                    f"q={metrics['threshold_q']:.4f} delta1={metrics['delta_1']:.4f} "
                    f"({run_time:.1f}s, total {elapsed/60:.1f}m)"
                )

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {CSV_PATH}")

    # Console summary
    print("\n=== Summary ===")
    eps1_default = Config().epsilon_1
    for block, knob, values in SWEEPS:
        knob_rows = [r for r in rows if r['knob'] == knob]
        if not knob_rows:
            continue
        vols = [r['volume'] for r in knob_rows]
        covs = [r['empirical_coverage'] for r in knob_rows]
        v_min, v_max = min(vols), max(vols)
        c_min, c_max = min(covs), max(covs)
        # "matters" if max/min volume ratio >= 2x
        ratio = v_max / max(v_min, 1e-12)
        matters = "MATTERS" if ratio >= 2.0 else "minor"

        # coverage sanity (all knob_rows should hit their cfg's epsilon_1 target)
        violations = sum(
            1 for r in knob_rows
            if r['empirical_coverage'] < (1.0 - r[f'cfg_epsilon_1']) - 0.01
        )
        cov_str = "OK" if violations == 0 else f"{violations} VIOLATIONS"

        print(
            f"  [{block}] {knob:12s}: vol [{v_min:.4f}, {v_max:.4f}] "
            f"({ratio:.2f}x) cov [{c_min:.4f}, {c_max:.4f}]  {matters}, cov {cov_str}"
        )

    total_min = (time.time() - overall_t0) / 60
    print(f"\nTotal wall time: {total_min:.1f} minutes")


if __name__ == '__main__':
    main()
