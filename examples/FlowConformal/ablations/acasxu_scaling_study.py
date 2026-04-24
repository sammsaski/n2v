"""Scaling study: find a flow-training config that produces healthy
calibrated q on 5D ACAS Xu.

Phase 2's smoke at flow_config='tight', flow_epochs=2000, n_train=5000
produced q = 0.045 — a pathologically small threshold indicating a
collapsed under-trained flow. This script sweeps 6 training configs
across 6 probe instances (3 properties x 2 networks) and reports q
plus the verdict, runtime, and coverage for each.

Acceptance: at least one config lands q in [1.0, 10.0] on all 6
probes. That config is then locked as the default for the full 186-
instance sweep in acasxu_sweep.py.

Usage (sequential; uses GPU when available):
    cd /home/sasakis/v/tools/n2v
    nohup /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.ablations.acasxu_scaling_study \
        > /tmp/acasxu_scaling.log 2>&1 &
    disown

Expected runtime: ~90 minutes (36 runs averaging ~2 min each).

Output: examples/FlowConformal/ablations/outputs/acasxu_scaling_study.csv
        with columns:
            instance_onnx, instance_vnnlib, config, q, verdict,
            coverage, epsilon_total, delta_total, train_s, verify_s,
            total_s, error
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'acasxu_scaling_study.csv'

# 6 probe instances: 2 from prop_1 (single halfspace, easiest), 2 from
# prop_2 (4-row AND spec, more structure), 2 from prop_3 (4-row AND, different subset).
# Networks chosen to span the 45-network grid: early, middle, late.
_PROBES = [
    ('ACASXU_run2a_1_1_batch_2000.onnx', 'prop_1.vnnlib'),
    ('ACASXU_run2a_3_3_batch_2000.onnx', 'prop_1.vnnlib'),
    ('ACASXU_run2a_2_1_batch_2000.onnx', 'prop_2.vnnlib'),
    ('ACASXU_run2a_4_5_batch_2000.onnx', 'prop_2.vnnlib'),
    ('ACASXU_run2a_1_5_batch_2000.onnx', 'prop_3.vnnlib'),
    ('ACASXU_run2a_5_5_batch_2000.onnx', 'prop_3.vnnlib'),
]

# 6 training configs spanning architecture and budget axes.
@dataclass
class _Config:
    name: str
    flow_config: str    # 'base' (h128, L4) or 'tight' (h256, L6, sinusoidal)
    n_train: int
    flow_epochs: int


_CONFIGS = [
    _Config('base-fast',  'base',  5_000,  2_000),
    _Config('base-med',   'base',  10_000, 5_000),
    _Config('base-long',  'base',  10_000, 10_000),
    _Config('tight-fast', 'tight', 5_000,  2_000),
    _Config('tight-med',  'tight', 10_000, 5_000),
    _Config('tight-long', 'tight', 20_000, 10_000),
]


def _load_instance(onnx_name: str, vnnlib_name: str):
    onnx_path = _ACASXU_ROOT / 'onnx' / onnx_name
    vnn_path = _ACASXU_ROOT / 'vnnlib' / vnnlib_name
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError(f'{vnnlib_name}: OR-of-input-regions not supported')
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    spec = _extract_spec(prop['prop'])
    return network, input_lb, input_ub, spec


def _run_one(probe, cfg: _Config) -> dict:
    onnx_name, vnnlib_name = probe
    try:
        network, lb, ub, spec = _load_instance(onnx_name, vnnlib_name)
    except ValueError as e:
        return {'error': str(e)}

    try:
        result = run_verification_pipeline(
            network=network,
            input_lb=lb, input_ub=ub, spec=spec,
            alpha=0.001,
            n_train=cfg.n_train, flow_epochs=cfg.flow_epochs,
            flow_config=cfg.flow_config,
            scenario_n_samples=2_000, scenario_beta=0.001,
            seed=0,
        )
        return result
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def main():
    _OUT_DIR.mkdir(exist_ok=True)

    rows = []
    total = len(_PROBES) * len(_CONFIGS)
    k = 0
    t_start = time.time()

    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'instance_onnx', 'instance_vnnlib', 'config',
            'q', 'verdict', 'coverage', 'epsilon_total', 'delta_total',
            'train_s', 'verify_s', 'total_s', 'error',
        ])
        f.flush()

        for probe in _PROBES:
            for cfg in _CONFIGS:
                k += 1
                elapsed = time.time() - t_start
                print(f'[{k}/{total}  t={elapsed:.0f}s] {probe[0]} + {probe[1]} | {cfg.name}',
                      flush=True)
                t0 = time.time()
                result = _run_one(probe, cfg)
                dt = time.time() - t0

                if 'error' in result:
                    row = [probe[0], probe[1], cfg.name,
                           '', '', '', '', '', '', '', dt, result['error']]
                else:
                    row = [
                        probe[0], probe[1], cfg.name,
                        result.get('q', ''),
                        result['verdict'],
                        f'{result["coverage_empirical"]:.4f}',
                        f'{result["epsilon_total"]:.4f}',
                        f'{result["delta_total"]:.4f}',
                        f'{result["flow_train_time_s"]:.1f}',
                        f'{result["verification_time_s"]:.1f}',
                        f'{result["total_time_s"]:.1f}',
                        '',
                    ]
                writer.writerow(row)
                f.flush()

                if 'error' not in result:
                    print(f'    verdict={result["verdict"]} '
                          f'q={result["q"]:.3f} '
                          f'cov={result["coverage_empirical"]:.4f} '
                          f'total={dt:.1f}s',
                          flush=True)

    print(f'\n=== Scaling study complete ===')
    print(f'Wrote {_OUT_CSV}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print()
    print('Next: open the CSV and pick the config where all 6 probes have q in [1.0, 10.0].')
    print('Lock that config in acasxu_sweep.py.')


if __name__ == '__main__':
    main()
