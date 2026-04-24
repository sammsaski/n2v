"""Full flow-conformal verification sweep on VNN-COMP ACAS Xu 2023.

Iterates the 186-instance instance list, calls run_verification_pipeline
with the scaling-study-locked training config, and writes a CSV that
joins 1:1 with acasxu_sweep_deterministic.csv (by (onnx_file, vnnlib_file)).

OR-of-ANDs specs (prop_5-10 in ACAS Xu, 6 instances total) are marked
SKIPPED; the remaining 180 instances use Phase 2's single-HalfSpace
dispatcher.

Usage:
    cd /home/sasakis/v/tools/n2v
    nohup /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.ablations.acasxu_sweep \
        > /tmp/acasxu_sweep_flow.log 2>&1 &
    disown

Expected runtime: depends on the Task-1-locked config. Conservative
estimate at base-med (h128/L4, 10k data, 5000 ep): 180 instances * ~90s
= ~4.5 hours. Worst case at tight-long: ~15 hours.

Output: examples/FlowConformal/ablations/outputs/acasxu_sweep_flow_conformal.csv
        with columns:
            onnx_file, vnnlib_file, verdict, coverage, q,
            epsilon_total, delta_total, train_s, verify_s, total_s,
            cex_x, cex_y, error

For SKIPPED instances: verdict='SKIPPED', error contains the reason.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_VNNCOMP_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
))
_INSTANCES_CSV = _VNNCOMP_ROOT / 'instances.csv'
_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'acasxu_sweep_flow_conformal.csv'
_PER_INSTANCE_TIMEOUT_S = 600  # 5x the VNN-COMP 116s; flow training costs more

# >>> LOCKED CONFIG from scaling study (`acasxu_scaling_study.csv`) <<<
# base-fast: tightest q across all 6 probes (4.5 to 17.95), fastest
# training (~26s/instance), verdict correctness 6/6. base-med/-long
# were unstable (q occasionally exploded to 1e2-1e4); tight-* were
# stable but consistently looser-q than base-fast. See
# docs/audits/2026-04-24-phase3-acasxu-sweep-results.md §scaling study.
_FLOW_CONFIG = 'base'
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_ALPHA = 0.001


def _raise_timeout(signum, frame):
    raise TimeoutError()


def _load_instance(onnx_rel: str, vnn_rel: str):
    """Load and normalize one instance. Returns (network, lb, ub, spec) or
    raises if the spec is OR-of-ANDs or input is OR-of-boxes."""
    onnx_path = _ACASXU_ROOT / onnx_rel.removeprefix('./')
    vnn_path = _ACASXU_ROOT / vnn_rel.removeprefix('./')
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError('OR-of-input-regions not supported')
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    spec = _extract_spec(prop['prop'])  # raises NotImplementedError on OR-of-ANDs
    return network, input_lb, input_ub, spec


def _run_one_instance(onnx_rel: str, vnn_rel: str) -> dict:
    """Run the full verification pipeline on one instance. Returns a
    row dict; always returns (never raises) so the outer loop can log
    errors to CSV instead of aborting."""
    try:
        network, lb, ub, spec = _load_instance(onnx_rel, vnn_rel)
    except (ValueError, NotImplementedError) as e:
        return {
            'verdict': 'SKIPPED',
            'error': f'{type(e).__name__}: {e}',
        }
    except Exception as e:
        return {
            'verdict': 'ERROR',
            'error': f'loadfailed {type(e).__name__}: {e}',
        }

    try:
        result = run_verification_pipeline(
            network=network,
            input_lb=lb, input_ub=ub, spec=spec,
            alpha=_ALPHA,
            n_train=_N_TRAIN, flow_epochs=_FLOW_EPOCHS,
            flow_config=_FLOW_CONFIG,
            scenario_n_samples=_SCENARIO_N, scenario_beta=0.001,
            seed=0,
        )
    except NotImplementedError as e:
        # e.g. OR-of-ANDs specs: deferred, not a real failure.
        return {
            'verdict': 'SKIPPED',
            'error': f'{type(e).__name__}: {e}',
        }
    except Exception as e:
        return {
            'verdict': 'ERROR',
            'error': f'runfailed {type(e).__name__}: {e}',
        }

    cex_x, cex_y = '', ''
    if result['counterexample'] is not None:
        ce = result['counterexample']
        cex_x = json.dumps(ce['x'].tolist())
        cex_y = json.dumps(ce['y'].tolist())
    return {
        'verdict': result['verdict'],
        'coverage': f'{result["coverage_empirical"]:.4f}',
        'q': f'{result.get("q", float("nan")):.4f}',
        'epsilon_total': f'{result["epsilon_total"]:.4f}',
        'delta_total': f'{result["delta_total"]:.4f}',
        'train_s': f'{result["flow_train_time_s"]:.1f}',
        'verify_s': f'{result["verification_time_s"]:.1f}',
        'total_s': f'{result["total_time_s"]:.1f}',
        'cex_x': cex_x, 'cex_y': cex_y,
        'error': '',
    }


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _raise_timeout)
    if not _INSTANCES_CSV.exists():
        print(f'instances.csv not found at {_INSTANCES_CSV}', file=sys.stderr)
        sys.exit(2)

    instances = []
    with open(_INSTANCES_CSV, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                int(row[2])
            except ValueError:
                continue  # skip header or malformed
            instances.append((row[0].strip(), row[1].strip()))
    print(f'Loaded {len(instances)} instances', flush=True)
    print(f'Config: flow_config={_FLOW_CONFIG}  n_train={_N_TRAIN}  '
          f'flow_epochs={_FLOW_EPOCHS}  alpha={_ALPHA}', flush=True)

    fields = ['onnx_file', 'vnnlib_file', 'verdict', 'coverage', 'q',
              'epsilon_total', 'delta_total', 'train_s', 'verify_s',
              'total_s', 'cex_x', 'cex_y', 'error']
    t_start = time.time()
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0, 'SKIPPED': 0,
              'ERROR': 0, 'TIMEOUT': 0}

    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()

        for k, (onnx_rel, vnn_rel) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s] '
                  f'{onnx_rel} + {vnn_rel}', flush=True)
            t0 = time.time()
            try:
                # SIGALRM fires at Python bytecode boundaries — long C-extension
                # calls (CUDA, LP solvers) may delay the TimeoutError until
                # control returns to Python. 600s is a soft budget, not a hard one.
                signal.alarm(_PER_INSTANCE_TIMEOUT_S)
                row = _run_one_instance(onnx_rel, vnn_rel)
            except TimeoutError:
                row = {'verdict': 'TIMEOUT',
                       'error': f'per-instance timeout {_PER_INSTANCE_TIMEOUT_S}s'}
            finally:
                signal.alarm(0)

            out_row = {_f: '' for _f in fields}
            out_row['onnx_file'] = Path(onnx_rel).name
            out_row['vnnlib_file'] = Path(vnn_rel).name
            for k2, v in row.items():
                out_row[k2] = v
            writer.writerow(out_row)
            f.flush()

            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={time.time()-t0:.1f}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {_OUT_CSV}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')


if __name__ == '__main__':
    main()
