"""Iterate the VNN-COMP ACAS Xu 2023 instance list, run n2v's
deterministic verifier (run_instance.py) on each, and write a CSV.

This is the baseline that the flow-conformal sweep is compared against.
Decoupled into its own script so the two sweeps can run on separate
processes in parallel.

Usage:
    cd /home/sasakis/v/tools/n2v
    nohup /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.ablations.acasxu_deterministic_baseline \
        > /tmp/acasxu_deterministic.log 2>&1 &
    disown

Expected runtime: ~2-4 hours (186 instances at average ~30-60 s each).

Output: examples/FlowConformal/ablations/outputs/acasxu_sweep_deterministic.csv
        with columns:
            onnx_file, vnnlib_file, verdict, time_s, method, cex
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path


_VNNCOMP_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
))
_INSTANCES_CSV = _VNNCOMP_ROOT / 'instances.csv'
_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_RUN_INSTANCE = _ACASXU_ROOT / 'run_instance.py'
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'acasxu_sweep_deterministic.csv'
_PY = '/home/sasakis/miniconda3/envs/n2v/bin/python'


def _parse_output(stdout: str) -> dict:
    """Parse run_instance.py's stdout. Returns a dict with keys
    verdict, time_s, method, cex (cex may be empty)."""
    result = {'verdict': 'ERROR', 'time_s': '', 'method': '', 'cex': ''}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith('RESULT:'):
            result['verdict'] = line[len('RESULT:'):].strip()
        elif line.startswith('TIME:'):
            result['time_s'] = line[len('TIME:'):].strip()
        elif line.startswith('METHOD:'):
            result['method'] = line[len('METHOD:'):].strip()
        elif line.startswith('CEX:'):
            result['cex'] = line[len('CEX:'):].strip()
    return result


def _run_one(onnx_rel: str, vnnlib_rel: str, timeout_s: int) -> dict:
    """Invoke run_instance.py as a subprocess with the given timeout."""
    onnx_path = _ACASXU_ROOT / onnx_rel.removeprefix('./')
    vnn_path = _ACASXU_ROOT / vnnlib_rel.removeprefix('./')

    if not onnx_path.exists() or not vnn_path.exists():
        return {'verdict': 'ERROR', 'time_s': 0.0,
                'method': 'missing-file', 'cex': ''}

    cmd = [
        'timeout', '--kill-after=5', f'{timeout_s}',
        _PY, str(_RUN_INSTANCE),
        str(onnx_path), str(vnn_path),
    ]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s + 10)
    except subprocess.TimeoutExpired:
        return {'verdict': 'TIMEOUT', 'time_s': timeout_s,
                'method': '', 'cex': ''}
    elapsed = time.time() - t0

    if res.returncode in (124, 137):
        return {'verdict': 'TIMEOUT', 'time_s': timeout_s,
                'method': '', 'cex': ''}

    parsed = _parse_output(res.stdout)
    if not parsed['time_s']:
        parsed['time_s'] = f'{elapsed:.3f}'
    return parsed


def main():
    _OUT_DIR.mkdir(exist_ok=True)

    if not _INSTANCES_CSV.exists():
        print(f'instances.csv not found at {_INSTANCES_CSV}', file=sys.stderr)
        sys.exit(2)

    instances = []
    with open(_INSTANCES_CSV, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                timeout_s = int(row[2])
            except ValueError:
                continue  # skip header or malformed row
            instances.append((row[0].strip(), row[1].strip(), timeout_s))
    print(f'Loaded {len(instances)} instances', flush=True)

    t_start = time.time()
    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['onnx_file', 'vnnlib_file', 'verdict',
                         'time_s', 'method', 'cex'])
        f.flush()

        for k, (onnx_rel, vnn_rel, timeout_s) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s] {onnx_rel} + {vnn_rel}',
                  flush=True)
            out = _run_one(onnx_rel, vnn_rel, timeout_s)
            writer.writerow([
                Path(onnx_rel).name, Path(vnn_rel).name,
                out['verdict'], out['time_s'], out['method'], out['cex'],
            ])
            f.flush()
            print(f'    {out["verdict"]}  t={out["time_s"]}s  {out["method"]}',
                  flush=True)

    print(f'\n=== Deterministic baseline complete ===')
    print(f'Wrote {_OUT_CSV}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
