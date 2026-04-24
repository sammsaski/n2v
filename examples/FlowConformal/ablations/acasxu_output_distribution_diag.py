"""Diagnostic: what does the push-forward output distribution look like on
ACAS Xu instances, and does it explain the pathologically small calibrated
`q` we saw in the scaling study?

Hypothesis: the ACAS Xu output distribution on a small input box is so
narrow that a flow trained with OT-CFM loss converges to near-identity,
making ``||phi(y)||`` track ``||y - mean(y)||`` rather than the expected
chi_5 (99.9% quantile ~4.5). If true, we should see:

    99.9% quantile of ||y - mean(y)||  ~= scaling-study q

on the same instance — for ALL training budgets.

Reads ``acasxu_scaling_study.csv`` (if present) so we can print the
observed q alongside the raw-output statistics.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.ablations.acasxu_output_distribution_diag
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks.test_acasxu_single import _ACASXuWrapper
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_SCALING_CSV = Path(__file__).parent / 'outputs' / 'acasxu_scaling_study.csv'

# Probe the same instances the scaling study covered.
_PROBES = [
    ('ACASXU_run2a_1_1_batch_2000.onnx', 'prop_1.vnnlib'),
    ('ACASXU_run2a_3_3_batch_2000.onnx', 'prop_1.vnnlib'),
    ('ACASXU_run2a_2_1_batch_2000.onnx', 'prop_2.vnnlib'),
    ('ACASXU_run2a_4_5_batch_2000.onnx', 'prop_2.vnnlib'),
    ('ACASXU_run2a_1_5_batch_2000.onnx', 'prop_3.vnnlib'),
    ('ACASXU_run2a_5_5_batch_2000.onnx', 'prop_3.vnnlib'),
]

_N_SAMPLES = 5_000
_SEED = 0


def _load_scaling_q_by_instance() -> dict:
    """Return {(onnx, vnnlib): {config: q}} from the scaling-study CSV."""
    if not _SCALING_CSV.exists():
        return {}
    out = defaultdict(dict)
    for r in csv.DictReader(open(_SCALING_CSV)):
        if not r.get('q'):
            continue
        key = (r['instance_onnx'], r['instance_vnnlib'])
        try:
            out[key][r['config']] = float(r['q'])
        except ValueError:
            continue
    return dict(out)


def _push_forward(onnx_name: str, vnnlib_name: str):
    onnx_path = _ACASXU_ROOT / 'onnx' / onnx_name
    vnn_path = _ACASXU_ROOT / 'vnnlib' / vnnlib_name
    net = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError('OR-of-input-regions not supported')
    lb = np.asarray(prop['lb']).flatten()
    ub = np.asarray(prop['ub']).flatten()

    rng = np.random.default_rng(_SEED)
    x = rng.uniform(lb, ub, size=(_N_SAMPLES, len(lb)))
    with torch.no_grad():
        y = net(torch.from_numpy(x.astype(np.float32))).reshape(_N_SAMPLES, -1).numpy()
    return lb, ub, y


def _report(onnx_name: str, vnnlib_name: str, scaling_q: dict) -> None:
    short = f'{onnx_name[13:-17]}/{vnnlib_name[:-7]}'
    print(f'\n=== {short} ===')
    try:
        lb, ub, y = _push_forward(onnx_name, vnnlib_name)
    except Exception as e:
        print(f'  (skip) {type(e).__name__}: {e}')
        return

    input_ranges = np.round(ub - lb, 4).tolist()
    print(f'  input box dim ranges: {input_ranges}')

    # Per-output-dim stats
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    print(f'  output per-dim  min: {np.round(y_min, 4)}')
    print(f'                  max: {np.round(y_max, 4)}')
    print(f'                 mean: {np.round(y_mean, 4)}')
    print(f'                  std: {np.round(y_std, 4)}')

    # Norm stats
    norms_raw = np.linalg.norm(y, axis=1)
    norms_centered = np.linalg.norm(y - y_mean, axis=1)
    qs = [0.5, 0.9, 0.99, 0.999]
    print(f'  ||y||       quantiles {qs}: '
          f'{np.round(np.quantile(norms_raw, qs), 4).tolist()}')
    print(f'  ||y-mean||  quantiles {qs}: '
          f'{np.round(np.quantile(norms_centered, qs), 4).tolist()}')

    # The critical comparison: 99.9%-quantile of centered norms vs scaling q
    q999 = float(np.quantile(norms_centered, 0.999))
    print(f'  99.9%-quantile ||y-mean|| = {q999:.4f}')

    cfgs = scaling_q.get((onnx_name, vnnlib_name), {})
    if cfgs:
        q_min = min(cfgs.values())
        q_max = max(cfgs.values())
        ratio_min = q_min / q999 if q999 > 0 else float('nan')
        ratio_max = q_max / q999 if q999 > 0 else float('nan')
        print(f'  scaling-study q range: [{q_min:.4f}, {q_max:.4f}] '
              f'across {len(cfgs)} configs')
        print(f'  q / (99.9%-||y-mean||) ratio: '
              f'[{ratio_min:.3f}, {ratio_max:.3f}] '
              f'(expect ~1 if flow is near-identity after centering)')

    # Compare to what a well-fit flow SHOULD give (chi_5 at 99.9% ~= 4.5).
    print(f'  target 99.9%-quantile under standard 5D normal: ~4.5 (chi_5)')


def main():
    scaling_q = _load_scaling_q_by_instance()
    if not scaling_q:
        print(f'(no {_SCALING_CSV.name} found; running raw diagnostic only)\n')
    for onnx, vnnlib in _PROBES:
        _report(onnx, vnnlib, scaling_q)

    print('\n=== Verdict ===')
    print('If the q / (99.9%-||y-mean||) ratios are ~1 across all probes, the')
    print('flow is acting near-identity after centering and the narrow output')
    print('distribution IS the root cause. Remedy: either (a) pre-whiten y to')
    print('unit variance before the flow sees it, or (b) switch to a score')
    print('insensitive to output scale (e.g. LogDetFlowScore).')


if __name__ == '__main__':
    main()
