"""Exp 1 — SaVer-Toolbox (Sivaramakrishnan et al., HSCC 2025) runner.

Mirrors :mod:`examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_saver`
but adapted to Exp 1's instance-loading interface (``list_instances``
returns ``(onnx_rel, vnn_rel, vnncomp_timeout_s)`` triples, not the
``(name, loader, timeout)`` triples Exp 2 uses).

SaVer applies to any sample-supporting network — includes ALL Exp 1
benchmarks (ACAS Xu, collins, dist_shift, linearizenn, tllverify,
malbeware, metaroom).

Usage::

    cd /path/to/n2v
    python -m \\
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_saver \\
        --benchmark acasxu_2023 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments._runner_utils import (
    append_csv_row_with_defaults,
)
from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    EXP1_BENCHMARKS,
    list_instances,
    load_one_instance,
)

_SEED = 47
_DEFAULT_N_INSTANCES = 100
_DEFAULT_BETA = 0.001
_DEFAULT_EPSILON = 0.01
# Bumped from 0.001 to 0.05: with DKW epsilon=0.01 at m=8000, delta=0.001
# is mathematically uncertifiable (DKW bound 0.01 > delta 0.001 always),
# which is why the original SaVer Phase 5 produced 100% UNKNOWN. Mirrors
# the patched default in baselines/run_saver.py.
_DEFAULT_DELTA = 0.05
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'instance', 'verdict', 'wall_s', 'timeout_s',
    'beta', 'dkw_epsilon', 'delta', 'n_samples',
    'k_disjuncts', 'beta_per', 'delta_per',
    'n_certified_disjuncts', 'worst_prob_unsafe',
    'union_upper_bound_unsafe', 'n_unsafe_samples',
    'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _make_process_one(args):
    """Wrap :mod:`baselines.run_saver`'s factory."""
    from examples.FlowConformal.experiments.baselines.run_saver import (
        _process_factory,
    )
    sv_args = argparse.Namespace(
        beta=args.beta,
        dkw_epsilon=args.dkw_epsilon,
        delta=args.delta,
        seed=args.seed,
    )
    return _process_factory(sv_args)


def _make_loader(benchmark, onnx_rel, vnn_rel, name):
    """Construct a deferred loader closure that returns the 4-tuple
    ``(network, boxes, spec, name)`` SaVer's ``process_one`` expects."""
    def _loader():
        net, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
        return net, boxes, spec, name
    return _loader


def _run_one_idx(idx, instances_meta, process_one, benchmark):
    onnx_rel, vnn_rel, timeout_s = instances_meta[idx]
    name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
    loader = _make_loader(benchmark, onnx_rel, vnn_rel, name)
    t0 = time.time()
    try:
        result = process_one(loader, name)
    except Exception as e:
        result = {'verdict': 'ERROR',
                  'error': f'process {type(e).__name__}: {e}'}
    wall_s = time.time() - t0

    row = {f: '' for f in _FIELDS}
    row.update({
        'benchmark': benchmark,
        'instance': name,
        'verdict': result.get('verdict', 'ERROR'),
        'wall_s': f'{wall_s:.2f}',
        'timeout_s': timeout_s,
        'beta': result.get('beta', ''),
        'dkw_epsilon': result.get('dkw_epsilon', ''),
        'delta': result.get('delta', ''),
        'n_samples': result.get('n_samples', ''),
        'k_disjuncts': result.get('k_disjuncts', ''),
        'beta_per': result.get('beta_per', ''),
        'delta_per': result.get('delta_per', ''),
        'n_certified_disjuncts': result.get('n_certified_disjuncts', ''),
        'worst_prob_unsafe': result.get('worst_prob_unsafe', ''),
        'union_upper_bound_unsafe': result.get('union_upper_bound_unsafe', ''),
        'n_unsafe_samples': result.get('n_unsafe_samples', ''),
        'error': result.get('error', ''),
        'timestamp': _now_iso(),
    })
    return row


def _write_timeout_row(out_csv, benchmark, name, timeout_s):
    append_csv_row_with_defaults(out_csv, _FIELDS, {
        'benchmark': benchmark, 'instance': name,
        'verdict': 'TIMEOUT', 'timeout_s': timeout_s,
        'error': 'shell timeout (run_cell.sh exit 124)',
        'timestamp': _now_iso(),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP1_BENCHMARKS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES)
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--beta', type=float, default=_DEFAULT_BETA,
                   help='DKW confidence parameter (default 1e-3).')
    p.add_argument('--dkw-epsilon', type=float, default=_DEFAULT_EPSILON,
                   help='DKW CDF tolerance (default 1e-2).')
    p.add_argument('--delta', type=float, default=_DEFAULT_DELTA,
                   help='Allowed P(unsafe) (default 1e-3).')
    p.add_argument('--seed', type=int, default=_SEED,
                   help=f'Master seed (default {_SEED}).')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances_meta = list_instances(benchmark)[:args.n_instances]

    if args.list_instances:
        for idx, (_o, _v, t) in enumerate(instances_meta):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp1_{benchmark}_saver.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances_meta)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_rel, vnn_rel, t = instances_meta[args.instance_idx]
        name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
        _write_timeout_row(out_csv, benchmark, name, t)
        return

    process_one = _make_process_one(args)

    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances_meta)):
            print(f'--instance-idx {args.instance_idx} out of range',
                  file=sys.stderr)
            sys.exit(2)
        target_indices = [args.instance_idx]
        append_mode = True
    else:
        target_indices = ([0] if args.smoke
                          else list(range(len(instances_meta))))
        append_mode = False

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'NOT_APPLICABLE': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader(); f.flush()

        for k, idx in enumerate(target_indices, start=1):
            elapsed = time.time() - t_start
            onnx_rel, vnn_rel, t = instances_meta[idx]
            name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
            print(f'[{benchmark} {k}/{len(target_indices)} t={elapsed:.0f}s '
                  f'budget={t}s] idx={idx} {name}', flush=True)
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            row = _run_one_idx(idx, instances_meta, process_one, benchmark)
            writer.writerow(row); f.flush()
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={row["wall_s"]}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next((v for v, c in counts.items() if c > 0), 'NONE')
        print(f'[smoke] PASS on {benchmark}: SaVer ran (verdict={actual}).')


if __name__ == '__main__':
    main()
