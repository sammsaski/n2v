"""Join the flow-conformal and deterministic sweep CSVs, print the
agreement matrix and per-property wall-clock summary.

Usage (defaults point to our Task 2 output + Task 3 output):
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \
        examples.FlowConformal.ablations.acasxu_analyze_sweep

Usage (use the cached run_benchmark.sh results instead of re-running
the deterministic sweep):
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \
        examples.FlowConformal.ablations.acasxu_analyze_sweep \
        --det-csv examples/ACASXu/outputs/benchmark_results.csv

Reads (by default):
    examples/FlowConformal/ablations/outputs/acasxu_sweep_flow_conformal.csv
    examples/FlowConformal/ablations/outputs/acasxu_sweep_deterministic.csv

Writes to stdout. Copy the output into the Phase 3 summary doc.

Schema normalization: the legacy run_benchmark.sh CSV uses columns
(result, time) where our Task 2 script uses (verdict, time_s). Both
schemas are accepted transparently.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


_OUT_DIR = Path(__file__).parent / 'outputs'
_DEFAULT_FLOW_CSV = _OUT_DIR / 'acasxu_sweep_flow_conformal.csv'
_DEFAULT_DET_CSV = _OUT_DIR / 'acasxu_sweep_deterministic.csv'


def _load(path: Path) -> dict:
    """Load a sweep CSV. Normalizes the legacy `(result, time)` column
    names from run_benchmark.sh to our canonical `(verdict, time_s)`."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if 'result' in r and 'verdict' not in r:
                r['verdict'] = r.pop('result')
            if 'time' in r and 'time_s' not in r:
                r['time_s'] = r.pop('time')
            key = (r['onnx_file'], r['vnnlib_file'])
            rows[key] = r
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flow-csv', type=Path, default=_DEFAULT_FLOW_CSV)
    p.add_argument('--det-csv', type=Path, default=_DEFAULT_DET_CSV)
    args = p.parse_args()

    if not args.flow_csv.exists():
        raise FileNotFoundError(f'missing {args.flow_csv}')
    if not args.det_csv.exists():
        raise FileNotFoundError(f'missing {args.det_csv}')

    print(f'flow CSV: {args.flow_csv}')
    print(f'det  CSV: {args.det_csv}')

    flow = _load(args.flow_csv)
    det = _load(args.det_csv)

    joined_keys = sorted(set(flow.keys()) & set(det.keys()))
    print(f'Joined {len(joined_keys)} instances '
          f'(flow: {len(flow)}, det: {len(det)})')

    # Agreement matrix: rows = flow verdict, cols = det verdict.
    matrix = defaultdict(Counter)
    for k in joined_keys:
        matrix[flow[k]['verdict']][det[k]['verdict']] += 1

    verdicts_flow = ['UNSAT', 'SAT', 'UNKNOWN', 'SKIPPED', 'TIMEOUT', 'ERROR']
    verdicts_det = ['UNSAT', 'SAT', 'UNKNOWN', 'TIMEOUT', 'ERROR']

    print()
    print('Agreement matrix (rows = flow-conformal verdict, cols = deterministic):')
    header = 'FLOW\\DET         ' + ' '.join(f'{v:>8}' for v in verdicts_det)
    print(header)
    for vf in verdicts_flow:
        row = [f'{vf:<16}'] + [f'{matrix[vf][vd]:>8d}' for vd in verdicts_det]
        print(' '.join(row))

    # Coverage sanity.
    bad_coverage = []
    for k in joined_keys:
        r = flow[k]
        try:
            c = float(r.get('coverage', ''))
        except ValueError:
            continue
        # Use the same Wald 3-sigma floor as acasxu_volume_validation,
        # with the scenario-sample size used in the sweep (2000).
        # 1 - alpha - 3*sqrt(alpha*(1-alpha)/N). alpha=0.001, N=2000.
        if c < 0.99 - 3 * (0.99 * 0.01 / 2000) ** 0.5:
            bad_coverage.append((k, c))
    print()
    print(f'Instances below conformal floor: {len(bad_coverage)}')

    # Wall-clock summary per property.
    print()
    print('Wall-clock per property (median train+verify vs. deterministic time):')
    per_prop_flow = defaultdict(list)
    per_prop_det = defaultdict(list)
    for k in joined_keys:
        prop = k[1]
        try:
            per_prop_flow[prop].append(float(flow[k].get('total_s', 'nan')))
        except ValueError:
            pass
        try:
            per_prop_det[prop].append(float(det[k].get('time_s', 'nan')))
        except ValueError:
            pass

    print(f'{"prop":<22} {"n":>5} {"flow-median":>14} {"det-median":>14}')
    for prop in sorted(set(per_prop_flow.keys()) | set(per_prop_det.keys())):
        fvals = sorted(v for v in per_prop_flow.get(prop, []) if v == v)
        dvals = sorted(v for v in per_prop_det.get(prop, []) if v == v)
        fmed = fvals[len(fvals)//2] if fvals else float('nan')
        dmed = dvals[len(dvals)//2] if dvals else float('nan')
        n = max(len(fvals), len(dvals))
        print(f'{prop:<22} {n:>5} {fmed:>14.1f} {dmed:>14.1f}')


if __name__ == '__main__':
    main()
