# VNN-COMP 2026 sweep harness

Drives n2v over the VNN-COMP 2026 benchmark corpus using the same competition
runner (`../vnncomp_runner.py`) the official harness invokes, so verdicts are
faithful. Two stages:

1. **Smoke test** (`run_smoke.py`) — load every distinct model, parse its spec,
   build the input set. No reachability. Surfaces load/parse errors that the
   runner would otherwise swallow into `unknown`. Subprocess-isolated per model
   (a giant spec or bad model can't take the run down).

2. **Full sweep** (`run_sweep.py`) — verify instances, honoring each instance's
   **exact** timeout from `instances.csv`. Bounded-parallel (K instances ×
   W workers); resumable; writes per-instance results + a master CSV +
   counterexamples.

## Corpus location

Set `N2V_VNNCOMP_BENCHMARKS` to the benchmark checkout (its root or the
`benchmarks/` subdir). Without it, both scripts fall back to the local path
`~/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks`.

## Usage

```bash
source ~/miniconda3/bin/activate n2v

# Smoke test the whole competition (distinct models, 32-way parallel):
python3 run_smoke.py --jobs 32 --timeout 300

# Full sweep, one instance per distinct network, bounded-parallel, resumable:
python3 run_sweep.py --mode different --jobs 6 --workers 18 --resume

# Preview the job list / timeout budget without running:
python3 run_sweep.py --mode different --dry-run
```

### `run_sweep.py` flags

| flag | default | meaning |
|---|---|---|
| `--mode different\|all\|first` | `different` | one per distinct ONNX / every instance / first only |
| `--jobs K` | 6 | instances run concurrently |
| `--workers W` | cores//K | `N2V_WORKERS` (LP workers) per instance; keep `K*W ≤ cores` |
| `--only / --exclude CAT,...` | — | restrict categories |
| `--out DIR` | `results/sweep` | output root |
| `--resume` | off | skip instances whose result file already exists |
| `--dry-run` | off | print job list + budget, run nothing |

## Timing caveat

In bounded-parallel mode K instances share the machine, each pinned to W cores,
so **runtimes are not competition-comparable** (the competition runs one
instance with all cores). Verdicts are faithful; timing is indicative. For
representative timing, run `--jobs 1` (serial, full cores per instance).

## Output (`results/sweep/`)

- `results.csv` — master: category, onnx, vnnlib, result, runtime_s, timeout_s
- `results/<cat>/<onnx>__<vnnlib>.txt` — per-instance verdict (+ counterexample if `sat`)
- `counterexamples/<cat>/…` — mirrored `sat` witnesses
- `logs/<cat>/…` — per-instance runner stdout/stderr

All of `results/` is gitignored.
