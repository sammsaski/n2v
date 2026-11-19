"""
Flow-based Demographic-Parity Verification of a fairness classifier (NN),
dataset-agnostic. The dataset (and thus its model, data, and sensitive
attribute) is selected via config['dataset'] / the DatasetAdapter loader
registry in adapter.py. Certifies the 80%-rule, bounding each group's favorable
rate with a two-sided Clopper-Pearson (exact-binomial) interval and applying the
three-valued ratio test. Two verdicts are reported per model:

  * flow -- rates on a per-group normalizing-flow population (flow_population);
    the guarantee is "fair w.r.t. the fitted flow P_{V|a}, confidence >= 1-delta"
    (the VeriFair framing, with n2v's learned flow as the population).
  * real -- the same test on the *actual* test rows; assumption-free (bounds only
    the empirical rate), so it reads inconclusive when a group is data-starved.

This script can be run standalone or called from run_group_fairness.py.
Standalone: default paths under the FairN2V dir (../models, ../data,
../results/group_fairness/<ts>).
Runner-driven: paths come from the `config` dict passed by the runner.
"""
from __future__ import annotations

import sys
import time
import datetime
from dataclasses import dataclass
from pathlib import Path

# data/, models/, results/ and adapter.py live in the FairN2V dir, one level up.
_FAIRN2V_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FAIRN2V_DIR))

from adapter import LOADERS, RUN_PROFILES
from flow_population import FlowGroupModel, _group_labels, make_favorable_classifier
from intervals import clopper_pearson

_LABEL = {True: 'fair', False: 'unfair', None: 'inconclusive'}


def parity_verdict(counts, *, c: float, delta: float):
    """Three-valued 80%-rule from per-group (s, n) using CP intervals. Budget
    split across the k groups (union bound), so all pairwise comparisons hold
    jointly at confidence >= 1 - delta. Returns (verdict, intervals)."""
    k = len(counts)
    alpha = delta / k
    ci = [clopper_pearson(s, n, alpha) for s, n in counts]
    los = [lo for lo, _ in ci]
    his = [hi for _, hi in ci]
    if any(his[a] < c * los[b] for a in range(k) for b in range(k) if a != b):
        return False, ci
    if all(los[a] >= c * his[b] for a in range(k) for b in range(k) if a != b):
        return True, ci
    return None, ci


@dataclass(frozen=True)
class FlowParityReport:
    """Demographic-parity verdicts for one model: the flow-population verdict next
    to the assumption-free real-data verdict (both the three-valued 80%-rule)."""

    dataset: str
    model_name: str
    verdict: bool | None                          # flow population
    flow_rates: tuple[float, ...]                 # favorable rate on flow samples
    intervals: tuple[tuple[float, float], ...]
    real_verdict: bool | None                     # real test rows (exact binomial)
    real_rates: tuple[float, ...]
    real_intervals: tuple[tuple[float, float], ...]
    group_sizes: tuple[int, ...]                  # real rows per group
    group_names: tuple[str, ...] | None
    n_samples: int
    c: float
    runtime: float = 0.0                          # wall-clock seconds to verify this model
    flow_fit: bool = True                         # False if a group was too small

    def summary(self) -> str:
        names = self.group_names or [f'g{i}' for i in range(len(self.real_rates))]

        def line(rates, ci):
            return '  '.join(f'{nm}={r:.3f}[{lo:.3f},{hi:.3f}]'
                             for nm, r, (lo, hi) in zip(names, rates, ci))

        flow = (f"  flow [{_LABEL[self.verdict].upper()}] (N={self.n_samples}): "
                f"{line(self.flow_rates, self.intervals)}"
                if self.flow_fit else "  flow: n/a (a group was too small to fit)")
        return (f"{self.dataset}/{self.model_name}  (c={self.c}, {self.runtime:.2f}s)\n"
                f"{flow}\n"
                f"  real [{_LABEL[self.real_verdict].upper()}]: "
                f"{line(self.real_rates, self.real_intervals)}  sizes={self.group_sizes}")


def verify_model_flow(
    dataset: str,
    model_name: str,
    *,
    data_dir: Path = _FAIRN2V_DIR / 'data',
    models_dir: Path = _FAIRN2V_DIR / 'models',
    c: float = 0.8,
    delta: float = 0.05,
    n_samples: int = 50_000,
    n_epochs: int = 300,
    seed: int = 0,
) -> FlowParityReport:
    """Certify demographic parity for one (dataset, model), both ways.

    Runs the three-valued Clopper-Pearson 80%-rule on the real per-group rows
    (empirical verdict) and on ``n_samples`` draws from a per-group flow
    (distributional verdict). If a group is too small to fit a flow, the flow
    verdict is reported as n/a and only the real-data verdict stands.
    """
    t0 = time.perf_counter()
    data_file = RUN_PROFILES[dataset]["data_file"]
    onnx_path = Path(models_dir) / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"model not found: {onnx_path}")
    adapter = LOADERS[dataset](data_dir, onnx_path, data_file)

    classify = make_favorable_classifier(adapter)
    labels = _group_labels(adapter)
    Xt = adapter.X.T
    k = 2 if adapter.sensitive_encoding == 'binary' else len(adapter.sensitive_features)

    # Real-data verdict: exact binomial on the actual per-group counts.
    real_counts = []
    for a in range(k):
        fav = classify(Xt[labels == a])
        real_counts.append((int(fav.sum()), int(fav.size)))
    real_rates = tuple(s / n if n else float('nan') for s, n in real_counts)
    real_verdict, real_intervals = parity_verdict(real_counts, c=c, delta=delta)
    group_sizes = tuple(n for _, n in real_counts)

    # Flow verdict: sample the fitted per-group population.
    try:
        model = FlowGroupModel(adapter, n_epochs=n_epochs, seed=seed)
        flow_counts, flow_rates = [], []
        for a in range(k):
            fav = classify(model.sample(a, n_samples))
            flow_counts.append((int(fav.sum()), n_samples))
            flow_rates.append(float(fav.mean()))
        verdict, intervals = parity_verdict(flow_counts, c=c, delta=delta)
        flow_fit = True
    except ValueError as e:
        print(f"    flow not fit ({e}); reporting real-data verdict only")
        verdict, flow_fit = None, False
        flow_rates = tuple(float('nan') for _ in range(k))
        intervals = tuple((float('nan'), float('nan')) for _ in range(k))

    return FlowParityReport(
        dataset=dataset, model_name=model_name, verdict=verdict,
        flow_rates=tuple(flow_rates), intervals=tuple(intervals),
        real_verdict=real_verdict, real_rates=real_rates,
        real_intervals=tuple(real_intervals), group_sizes=group_sizes,
        group_names=tuple(adapter.group_names) if adapter.group_names else None,
        n_samples=n_samples, c=c, runtime=time.perf_counter() - t0,
        flow_fit=flow_fit,
    )


def _csv_row(rep: FlowParityReport) -> str:
    """One CSV line for a report; per-group fields are ';'-joined, and an interval
    is 'lo:hi' (plot_parity parses this back)."""
    def ivals(xs):
        return ';'.join(f"{lo:.4f}:{hi:.4f}" for lo, hi in xs)
    names = ';'.join(rep.group_names) if rep.group_names else ''
    rates = ';'.join(f"{x:.4f}" for x in rep.flow_rates)
    real = ';'.join(f"{x:.4f}" for x in rep.real_rates)
    sizes = ';'.join(map(str, rep.group_sizes))
    return (f"{rep.dataset},{rep.model_name},{_LABEL[rep.verdict]},"
            f"{_LABEL[rep.real_verdict]},{names},{rates},{ivals(rep.intervals)},"
            f"{real},{ivals(rep.real_intervals)},{rep.n_samples},{sizes},{rep.c},"
            f"{rep.runtime:.4f}\n")


def main(config=None):
    ## Setup
    # Check if config exists (set by runner script), otherwise use defaults
    if config is None:
        # Default configuration for standalone execution
        # Paths are relative to the FairN2V dir (one level up from this script)
        ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        config = {
            'models_dir': _FAIRN2V_DIR / 'models',
            'data_dir': _FAIRN2V_DIR / 'data',
            'output_dir': _FAIRN2V_DIR / 'results' / 'group_fairness' / ts,
            'dataset': 'adult',
            'model_list': ['AC-1', 'AC-3', 'AC-4'],
            'n_samples': 50_000,
            'n_epochs': 300,
            'random_seed': 0,
            'c': 0.8,
            'delta': 0.05,
        }

    dataset = config.get('dataset', 'adult')
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    print(f"======= Demographic parity (flow + real-data) :: {dataset} ==========")
    print(f"c={config['c']}  delta={config['delta']}  n_epochs={config['n_epochs']}  "
          f"n_samples={config['n_samples']}")
    print(" ")

    ## Loop through each model
    reports = []
    for model_name in config['model_list']:
        try:
            rep = verify_model_flow(
                dataset, model_name,
                data_dir=config['data_dir'], models_dir=config['models_dir'],
                c=config['c'], delta=config['delta'],
                n_samples=config['n_samples'], n_epochs=config['n_epochs'],
                seed=config['random_seed'])
        except FileNotFoundError as e:
            print(f"  skip {model_name}: {e}")
            continue
        reports.append(rep)
        print(rep.summary())
        print(" ")

    if not reports:
        raise FileNotFoundError(f"No models verified for {dataset}; nothing written.")

    ## Save results to CSV
    csv_path = config['output_dir'] / f"parity_{dataset}.csv"
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("Dataset,Model,Verdict,RealVerdict,GroupNames,FlowRates,"
                   "Intervals,RealRates,RealIntervals,NSamples,GroupSizes,C,Runtime\n")
        for rep in reports:
            file.write(_csv_row(rep))
    print(f"Parity results saved to {csv_path}")

    runtimes = sorted(r.runtime for r in reports)
    median = runtimes[len(runtimes) // 2]
    print(f"Median runtime: {median:.2f}s over {len(runtimes)} model(s)")

    print(" ")
    print("======= FLOW PARITY VERIFICATION COMPLETE ==========")
    return reports


if __name__ == "__main__":
    main()
