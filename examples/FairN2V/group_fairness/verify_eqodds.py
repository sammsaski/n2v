"""
Flow-based Equal-Opportunity + Equalized-Odds Verification of a fairness
classifier (NN), dataset-agnostic. Where verify_parity.py certifies *demographic
parity* (the marginal favorable rate per group), this conditions on the *true
label* (adapter.y) and certifies two nested notions:

  * equal opportunity -- the true-positive rate (TPR) is close across groups;
  * equalized odds     -- BOTH the TPR and the false-positive rate (FPR) are
                          close across groups (equal opportunity + the FPR gap).

So equal opportunity is exactly the TPR half of equalized odds -- both come from
the same TPR/FPR cells, no extra work. Conditioning on the label lets these
distinguish "treated worse" from "less qualified".

Construction: for each true-label cell we fit a per-group flow (flow_population
with a label mask) -- a group's favorable-label flow models
``P(features | group=a, Y=favorable)`` (its samples estimate TPR_a), its
unfavorable-label flow gives FPR_a. Each rate gets a two-sided Clopper-Pearson
interval; a gap holds iff it is within tolerance ``tau`` across every group pair,
at confidence >= 1 - delta. Two verdicts are reported: the flow (distributional)
one and an assumption-free real-data one on the actual cells -- the latter still
stands when a cell is too small to fit a flow.

This script can be run standalone or called from a runner (see verify_parity.py
for the config/main pattern). Guarantee is relative to the fitted flow cells.
"""
from __future__ import annotations

import sys
import time
import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FAIRN2V_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FAIRN2V_DIR))

from adapter import LOADERS, RUN_PROFILES
from flow_population import FlowGroupModel, _group_labels, make_favorable_classifier
from intervals import clopper_pearson

_LABEL = {True: 'fair', False: 'unfair', None: 'inconclusive'}


def gap_verdict(counts, *, tau: float, alpha: float):
    """Three-valued absolute-gap test on per-group (s, n): |rate_a - rate_b| <= tau
    for every pair, using CP intervals at per-group level ``alpha``. Returns
    (verdict, intervals). unfair if some pair is forced apart by > tau (the
    guaranteed gap exceeds tau); fair if every pair's worst-case gap <= tau."""
    ci = [clopper_pearson(s, n, alpha) for s, n in counts]
    los = [lo for lo, _ in ci]
    his = [hi for _, hi in ci]
    k = len(ci)
    pairs = [(a, b) for a in range(k) for b in range(k) if a < b]
    # guaranteed (minimum plausible) gap between a, b
    if any(max(los[a] - his[b], los[b] - his[a]) > tau for a, b in pairs):
        return False, ci
    # worst-case (maximum plausible) gap between a, b
    if all(max(his[a] - los[b], his[b] - los[a]) <= tau for a, b in pairs):
        return True, ci
    return None, ci


def _combine(tpr: bool | None, fpr: bool | None) -> bool | None:
    """Equalized odds = TPR-parity AND FPR-parity."""
    if tpr is False or fpr is False:
        return False
    if tpr is True and fpr is True:
        return True
    return None


@dataclass(frozen=True)
class EqOddsReport:
    """Equal-opportunity + equalized-odds verdicts for one model, each computed
    two ways: on the fitted flow cells and on the real rows (exact binomial). The
    ``verdict``/``tpr_verdict`` fields are the flow ones; the ``real_*`` fields are
    the assumption-free counterparts."""

    dataset: str
    model_name: str
    verdict: bool | None                # flow equalized odds (TPR-parity AND FPR-parity)
    tpr_verdict: bool | None            # flow equal opportunity (TPR-parity alone)
    fpr_verdict: bool | None
    tpr_rates: tuple[float, ...]
    fpr_rates: tuple[float, ...]
    tpr_intervals: tuple[tuple[float, float], ...]
    fpr_intervals: tuple[tuple[float, float], ...]
    real_verdict: bool | None           # real-data equalized odds
    real_tpr_verdict: bool | None       # real-data equal opportunity
    real_fpr_verdict: bool | None
    real_tpr: tuple[float, ...]
    real_fpr: tuple[float, ...]
    real_tpr_intervals: tuple[tuple[float, float], ...]
    real_fpr_intervals: tuple[tuple[float, float], ...]
    pos_sizes: tuple[int, ...]          # per-group truly-favorable cell sizes
    neg_sizes: tuple[int, ...]          # per-group truly-unfavorable cell sizes
    group_names: tuple[str, ...] | None
    n_samples: int
    tau: float
    runtime: float = 0.0                # wall-clock seconds to verify this model
    flow_fit: bool = True               # False if a cell was too small to fit a flow

    @property
    def eqopp_verdict(self) -> bool | None:
        """Flow equal opportunity = the TPR-parity condition alone."""
        return self.tpr_verdict

    @property
    def real_eqopp_verdict(self) -> bool | None:
        """Real-data equal opportunity = the real TPR-parity condition alone."""
        return self.real_tpr_verdict

    def _line(self, label, f_rates, f_ci, f_v, r_rates, r_ci, r_v, sizes):
        names = self.group_names or [f'g{i}' for i in range(len(r_rates))]

        def d(rates, ci):
            return '  '.join(f'{nm}={r:.3f}[{lo:.3f},{hi:.3f}]'
                             for nm, r, (lo, hi) in zip(names, rates, ci))

        flow = f"flow [{_LABEL[f_v]}]: {d(f_rates, f_ci)}" if self.flow_fit else "flow: n/a"
        return (f"  {label}  {flow}\n"
                f"       real [{_LABEL[r_v]}]: {d(r_rates, r_ci)}   cell sizes={sizes}")

    def summary(self) -> str:
        def pair(f, r):
            return f"flow: {_LABEL[f].upper():<13} real: {_LABEL[r].upper()}"
        return (f"{self.dataset}/{self.model_name}  (tau={self.tau}, N={self.n_samples}, {self.runtime:.2f}s)\n"
                f"  equal opportunity   {pair(self.tpr_verdict, self.real_tpr_verdict)}\n"
                f"  equalized odds      {pair(self.verdict, self.real_verdict)}\n"
                + self._line('TPR', self.tpr_rates, self.tpr_intervals, self.tpr_verdict,
                             self.real_tpr, self.real_tpr_intervals, self.real_tpr_verdict,
                             self.pos_sizes) + "\n"
                + self._line('FPR', self.fpr_rates, self.fpr_intervals, self.fpr_verdict,
                             self.real_fpr, self.real_fpr_intervals, self.real_fpr_verdict,
                             self.neg_sizes))


def verify_model_eqodds(
    dataset: str,
    model_name: str,
    *,
    data_dir: Path = _FAIRN2V_DIR / 'data',
    models_dir: Path = _FAIRN2V_DIR / 'models',
    tau: float = 0.1,
    delta: float = 0.05,
    n_samples: int = 50_000,
    n_epochs: int = 300,
    seed: int = 0,
) -> EqOddsReport:
    """Certify equal opportunity + equalized odds for one (dataset, model), both
    ways: on the real per-(group, label) cells (exact binomial) and on ``n_samples``
    draws from per-cell flows. If a cell is too small to fit a flow, the flow
    verdict is reported as n/a and only the real-data verdict stands."""
    t0 = time.perf_counter()
    data_file = RUN_PROFILES[dataset]["data_file"]
    onnx_path = Path(models_dir) / f"{model_name}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"model not found: {onnx_path}")
    adapter = LOADERS[dataset](data_dir, onnx_path, data_file)

    y = np.asarray(adapter.y)
    pos_mask = y == adapter.positive_class          # truly-favorable rows
    neg_mask = ~pos_mask
    classify = make_favorable_classifier(adapter)
    lab = _group_labels(adapter)
    Xt = adapter.X.T
    k = 2 if adapter.sensitive_encoding == 'binary' else len(adapter.sensitive_features)
    # Two claims (TPR, FPR); split delta/2, then delta/2 again over k intervals.
    alpha = delta / 2 / k

    # Real-data verdict: exact binomial on the actual per-cell counts.
    def real_counts(mask):
        out = []
        for a in range(k):
            fav = classify(Xt[(lab == a) & mask])
            out.append((int(fav.sum()), int(fav.size)))
        return out
    tpr_real_counts = real_counts(pos_mask)
    fpr_real_counts = real_counts(neg_mask)
    pos_sizes = tuple(n for _, n in tpr_real_counts)
    neg_sizes = tuple(n for _, n in fpr_real_counts)
    real_tpr = tuple(s / n if n else float('nan') for s, n in tpr_real_counts)
    real_fpr = tuple(s / n if n else float('nan') for s, n in fpr_real_counts)
    real_tpr_verdict, real_tpr_ci = gap_verdict(tpr_real_counts, tau=tau, alpha=alpha)
    real_fpr_verdict, real_fpr_ci = gap_verdict(fpr_real_counts, tau=tau, alpha=alpha)
    real_verdict = _combine(real_tpr_verdict, real_fpr_verdict)

    # Flow verdict: sample per-(group, label) flows.
    try:
        model_pos = FlowGroupModel(adapter, mask=pos_mask, n_epochs=n_epochs, seed=seed)
        model_neg = FlowGroupModel(adapter, mask=neg_mask, n_epochs=n_epochs, seed=seed)

        def flow_rates(model):
            counts, rates = [], []
            for a in range(k):
                fav = classify(model.sample(a, n_samples))
                counts.append((int(fav.sum()), n_samples))
                rates.append(float(fav.mean()))
            return counts, rates

        tpr_counts, tpr_rates = flow_rates(model_pos)
        fpr_counts, fpr_rates = flow_rates(model_neg)
        tpr_verdict, tpr_ci = gap_verdict(tpr_counts, tau=tau, alpha=alpha)
        fpr_verdict, fpr_ci = gap_verdict(fpr_counts, tau=tau, alpha=alpha)
        verdict = _combine(tpr_verdict, fpr_verdict)
        flow_fit = True
    except ValueError as e:
        print(f"    flow not fit ({e}); reporting real-data verdict only")
        tpr_verdict = fpr_verdict = verdict = None
        tpr_rates = fpr_rates = tuple(float('nan') for _ in range(k))
        tpr_ci = fpr_ci = tuple((float('nan'), float('nan')) for _ in range(k))
        flow_fit = False

    return EqOddsReport(
        dataset=dataset, model_name=model_name, verdict=verdict,
        tpr_verdict=tpr_verdict, fpr_verdict=fpr_verdict,
        tpr_rates=tuple(tpr_rates), fpr_rates=tuple(fpr_rates),
        tpr_intervals=tuple(tpr_ci), fpr_intervals=tuple(fpr_ci),
        real_verdict=real_verdict,
        real_tpr_verdict=real_tpr_verdict, real_fpr_verdict=real_fpr_verdict,
        real_tpr=real_tpr, real_fpr=real_fpr,
        real_tpr_intervals=tuple(real_tpr_ci), real_fpr_intervals=tuple(real_fpr_ci),
        pos_sizes=pos_sizes, neg_sizes=neg_sizes,
        group_names=tuple(adapter.group_names) if adapter.group_names else None,
        n_samples=n_samples, tau=tau, runtime=time.perf_counter() - t0,
        flow_fit=flow_fit,
    )


def _csv_row(rep: EqOddsReport) -> str:
    """One CSV line; per-group fields ';'-joined, an interval is 'lo:hi'."""
    def groups(xs):
        return ';'.join(f"{x:.4f}" for x in xs)
    def ivals(xs):
        return ';'.join(f"{lo:.4f}:{hi:.4f}" for lo, hi in xs)
    names = ';'.join(rep.group_names) if rep.group_names else ''
    return (f"{rep.dataset},{rep.model_name},"
            f"{_LABEL[rep.eqopp_verdict]},{_LABEL[rep.verdict]},"
            f"{_LABEL[rep.real_eqopp_verdict]},{_LABEL[rep.real_verdict]},"
            f"{_LABEL[rep.tpr_verdict]},{_LABEL[rep.fpr_verdict]},"
            f"{_LABEL[rep.real_tpr_verdict]},{_LABEL[rep.real_fpr_verdict]},{names},"
            f"{groups(rep.tpr_rates)},{ivals(rep.tpr_intervals)},"
            f"{groups(rep.fpr_rates)},{ivals(rep.fpr_intervals)},"
            f"{groups(rep.real_tpr)},{ivals(rep.real_tpr_intervals)},"
            f"{groups(rep.real_fpr)},{ivals(rep.real_fpr_intervals)},"
            f"{';'.join(map(str, rep.pos_sizes))},{';'.join(map(str, rep.neg_sizes))},"
            f"{rep.n_samples},{rep.tau},{rep.runtime:.4f}\n")


def main(config=None):
    ## Setup
    if config is None:
        # Default configuration for standalone execution
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
            'tau': 0.1,
            'delta': 0.05,
        }

    dataset = config.get('dataset', 'adult')
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    print(f"======= Equal opportunity + equalized odds (flow + real-data) :: {dataset} ==========")
    print(f"tau={config['tau']}  delta={config['delta']}  n_epochs={config['n_epochs']}  "
          f"n_samples={config['n_samples']}")
    print(" ")

    reports = []
    for model_name in config['model_list']:
        try:
            rep = verify_model_eqodds(
                dataset, model_name,
                data_dir=config['data_dir'], models_dir=config['models_dir'],
                tau=config['tau'], delta=config['delta'],
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

    csv_path = config['output_dir'] / f"eqodds_{dataset}.csv"
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("Dataset,Model,EqOpportunity,EqOdds,RealEqOpportunity,RealEqOdds,"
                   "TPRVerdict,FPRVerdict,RealTPRVerdict,RealFPRVerdict,GroupNames,"
                   "TPRRates,TPRIntervals,FPRRates,FPRIntervals,"
                   "RealTPR,RealTPRIntervals,RealFPR,RealFPRIntervals,"
                   "PosSizes,NegSizes,NSamples,Tau,Runtime\n")
        for rep in reports:
            file.write(_csv_row(rep))
    print(f"Equal-opportunity + equalized-odds results saved to {csv_path}")

    runtimes = sorted(r.runtime for r in reports)
    median = runtimes[len(runtimes) // 2]
    print(f"Median runtime: {median:.2f}s over {len(runtimes)} model(s)")

    print(" ")
    print("======= EQUAL-OPPORTUNITY + EQUALIZED-ODDS VERIFICATION COMPLETE ==========")
    return reports


if __name__ == "__main__":
    main()
