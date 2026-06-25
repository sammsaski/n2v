"""SaVer-Toolbox runner (Sivaramakrishnan et al., HSCC 2025, vigsiv/SaVer-Toolbox).

Calls the DKW empirical-CDF verifier from
``~/v/other/SaVer-Toolbox/SaVer_Toolbox/verify.py``. For each instance:

1. Sample ``N`` inputs uniformly from the input box.
2. Push them through the network.
3. Build a polytope SDF for each unsafe disjunct (OR-of-ANDs becomes
   k Bonferroni-corrected single-polytope checks at ``epsilon = eps/k``).
4. Feed samples to the DKW verifier per disjunct.
5. UNSAT iff every disjunct certifies; SAT if any sample lands in any
   disjunct; UNKNOWN otherwise.

NOTE on Gurobi: SaVer's *Scenario* method requires Gurobi via cvxpy.
The DKW path used here is Gurobi-free for *non-counterexample* spec
shapes (norm balls and polytopes whose SDF can be evaluated without
solving an LP). Polytope SDFs WITH points outside the polytope fall
back to a cvxpy projection — that's the only place Gurobi (or another
LP solver) is invoked, and only when the network produces samples
outside the unsafe set (i.e. nearly always, for safe networks). The
runner uses cvxpy with the default solver chain (ECOS/SCS), which does
not require Gurobi.

Multi-disjunct (Bonferroni-union) mode: for an unsafe region
``U = A_1 \\/ ... \\/ A_k`` (each ``A_i`` a polytope), we run DKW on
each ``A_i`` separately at Bonferroni-corrected ``Delta_i = Delta / k``.
If every disjunct certifies (``P(y in A_i) <= Delta/k`` with confidence
``1 - beta``), then by union bound ``P(y in U) <= Delta`` with the same
confidence (loose but valid). Beta is NOT split — each per-disjunct
check is independent of the others' confidence sets, but to stay
strictly sound we likewise split ``beta`` over the k checks.

Usage:
    cd /path/to/n2v
    python -u -m \\
        examples.FlowConformal.experiments.baselines.run_saver \\
        --benchmark <name> --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from examples.FlowConformal.experiments.baselines._common import (
    add_common_args, extract_disjuncts_from_spec,
    load_benchmark_instances, resolve_n_instances, resolve_output_csv,
    run_baseline_sweep, torch_callable,
)


_BASELINE = 'saver'
_SAVER_PATH = Path(os.path.expanduser('~/v/other/SaVer-Toolbox'))
_DEFAULT_BETA = 0.001  # confidence
_DEFAULT_EPSILON = 0.01  # DKW CDF tolerance
# DKW certifies UNSAT iff (empirical_unsafe + epsilon) <= delta. Since the
# DKW tolerance is 0.01 at m=8000, delta must exceed 0.01 for any sample
# distribution to ever certify. Using delta=0.05 (vs the tighter 0.001 we
# used earlier) preserves soundness of the probabilistic guarantee
# Pr[unsafe] <= delta and unblocks SaVer from emitting only UNKNOWNs.
_DEFAULT_DELTA = 0.05  # P(unsafe) bound


def _try_import_saver():
    if str(_SAVER_PATH) not in sys.path:
        sys.path.insert(0, str(_SAVER_PATH))
    try:
        from SaVer_Toolbox import verify as saver_verify  # type: ignore
        from SaVer_Toolbox import (  # type: ignore
            signedDistanceFunction as saver_sdf,
        )
        return (saver_verify, saver_sdf), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


class _FastUnsafePolytopeSDF:
    """A drop-in replacement for ``saver_sdf.polytope`` that ONLY needs
    sign-correctness at the threshold ``0`` (which is all that DKW's
    ``empiricalCDF(0)`` consumes).

    For an unsafe disjunct ``U_i = {y : G y <= g}``, we define::

        sdf(y) = max_i (G_i^T y - g_i)

    so ``sdf(y) <= 0`` iff y satisfies every row of ``G y <= g`` (i.e.
    y is INSIDE the unsafe disjunct), and ``sdf(y) > 0`` otherwise. The
    magnitude is NOT a true Euclidean signed distance, but DKW only
    consumes ``CDF(0)`` so the magnitude is irrelevant. This avoids the
    per-sample ``cvxpy`` LP fallback in SaVer's ``signed_distance_function``,
    which is prohibitively slow at the sample counts DKW requires
    (~3.8e4 samples per disjunct).
    """

    def __init__(self, G: np.ndarray, g: np.ndarray):
        self.G = np.asarray(G, dtype=np.float64)
        self.g = np.asarray(g, dtype=np.float64).flatten()

    def eval(self, points: np.ndarray, zero_radius: float):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        # (n, n_rows): G_i^T y - g_i for every row of every point
        residuals = pts @ self.G.T - self.g
        # max over rows: <=0 iff inside, >0 iff outside
        return residuals.max(axis=1) - zero_radius


def _verify_one_disjunct(saver_verify, saver_sdf, ys, G, g,
                         beta_per, eps_per, delta_per):
    """Run DKW on a single unsafe disjunct ``{y : G y <= g}``.

    Returns ``{prob_unsafe, upper_bound_unsafe, certified, n_unsafe,
    n_samples_used}``. ``certified`` is True iff the DKW upper-bound
    ``empirical + eps_per`` is at most ``delta_per`` (the per-disjunct
    Bonferroni-corrected target). ``saver_sdf`` is unused here — we
    use the fast vectorized SDF below — but kept in the signature for
    drop-in parity with future SaVer versions.
    """
    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.shape[1] != ys.shape[1]:
        raise ValueError(
            f'disjunct dim mismatch: G has {G_arr.shape[1]} cols but y '
            f'has {ys.shape[1]} dims')

    # Use the fast vectorized SDF (sign-correct at 0; equivalent to
    # SaVer's polytope SDF for DKW's CDF-at-0 query, but skips cvxpy
    # projection for points outside the polytope). ``saver_sdf`` is
    # accepted for parity with the original code path but unused.
    _ = saver_sdf  # silence unused-arg warning
    sdf = _FastUnsafePolytopeSDF(G_arr, np.asarray(g).flatten())
    verifier = saver_verify.usingDKW(
        beta=beta_per, epsilon=eps_per, Delta=delta_per)
    verifier.specification(sdf)
    verifier.addSamples(ys)

    # The SDF is negative when y is INSIDE the unsafe disjunct, so
    # ``empiricalCDF(0)`` counts ``samples <= 0`` = fraction-unsafe.
    # DKW: with probability >= 1 - beta_per, the true CDF lies within
    # ``eps_per`` of the empirical, so an UPPER bound on
    # ``P(y in U_i)`` is ``empirical + eps_per``. We certify the
    # disjunct iff that upper bound is <= delta_per.
    prob_unsafe = float(verifier.empiricalCDF(0))
    upper_bound_unsafe = prob_unsafe + eps_per
    certified = (upper_bound_unsafe <= delta_per)

    # n_unsafe: count of samples that fell INSIDE the unsafe disjunct.
    sdf_vals = sdf.eval(ys, 0)
    n_unsafe = int(np.sum(sdf_vals <= 0))
    return {
        'prob_unsafe': prob_unsafe,
        'upper_bound_unsafe': upper_bound_unsafe,
        'certified': bool(certified),
        'n_unsafe': n_unsafe,
        'n_samples_used': int(ys.shape[0]),
    }


def _process_factory(args):
    saver_modules, import_err = _try_import_saver()
    if saver_modules is None:
        def _err(loader, name):
            return {'verdict': 'ERROR',
                    'error': f'saver_import_failed: {import_err}'}
        return _err
    saver_verify, saver_sdf = saver_modules

    beta = args.beta
    eps = args.dkw_epsilon
    delta = args.delta
    seed = args.seed

    def process_one(loader, name):
        try:
            net, boxes, spec, _ = loader()
        except FileNotFoundError as e:
            return {'verdict': 'ERROR', 'error': f'load_missing: {e}'}
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'load {type(e).__name__}: {e}'}

        disjuncts = extract_disjuncts_from_spec(spec)
        if disjuncts is None:
            return {'verdict': 'NOT_APPLICABLE',
                    'error': 'spec is not a flat OR of polytopes '
                             '(multi-group AND-of-OR-of-ANDs)'}

        if len(boxes) != 1:
            return {'verdict': 'NOT_APPLICABLE',
                    'error': 'OR-of-input-regions not supported'}

        k = len(disjuncts)
        # Bonferroni-corrected per-disjunct levels. Splitting both
        # ``Delta`` and ``beta`` over k disjuncts gives the loosest valid
        # joint guarantee; we keep DKW's ``epsilon`` at the user-supplied
        # value (it's the *empirical* CDF tolerance, not part of the
        # union bound).
        delta_per = delta / k
        beta_per = beta / k
        eps_per = eps  # CDF tolerance unchanged

        # Use the largest per-disjunct sample count so the same sample
        # set works for every disjunct.
        sample_count_dummy = saver_verify.usingDKW(
            beta=beta_per, epsilon=eps_per, Delta=delta_per)
        n_samples = int(sample_count_dummy.numSamples)

        lb, ub = boxes[0]
        lb = np.asarray(lb, dtype=np.float64).flatten()
        ub = np.asarray(ub, dtype=np.float64).flatten()

        rng = np.random.default_rng(seed)
        xs = rng.uniform(lb, ub, size=(n_samples, lb.size)).astype(np.float32)
        model_fn = torch_callable(net)
        try:
            ys = model_fn(xs)
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'model_inference {type(e).__name__}: {e}'}
        ys = np.asarray(ys, dtype=np.float64)

        # Run DKW per disjunct.
        certified_flags = []
        per_unsafe_probs = []
        per_unsafe_counts = []
        any_unsafe = False
        try:
            for (G, g) in disjuncts:
                res = _verify_one_disjunct(
                    saver_verify, saver_sdf, ys, G, g,
                    beta_per=beta_per, eps_per=eps_per,
                    delta_per=delta_per)
                certified_flags.append(res['certified'])
                per_unsafe_probs.append(res['prob_unsafe'])
                per_unsafe_counts.append(res['n_unsafe'])
                if res['n_unsafe'] > 0:
                    any_unsafe = True
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'disjunct_verify {type(e).__name__}: {e}'}

        all_certified = bool(np.all(certified_flags))
        if all_certified:
            verdict = 'UNSAT'
        elif any_unsafe:
            verdict = 'SAT'
        else:
            verdict = 'UNKNOWN'

        # Aggregated metrics for the CSV. The Bonferroni-union UPPER
        # bound on P(y in U) is the SUM of per-disjunct upper bounds
        # (each <= delta/k under certification, total <= delta).
        worst_prob_unsafe = float(max(per_unsafe_probs)) if per_unsafe_probs else float('nan')
        union_upper_bound = float(sum(p + eps_per for p in per_unsafe_probs))
        total_unsafe = int(sum(per_unsafe_counts))

        return {
            'verdict': verdict,
            'beta': beta, 'dkw_epsilon': eps, 'delta': delta,
            'n_samples': n_samples,
            'k_disjuncts': k,
            'beta_per': beta_per, 'delta_per': delta_per,
            'n_certified_disjuncts': int(sum(certified_flags)),
            'worst_prob_unsafe': worst_prob_unsafe,
            'union_upper_bound_unsafe': union_upper_bound,
            'n_unsafe_samples': total_unsafe,
            'error': '',
        }

    return process_one


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--beta', type=float, default=_DEFAULT_BETA,
                        help='DKW confidence parameter (default 1e-3).')
    parser.add_argument('--dkw-epsilon', type=float,
                        default=_DEFAULT_EPSILON,
                        help='DKW CDF tolerance (default 1e-2).')
    parser.add_argument('--delta', type=float, default=_DEFAULT_DELTA,
                        help='Allowed P(unsafe) (default 1e-3).')
    args = parser.parse_args()

    n = resolve_n_instances(args)
    try:
        instances = load_benchmark_instances(args.benchmark, n)
    except FileNotFoundError as e:
        print(f'[{_BASELINE}] TODO/load failed: {e}', file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f'[{_BASELINE}] load error: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(2)
    if not instances:
        print(f'[{_BASELINE}] no instances', file=sys.stderr)
        sys.exit(0)

    out_csv = resolve_output_csv(args, _BASELINE)
    extra_fields = [
        'beta', 'dkw_epsilon', 'delta',
        'n_samples',
        'k_disjuncts', 'beta_per', 'delta_per',
        'n_certified_disjuncts',
        'worst_prob_unsafe', 'union_upper_bound_unsafe',
        'n_unsafe_samples',
    ]
    run_baseline_sweep(
        baseline=_BASELINE, benchmark=args.benchmark,
        instances=instances, out_csv=out_csv,
        extra_fields=extra_fields,
        process_one=_process_factory(args),
    )


if __name__ == '__main__':
    main()
