"""
Coverage Validation Experiment

Empirically validates the three probabilistic claims of the flow-conformal
pipeline:
  1. Conformal coverage: Pr[f(x) in R] >= 1 - epsilon_1
  2. Scenario violation: Pr[violation | y in R] <= epsilon_2
  3. Joint claim: Pr[f(x) satisfies spec] >= (1 - epsilon_1)(1 - epsilon_2)

Runs the full flow-conformal pipeline on a synthetic 3-class classifier
with 5 test inputs spanning robustness difficulty levels and 5 seeds per
input. Writes results to a CSV for downstream plotting.

The companion script plot_coverage_validation.py produces figures from
the CSV.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from n2v.probabilistic.flow import (
    VelocityField, FlowODE, train_flow, calibrate, compute_guarantee,
    FlowScore, ProbabilisticSet, verify_robustness,
    sample_empirical_latent_ball,
)
from n2v.sets.box import Box
from n2v.nn import NeuralNetwork
from n2v.utils.lpsolver import solve_lp
from FlowConformal.networks import ThreeBlobClassifier


def _project_max_halfspace(star, direction):
    """Compute max of direction^T y over a Star via LP.
    Returns the max value, or None if the LP fails."""
    offset = float(direction @ star.V[:, 0])
    if star.nVar == 0:
        return offset
    obj = (direction @ star.V[:, 1:]).flatten().reshape(-1, 1)
    A = star.C if (star.C is not None and star.C.size > 0) else None
    b = star.d if (star.d is not None and star.d.size > 0) else None
    lb = star.predicate_lb if star.predicate_lb is not None else None
    ub = star.predicate_ub if star.predicate_ub is not None else None
    x_opt, fval, status, info = solve_lp(
        f=obj, A=A, b=b, lb=lb, ub=ub, minimize=False,
    )
    if fval is None:
        return None
    return offset + fval


def _verify_spec_on_stars(output_stars, true_class, n_classes):
    """Check classification robustness spec on all output Stars.
    Returns (all_verified, failures) where failures is a list of
    (star_idx, wrong_class, max_margin).
    """
    failures = []
    for star_idx, star in enumerate(output_stars):
        for k in range(n_classes):
            if k == true_class:
                continue
            w = np.zeros(n_classes)
            w[k] = 1.0
            w[true_class] = -1.0
            max_margin = _project_max_halfspace(star, w)
            if max_margin is None or max_margin > 0:
                failures.append((star_idx, k, max_margin))
    return len(failures) == 0, failures


def compute_n2v_ground_truth(classifier, x_center, true_class, radius,
                              exact_timeout_seconds=120):
    """
    Run n2v's sound verifier (approx first, exact fallback) on the
    classification robustness spec for a single test input.

    Returns a dict with:
        'spec_holds': True if approx or exact verified, False if exact
                      falsified, None if exact timed out
        'method': 'approx' or 'exact' or 'timeout' or 'approx-fail-exact-timeout'
        'max_failing_margin': largest failing margin from exact (0.0 if
                              verified, None if timeout)
        'n_output_stars': number of output stars from the final method used
        'time': wall clock time
    """
    import signal
    import time as _time

    class _VerificationTimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _VerificationTimeoutError("Exact verification timed out")

    x_center_np = x_center.numpy() if hasattr(x_center, 'numpy') else np.asarray(x_center)
    lb_np = (x_center_np - radius).reshape(-1, 1)
    ub_np = (x_center_np + radius).reshape(-1, 1)
    box = Box(lb_np, ub_np)
    input_star = box.to_star()
    nn_wrapper = NeuralNetwork(classifier.net)

    t0 = _time.time()

    # Try approx first
    try:
        output_stars = nn_wrapper.reach(input_star, method='approx')
        verified, failures = _verify_spec_on_stars(
            output_stars, true_class, n_classes=3,
        )
        if verified:
            return {
                'spec_holds': True,
                'method': 'approx',
                'max_failing_margin': 0.0,
                'n_output_stars': len(output_stars),
                'time': _time.time() - t0,
            }
    except Exception as e:
        print(f"  approx: ERROR: {e}")

    # Fall back to exact with timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(exact_timeout_seconds)
    try:
        output_stars = nn_wrapper.reach(input_star, method='exact')
        signal.alarm(0)
        verified, failures = _verify_spec_on_stars(
            output_stars, true_class, n_classes=3,
        )
        max_margin = 0.0
        if not verified and failures:
            margin_values = [f[2] for f in failures if f[2] is not None]
            if margin_values:
                max_margin = max(margin_values)
        return {
            'spec_holds': verified,
            'method': 'exact',
            'max_failing_margin': max_margin,
            'n_output_stars': len(output_stars),
            'time': _time.time() - t0,
        }
    except _VerificationTimeoutError:
        signal.alarm(0)
        return {
            'spec_holds': None,
            'method': 'approx-fail-exact-timeout',
            'max_failing_margin': None,
            'n_output_stars': 0,
            'time': _time.time() - t0,
        }
    except Exception as e:
        signal.alarm(0)
        return {
            'spec_holds': None,
            'method': 'error',
            'max_failing_margin': None,
            'n_output_stars': 0,
            'time': _time.time() - t0,
        }


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_coverage_validation.csv')

# ---- Experiment configuration (preliminary scale) ----
N_TEST_INPUTS = 5
N_SEEDS = 5
PERTURBATION_RADIUS = 0.5

# Per-run pipeline parameters
N_TRAIN = 2000
N_CALIB = 8000  # Hashemi standard, gives delta_1 ~ 0.997 with epsilon_1=0.001
N_TEST = 10000  # more samples for tighter empirical rate estimates

# Flow training
FLOW_HIDDEN = 64
FLOW_N_LAYERS = 4  # more capacity for a tighter flow
FLOW_N_EPOCHS = 300  # more training to reduce flow set slack
FLOW_BATCH_SIZE = 256
FLOW_COUPLING = 'sinkhorn'

# Conformal and scenario parameters
EPSILON_1 = 0.001
BETA_2 = 0.001
N_SCENARIO_SAMPLES = 20000  # tighter epsilon_2 and lower chance of false hallucination finds

# Ground truth robustness MC
GROUND_TRUTH_SAMPLES = 100_000


class CenteredFlowScore:
    """Wraps a FlowScore to center inputs before scoring."""

    def __init__(self, flow_score, center):
        self.flow_score = flow_score
        self.center = center

    def __call__(self, y):
        return self.flow_score(y - self.center)


def select_test_inputs(classifier):
    """
    Select 5 test inputs spanning robustness difficulty.

    Samples a large pool of candidates, computes each candidate's
    softmax margin, and picks 5 candidates whose margins are evenly
    spaced in VALUE (not rank) between min and max. This gives a
    smoother difficulty spectrum than rank-based selection.
    """
    torch.manual_seed(999)
    n_candidates = 5000
    candidates, true_labels = classifier.sample_data(
        n_candidates, seed=999
    )
    with torch.no_grad():
        logits = classifier(candidates)
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    margins = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()

    # Pick N_TEST_INPUTS target margins evenly spaced between
    # the 5th percentile and 95th percentile (avoids extreme
    # outliers that would dominate the range)
    lo_margin = float(np.percentile(margins, 5))
    hi_margin = float(np.percentile(margins, 95))
    target_margins = np.linspace(lo_margin, hi_margin, N_TEST_INPUTS)

    selected_indices = []
    used = set()
    for target in target_margins:
        # Pick the candidate whose margin is closest to the target,
        # excluding already-selected candidates
        distances = np.abs(margins - target)
        for idx in np.argsort(distances):
            idx = int(idx)
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                break

    selected_indices = np.array(selected_indices)
    selected_x = candidates[selected_indices]
    selected_labels = true_labels[selected_indices]
    selected_margins = margins[selected_indices]
    return selected_x, selected_labels, selected_margins


def compute_ground_truth_robustness(classifier, x_center, true_label,
                                    radius, n_samples, seed):
    """
    Compute the empirical robustness rate by dense MC.

    Samples n_samples perturbed inputs uniformly from the L_inf ball
    of radius `radius` around `x_center`, classifies them, and returns
    the fraction correctly classified as `true_label`.
    """
    gen = torch.Generator().manual_seed(seed)
    # Uniform in L_inf ball = uniform per dimension in [-radius, radius]
    perturbations = (torch.rand(n_samples, 2, generator=gen) * 2 - 1) * radius
    x_perturbed = x_center + perturbations
    with torch.no_grad():
        preds = classifier(x_perturbed).argmax(dim=1)
    return (preds == true_label).float().mean().item()


def run_single_pipeline(classifier, x_center, true_label, radius, seed):
    """
    Run the full flow-conformal pipeline for a single
    (test_input, seed) combination.

    Returns a dict with all measurements needed for the CSV row.
    On flow training failure, returns a dict with NaN measurements
    and an error message.
    """
    run_start = time.time()
    try:
        gen = torch.Generator().manual_seed(seed)

        # Sample train/calib/test inputs uniformly from L_inf ball
        def sample_inputs(n):
            perturbations = (
                torch.rand(n, 2, generator=gen) * 2 - 1
            ) * radius
            return x_center + perturbations

        x_train = sample_inputs(N_TRAIN)
        x_calib = sample_inputs(N_CALIB)
        x_test = sample_inputs(N_TEST)

        with torch.no_grad():
            y_train = classifier(x_train)
            y_calib = classifier(x_calib)
            y_test = classifier(x_test)

        center = y_train.mean(dim=0)
        y_train_centered = y_train - center

        # Train flow
        torch.manual_seed(seed)
        vf = VelocityField(
            dim=3, hidden=FLOW_HIDDEN, n_layers=FLOW_N_LAYERS,
        )
        flow_ode = FlowODE(vf)
        flow_train_start = time.time()
        train_flow(
            vf, y_train_centered,
            n_epochs=FLOW_N_EPOCHS,
            batch_size=FLOW_BATCH_SIZE,
            lr=1e-3,
            coupling=FLOW_COUPLING,
        )
        flow_training_time = time.time() - flow_train_start

        # Calibrate flow score at t=1
        flow_score_raw = FlowScore(flow_ode, t=1.0)
        flow_score = CenteredFlowScore(flow_score_raw, center)
        with torch.no_grad():
            calib_scores = flow_score(y_calib)
        ell = N_CALIB - 1
        threshold_q = calibrate(calib_scores, ell).item()
        _, delta_1 = compute_guarantee(
            m=N_CALIB, ell=ell, epsilon=EPSILON_1
        )

        # Build ProbabilisticSet for membership queries
        pset = ProbabilisticSet(
            score_fn=flow_score,
            threshold=threshold_q,
            m=N_CALIB, ell=ell, epsilon=EPSILON_1,
            dim=3,
        )

        # Three empirical measurements on the test set
        with torch.no_grad():
            in_R = pset.contains(y_test)
            preds = y_test.argmax(dim=1)
            correct = (preds == true_label)

        conformal_coverage = in_R.float().mean().item()
        n_in_R = in_R.sum().item()
        if n_in_R > 0:
            scenario_violation_rate = (
                (~correct & in_R).sum().item() / n_in_R
            )
        else:
            scenario_violation_rate = float('nan')
        joint_spec_satisfaction = (in_R & correct).float().mean().item()

        # Run scenario verification with preimage search.
        # The flow operates on centered outputs; we tell verify_robustness
        # about the shift so it can correctly reconstruct raw outputs for
        # the classification spec check. target_fn is the RAW classifier.
        def target_fn(x):
            return classifier(x)

        output_shift_np = center.numpy()

        lb = (x_center - radius).numpy()
        ub = (x_center + radius).numpy()
        epsilon_2 = -np.log(BETA_2) / N_SCENARIO_SAMPLES
        delta_2 = 1.0 - BETA_2
        epsilon_total = 1.0 - (1.0 - EPSILON_1) * (1.0 - epsilon_2)
        delta_total = delta_1 * delta_2

        # Sample from empirical latent distribution instead of the
        # default truncated Gaussian prior. This keeps samples near the
        # real data manifold and avoids flow hallucinations.
        empirical_latent_samples = sample_empirical_latent_ball(
            flow_ode=flow_ode,
            y_train_centered=y_train_centered,
            q=threshold_q,
            n_samples=N_SCENARIO_SAMPLES,
            noise_sigma=0.1,
            seed=seed + 500,
        )

        rob_result = verify_robustness(
            flow_ode=flow_ode,
            threshold_q=threshold_q,
            true_class=int(true_label),
            n_classes=3,
            epsilon_1=EPSILON_1,
            delta_1=delta_1,
            n_samples=N_SCENARIO_SAMPLES,
            beta_2=BETA_2,
            t=1.0,
            target_fn=target_fn,
            input_set_bounds=(lb, ub),
            preimage_n_restarts=5,
            preimage_n_steps=100,
            preimage_tolerance=0.05,
            output_shift=output_shift_np,
            latent_samples=empirical_latent_samples,
        )
        outcome = rob_result.outcome

        per_run_time = time.time() - run_start

        return {
            'conformal_coverage': conformal_coverage,
            'scenario_violation_rate': scenario_violation_rate,
            'joint_spec_satisfaction': joint_spec_satisfaction,
            'outcome': outcome,
            'threshold_q': threshold_q,
            'epsilon_1': EPSILON_1,
            'delta_1': delta_1,
            'epsilon_2': epsilon_2,
            'delta_2': delta_2,
            'epsilon_total': epsilon_total,
            'delta_total': delta_total,
            'n_train': N_TRAIN,
            'n_calib': N_CALIB,
            'n_test': N_TEST,
            'n_scenario_samples': N_SCENARIO_SAMPLES,
            'flow_training_time': flow_training_time,
            'per_run_time': per_run_time,
            'error': None,
        }
    except Exception as e:
        warnings.warn(
            f"Pipeline run failed: {e}. Recording NaN row."
        )
        return {
            'conformal_coverage': float('nan'),
            'scenario_violation_rate': float('nan'),
            'joint_spec_satisfaction': float('nan'),
            'outcome': 'error',
            'threshold_q': float('nan'),
            'epsilon_1': EPSILON_1,
            'delta_1': float('nan'),
            'epsilon_2': float('nan'),
            'delta_2': float('nan'),
            'epsilon_total': float('nan'),
            'delta_total': float('nan'),
            'n_train': N_TRAIN,
            'n_calib': N_CALIB,
            'n_test': N_TEST,
            'n_scenario_samples': N_SCENARIO_SAMPLES,
            'flow_training_time': float('nan'),
            'per_run_time': time.time() - run_start,
            'error': str(e),
        }


def print_summary(df):
    """Print summary statistics of the results."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal runs: {len(df)}")
    print(f"Outcome distribution:")
    for outcome, count in df['outcome'].value_counts().items():
        print(f"  {outcome:>12}: {count}")

    # Theoretical bounds (all runs share the same parameters in preliminary)
    eps1 = EPSILON_1
    eps2 = -np.log(BETA_2) / N_SCENARIO_SAMPLES
    conformal_bound = 1 - eps1
    scenario_bound = eps2
    joint_bound = (1 - eps1) * (1 - eps2)

    clean_df = df.dropna(subset=['conformal_coverage'])

    def pct_satisfies(col, bound, direction):
        series = clean_df[col].dropna()
        if len(series) == 0:
            return float('nan')
        if direction == '>=':
            return (series >= bound).mean() * 100
        else:
            return (series <= bound).mean() * 100

    print(f"\nMeasurement statistics (across {len(clean_df)} successful runs):")
    print(f"  {'metric':<30} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("  " + "-" * 70)
    for col in ['conformal_coverage', 'scenario_violation_rate',
                'joint_spec_satisfaction', 'ground_truth_robustness']:
        if col in clean_df.columns:
            s = clean_df[col].dropna()
            if len(s) > 0:
                print(
                    f"  {col:<30} {s.mean():>10.4f} {s.std():>10.4f} "
                    f"{s.min():>10.4f} {s.max():>10.4f}"
                )

    print(f"\nTheoretical bounds:")
    print(f"  conformal coverage >= {conformal_bound:.6f}: "
          f"satisfied in {pct_satisfies('conformal_coverage', conformal_bound, '>='):.1f}% of runs")
    print(f"  scenario violation <= {scenario_bound:.6f}: "
          f"satisfied in {pct_satisfies('scenario_violation_rate', scenario_bound, '<='):.1f}% of runs")
    print(f"  joint satisfaction >= {joint_bound:.6f}: "
          f"satisfied in {pct_satisfies('joint_spec_satisfaction', joint_bound, '>='):.1f}% of runs")

    # Flag bound violations that exceed statistical expectation.
    # With confidence delta_1, we expect (1 - delta_1) fraction of runs
    # to fail the conformal bound by chance. We only warn if the observed
    # number significantly exceeds the expected number (3-sigma band).
    delta_1_value = float(clean_df['delta_1'].iloc[0]) if len(clean_df) > 0 else 1.0
    n_runs = len(clean_df)
    expected_failures = n_runs * (1 - delta_1_value)
    std_failures = np.sqrt(max(n_runs * (1 - delta_1_value) * delta_1_value, 0))
    violations = clean_df[clean_df['conformal_coverage'] < conformal_bound]
    n_violations = len(violations)
    three_sigma_threshold = expected_failures + 3 * std_failures

    print(
        f"\nConformal bound violations: {n_violations} observed, "
        f"~{expected_failures:.2f} expected (based on delta_1={delta_1_value:.4f}), "
        f"3-sigma threshold = {three_sigma_threshold:.2f}"
    )
    if n_violations > three_sigma_threshold and n_violations > 2:
        print(
            f"WARNING: observed violations exceed 3-sigma statistical "
            f"expectation. Framework may have a bug."
        )
    else:
        print(
            "Violations are within statistical expectation "
            "for the chosen delta_1."
        )

    # Cross-tabulate outcome vs n2v ground truth
    print(f"\nOutcome vs n2v ground truth:")
    print(f"{'n2v says':<15} {'verified':>10} {'falsified':>10} {'unknown':>10}")
    print("  " + "-" * 50)
    for n2v_label, n2v_value in [
        ('ROBUST (True)', True),
        ('NON-ROBUST', False),
        ('TIMEOUT', None),
    ]:
        subset = clean_df[clean_df['n2v_spec_holds'] == n2v_value] if n2v_value is not None else clean_df[clean_df['n2v_spec_holds'].isna()]
        n_verified = (subset['outcome'] == 'verified').sum()
        n_falsified = (subset['outcome'] == 'falsified').sum()
        n_unknown = (subset['outcome'] == 'unknown').sum()
        print(f"  {n2v_label:<15} {n_verified:>10} {n_falsified:>10} {n_unknown:>10}")

    # Flag any soundness violations
    sound_violations = clean_df[
        (clean_df['n2v_spec_holds'] == False) & (clean_df['outcome'] == 'verified')
    ]
    if len(sound_violations) > 0:
        print(f"\n*** SOUNDNESS VIOLATION: "
              f"{len(sound_violations)} runs returned 'verified' for "
              f"test inputs where n2v proved the spec is FALSE ***")

    # Flag false positives
    false_positives = clean_df[
        (clean_df['n2v_spec_holds'] == True) & (clean_df['outcome'] == 'falsified')
    ]
    if len(false_positives) > 0:
        print(f"\n*** FALSE POSITIVE: "
              f"{len(false_positives)} runs returned 'falsified' for "
              f"test inputs where n2v proved the spec holds ***")

    print("=" * 70)


def run():
    print("=" * 70)
    print("COVERAGE VALIDATION EXPERIMENT")
    print("=" * 70)

    # Build classifier (deterministic)
    print("\nTraining ThreeBlobClassifier...")
    torch.manual_seed(0)
    classifier = ThreeBlobClassifier()
    print("Classifier trained.")

    # Select test inputs spanning difficulty
    print(f"\nSelecting {N_TEST_INPUTS} test inputs...")
    test_inputs, test_labels, test_margins = select_test_inputs(classifier)
    print(f"{'id':>4} {'x[0]':>10} {'x[1]':>10} {'label':>6} {'margin':>10}")
    for i in range(N_TEST_INPUTS):
        print(f"{i:>4} {test_inputs[i, 0].item():>10.3f} "
              f"{test_inputs[i, 1].item():>10.3f} "
              f"{test_labels[i].item():>6} {test_margins[i]:>10.3f}")

    # Compute ground truth robustness (once per test input)
    print(f"\nComputing ground truth robustness "
          f"({GROUND_TRUTH_SAMPLES} MC samples per input)...")
    ground_truths = []
    for i in range(N_TEST_INPUTS):
        gt = compute_ground_truth_robustness(
            classifier=classifier,
            x_center=test_inputs[i],
            true_label=int(test_labels[i]),
            radius=PERTURBATION_RADIUS,
            n_samples=GROUND_TRUTH_SAMPLES,
            seed=1000 + i,
        )
        ground_truths.append(gt)
        print(f"  test input {i}: ground truth robustness = {gt:.4f}")

    # Compute n2v sound ground truth (once per test input)
    print(f"\nComputing n2v sound ground truth (approx + exact fallback)...")
    n2v_truths = []
    for i in range(N_TEST_INPUTS):
        n2v_result = compute_n2v_ground_truth(
            classifier=classifier,
            x_center=test_inputs[i],
            true_class=int(test_labels[i]),
            radius=PERTURBATION_RADIUS,
        )
        n2v_truths.append(n2v_result)
        spec_holds_str = (
            'ROBUST' if n2v_result['spec_holds'] is True
            else 'NON-ROBUST' if n2v_result['spec_holds'] is False
            else 'TIMEOUT'
        )
        margin_str = (
            f"margin={n2v_result['max_failing_margin']:.4f}"
            if n2v_result['max_failing_margin'] is not None
            else "margin=N/A"
        )
        print(
            f"  test input {i}: {spec_holds_str} "
            f"({n2v_result['method']}, {n2v_result['n_output_stars']} stars, "
            f"{margin_str}, {n2v_result['time']:.2f}s)"
        )

    # Run pipeline for each (test_input, seed)
    print(f"\nRunning {N_TEST_INPUTS * N_SEEDS} pipeline runs "
          f"({N_TEST_INPUTS} test inputs x {N_SEEDS} seeds)...")
    rows = []
    for i in range(N_TEST_INPUTS):
        for s in range(N_SEEDS):
            seed = i * 1000 + s
            print(f"\n  [run {len(rows) + 1}/"
                  f"{N_TEST_INPUTS * N_SEEDS}] "
                  f"test_input={i} seed={seed}...")
            measurements = run_single_pipeline(
                classifier=classifier,
                x_center=test_inputs[i],
                true_label=int(test_labels[i]),
                radius=PERTURBATION_RADIUS,
                seed=seed,
            )
            row = {
                'test_input_id': i,
                'seed_index': s,
                'seed': seed,
                'test_x0': test_inputs[i, 0].item(),
                'test_x1': test_inputs[i, 1].item(),
                'test_label': int(test_labels[i]),
                'test_margin': float(test_margins[i]),
                'perturbation_radius': PERTURBATION_RADIUS,
                'ground_truth_robustness': ground_truths[i],
                'n2v_spec_holds': n2v_truths[i]['spec_holds'],
                'n2v_method': n2v_truths[i]['method'],
                'n2v_max_failing_margin': n2v_truths[i]['max_failing_margin'],
                'n2v_n_output_stars': n2v_truths[i]['n_output_stars'],
                'n2v_verification_time': n2v_truths[i]['time'],
                **measurements,
            }
            rows.append(row)
            if measurements.get('error') is None:
                print(
                    f"    coverage={measurements['conformal_coverage']:.4f} "
                    f"violation={measurements['scenario_violation_rate']:.4f} "
                    f"joint={measurements['joint_spec_satisfaction']:.4f} "
                    f"outcome={measurements['outcome']} "
                    f"time={measurements['per_run_time']:.1f}s"
                )
            else:
                print(f"    ERROR: {measurements['error']}")

    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved CSV to {CSV_PATH}")

    # Print summary
    print_summary(df)


if __name__ == '__main__':
    run()
