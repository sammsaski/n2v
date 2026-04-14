"""
Hashemi Comparison Experiment.

Head-to-head comparison of three methods for probabilistic reach-set
verification on two small networks:

  1. Flow-conformal (ours): flow matching + conformal calibration.
  2. Clipping-block conformal (Hashemi Paper 2): convex-hull surrogate
     via n2v's `verify()` — now audited + patched to match the paper.
  3. Exact reach set (ground truth): n2v Star propagation.

Networks (preliminary scale, both small enough for sound ground truth):

  - RotatedBananaNet          (2 -> 2, regression)    halfspace spec
  - ThreeBlobClassifier       (2 -> 3, logits)        classification robustness

For each (network, test_input, radius, seed):
  - sample train/calib/test inputs from an L_inf ball around test_input
  - flow path: train + calibrate + volume + coverage + verdict
  - clip path: n2v.verify() + volume + coverage + verdict
  - exact path (cached per test_input/radius): star propagation + volume (2D)
  - record all metrics into exp_hashemi_comparison.csv

The companion plot_hashemi_comparison.py produces the figures.
"""

import os
import sys
import time
import warnings
import copy

import numpy as np
import pandas as pd
import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from scipy.spatial import ConvexHull
import matplotlib.path as mpath
from scipy.ndimage import binary_closing, binary_fill_holes

from n2v.probabilistic import verify as n2v_verify
from n2v.probabilistic.flow import (
    VelocityField, FlowODE, FlowScore, calibrate, compute_guarantee,
    ProbabilisticSet, verify_robustness, sample_empirical_latent_ball,
    scenario_verify_halfspace, train_flow,
)
from n2v.sets.box import Box
from n2v.nn import NeuralNetwork
from n2v.utils.lpsolver import solve_lp

from FlowConformal.networks import RotatedBananaNet, ThreeBlobClassifier


# ---------- Experiment configuration ----------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison.csv')

N_SEEDS = 3

# Calibration / flow hyperparameters (match coverage_validation defaults).
N_TRAIN = 2000
N_CALIB = 8000
N_TEST = 5000
FLOW_HIDDEN = 64
FLOW_N_LAYERS = 4
FLOW_EPOCHS = 100
FLOW_BATCH_SIZE = 256
FLOW_LR = 1e-3
FLOW_COUPLING = 'sinkhorn'

EPSILON_1 = 0.001
BETA_2 = 0.001
N_SCENARIO_SAMPLES = 20000
SCENARIO_NOISE_SIGMA = 0.1

# Volume MC
N_VOLUME_MC = 100_000


# ---------- Network and spec definitions ----------

def build_networks():
    torch.manual_seed(0)
    banana = RotatedBananaNet()
    torch.manual_seed(0)
    classifier = ThreeBlobClassifier()
    return {'RotatedBananaNet': banana, 'ThreeBlobClassifier': classifier}


def banana_test_inputs():
    """Three fixed test inputs in the banana's [0, 1]^2 input range."""
    return torch.tensor([
        [0.2, 0.2],
        [0.5, 0.5],
        [0.8, 0.8],
    ], dtype=torch.float32)


BANANA_RADII = [0.05, 0.1, 0.2, 0.4]

# Halfspace spec for the banana: y_0 <= 0.6.
# w = [1, 0], b = 0.6, check w^T y <= b.
BANANA_SPEC = {'type': 'halfspace', 'w': np.array([1.0, 0.0]), 'b': 0.6}


def classifier_test_inputs(classifier):
    """Three test inputs spanning the robustness difficulty spectrum of
    the three-blob classifier, selected by softmax margin."""
    torch.manual_seed(999)
    n_candidates = 2000
    candidates, labels = classifier.sample_data(n_candidates, seed=999)
    with torch.no_grad():
        logits = classifier(candidates)
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    margins = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()

    lo = float(np.percentile(margins, 5))
    hi = float(np.percentile(margins, 95))
    target_margins = np.linspace(lo, hi, 3)

    selected = []
    used = set()
    for target in target_margins:
        idx_order = np.argsort(np.abs(margins - target))
        for idx in idx_order:
            idx = int(idx)
            if idx not in used:
                selected.append(idx)
                used.add(idx)
                break
    selected = np.array(selected)
    return (
        candidates[selected],
        labels[selected],
        margins[selected],
    )


CLASSIFIER_RADII = [0.1, 0.25, 0.5, 1.0]


# ---------- Helper: sample L_inf ball ----------

def sample_l_inf_ball(x_center, radius, n_samples, seed, dim):
    gen = torch.Generator().manual_seed(seed)
    perturbations = (torch.rand(n_samples, dim, generator=gen) * 2 - 1) * radius
    return x_center + perturbations


# ---------- Helper: halfspace LP on a Star ----------

def _project_max_halfspace(star, direction):
    """Max of direction^T y over a Star (polytope image). Returns None if LP fails."""
    offset = float(direction @ star.V[:, 0])
    if star.nVar == 0:
        return offset
    obj = (direction @ star.V[:, 1:]).flatten().reshape(-1, 1)
    A = star.C if (star.C is not None and star.C.size > 0) else None
    b = star.d if (star.d is not None and star.d.size > 0) else None
    lb = star.predicate_lb if star.predicate_lb is not None else None
    ub = star.predicate_ub if star.predicate_ub is not None else None
    _, fval, _, _ = solve_lp(f=obj, A=A, b=b, lb=lb, ub=ub, minimize=False)
    if fval is None:
        return None
    return offset + fval


def verify_halfspace_on_stars(stars, w, b):
    """Check if every star's image satisfies w^T y <= b."""
    max_over_all = -np.inf
    for star in stars:
        m = _project_max_halfspace(star, w)
        if m is None:
            return ('unknown', None)
        max_over_all = max(max_over_all, m)
    return (
        ('verified', max_over_all) if max_over_all <= b
        else ('falsified', max_over_all)
    )


def verify_classification_on_stars(stars, true_class, n_classes):
    """Classification robustness: all stars must have max(y_k - y_true) <= 0
    for every wrong class k."""
    worst_margin = -np.inf
    for star in stars:
        for k in range(n_classes):
            if k == true_class:
                continue
            w = np.zeros(n_classes)
            w[k] = 1.0
            w[true_class] = -1.0
            m = _project_max_halfspace(star, w)
            if m is None:
                return ('unknown', None)
            worst_margin = max(worst_margin, m)
            if worst_margin > 0:
                return ('falsified', worst_margin)
    return ('verified', worst_margin)


# ---------- Helper: exact reach + volume (2D only) ----------

def compute_exact_reach(net, x_center_np, radius, output_dim):
    """Run n2v exact star propagation through the network on the L_inf
    input box. Returns (stars, time, n_stars)."""
    lb = (x_center_np - radius).reshape(-1, 1)
    ub = (x_center_np + radius).reshape(-1, 1)
    input_star = Box(lb, ub).to_star()
    wrapper = NeuralNetwork(net.net)
    t0 = time.time()
    try:
        stars = wrapper.reach(input_star, method='approx')
        method = 'approx'
    except Exception:
        stars = None
        method = 'approx-failed'
    # Fall back to exact if approx returns a single loose box
    if stars is None or len(stars) == 1:
        try:
            stars = wrapper.reach(input_star, method='exact')
            method = 'exact'
        except Exception as e:
            return {'stars': [], 'method': f'error:{e}', 'time': time.time() - t0}
    return {
        'stars': stars,
        'method': method,
        'time': time.time() - t0,
    }


def star_polygon_2d(star, rng, n_samples=600):
    """For a 2D output star, sample the predicate polytope and return the
    convex hull of the image points. None if degenerate."""
    V = star.V
    offset = V[:, 0]
    basis = V[:, 1:]
    nVar = star.nVar
    if nVar == 0:
        return None
    plb = star.predicate_lb.flatten()
    pub = star.predicate_ub.flatten()
    accepted = []
    for _ in range(40):
        alpha = rng.uniform(plb, pub, size=(n_samples * 4, nVar))
        if star.C is not None and star.C.size > 0:
            mask = (star.C @ alpha.T <= star.d + 1e-9).all(axis=0)
            alpha = alpha[mask]
        if len(alpha) > 0:
            accepted.append(alpha)
            if sum(len(a) for a in accepted) >= n_samples:
                break
    if not accepted:
        return None
    alpha_all = np.vstack(accepted)
    y_samples = offset[None, :] + alpha_all @ basis.T
    if len(y_samples) < 3:
        return None
    if np.ptp(y_samples, axis=0).min() < 1e-6:
        return None
    try:
        hull = ConvexHull(y_samples)
    except Exception:
        return None
    return y_samples[hull.vertices]


def exact_volume_2d(stars, bbox, resolution=500):
    """2D volume of the union of star polygons, via occupancy rasterization."""
    rng = np.random.default_rng(0)
    polygons = []
    for s in stars:
        poly = star_polygon_2d(s, rng)
        if poly is not None:
            polygons.append(poly)
    if not polygons:
        return 0.0
    xmin, ymin = bbox[0]
    xmax, ymax = bbox[1]
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    occ = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        occ |= mpath.Path(poly).contains_points(pts)
    mask = occ.reshape(resolution, resolution)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    cell_area = (xmax - xmin) * (ymax - ymin) / (resolution * resolution)
    return float(mask.sum() * cell_area)


# ---------- Centered flow score wrapper ----------

class CenteredFlowScore:
    def __init__(self, flow_score, center):
        self.flow_score = flow_score
        self.center = center

    def __call__(self, y):
        return self.flow_score(y - self.center)


# ---------- Flow pipeline for a single run ----------

def run_flow_pipeline(
    network,
    x_train, x_calib, x_test, y_train, y_calib, y_test,
    spec, output_dim, mc_bbox, x_center_np, radius, seed,
):
    t0 = time.time()

    center = y_train.mean(dim=0)
    y_train_c = y_train - center

    torch.manual_seed(seed)
    vf = VelocityField(dim=output_dim, hidden=FLOW_HIDDEN, n_layers=FLOW_N_LAYERS)
    flow_ode = FlowODE(vf)
    train_flow(
        vf, y_train_c,
        n_epochs=FLOW_EPOCHS,
        batch_size=FLOW_BATCH_SIZE,
        lr=FLOW_LR,
        coupling=FLOW_COUPLING,
    )

    flow_score_raw = FlowScore(flow_ode, t=1.0)
    flow_score = CenteredFlowScore(flow_score_raw, center)

    with torch.no_grad():
        calib_scores = flow_score(y_calib)
    ell = N_CALIB - 1
    threshold = calibrate(calib_scores, ell).item()
    _, delta_1 = compute_guarantee(m=N_CALIB, ell=ell, epsilon=EPSILON_1)

    pset = ProbabilisticSet(
        score_fn=flow_score, threshold=threshold,
        m=N_CALIB, ell=ell, epsilon=EPSILON_1, dim=output_dim,
    )

    volume, _ = pset.estimate_volume(n_samples=N_VOLUME_MC, bounding_box=mc_bbox)

    with torch.no_grad():
        coverage = pset.contains(y_test).float().mean().item()

    # Verdict via scenario verification.
    if spec['type'] == 'halfspace':
        empirical_latent = sample_empirical_latent_ball(
            flow_ode=flow_ode,
            y_train_centered=y_train_c,
            q=threshold,
            n_samples=N_SCENARIO_SAMPLES,
            noise_sigma=SCENARIO_NOISE_SIGMA,
            seed=seed + 500,
        )
        lb_in = x_center_np - radius
        ub_in = x_center_np + radius
        result = scenario_verify_halfspace(
            flow_ode=flow_ode,
            threshold_q=threshold,
            w=spec['w'],
            b=spec['b'],
            n_samples=N_SCENARIO_SAMPLES,
            beta_2=BETA_2,
            t=1.0,
            output_shift=center.numpy(),
            latent_samples=empirical_latent,
            target_fn=lambda x_t: network(x_t),
            input_set_bounds=(lb_in, ub_in),
            preimage_n_restarts=5,
            preimage_n_steps=100,
            preimage_tolerance=0.05,
        )
        verdict = result.outcome
    else:
        true_class = int(spec['true_class'])
        n_classes = output_dim
        empirical_latent = sample_empirical_latent_ball(
            flow_ode=flow_ode,
            y_train_centered=y_train_c,
            q=threshold,
            n_samples=N_SCENARIO_SAMPLES,
            noise_sigma=SCENARIO_NOISE_SIGMA,
            seed=seed + 500,
        )
        lb_in = x_center_np - radius
        ub_in = x_center_np + radius
        result = verify_robustness(
            flow_ode=flow_ode,
            threshold_q=threshold,
            true_class=true_class,
            n_classes=n_classes,
            epsilon_1=EPSILON_1,
            delta_1=delta_1,
            n_samples=N_SCENARIO_SAMPLES,
            beta_2=BETA_2,
            t=1.0,
            target_fn=lambda x_t: network(x_t),
            input_set_bounds=(lb_in, ub_in),
            preimage_n_restarts=5,
            preimage_n_steps=100,
            preimage_tolerance=0.05,
            output_shift=center.numpy(),
            latent_samples=empirical_latent,
        )
        verdict = result.outcome

    runtime = time.time() - t0
    return {
        'flow_volume': volume,
        'flow_coverage': coverage,
        'flow_threshold': threshold,
        'flow_delta_1': delta_1,
        'flow_verdict': verdict,
        'flow_runtime': runtime,
    }


# ---------- Clip pipeline for a single run ----------

def run_clip_pipeline(
    network, x_center_np, radius, spec, output_dim, y_test, seed,
):
    t0 = time.time()

    lb_in = (x_center_np - radius).reshape(-1, 1)
    ub_in = (x_center_np + radius).reshape(-1, 1)
    input_box = Box(lb_in, ub_in)

    def model_fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return network(x_t).numpy()

    clip = n2v_verify(
        model=model_fn,
        input_set=input_box,
        m=N_CALIB,
        ell=N_CALIB - 1,
        epsilon=EPSILON_1,
        surrogate='clipping_block',
        training_samples=N_TRAIN,
        seed=seed,
        verbose=False,
    )

    clip_lb = clip.lb.flatten()
    clip_ub = clip.ub.flatten()
    volume = float(np.prod(clip_ub - clip_lb))

    with torch.no_grad():
        clip_lb_t = torch.tensor(clip_lb, dtype=torch.float32)
        clip_ub_t = torch.tensor(clip_ub, dtype=torch.float32)
        inside = ((y_test >= clip_lb_t) & (y_test <= clip_ub_t)).all(dim=1)
    coverage = inside.float().mean().item()

    # Verdict: check spec against the box (verified / unknown).
    if spec['type'] == 'halfspace':
        w = spec['w']
        b = spec['b']
        max_wty = np.sum(
            np.maximum(w * clip_lb, w * clip_ub)
        )
        verdict = 'verified' if max_wty <= b else 'unknown'
    else:
        true_class = int(spec['true_class'])
        worst_margin = -np.inf
        for k in range(output_dim):
            if k == true_class:
                continue
            max_margin = clip_ub[k] - clip_lb[true_class]
            if max_margin > worst_margin:
                worst_margin = max_margin
        verdict = 'verified' if worst_margin <= 0 else 'unknown'

    runtime = time.time() - t0
    return {
        'clip_volume': volume,
        'clip_coverage': coverage,
        'clip_lb': clip_lb.tolist(),
        'clip_ub': clip_ub.tolist(),
        'clip_verdict': verdict,
        'clip_runtime': runtime,
    }


# ---------- Exact ground truth (cached) ----------

class ExactCache:
    def __init__(self):
        self.cache = {}

    def get(self, network_name, network, x_center_np, radius, spec, output_dim, mc_bbox):
        key = (network_name, tuple(x_center_np.tolist()), radius)
        if key in self.cache:
            return self.cache[key]
        reach = compute_exact_reach(network, x_center_np, radius, output_dim)
        stars = reach.get('stars', [])
        exact_result = {
            'n_stars': len(stars),
            'method': reach.get('method'),
            'time': reach.get('time'),
        }
        if not stars:
            exact_result.update({
                'verdict': 'unknown',
                'volume': float('nan'),
                'stars_handle': None,
            })
            self.cache[key] = exact_result
            return exact_result

        # Verdict
        if spec['type'] == 'halfspace':
            verdict, worst = verify_halfspace_on_stars(stars, spec['w'], spec['b'])
        else:
            verdict, worst = verify_classification_on_stars(
                stars, int(spec['true_class']), output_dim
            )
        exact_result['verdict'] = verdict
        exact_result['worst_margin'] = worst

        # Volume (2D only)
        if output_dim == 2:
            exact_result['volume'] = exact_volume_2d(stars, mc_bbox)
        else:
            exact_result['volume'] = float('nan')

        exact_result['stars_handle'] = stars
        self.cache[key] = exact_result
        return exact_result


# ---------- Per-run orchestrator ----------

def compute_mc_bbox(network, x_center_np, radius, output_dim, n_samples=5000, pad=1.0):
    """Bounding box covering the reach set for MC volume estimation."""
    rng = torch.Generator().manual_seed(12345)
    x = torch.tensor(x_center_np, dtype=torch.float32) + (
        torch.rand(n_samples, x_center_np.shape[0], generator=rng) * 2 - 1
    ) * radius
    with torch.no_grad():
        y = network(x).numpy()
    lo = y.min(axis=0) - pad
    hi = y.max(axis=0) + pad
    if output_dim == 2:
        return (
            torch.tensor([lo[0], lo[1]], dtype=torch.float32),
            torch.tensor([hi[0], hi[1]], dtype=torch.float32),
        )
    elif output_dim == 3:
        return (
            torch.tensor([lo[0], lo[1], lo[2]], dtype=torch.float32),
            torch.tensor([hi[0], hi[1], hi[2]], dtype=torch.float32),
        )
    else:
        return (
            torch.tensor(lo, dtype=torch.float32),
            torch.tensor(hi, dtype=torch.float32),
        )


def cross_check_verdict(method_verdict, exact_verdict):
    """Agreement label for the soundness cross-check."""
    if exact_verdict == 'unknown' or method_verdict == 'unknown':
        if method_verdict == 'unknown':
            return 'conservative_unknown'
        return 'exact_unknown'
    if method_verdict == 'verified' and exact_verdict == 'verified':
        return 'match'
    if method_verdict == 'falsified' and exact_verdict == 'falsified':
        return 'match'
    if method_verdict == 'verified' and exact_verdict == 'falsified':
        return 'false_positive'  # SOUNDNESS BUG
    if method_verdict == 'falsified' and exact_verdict == 'verified':
        return 'false_negative'  # METHOD WRONG
    return 'unknown'


def run_single(
    network_name, network, test_input_id, x_center, radius, seed,
    spec, output_dim, exact_cache,
):
    """Sample data, run flow + clip + exact, return flat dict of metrics."""
    x_center_np = x_center.numpy()

    # Sample shared calibration data for flow + clip
    x_train = sample_l_inf_ball(x_center, radius, N_TRAIN,
                                 seed=seed, dim=x_center.shape[0])
    x_calib = sample_l_inf_ball(x_center, radius, N_CALIB,
                                 seed=seed + 1_000_000, dim=x_center.shape[0])
    x_test = sample_l_inf_ball(x_center, radius, N_TEST,
                                seed=seed + 2_000_000, dim=x_center.shape[0])

    with torch.no_grad():
        y_train = network(x_train)
        y_calib = network(x_calib)
        y_test = network(x_test)

    mc_bbox = compute_mc_bbox(network, x_center_np, radius, output_dim)

    # Flow
    flow_metrics = run_flow_pipeline(
        network=network,
        x_train=x_train, x_calib=x_calib, x_test=x_test,
        y_train=y_train, y_calib=y_calib, y_test=y_test,
        spec=spec, output_dim=output_dim, mc_bbox=mc_bbox,
        x_center_np=x_center_np, radius=radius, seed=seed,
    )

    # Clip
    clip_metrics = run_clip_pipeline(
        network=network, x_center_np=x_center_np, radius=radius,
        spec=spec, output_dim=output_dim, y_test=y_test, seed=seed,
    )

    # Exact (cached per test_input, radius)
    exact_metrics = exact_cache.get(
        network_name, network, x_center_np, radius, spec, output_dim, mc_bbox,
    )

    # Cross-checks against exact ground truth
    flow_vs_exact = cross_check_verdict(
        flow_metrics['flow_verdict'], exact_metrics['verdict']
    )
    clip_vs_exact = cross_check_verdict(
        clip_metrics['clip_verdict'], exact_metrics['verdict']
    )

    # Volume ratios
    flow_vol = flow_metrics['flow_volume']
    clip_vol = clip_metrics['clip_volume']
    exact_vol = exact_metrics['volume']
    ratio_clip_over_flow = clip_vol / max(flow_vol, 1e-12)
    ratio_clip_over_exact = (
        clip_vol / max(exact_vol, 1e-12)
        if exact_vol and np.isfinite(exact_vol) else float('nan')
    )
    ratio_flow_over_exact = (
        flow_vol / max(exact_vol, 1e-12)
        if exact_vol and np.isfinite(exact_vol) else float('nan')
    )

    row = {
        'network': network_name,
        'test_input_id': test_input_id,
        'test_x0': float(x_center[0]),
        'test_x1': float(x_center[1]),
        'perturbation_radius': radius,
        'seed': seed,
        'output_dim': output_dim,
        'spec_type': spec['type'],
        'spec_params': (
            f"w={spec['w'].tolist()},b={spec['b']}"
            if spec['type'] == 'halfspace'
            else f"true_class={spec['true_class']}"
        ),
        **flow_metrics,
        **clip_metrics,
        'exact_volume': exact_vol,
        'exact_verdict': exact_metrics['verdict'],
        'exact_n_stars': exact_metrics['n_stars'],
        'exact_time': exact_metrics['time'],
        'exact_method': exact_metrics['method'],
        'volume_ratio_clip_over_flow': ratio_clip_over_flow,
        'volume_ratio_clip_over_exact': ratio_clip_over_exact,
        'volume_ratio_flow_over_exact': ratio_flow_over_exact,
        'flow_vs_exact': flow_vs_exact,
        'clip_vs_exact': clip_vs_exact,
    }
    return row


def run():
    print("=" * 72)
    print("HASHEMI COMPARISON EXPERIMENT")
    print("=" * 72)

    networks = build_networks()
    exact_cache = ExactCache()
    rows = []
    run_idx = 0

    # ---- RotatedBananaNet ----
    banana = networks['RotatedBananaNet']
    banana_inputs = banana_test_inputs()
    n_runs_banana = banana_inputs.shape[0] * len(BANANA_RADII) * N_SEEDS
    print(f"\nBanana: {banana_inputs.shape[0]} test inputs × {len(BANANA_RADII)} radii × {N_SEEDS} seeds = {n_runs_banana} runs")

    for ti_id in range(banana_inputs.shape[0]):
        x_center = banana_inputs[ti_id]
        for radius in BANANA_RADII:
            for s in range(N_SEEDS):
                run_idx += 1
                seed = ti_id * 10000 + int(radius * 1000) * 100 + s
                t0 = time.time()
                row = run_single(
                    network_name='RotatedBananaNet',
                    network=banana,
                    test_input_id=ti_id,
                    x_center=x_center,
                    radius=radius,
                    seed=seed,
                    spec=BANANA_SPEC,
                    output_dim=2,
                    exact_cache=exact_cache,
                )
                rows.append(row)
                dt = time.time() - t0
                print(
                    f"  [{run_idx}] banana ti={ti_id} r={radius} s={s} "
                    f"volF={row['flow_volume']:.3f} volC={row['clip_volume']:.3f} "
                    f"volE={row['exact_volume']:.3f} "
                    f"vF={row['flow_verdict']} vC={row['clip_verdict']} vE={row['exact_verdict']} "
                    f"({dt:.1f}s)"
                )

    # ---- ThreeBlobClassifier ----
    classifier = networks['ThreeBlobClassifier']
    clf_inputs, clf_labels, clf_margins = classifier_test_inputs(classifier)
    n_runs_clf = clf_inputs.shape[0] * len(CLASSIFIER_RADII) * N_SEEDS
    print(f"\nClassifier: {clf_inputs.shape[0]} test inputs × {len(CLASSIFIER_RADII)} radii × {N_SEEDS} seeds = {n_runs_clf} runs")
    for i in range(clf_inputs.shape[0]):
        print(f"  TI {i}: x={clf_inputs[i].tolist()} label={int(clf_labels[i])} margin={clf_margins[i]:.2f}")

    for ti_id in range(clf_inputs.shape[0]):
        x_center = clf_inputs[ti_id]
        true_class = int(clf_labels[ti_id])
        spec = {'type': 'classification_robustness', 'true_class': true_class}
        for radius in CLASSIFIER_RADII:
            for s in range(N_SEEDS):
                run_idx += 1
                seed = 100_000 + ti_id * 10000 + int(radius * 1000) * 100 + s
                t0 = time.time()
                row = run_single(
                    network_name='ThreeBlobClassifier',
                    network=classifier,
                    test_input_id=ti_id,
                    x_center=x_center,
                    radius=radius,
                    seed=seed,
                    spec=spec,
                    output_dim=3,
                    exact_cache=exact_cache,
                )
                rows.append(row)
                dt = time.time() - t0
                print(
                    f"  [{run_idx}] classifier ti={ti_id} r={radius} s={s} "
                    f"volF={row['flow_volume']:.3f} volC={row['clip_volume']:.3f} "
                    f"vF={row['flow_verdict']} vC={row['clip_verdict']} vE={row['exact_verdict']} "
                    f"({dt:.1f}s)"
                )

    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {CSV_PATH}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for network_name in ['RotatedBananaNet', 'ThreeBlobClassifier']:
        sub = df[df['network'] == network_name]
        if len(sub) == 0:
            continue
        print(f"\n[{network_name}] {len(sub)} runs")

        for radius in sorted(sub['perturbation_radius'].unique()):
            r_sub = sub[sub['perturbation_radius'] == radius]
            mean_ratio = r_sub['volume_ratio_clip_over_flow'].mean()
            median_ratio = r_sub['volume_ratio_clip_over_flow'].median()
            flow_v = (r_sub['flow_verdict'] == 'verified').sum()
            clip_v = (r_sub['clip_verdict'] == 'verified').sum()
            exact_v = (r_sub['exact_verdict'] == 'verified').sum()
            print(
                f"  r={radius:.2f}: "
                f"clip/flow ratio mean={mean_ratio:.2f} median={median_ratio:.2f}  "
                f"verified counts F={flow_v} C={clip_v} E={exact_v} / {len(r_sub)}"
            )

    # Soundness cross-check
    fp_flow = df[df['flow_vs_exact'] == 'false_positive']
    fp_clip = df[df['clip_vs_exact'] == 'false_positive']
    if len(fp_flow) > 0:
        print(f"\n*** SOUNDNESS: {len(fp_flow)} flow false-positives "
              "(flow=verified, exact=falsified) ***")
    if len(fp_clip) > 0:
        print(f"\n*** SOUNDNESS: {len(fp_clip)} clip false-positives "
              "(clip=verified, exact=falsified) ***")
    if len(fp_flow) == 0 and len(fp_clip) == 0:
        print("\nNo soundness violations across all runs.")

    # Runtime summary
    mean_flow_time = df['flow_runtime'].mean()
    mean_clip_time = df['clip_runtime'].mean()
    print(f"\nMean runtime: flow={mean_flow_time:.2f}s, clip={mean_clip_time:.2f}s")


if __name__ == '__main__':
    run()
