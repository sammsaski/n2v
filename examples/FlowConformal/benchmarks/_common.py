"""Shared pipeline runner for PoC benchmarks with curved output sets.

Unlike golden-path (identity / rotated linear), these benchmarks have no
analytical exact reach-set volume. The ground-truth 1-alpha probabilistic
reachset volume is estimated from the Star-union pushforward:

    vol_exact(1-alpha) ~= (1 - alpha) * vol(Star_union)

because P_X is uniform on B(x_0, eps) and the Star union is the exact
deterministic pushforward of that input box. The smallest 1-alpha reachset
under a uniform distribution on a set is the set itself times (1-alpha)
mass — this is the tightness floor any conformal score is measured against.

Compares hyperrect / ball / flow scores against the Star-union reference.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.utils import compute_exact_reach
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore  # noqa: F401 (available to callers)
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sampling import sample_l_inf_ball
from n2v.probabilistic.flow.scores import BallScore, FlowScore, HyperrectScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.train import train_flow
from n2v.sets.volume import (
    compute_mc_bbox, exact_volume_2d, star_union_volume_mc,
)
from n2v.sets.halfspace import HalfSpace
from n2v.utils.falsify import falsify


# ---- Whitening glue for run_verification_pipeline ----
#
# ACAS Xu networks produce outputs whose per-dim std is 1e-4 to 1e-2 on
# the tight VNN-LIB input boxes. A flow trained with OT-CFM cannot bridge
# that ~1000x scale gap in finite time — it converges to a near-identity
# that leaves ||phi(y)|| dominated by the data-space offset ||mu||.
#
# The fix below pre-whitens y in the pipeline before the flow sees it,
# trains with the model's internal standardize_outputs disabled (to avoid
# double-whitening), and transforms the halfspace spec into whitened
# coordinates so verify_spec_on_flow operates in the same frame. All
# joint conformal-scenario guarantees carry over unchanged because the
# whitening transform (y - mu) / sigma is a deterministic invertible
# affine map and the flow is the interesting non-linear part downstream.
#
# Future work: promote this glue to an ``n2v.probabilistic.flow.WhiteningLayer``
# ``nn.Module`` that wraps the network with full covariance whitening
# (Sigma^(-1/2) instead of per-dim sigma), and register it as a buffer on
# the VelocityField so the flow model carries its own coordinate frame
# without pipeline-level coordination. See docs/audits/2026-04-24-
# phase3-acasxu-sweep-results.md "Future work".


class _WhitenedNetwork:
    """Callable wrapper around a network that whitens its outputs.

    ``whitened_network(x) = (network(x) - mu) / sigma`` (per-dim).

    Used to hand scenario-verify / preimage-search a network whose
    outputs live in the same whitened coordinates as the flow.
    """

    def __init__(self, net, y_mean: torch.Tensor, y_std: torch.Tensor):
        self.net = net
        self.y_mean = y_mean
        self.y_std = y_std

    def __call__(self, x):
        y = self.net(x)
        dev = y.device
        return (y - self.y_mean.to(dev)) / self.y_std.to(dev)

    def eval(self):
        if hasattr(self.net, 'eval'):
            self.net.eval()
        return self

    def parameters(self):
        if hasattr(self.net, 'parameters'):
            yield from self.net.parameters()


class _WhiteningFlowScore:
    """Score function that whitens its input before delegating.

    Lets callers (e.g. volume validation) keep passing raw network
    outputs: whitening happens transparently before the underlying
    :class:`FlowScore` operates.
    """

    def __init__(self, base_score_fn, y_mean: torch.Tensor, y_std: torch.Tensor):
        self.base = base_score_fn
        self.y_mean = y_mean
        self.y_std = y_std

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        dev = y.device
        y_w = (y - self.y_mean.to(dev)) / self.y_std.to(dev)
        return self.base(y_w)

    @property
    def flow_model(self):
        return self.base.flow_model


def _whiten_halfspace(spec: HalfSpace, y_mean: np.ndarray,
                       y_std: np.ndarray) -> HalfSpace:
    """Transform ``G @ y <= g`` to the equivalent constraint on whitened
    coordinates ``y_w = (y - mu) / sigma``:

        G @ y <= g
        G @ (sigma * y_w + mu) <= g
        (G * sigma) @ y_w <= g - G @ mu
    """
    sigma = np.asarray(y_std, dtype=np.float64).flatten()
    mu = np.asarray(y_mean, dtype=np.float64).flatten()
    G_white = spec.G * sigma[None, :]  # row-wise elementwise scale
    g_white = spec.g.flatten() - spec.G @ mu
    return HalfSpace(G_white, g_white.reshape(-1, 1))


@dataclass
class MethodResult:
    name: str
    threshold: float
    volume: float
    volume_se: float
    empirical_coverage: float
    fit_time_s: float


def _forward(net, x):
    with torch.no_grad():
        return net(torch.as_tensor(x, dtype=torch.float32))


def _train_flow(y_train: torch.Tensor, dim: int, n_epochs: int, seed: int,
                batch_size: int = 2048, sinkhorn_iters: int = 10,
                hidden: int = 128, n_layers: int = 4,
                time_embed: str = 'concat',
                time_sampling: str = 'uniform',
                internal_standardize: bool = True) -> FlowODE:
    """Production-grade OT-CFM. Runs GPU-end-to-end.

    ``internal_standardize``: pass-through to ``train_flow``'s
    ``standardize_outputs`` argument. Callers that pre-whiten the
    training data externally (e.g. ``run_verification_pipeline``) must
    pass False to avoid double-whitening and to keep the flow operating
    end-to-end in whitened coordinates rather than data coordinates.
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vf = VelocityField(dim=dim, hidden=hidden, n_layers=n_layers,
                       activation='silu', time_embed=time_embed).to(device)
    y_train = y_train.to(device)
    vf, _ = train_flow(
        vf, y_train, n_epochs=n_epochs, batch_size=batch_size, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=sinkhorn_iters,
        use_ema=True, standardize_outputs=internal_standardize,
        time_sampling=time_sampling,
    )
    vf.eval()
    return FlowODE(vf)


def _train_flow_tight(y_train: torch.Tensor, dim: int, n_epochs: int,
                      seed: int, internal_standardize: bool = True) -> FlowODE:
    """Higher-capacity, longer-training config for ThreeBlobClassifier3D-
    class multimodal output distributions.

    hidden=256, L=6, sinusoidal time embedding, logit-normal time sampling
    (concentrates t near 0.5 where interpolation is hardest). Meets the
    (c)+(e) experiment spec. Training cost scales linearly with n_epochs.
    """
    return _train_flow(
        y_train, dim=dim, n_epochs=n_epochs, seed=seed,
        batch_size=2048, sinkhorn_iters=10,
        hidden=256, n_layers=6,
        time_embed='sinusoidal', time_sampling='logit_normal',
        internal_standardize=internal_standardize,
    )


def exact_star_union_volume(net, x_center: np.ndarray, radius: float,
                            output_dim: int, n_mc: int = 500_000,
                            seed: int = 42) -> tuple[float, list]:
    """Star-union ground-truth volume (an exact deterministic over-approx of
    f_#P_X's support; the 1-alpha reachset is smaller by (1-alpha)).

    Returns (volume_mean, stars). The MC estimate is used because the Star
    union can have thousands of overlapping polytopes whose exact volume
    requires inclusion-exclusion.
    """
    reach = compute_exact_reach(net, x_center, radius, output_dim=output_dim)
    stars = reach['stars']
    if output_dim == 2:
        # 2D has a cheap rasterization method, which we use as the ground-
        # truth reference rather than MC on a box (the 2D Star union is a
        # measure-zero manifold in some cases, so MC-on-a-box would give 0).
        y_bbox = compute_mc_bbox(net, x_center, radius, output_dim=output_dim,
                                 n_samples=5000, pad=1.0)
        vol = exact_volume_2d(stars, (y_bbox[0].numpy(), y_bbox[1].numpy()),
                              resolution=500)
        return float(vol), stars
    ve = star_union_volume_mc(
        stars, n_samples=n_mc, batch_size=25_000, seed=seed,
        contains_method='algebraic',
    )
    return float(ve.mean), stars


def run_pipeline(
    net,
    x_center: np.ndarray,
    radius: float,
    output_dim: int,
    star_union_volume: float,
    alpha: float = 0.01,
    n_train: int = 10_000,
    n_calib: int = 2_000,
    n_test: int = 2_000,
    seed: int = 0,
    flow_epochs: int = 2000,
    n_mc_volume: int = 400_000,
    flow_config: str = 'base',
    infer_solver: str = 'rk4',
    infer_atol: float = 1e-5,
    infer_rtol: float = 1e-5,
    infer_steps: int = 30,
    flow_score_class=FlowScore,
    flow_score_infer_batch_size: int = 65536,
) -> dict:
    """Run the flow-conformal pipeline.

    flow_config:
      'base'  — hidden=128, L=4, concat time, uniform time (original).
      'tight' — hidden=256, L=6, sinusoidal time, logit-normal time
                (experiment (c)+(e)).
    infer_solver / atol / rtol / steps:
      Control the ODE solver used when scoring y_calib, y_test and MC
      samples. 'rk4' is a fast fixed-step solver (30 steps is plenty if the
      flow has converged). 'dopri5' with atol/rtol ~ 1e-4 is 2-3x slower but
      more accurate for a poorly-converged flow (experiment (d)).
    flow_score_class: callable producing a NonconformityScore around the
        trained flow. Defaults to the naive ``FlowScore``. Pass
        ``LogDetFlowScore`` for the log-density-corrected variant.
        The constructor is called with the same kwargs we currently pass
        to FlowScore (t, n_steps, method, atol, rtol, batch_size).
    flow_score_infer_batch_size: chunk size for score evaluations on MC
        points. 65536 is the current production default; reduce if GPU
        memory is tight.
    """
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))
    torch.manual_seed(seed)
    dim_in = x_center.shape[0]
    x_center_t = torch.as_tensor(x_center, dtype=torch.float32)

    x_tr = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_train, seed=seed, dim=dim_in,
    )
    x_ca = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_calib,
        seed=seed + 1_000_000, dim=dim_in,
    )
    x_te = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_test,
        seed=seed + 2_000_000, dim=dim_in,
    )
    y_tr = _forward(net, x_tr)
    y_ca = _forward(net, x_ca)
    y_te = _forward(net, x_te)

    y_all = torch.cat([y_tr, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)

    results: list[MethodResult] = []
    for name, builder in (
        ('hyperrect', lambda: HyperrectScore(
            center=y_ca.mean(dim=0),
            scales=y_ca.std(dim=0).clamp(min=1e-8),
        )),
        ('ball', lambda: BallScore(center=y_ca.mean(dim=0))),
    ):
        t0 = time.time()
        score_fn = builder()
        thresh = calibrate(score_fn(y_ca), ell).item()
        s = ProbabilisticSet(
            score_fn=score_fn, threshold=thresh,
            m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
        )
        vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
        cov = s.contains(y_te).float().mean().item()
        results.append(MethodResult(
            name=name, threshold=thresh, volume=vol, volume_se=se,
            empirical_coverage=cov, fit_time_s=time.time() - t0,
        ))

    # Flow
    t0 = time.time()
    if flow_config == 'base':
        flow = _train_flow(y_tr, output_dim, flow_epochs, seed)
    elif flow_config == 'tight':
        flow = _train_flow_tight(y_tr, output_dim, flow_epochs, seed)
    else:
        raise ValueError(f"unknown flow_config {flow_config!r}")
    train_time = time.time() - t0

    score_fn = flow_score_class(
        flow, t=1.0, n_steps=infer_steps, method=infer_solver,
        batch_size=flow_score_infer_batch_size,
        atol=infer_atol, rtol=infer_rtol,
    )
    t1 = time.time()
    thresh = calibrate(score_fn(y_ca), ell).item()
    s = ProbabilisticSet(
        score_fn=score_fn, threshold=thresh,
        m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
    cov = s.contains(y_te).float().mean().item()
    infer_time = time.time() - t1
    results.append(MethodResult(
        name='flow', threshold=thresh, volume=vol, volume_se=se,
        empirical_coverage=cov, fit_time_s=train_time + infer_time,
    ))

    return {
        'results': results,
        'bbox': bbox,
        'y_train': y_tr, 'y_calib': y_ca, 'y_test': y_te,
        'star_union_volume': star_union_volume,
        'alpha': alpha,
        'flow_train_time_s': train_time,
        'flow_infer_time_s': infer_time,
    }


def print_report(bundle: dict):
    results = bundle['results']
    su = bundle['star_union_volume']
    alpha = bundle['alpha']
    floor = (1 - alpha) * su  # tightness floor for any 1-alpha reachset
    print(f"\n  Star-union volume      = {su:.4f}")
    print(f"  (1-alpha)*Star-union   = {floor:.4f}  <- tightness floor")
    print(f"  alpha = {alpha}  coverage floor = {1 - alpha}")
    print(f"  {'method':<10} {'vol':>10} {'+/-SE':>10} {'vol/floor':>10} "
          f"{'cov':>8} {'fit(s)':>8}")
    for r in results:
        ratio = r.volume / floor if floor > 0 else float('nan')
        print(f"  {r.name:<10} {r.volume:>10.4f} {r.volume_se:>10.4f} "
              f"{ratio:>10.3f} {r.empirical_coverage:>8.4f} {r.fit_time_s:>8.1f}")
    print(f"  (flow: train {bundle['flow_train_time_s']:.1f}s, "
          f"infer {bundle['flow_infer_time_s']:.1f}s)")


# Separate imports for verification pipeline (placed here to keep the
# volume-pipeline-only imports above unchanged and easy to read).
from scipy.stats import beta as _beta_dist

from examples.FlowConformal.benchmarks._spec import spec_summary, verify_spec_on_flow
from n2v.probabilistic.flow.sampling import sample_box as _sample_box


def run_verification_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    alpha: float = 0.001,
    m: int = 8000,
    ell: int = 7999,
    scenario_n_samples: int = 10_000,
    scenario_beta: float = 0.001,
    n_train: int = 10_000,
    flow_epochs: int = 5000,
    flow_config: str = 'tight',
    seed: int = 0,
    infer_solver: str = 'rk4',
    infer_steps: int = 30,
    enable_preimage_search: bool = True,
    preimage_tolerance: float = 0.1,
    sat_backend: str = 'random+pgd',
    sat_backend_kwargs: dict | None = None,
) -> dict:
    """Train-calibrate-verify pipeline for flow-conformal probabilistic
    reachability against a VNN-LIB-style halfspace spec.

    Returns a dict with the verification verdict and joint conformal-
    scenario certificate parameters. See the Phase 2 design doc for the
    formal statement of the joint ``(epsilon_total, delta_total)`` bound.

    Args:
        network: A callable ``network(x_tensor) -> y_tensor``; typically a
            :class:`torch.nn.Module` loaded via ``load_onnx``.
        input_lb, input_ub: ``(input_dim,)`` numpy arrays defining the
            input box (L∞ or general). All training / calibration / test
            samples are drawn uniformly from this box.
        spec: Output spec. A single :class:`HalfSpace` (with 1 or more
            rows, = AND-of-constraints) is supported. OR-of-ANDs lists
            will raise ``NotImplementedError``.
        alpha: conformal miscoverage (``ε_1`` in the joint bound).
        m, ell: calibration size and Hashemi double-step rank.
        scenario_n_samples: ``N`` for scenario verification.
        scenario_beta: scenario confidence-failure ``β_2``.
        n_train, flow_epochs, flow_config: as in :func:`run_pipeline`.
        seed: global RNG seed.
        infer_solver, infer_steps: ODE solver at scoring / verification time.
        enable_preimage_search: if True, flow-set violations are checked
            against the real network via :func:`preimage_search`.
        preimage_tolerance: L2 distance threshold below which a preimage
            search result counts as "found". Semantics: whitened output
            space (unit variance per dim), because run_verification_pipeline
            pre-whitens the output coordinates before the flow/scenario-
            verify stack operates on them. Default 0.1 corresponds to
            ~10% of a per-dim std in the original output scale. Only used
            when ``sat_backend == 'flow_preimage'``.
        sat_backend: SAT-detection strategy when the flow can't certify
            UNSAT. Options:
              - 'random+pgd' (default): try n2v's random sampling + PGD
                falsifier on the raw (unwhitened) network. Fast, well-tested,
                and handles AND-of-OR specs correctly via the existing
                ``n2v.utils.falsify`` implementation.
              - 'pgd': PGD only (no random pre-pass).
              - 'random': random sampling only.
              - 'flow_preimage': the flow-targeted preimage search currently
                embedded in scenario_verify_halfspace. Limited — only tries
                the single worst-margin flow sample. Kept for comparison.
              - 'none': no SAT detection; verdicts limited to UNSAT/UNKNOWN.
            The default 'random+pgd' is the recommended setting: the flow-
            conformal method's contribution is the probabilistic UNSAT
            certificate; SAT detection is delegated to the strongest
            available falsifier. Extending counterexample generation to
            use the flow directly is future work.
        sat_backend_kwargs: dict of kwargs passed through to
            ``n2v.utils.falsify`` (e.g. ``{'n_restarts': 20, 'n_steps': 100}``).

    Returns:
        Dict with keys
            - 'verdict': 'SAT' | 'UNSAT' | 'UNKNOWN'
            - 'epsilon_total', 'delta_total': joint probabilistic bound
            - 'epsilon_1', 'delta_1': conformal-layer bound
            - 'epsilon_2', 'delta_2': scenario-layer bound
            - 'counterexample': Optional[dict] with 'x' (real input) and
              'y' (network output at x) when verdict == 'SAT'
            - 'flow_train_time_s', 'verification_time_s', 'total_time_s'
            - 'coverage_empirical': fraction of held-out y_test inside S
            - 'spec_summary': str
    """
    import time

    # 1. Sample input box.
    lb_t = torch.as_tensor(input_lb, dtype=torch.float32)
    ub_t = torch.as_tensor(input_ub, dtype=torch.float32)
    x_tr = _sample_box(lb_t, ub_t, n_samples=n_train, seed=seed)
    x_ca = _sample_box(lb_t, ub_t, n_samples=m, seed=seed + 1_000_000)
    x_te = _sample_box(lb_t, ub_t, n_samples=2_000, seed=seed + 2_000_000)
    y_tr = _forward(network, x_tr)
    y_ca = _forward(network, x_ca)
    y_te = _forward(network, x_te)

    # 2. Whiten. The flow operates end-to-end on coordinates
    #   y_w = (y - y_mean) / y_std
    # and everything downstream (calibration, scenario-verify, preimage
    # search, spec) is transformed into the same frame. See the module-
    # level comment "Whitening glue for run_verification_pipeline".
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std
    y_te_w = (y_te - y_mean) / y_std

    # 3. Train flow on pre-whitened data (no internal double-whitening).
    t0 = time.time()
    output_dim = y_tr_w.shape[1]
    if flow_config == 'base':
        flow = _train_flow(y_tr_w, output_dim, flow_epochs, seed,
                           internal_standardize=False)
    elif flow_config == 'tight':
        flow = _train_flow_tight(y_tr_w, output_dim, flow_epochs, seed,
                                 internal_standardize=False)
    else:
        raise ValueError(f"unknown flow_config {flow_config!r}")
    # Move the trained flow to CPU so downstream scenario_verify (which
    # constructs CPU latent samples and uses CPU target_fn) is
    # device-consistent. FlowScore already cross-device-handles, so
    # calibration still works.
    flow = flow.to('cpu').eval()
    train_time = time.time() - t0

    # 4. Calibrate on whitened calibration samples.
    base_score_fn = FlowScore(
        flow, t=1.0, n_steps=infer_steps, method=infer_solver,
        batch_size=65536,
    )
    calib_scores = base_score_fn(y_ca_w)
    q = calibrate(calib_scores, ell).item()

    # 5. Empirical coverage (diagnostic) on whitened test samples.
    s = ProbabilisticSet(
        score_fn=base_score_fn, threshold=q,
        m=m, ell=ell, epsilon=alpha, dim=output_dim,
    )
    coverage_empirical = s.contains(y_te_w).float().mean().item()

    # 6. Hashemi double-step confidence δ_1.
    delta_1 = 1.0 - float(_beta_dist.cdf(1.0 - alpha, ell, m + 1 - ell))
    epsilon_1 = alpha

    # Before calling verify_spec_on_flow, normalize single-element lists
    # down to the bare HalfSpace (the len-1 list is a load_vnnlib quirk;
    # the VNN-LIB semantics are identical to a single HalfSpace).
    if isinstance(spec, list) and len(spec) == 1:
        spec = spec[0]

    # 7. Transform the spec + network to whitened coordinates so the
    # scenario verifier, preimage search, and flow all live in the same
    # frame.
    y_mean_np = y_mean.detach().cpu().numpy()
    y_std_np = y_std.detach().cpu().numpy()
    if isinstance(spec, HalfSpace):
        spec_whitened = _whiten_halfspace(spec, y_mean_np, y_std_np)
    else:
        # OR-of-ANDs is not yet supported; let verify_spec_on_flow raise.
        spec_whitened = spec
    whitened_network = _WhitenedNetwork(network, y_mean.cpu(), y_std.cpu())

    # 8. Dispatch the spec. When sat_backend is not 'flow_preimage',
    # disable the scenario verifier's internal flow-targeted preimage
    # search: we use n2v's falsifier (step 9) as the SAT backend instead.
    use_flow_preimage = (sat_backend == 'flow_preimage') and enable_preimage_search
    t1 = time.time()
    spec_result = verify_spec_on_flow(
        flow_ode=flow,
        threshold_q=q,
        spec=spec_whitened,
        input_lb=input_lb,
        input_ub=input_ub,
        network=(whitened_network if use_flow_preimage else None),
        alpha=alpha,
        delta_1=delta_1,
        beta_2=scenario_beta,
        n_samples=scenario_n_samples,
        preimage_tolerance=preimage_tolerance,
    )
    verify_time = time.time() - t1

    # 7. Compose the joint certificate.
    epsilon_2 = spec_result['epsilon_2']
    delta_2 = spec_result['delta_2']
    epsilon_total = 1.0 - (1.0 - epsilon_1) * (1.0 - epsilon_2)
    delta_total = delta_1 * delta_2

    counterexample = None
    counterexample_source = None
    sat_backend_time = 0.0
    if spec_result['verdict'] == 'SAT':
        # Only reachable via the legacy flow_preimage backend. AND
        # semantics: a row-level falsification can be spurious — the real
        # x produces a y hitting row i's unsafe region but missing some
        # other row. Accept SAT only if the counterexample y satisfies
        # the FULL joint unsafe region (G y <= g componentwise).
        if isinstance(spec, HalfSpace):
            G_full = spec.G
            g_full = spec.g.flatten()
            for r in spec_result['per_constraint_results']:
                if r.outcome == 'falsified' and r.genuine_input is not None:
                    x_candidate = r.genuine_input
                    y_at_x = network(
                        torch.as_tensor(x_candidate,
                                        dtype=torch.float32).unsqueeze(0)
                    ).detach().numpy().flatten()
                    if (G_full @ y_at_x - g_full <= 0).all():
                        counterexample = {
                            'x': x_candidate,
                            'y': y_at_x,
                        }
                        counterexample_source = 'flow_preimage'
                        break
        if counterexample is None:
            spec_result = dict(spec_result)
            spec_result['verdict'] = 'UNKNOWN'

    # 9. SAT backend: when the flow can't certify UNSAT, delegate to
    # n2v's falsifier for counterexample search. This is the primary
    # SAT path under default settings; the flow-targeted preimage search
    # above is kept only for 'flow_preimage' mode (legacy/comparison).
    if spec_result['verdict'] != 'UNSAT' and sat_backend not in (
            'none', 'flow_preimage'):
        t_sat = time.time()
        fals_kwargs = dict(sat_backend_kwargs or {})
        try:
            fals_result, fals_cex = falsify(
                model=network, lb=input_lb, ub=input_ub, property=spec,
                method=sat_backend, seed=seed, **fals_kwargs,
            )
        except Exception as e:
            # Falsifier shouldn't break the pipeline; log and continue.
            fals_result, fals_cex = 2, None
            print(f'[run_verification_pipeline] falsify({sat_backend}) '
                  f'raised {type(e).__name__}: {e}', file=sys.stderr)
        sat_backend_time = time.time() - t_sat
        if fals_result == 0 and fals_cex is not None:
            cex_x, cex_y = fals_cex
            counterexample = {
                'x': np.asarray(cex_x).flatten(),
                'y': np.asarray(cex_y).flatten(),
            }
            counterexample_source = sat_backend
            spec_result = dict(spec_result)
            spec_result['verdict'] = 'SAT'

    # Wrap score_fn so external callers can still pass RAW network
    # outputs; whitening happens inside the wrapper.
    score_fn = _WhiteningFlowScore(base_score_fn, y_mean.cpu(), y_std.cpu())

    return {
        'verdict': spec_result['verdict'],
        'epsilon_total': epsilon_total,
        'delta_total': delta_total,
        'epsilon_1': epsilon_1, 'delta_1': delta_1,
        'epsilon_2': epsilon_2, 'delta_2': delta_2,
        'counterexample': counterexample,
        'counterexample_source': counterexample_source,
        'flow_train_time_s': train_time,
        'verification_time_s': verify_time,
        'sat_backend_time_s': sat_backend_time,
        'total_time_s': train_time + verify_time + sat_backend_time,
        'coverage_empirical': coverage_empirical,
        'q': q,
        'flow': flow,
        'score_fn': score_fn,
        'y_mean': y_mean_np,
        'y_std': y_std_np,
        'spec_summary': spec_summary(spec),
    }
