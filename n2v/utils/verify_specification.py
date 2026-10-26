"""
Verify a specification against a computed reach set.

The router :func:`verify_specification` parses the spec, detects the reach
set's type, and dispatches to a type-specific check:

  * Sound dispatch: :class:`Star` / :class:`Box` list (output of
    :meth:`NeuralNetwork.reach` for sound methods). Uses interval
    arithmetic + LP-based disjointness; returns a deterministic verdict.
  * Probabilistic dispatch: :class:`ProbabilisticSet` (output of
    :func:`n2v.probabilistic.flow_reach` / ``method='flow_matching'``).
    Whitens the spec via the set's ``affine_transform`` and runs one of
    the sampling-based certify methods (scenario, AMLS, AMLS-bounded,
    IS-tilted, raw-MC).

Both dispatch paths return a single :class:`VerificationResult`. Sound
verdicts are ``'UNSAT'`` (provably safe) or ``'UNKNOWN'`` (might be
unsafe — over-approximation cannot rule it out). ``'SAT'`` (counterexample
found) is produced only by the falsifier lane
(:func:`n2v.utils.falsify.falsify`), never by ``verify_specification``.
"""

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
from scipy.optimize import linprog

from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets import Star, Box, HalfSpace, ImageZono
from n2v.utils.lpsolver import LPFailure, confirm_infeasible
from n2v.utils.ns_certificate import (
    _ETA, _SAFETY, _gamma_rows, outward_interval_bounds, outward_spec_rows,
)

logger = logging.getLogger(__name__)

# Matrix elements per row block of the Box fast check (64 KB of float64):
# temporaries this small are reused by the allocator instead of being
# page-faulted in on every call.
_BOX_CHECK_BLOCK_ELEMS = 1 << 13

SpecLike = Union[HalfSpace, list, dict]


@dataclass(frozen=True)
class VerificationResult:
    """Result of :func:`verify_specification`.

    The same dataclass is returned by both sound and probabilistic dispatch
    paths; field population is partial — only the fields populated by the
    path that produced this result are non-None. Callers check ``verdict``
    first; method-specific fields are documented inline below.

    Use :func:`dataclasses.replace` to create modified copies (e.g. when
    aggregating verdicts across multiple input boxes in an
    OR-of-input-regions specification).

    Attributes:
        verdict: ``'SAT'`` (property violated, counterexample found),
            ``'UNSAT'`` (property certified safe), or ``'UNKNOWN'``
            (verifier cannot decide). Always populated.

        intersecting_groups: Sound dispatch only. Indices of AND-property
            groups that intersect the reach set; populated when the caller
            asks for diagnostic detail. ``None`` otherwise.

        epsilon_total: Probabilistic dispatch only. Joint conformal +
            scenario miscoverage bound:
            ``epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)``.
        delta_total: Probabilistic dispatch only. Joint confidence:
            ``delta_total = delta_1 * delta_2``.
        n_samples_used: Probabilistic dispatch only. Scenario-verify
            sample count actually drawn (after adaptive escalation).
        coverage_empirical: Probabilistic dispatch only. Held-out
            empirical coverage; diagnostic, not the guarantee.
        q: Probabilistic dispatch only. Calibrated conformal threshold
            (rank-``ell`` nonconformity score).
        amls_levels_used: AMLS verification method only. Number of
            adaptive levels used; ``None`` for other methods.
        amls_bounded_eps_2_upper: Bounded-AMLS only. Upper bound on
            ``epsilon_2`` produced by the bounded-AMLS estimator.

        counterexample_x: SAT only. Input that violates the property.
        counterexample_y: SAT only. Network output at ``counterexample_x``
            confirming spec violation.

        total_time_s: End-to-end wall time of the verification call.
        flow_train_time_s: Probabilistic dispatch only. Flow-training
            time as a sub-component of ``total_time_s``.
        verification_time_s: Spec-disjointness check time as a
            sub-component of ``total_time_s``.
    """

    verdict: Literal['SAT', 'UNSAT', 'UNKNOWN']

    # Sound dispatch only.
    intersecting_groups: Optional[List[int]] = None

    # Probabilistic dispatch only.
    epsilon_total: Optional[float] = None
    delta_total: Optional[float] = None
    n_samples_used: Optional[int] = None
    coverage_empirical: Optional[float] = None
    q: Optional[float] = None

    # AMLS-method-specific (probabilistic dispatch).
    amls_levels_used: Optional[int] = None
    amls_bounded_eps_2_upper: Optional[float] = None
    amls_bounded_detected_unsafe: Optional[bool] = None

    # Counterexample (SAT result, sound or via falsifier).
    counterexample_x: Optional[np.ndarray] = None
    counterexample_y: Optional[np.ndarray] = None

    # Timing (best-effort; always optional).
    total_time_s: Optional[float] = None
    flow_train_time_s: Optional[float] = None
    verification_time_s: Optional[float] = None


# Internal-to-Literal mapping for VerificationResult.verdict.
_INT_TO_VERDICT: dict[int, Literal['SAT', 'UNSAT', 'UNKNOWN']] = {
    0: 'SAT',      # property failed — intersection found (counterexample present)
    1: 'UNSAT',    # property satisfied — no intersection (provably safe)
    2: 'UNKNOWN',  # cannot decide
}


# All probabilistic verification methods currently exposed via
# ProbVerifyConfig. Each maps to a ``*_certify_spec`` function in
# ``n2v.probabilistic.flow.*``. Adding a method here requires extending
# the dispatch table in ``_verify_specification_probabilistic`` below.
ProbVerifyMethod = Literal[
    'scenario',
    'amls',
    'amls_bounded',
    'amls_bounded_union',
    'is_tilted',
    'raw_mc_uniform',
]


@dataclass(frozen=True)
class ProbVerifyConfig:
    """Configuration for the probabilistic branch of :func:`verify_specification`.

    Only consumed when ``reach_set`` is a :class:`ProbabilisticSet`; the
    sound branch (``Star`` / ``Box`` list) ignores configuration entirely
    and operates deterministically.

    The config is flat to keep the surface discoverable — all knobs across
    all 6 methods live here. ``__post_init__`` performs basic sanity
    checks but does NOT reject unused-for-this-method knobs (callers can
    set them in advance and switch methods freely).

    Attributes:
        method: Which probabilistic certify family to use. Choices:

          * ``'scenario'`` (default): scenario-style sampling on the
            conformal level set; the canonical paper method.
          * ``'amls'``: Adaptive Multi-Level Splitting; detection-style.
          * ``'amls_bounded'``: bounded variant returning a tight
            ``epsilon_2`` upper bound. Requires
            ``amls_bounded_eps_2_target``.
          * ``'amls_bounded_union'``: AMLS-bounded estimating per-group
            union mass directly.
          * ``'is_tilted'``: tilted importance-sampling estimator.
          * ``'raw_mc_uniform'``: raw uniform Monte Carlo over the
            conformal level set. Requires ``amls_bounded_eps_2_target``.

        n_samples: Sample budget. Per-call for ``'scenario'``,
            ``'is_tilted'``, ``'raw_mc_uniform'``; per-level for AMLS
            methods (overridable per-method below).
        beta: Confidence level for the per-method significance.
            Default ``0.001``.
        seed: RNG seed for the verification phase. Default ``0``.

        t: ODE end-time for flow-set membership tests. Default ``1.0``.
        n_ode_steps: ODE step count. Default ``30``.
        ode_method: ODE solver. Default ``'rk4'``.
        ode_atol, ode_rtol: ODE tolerances. Defaults ``1e-5``.

        adaptive_threshold: ``'scenario'``-only. Margin threshold for
            adaptive re-sampling escalation. ``None`` disables.
        adaptive_n_samples: ``'scenario'``-only. Post-escalation sample
            count. ``None`` disables.
        sampling_strategy: ``'scenario'``-only. ``'uniform'`` (default)
            samples on the truncated ball; ``'qmc'`` uses Sobol quasi-MC.

        amls_quantile: AMLS / AMLS-bounded quantile. Default ``0.1``.
        amls_max_levels: AMLS / AMLS-bounded max levels. Default ``30``.
        amls_n_mcmc_steps: AMLS MCMC steps per level. Default ``10``.
        amls_mcmc_step_size: AMLS MCMC step size. Default ``0.3``.

        amls_bounded_eps_2_target: AMLS-bounded / raw-MC target ε₂
            (a.k.a. η — the *verification-layer* miscoverage).
            Required for ``'amls_bounded'``, ``'amls_bounded_union'``,
            ``'raw_mc_uniform'``.

            *Not* the same as :attr:`FlowReachConfig.epsilon` (a.k.a. α,
            the *conformal-layer* miscoverage of the
            :class:`ProbabilisticSet`). The joint guarantee is
            ``epsilon_total = 1 - (1 - epsilon)(1 - eps_2_target)``;
            ``FlowReachConfig.epsilon`` controls only the conformal layer
            (the calibrated reach set), and this field controls only the
            verification layer (the spec-disjointness test).
        amls_bounded_adaptive_step: AMLS-bounded adaptive step size.
            Default ``False``.

        is_lambda_tilt: ``'is_tilted'``-only tilt strength. Default ``5.0``.
    """

    method: ProbVerifyMethod = 'scenario'

    # Common: sample budget + confidence + seed
    n_samples: int = 10_000
    beta: float = 0.001
    seed: int = 0

    # Common ODE-inference settings
    t: float = 1.0
    n_ode_steps: int = 30
    ode_method: str = 'rk4'
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5

    # scenario-only
    adaptive_threshold: Optional[float] = None
    adaptive_n_samples: Optional[int] = None
    sampling_strategy: str = 'uniform'

    # AMLS family
    amls_quantile: float = 0.1
    amls_max_levels: int = 30
    amls_n_mcmc_steps: int = 10
    amls_mcmc_step_size: float = 0.3

    # AMLS-bounded + raw-MC
    amls_bounded_eps_2_target: Optional[float] = None
    amls_bounded_adaptive_step: bool = False

    # IS-tilted
    is_lambda_tilt: float = 5.0

    def __post_init__(self):
        valid_methods = (
            'scenario', 'amls', 'amls_bounded',
            'amls_bounded_union', 'is_tilted', 'raw_mc_uniform',
        )
        if self.method not in valid_methods:
            raise ValueError(
                f"ProbVerifyConfig.method must be one of {valid_methods}, "
                f"got {self.method!r}"
            )
        if self.n_samples < 1:
            raise ValueError(
                f"ProbVerifyConfig.n_samples must be >= 1, got {self.n_samples}"
            )
        if not 0.0 < self.beta < 1.0:
            raise ValueError(
                f"ProbVerifyConfig.beta must be in (0, 1), got {self.beta}"
            )
        if self.n_ode_steps < 1:
            raise ValueError(
                f"ProbVerifyConfig.n_ode_steps must be >= 1, "
                f"got {self.n_ode_steps}"
            )
        if self.sampling_strategy not in ('uniform', 'qmc'):
            raise ValueError(
                f"ProbVerifyConfig.sampling_strategy must be 'uniform' or "
                f"'qmc', got {self.sampling_strategy!r}"
            )
        if not 0.0 < self.amls_quantile < 1.0:
            raise ValueError(
                f"ProbVerifyConfig.amls_quantile must be in (0, 1), "
                f"got {self.amls_quantile}"
            )
        if self.amls_max_levels < 1:
            raise ValueError(
                f"ProbVerifyConfig.amls_max_levels must be >= 1, "
                f"got {self.amls_max_levels}"
            )
        if self.method in ('amls_bounded', 'amls_bounded_union',
                           'raw_mc_uniform'):
            if self.amls_bounded_eps_2_target is None:
                raise ValueError(
                    f"method={self.method!r} requires "
                    f"amls_bounded_eps_2_target to be set"
                )


def verify_specification(
    reach_set,
    spec: Union[dict, List[dict], HalfSpace, List[HalfSpace]],
    *,
    config: Optional[ProbVerifyConfig] = None,
    verbose: bool = False,
) -> VerificationResult:
    """Verify a spec against a reach set; sound OR probabilistic dispatch.

    The spec represents the un-robust / unsafe region we want to prove
    DOES NOT intersect with the reachable set. Dispatch is based on the
    type of ``reach_set``:

      * :class:`ProbabilisticSet` — uses ``config`` to run one of the
        sampling-based certify methods. Whitens the spec via the set's
        ``affine_transform`` (when present) before disjointness checks.
      * ``list[Star]`` / ``list[Box]`` (or anything with ``.to_star()`` /
        Box-shaped ``.lb``/``.ub``) — uses interval arithmetic + LP-based
        disjointness, returns a deterministic verdict.

    Args:
        reach_set: Either a single :class:`ProbabilisticSet` or a list
            of sound reach sets (``Star``/``Box``).
        spec: Property specification, in any of the shapes
            :func:`_parse_property_groups` accepts (single ``HalfSpace``,
            ``list[HalfSpace]``, ``dict {'Hg': ...}``, or ``list[dict]``).
        config: Required for probabilistic dispatch; ignored (and
            rejected) for sound dispatch. Defaults to ``ProbVerifyConfig()``
            when ``None`` and the reach set is probabilistic.
        verbose: If ``True``, print :func:`spec_summary` of ``spec`` plus
            the verdict and key result fields. Off by default; results
            are unchanged either way.

    Returns:
        :class:`VerificationResult` with ``verdict`` populated:
          * ``'UNSAT'`` — provably disjoint (sound) or certifiably
            disjoint within ``epsilon_total`` confidence (probabilistic).
          * ``'UNKNOWN'`` — over-approximation cannot rule out
            intersection (sound) or sampling cannot certify (probabilistic).

        ``'SAT'`` is not produced by this function — it comes from the
        falsifier lane (:func:`n2v.utils.falsify.falsify`) finding a
        concrete counterexample. Probabilistic-dispatch results also
        populate ``epsilon_total``, ``delta_total``, ``q``, ``n_samples_used``,
        and method-specific fields (see :class:`VerificationResult`).

    Raises:
        TypeError: If ``config`` is given but ``reach_set`` is sound.

    Notes:
        - Multiple property groups (``list[dict]``): AND across groups.
          The unsafe region is the intersection of all groups. If ANY
          group is disjoint from the reach set → safe.
        - Within a single group with multiple HalfSpaces: OR. If ANY
          halfspace intersects ANY reach set → that group intersects.
        - Single HalfSpace: ALL reach sets must NOT intersect.
    """
    # ---- Probabilistic dispatch ----
    if isinstance(reach_set, ProbabilisticSet):
        if config is None:
            config = ProbVerifyConfig()
        return _verify_specification_probabilistic(
            reach_set, spec, config, verbose=verbose,
        )

    # ---- Sound dispatch ----
    if config is not None:
        raise TypeError(
            "config= is only valid when reach_set is a ProbabilisticSet; "
            f"got reach_set of type {type(reach_set).__name__}. Sound "
            "verification is parameter-free."
        )

    # Parse spec into canonical AND-of-OR groups.
    groups = _parse_property_groups(spec)

    # For AND-across-groups: if ANY group is fully disjoint → UNSAT (safe).
    int_verdict = 2  # default UNKNOWN
    for k, group in enumerate(groups):
        try:
            disjoint = _group_disjoint_from_reach_set(group, reach_set)
        except LPFailure as e:
            # SOUNDNESS: a solver failure inside a disjointness test is not a
            # disjointness certificate, so this group stays undecided (UNSAT is
            # verdict-bearing: it prunes as safe). The groups are ANDed, so a
            # later group that is certified disjoint still proves UNSAT alone.
            logger.warning("verify_specification: LP did not decide for "
                           "property group %d of %d (%s); group left "
                           "undecided", k + 1, len(groups), e)
            continue
        if disjoint:
            int_verdict = 1  # UNSAT
            break

    verdict = _INT_TO_VERDICT[int_verdict]
    if verbose:
        print(f"spec:    {spec_summary(spec)}")
        print(f"verdict: {verdict}")
    return VerificationResult(verdict=verdict)


def _verify_specification_probabilistic(
    prob_set: ProbabilisticSet,
    spec: Union[dict, List[dict], HalfSpace, List[HalfSpace]],
    config: ProbVerifyConfig,
    *,
    verbose: bool = False,
) -> VerificationResult:
    """Probabilistic branch of :func:`verify_specification`.

    Whitens the spec into the set's coordinate frame (via
    ``prob_set.affine_transform`` when present), then dispatches on
    ``config.method`` to one of the ``flow.*_certify_spec`` functions.
    Packages the result as a :class:`VerificationResult` with the joint
    conformal+verification guarantee in ``epsilon_total``/``delta_total``.

    For methods that don't produce an ``epsilon_2`` per construction
    (``'amls'``, ``'is_tilted'``), ``epsilon_total`` is ``None`` — only
    the conformal-layer ``epsilon`` from the set itself applies, and
    the verification is detection-style (UNKNOWN on any detection).
    """
    import time

    # Eager imports of the method dispatch table. None of these reach
    # back into ``n2v.utils.verify_specification`` so there is no
    # circular risk.
    from n2v.probabilistic.flow.amls import amls_certify_spec
    from n2v.probabilistic.flow.amls_bounded import (
        amls_bounded_certify_spec,
        amls_bounded_certify_spec_union,
    )
    from n2v.probabilistic.flow.importance_sampling import (
        is_tilted_certify_spec,
    )
    from n2v.probabilistic.flow.raw_mc_uniform import raw_mc_certify_spec
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint

    t_start = time.time()

    # Whiten the spec into the set's frame (when affine_transform set).
    # The set's score function operates on whitened coordinates; the spec
    # was provided in raw output coordinates.
    raw_groups = _parse_property_groups(spec)
    if prob_set.affine_transform is not None:
        spec_groups = [
            [prob_set.affine_transform.transform_halfspace(hs)
             for hs in group]
            for group in raw_groups
        ]
    else:
        spec_groups = raw_groups

    # Extract the underlying FlowODE from the score function. All
    # flow-based score functions used by ``flow_reach`` (FlowScore,
    # _WhiteningFlowScore) expose ``.flow_model``.
    if not hasattr(prob_set.score_fn, 'flow_model'):
        raise TypeError(
            "verify_specification's probabilistic dispatch requires a "
            "flow-based score function (with .flow_model attribute); "
            f"got {type(prob_set.score_fn).__name__}. The non-flow "
            "score functions (HyperrectScore, etc.) only support the "
            "set's own .contains() / .estimate_volume()."
        )
    flow_ode = prob_set.score_fn.flow_model

    method = config.method
    q = prob_set.threshold

    if method == 'scenario':
        result = certify_spec_disjoint(
            flow_ode=flow_ode, threshold_q=q, spec_groups=spec_groups,
            n_samples=config.n_samples, beta_2=config.beta,
            t=config.t, n_ode_steps=config.n_ode_steps,
            ode_method=config.ode_method,
            ode_atol=config.ode_atol, ode_rtol=config.ode_rtol,
            seed=config.seed,
            adaptive_threshold=config.adaptive_threshold,
            adaptive_n_samples=config.adaptive_n_samples,
            sampling_strategy=config.sampling_strategy,
        )
        unsat = result.unsat_certified
        eps_2 = result.epsilon_2
        n_used = result.n_samples_used
        amls_levels = None
        amls_eps2_upper = None
        amls_detected = None

    elif method == 'amls':
        result = amls_certify_spec(
            flow_ode=flow_ode, spec_groups=spec_groups,
            n_samples_per_level=config.n_samples,
            quantile=config.amls_quantile,
            max_levels=config.amls_max_levels,
            n_mcmc_steps=config.amls_n_mcmc_steps,
            mcmc_step_size=config.amls_mcmc_step_size,
            beta=config.beta, seed=config.seed,
            t=config.t, n_ode_steps=config.n_ode_steps,
            ode_method=config.ode_method,
            ode_atol=config.ode_atol, ode_rtol=config.ode_rtol,
        )
        unsat = result.unsat_certified
        eps_2 = None  # AMLS is detection-style; no finite-sample eps_2
        n_used = config.n_samples * config.amls_max_levels
        amls_levels = config.amls_max_levels
        amls_eps2_upper = None
        amls_detected = None

    elif method in ('amls_bounded', 'amls_bounded_union'):
        certify_fn = (amls_bounded_certify_spec_union
                      if method == 'amls_bounded_union'
                      else amls_bounded_certify_spec)
        result = certify_fn(
            flow_ode=flow_ode, spec_groups=spec_groups,
            q=q,
            eps_2_target=config.amls_bounded_eps_2_target,
            n_samples_per_level=config.n_samples,
            quantile=config.amls_quantile,
            max_levels=config.amls_max_levels,
            n_mcmc_steps=config.amls_n_mcmc_steps,
            mcmc_step_size=config.amls_mcmc_step_size,
            adaptive_step=config.amls_bounded_adaptive_step,
            beta=config.beta, seed=config.seed,
            t=config.t, n_ode_steps=config.n_ode_steps,
            ode_method=config.ode_method,
            ode_atol=config.ode_atol, ode_rtol=config.ode_rtol,
        )
        unsat = result.unsat_certified
        eps_2 = config.amls_bounded_eps_2_target
        n_used = config.n_samples * config.amls_max_levels
        amls_levels = config.amls_max_levels
        # Best-effort: extract bounded diagnostics (struct shape varies
        # slightly between amls_bounded_certify_spec and *_union).
        amls_eps2_upper = getattr(result, 'eps_2_upper', None)
        amls_detected = getattr(result, 'detected_unsafe',
                                getattr(result, 'detected_any', None))

    elif method == 'is_tilted':
        result = is_tilted_certify_spec(
            flow_ode=flow_ode, spec_groups=spec_groups,
            n_samples=config.n_samples,
            lambda_tilt=config.is_lambda_tilt,
            beta=config.beta, seed=config.seed,
            t=config.t, n_ode_steps=config.n_ode_steps,
            ode_method=config.ode_method,
            ode_atol=config.ode_atol, ode_rtol=config.ode_rtol,
        )
        unsat = result.unsat_certified
        eps_2 = None  # Detection-style
        n_used = config.n_samples
        amls_levels = None
        amls_eps2_upper = None
        amls_detected = None

    elif method == 'raw_mc_uniform':
        result = raw_mc_certify_spec(
            flow_ode=flow_ode, spec_groups=spec_groups,
            q=q,
            eps_2_target=config.amls_bounded_eps_2_target,
            n_samples=config.n_samples,
            beta=config.beta, seed=config.seed,
            t=config.t, n_ode_steps=config.n_ode_steps,
            ode_method=config.ode_method,
            ode_atol=config.ode_atol, ode_rtol=config.ode_rtol,
        )
        unsat = result.unsat_certified
        eps_2 = config.amls_bounded_eps_2_target
        n_used = config.n_samples
        amls_levels = None
        amls_eps2_upper = None
        amls_detected = None

    else:
        # Unreachable — ProbVerifyConfig.__post_init__ enforces method.
        raise ValueError(f"unknown probabilistic method {method!r}")

    verdict: Literal['SAT', 'UNSAT', 'UNKNOWN'] = (
        'UNSAT' if unsat else 'UNKNOWN'
    )

    # Joint guarantee. Conformal layer carries
    # (epsilon_1=prob_set.epsilon, delta_1=prob_set.confidence). The
    # verification layer contributes (eps_2, delta_2=1-beta) when its
    # certify method produces a finite-sample eps_2.
    eps_1 = prob_set.epsilon
    delta_2 = 1.0 - config.beta
    if eps_2 is None:
        epsilon_total = None
    else:
        epsilon_total = 1.0 - (1.0 - eps_1) * (1.0 - eps_2)
    delta_total = prob_set.confidence * delta_2

    total_time = time.time() - t_start

    if verbose:
        print(f"spec:    {spec_summary(spec)}")
        print(f"method:  {method}")
        print(f"verdict: {verdict}")
        if epsilon_total is not None:
            print(f"eps_total: {epsilon_total:.6e}")
        print(f"q:       {q:.4f}")

    return VerificationResult(
        verdict=verdict,
        epsilon_total=epsilon_total,
        delta_total=delta_total,
        n_samples_used=n_used,
        q=q,
        amls_levels_used=amls_levels,
        amls_bounded_eps_2_upper=amls_eps2_upper,
        amls_bounded_detected_unsafe=amls_detected,
        total_time_s=total_time,
        verification_time_s=total_time,
    )


def _parse_property_groups(property: Union[dict, List, HalfSpace]) -> List[List[HalfSpace]]:
    """
    Normalize property input into a list of groups (AND of OR).

    Each group is a list of HalfSpace objects (OR within group).
    Multiple groups are ANDed together.
    """
    # List of dicts: each dict is a property group (AND across groups)
    if isinstance(property, list) and len(property) > 0 and isinstance(property[0], dict):
        groups = []
        for p in property:
            hg = p['Hg']
            if isinstance(hg, HalfSpace):
                groups.append([hg])
            elif isinstance(hg, list):
                groups.append(hg)
            else:
                raise TypeError(f"Property group 'Hg' must be HalfSpace or list, got {type(hg)}")
        return groups
    elif isinstance(property, dict):
        hg = property['Hg']
        if isinstance(hg, HalfSpace):
            return [[hg]]
        elif isinstance(hg, list):
            return [hg]
        else:
            raise TypeError(f"Property 'Hg' must be HalfSpace or list, got {type(hg)}")

    # Single HalfSpace → one group with one halfspace
    if isinstance(property, HalfSpace):
        return [[property]]
    # List of HalfSpaces → one group with OR logic
    elif isinstance(property, list):
        return [property]
    else:
        raise TypeError(f"Property must be HalfSpace, list of HalfSpace, or dict with 'Hg' field, got {type(property)}")


def spec_summary(spec: SpecLike) -> str:
    """One-line human-readable description of a VNN-LIB spec.

    Accepts the same shapes :func:`_parse_property_groups` does:
      - Single ``HalfSpace`` (may have multiple rows = AND-of-halfspaces).
      - ``list[HalfSpace]`` (= OR-of-ANDs).
      - ``dict`` with an ``'Hg'`` field (single group from ``load_vnnlib``).
      - ``list[dict]`` (AND-across-groups from ``load_vnnlib``).

    Returns a short string like ``"HalfSpace dim=5, 4 constraints (AND)"``
    or ``"OR of 2 HalfSpace groups"``. Pure utility; no side effects.
    """
    if isinstance(spec, HalfSpace):
        n_rows = spec.G.shape[0]
        suffix = " (AND)" if n_rows > 1 else ""
        return (
            f"HalfSpace dim={spec.dim}, "
            f"{n_rows} constraint{'s' if n_rows != 1 else ''}{suffix}"
        )
    if isinstance(spec, dict) and 'Hg' in spec:
        hg = spec['Hg']
        if isinstance(hg, HalfSpace):
            n_rows = hg.G.shape[0]
            return (
                f"AND group: HalfSpace dim={hg.dim}, "
                f"{n_rows} constraint{'s' if n_rows != 1 else ''}"
            )
        if isinstance(hg, list):
            return f"AND group: OR of {len(hg)} HalfSpaces"
    if isinstance(spec, list):
        if len(spec) == 0:
            return "empty spec"
        if isinstance(spec[0], dict):
            return f"AND of {len(spec)} property groups"
        return f"OR of {len(spec)} HalfSpace groups"
    raise TypeError(f"unsupported spec type: {type(spec).__name__}")


def distribute_and_of_or_of_and(
    groups: List[List[HalfSpace]],
) -> List[List[HalfSpace]]:
    """Distribute AND-of-OR-of-AND-of-rows into a single OR-of-AND-of-rows.

    Input is the canonical n2v ``List[List[HalfSpace]]`` shape returned
    by :func:`_parse_property_groups`:

    * Outer list: ``AND`` across groups (a point is unsafe iff every
      group contains it).
    * Each group is a list of ``HalfSpace`` (``OR`` within a group).
    * Each ``HalfSpace`` itself encodes ``G y <= g`` row-wise (``AND``
      over rows).

    Output is a single-group ``List[List[HalfSpace]]`` (outer length 1)
    where the inner list is the cartesian-product OR over all
    per-group choices, and each compound ``HalfSpace`` stacks the rows
    of the chosen per-group HalfSpaces. The unsafe-region semantics are
    mathematically identical: ``OR over (H_i1, H_i2, ..., H_ik)`` of
    ``(in H_i1) AND (in H_i2) AND ... AND (in H_ik)``.

    This is the form AMLS-bounded estimators can correctly bound: each
    compound disjunct's mass is the AND-conjunction mass (which is
    what the verdict gate is actually trying to bound).

    Cost: ``O(prod(group_sizes))`` compound HalfSpaces. For the
    multi-group benchmarks in our sweep (lsnc_relu 13×1, metaroom 19×1,
    malbeware 25×1) this is bounded and tractable.

    Args:
        groups: AND-of-OR shape from :func:`_parse_property_groups`.

    Returns:
        Single-group ``List[List[HalfSpace]]`` (outer length 1) in
        OR-of-AND-of-rows form. If the input has length 1 and is
        already a single-group OR (the common single-spec case), it is
        returned unchanged for parity. If any group is empty, returns
        ``[[]]`` representing a vacuously-empty unsafe region.
    """
    if len(groups) == 0:
        # Empty AND of nothing is trivially True everywhere → unsafe
        # region is the whole output space. We return a single-group
        # empty disjunction; caller should treat that as "trivially
        # unsafe" or raise.
        return [[]]
    if any(len(g) == 0 for g in groups):
        # Empty OR within some group → that group is never satisfied
        # → AND with it is always False → unsafe region empty. Return
        # an empty disjunction (the caller's UNSAT check should fire
        # trivially: no halfspaces to violate).
        return [[]]
    if len(groups) == 1:
        # Already a single OR-of-AND group; no distribution to do.
        return [list(groups[0])]

    import itertools

    compound_disjuncts: List[HalfSpace] = []
    for combo in itertools.product(*groups):
        # ``combo`` is a tuple of one HalfSpace from each group. The
        # AND of these is a single HalfSpace whose rows are the row-
        # concatenation of every member's rows, with the same RHS
        # row-concatenation. Sound-by-construction: a point satisfies
        # the AND iff it satisfies every row, iff it satisfies every
        # member HalfSpace.
        Gs = [np.asarray(hs.G, dtype=np.float64) for hs in combo]
        gs = [np.asarray(hs.g, dtype=np.float64).reshape(-1, 1)
              for hs in combo]
        G_combined = np.concatenate(Gs, axis=0)
        g_combined = np.concatenate(gs, axis=0)
        compound_disjuncts.append(HalfSpace(G_combined, g_combined))

    return [compound_disjuncts]


def _group_disjoint_from_reach_set(group: List[HalfSpace], reach_set: list) -> bool:
    """
    Check if a group of halfspaces (OR) is disjoint from all reach sets.

    A group is disjoint from the reach set if for every reach set S and every
    halfspace in the group, S is disjoint from the halfspace.

    If the group has a single halfspace: all S must be disjoint from it.
    If the group has multiple halfspaces (OR): if any hs intersects any S → not disjoint.
    """
    for hs in group:
        for S in reach_set:
            if not _is_disjoint(S, hs):
                return False  # this halfspace intersects → group not disjoint
    return True  # all halfspaces disjoint from all reach sets


def _is_disjoint(S: Union[Star, Box], halfspace: HalfSpace) -> bool:
    """Check if set S is disjoint from halfspace. Dispatches by set type."""
    if hasattr(S, 'flatten_to_star'):
        # ImageStar: its ``to_star()`` rows are HWC-ordered while specs index
        # outputs in CHW, so a spec column lands on the wrong pixel/channel
        # (measured: false UNSAT on multi-channel spatial outputs).
        # ``flatten_to_star()`` produces the CHW ordering the spec expects.
        return _verify_specification_star(S.flatten_to_star(), halfspace)
    if isinstance(S, ImageZono):
        # ImageZono rows are HWC-ordered as well (it subclasses Zono, so the
        # generic ``to_star()`` branch below would keep that order and check
        # CHW spec columns against the wrong pixel/channel). Reorder to CHW
        # exactly as the Flatten layer does. Imported here: n2v.nn imports
        # this module.
        from n2v.nn.layer_ops.flatten_reach import _flatten_imagezono_chw
        return _verify_specification_star(_flatten_imagezono_chw(S).to_star(),
                                          halfspace)
    if isinstance(S, Box):
        return _verify_specification_box(S, halfspace)
    elif isinstance(S, Star):
        return _verify_specification_star(S, halfspace)
    elif hasattr(S, 'to_star'):
        return _verify_specification_star(S.to_star(), halfspace)
    else:
        raise TypeError(f"Cannot verify specification for {type(S)}")


def _verify_specification_box(box: Box, halfspace: HalfSpace) -> bool:
    """
    Check if a Box is disjoint from a halfspace using interval arithmetic.

    For halfspace Gx <= g with box [lb, ub], the intersection is empty iff
    there is no x in [lb, ub] satisfying all rows of Gx <= g.

    For each row i: min_{x in box} G_i·x = Σ_j min(G_ij*lb_j, G_ij*ub_j).
    If min > g_i for any row, by more than the rounding error of computing
    min → that constraint is infeasible → disjoint.

    If all rows are individually feasible, we solve a box-bounded LP to check
    simultaneous feasibility (needed for multi-row halfspaces).

    Returns:
        True if disjoint (empty intersection), False if intersection may exist.
    """
    G = halfspace.G.astype(np.float64)
    g = halfspace.g.astype(np.float64).flatten()
    lb = box.lb.flatten()
    ub = box.ub.flatten()

    n_rows = G.shape[0]

    # Fast check: for each row, compute min(G_i · x) over the box.
    # min(G_i · x) = Σ_j min(G_ij * lb_j, G_ij * ub_j) = G_i · x_min, where
    # x_min is the minimizing corner. Rows are processed in blocks, so the
    # (rows, n) temporaries stay bounded.
    step = max(1, _BOX_CHECK_BLOCK_ELEMS // max(lb.size, 1))
    # Overflow and inf - inf only reach min_val / slack as inf or NaN, which
    # fail the comparison below (not disjoint).
    with np.errstate(invalid='ignore', over='ignore'):
        for start in range(0, n_rows, step):
            Gb = G[start:start + step]
            # Rounding is monotone, so this is the rounded product at x_min.
            terms = Gb * lb
            np.minimum(terms, Gb * ub, out=terms)
            min_val = terms.sum(axis=1)
            if np.isnan(min_val).any():
                # A zero coefficient contributes exactly 0, also on an
                # unbounded coordinate (where the product is 0 * inf = NaN).
                terms[Gb == 0.0] = 0.0
                min_val = terms.sum(axis=1)
            # SOUNDNESS: min_val is a round-to-nearest sum of rounded products
            # and can land a few ulps ABOVE the exact minimum, so a box whose
            # corner touches the halfspace would be declared disjoint -- a
            # false UNSAT (measured: 2117 of 20000 random touching boxes). A
            # row certifies disjointness only when its margin clears a
            # rigorous bound on the rounding of G_i · x_min. It is
            # ns_certificate.outward_interval_slack's bound, built from that
            # module's constants so the two stay in sync, but charged at the
            # minimizing corner: for k nonzero terms, _SAFETY * (gamma_{k+1}
            # * Σ|terms| + k * _ETA), where _SAFETY also covers taking |terms|
            # after rounding and the rounding of the slack itself. Charging
            # the corner rather than the whole box keeps the slack finite on
            # a half-unbounded box whose minimizing corner is finite. (Calling
            # outward_interval_slack itself, with each row split by sign
            # against lb and ub, gives the same slack at 3-5x the cost of
            # this whole check.)
            k = (Gb != 0.0).sum(axis=1)
            slack = _SAFETY * (_gamma_rows(k)
                               * np.abs(terms, out=terms).sum(axis=1)
                               + k * _ETA)
            if (min_val - slack > g[start:start + step]).any():
                return True  # some constraint infeasible → disjoint

    # If only one row and it's feasible, intersection exists
    if n_rows == 1:
        return False

    # Multiple rows all individually feasible — check simultaneous feasibility
    # via box-bounded LP (only variable bounds + halfspace constraints)
    # Feasibility LP: minimize 0 subject to Gx <= g, lb <= x <= ub
    n = len(lb)
    c = np.zeros(n)
    bounds = list(zip(lb, ub))

    result = linprog(c, A_ub=G, b_ub=g, bounds=bounds, method='highs')

    # SOUNDNESS: "disjoint" is verdict-bearing (it prunes the branch as safe),
    # so only a certified INFEASIBLE result may return True. ``not
    # result.success`` also covers iteration limits, numerical failure and
    # unboundedness, each of which would prune a branch the solver never
    # decided -- a false UNSAT. scipy status codes: 0 optimal, 2 infeasible.
    if result.status == 0:
        return False  # feasible intersection
    if result.status == 2:
        # SOUNDNESS: scipy's presolve also mislabels tiny-scale feasible
        # systems; only a robustly confirmed infeasibility is disjointness.
        return confirm_infeasible(G, g, lb, ub)
    raise LPFailure(
        f"specification-box feasibility LP did not decide: scipy status "
        f"{result.status} ({result.message!r}); not evidence of disjointness"
    )


def _verify_specification_star(star: Star, halfspace: HalfSpace) -> bool:
    """
    Check if a Star is disjoint from a halfspace using LP-based intersection.

    Returns:
        True if disjoint (empty intersection), False if intersection may exist.
    """
    G = halfspace.G.astype(np.float64)
    g = halfspace.g.astype(np.float64)

    # Interval pre-check, no LP: a row whose pulled-back halfspace misses the
    # predicate box already rules the unsafe region (all rows at once) out.
    # ``outward_spec_rows`` moves the right-hand side out by the rounding of
    # the pull-back and ``outward_interval_bounds`` gives a rigorous lower
    # bound over the box, so ``lo > rhs`` proves the row infeasible. It needs
    # a finite box (the slack's magnitude term). This also decides stars
    # whose LP emptiness check a solver leaves at 'optimal' on the slackened
    # system although the interval bound clears the threshold.
    plb, pub = star.predicate_lb, star.predicate_ub
    if (plb is not None and pub is not None
            and np.all(np.isfinite(plb)) and np.all(np.isfinite(pub))):
        H_alpha, rhs = outward_spec_rows(G, g, star.V, plb, pub)
        lo, _ = outward_interval_bounds(np.zeros(H_alpha.shape[0]), H_alpha,
                                        np.ravel(plb), np.ravel(pub))
        if np.any(lo > rhs):
            return True

    S = star.intersect_half_space(G, g)

    if S is None or (isinstance(S, list) and len(S) == 0) or (isinstance(S, Star) and S.is_empty_set()):
        return True  # empty intersection → disjoint
    return False
