"""Small helpers for turning VNN-LIB ``HalfSpace`` specs into human-
readable strings and dispatching them to scenario verification.

Not library code — lives alongside the benchmark harness because the
mapping from spec structure → scenario_verify call is benchmark-level
glue, not a reusable abstraction.
"""
from __future__ import annotations

from typing import Union

from n2v.sets.halfspace import HalfSpace

SpecLike = Union[HalfSpace, list]


def spec_summary(spec: SpecLike) -> str:
    """One-line human-readable description of a VNN-LIB spec.

    Supported inputs (matching ``n2v.utils.load_vnnlib``'s ``prop`` field):
      - Single ``HalfSpace`` (may have multiple rows = AND-of-halfspaces).
      - ``list[HalfSpace]`` (= OR-of-ANDs).

    Args:
        spec: The parsed spec.

    Returns:
        A short string like ``"HalfSpace dim=5, 4 constraints (AND)"`` or
        ``"OR of 2 HalfSpace groups"``.
    """
    if isinstance(spec, HalfSpace):
        n_rows = spec.G.shape[0]
        suffix = " (AND)" if n_rows > 1 else ""
        return (
            f"HalfSpace dim={spec.dim}, "
            f"{n_rows} constraint{'s' if n_rows != 1 else ''}{suffix}"
        )
    if isinstance(spec, list):
        if len(spec) == 0:
            return "empty spec"
        return f"OR of {len(spec)} HalfSpace groups"
    raise TypeError(f"unsupported spec type: {type(spec).__name__}")


from typing import Callable, Optional

import numpy as np

from n2v.probabilistic.flow.scenario_verify import scenario_verify_halfspace


def verify_spec_on_flow(
    flow_ode,
    threshold_q: float,
    spec: SpecLike,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    network: Optional[Callable],
    alpha: float,
    delta_1: float,
    beta_2: float,
    n_samples: int,
    t: float = 1.0,
    n_ode_steps: int = 30,
    preimage_n_restarts: int = 10,
    preimage_n_steps: int = 200,
    preimage_lr: float = 0.05,
    preimage_tolerance: float = 1e-3,
    output_shift: Optional[np.ndarray] = None,
    ode_method: str = 'rk4',
) -> dict:
    """Run scenario-based verification of ``spec`` on a calibrated flow.

    The spec dispatcher for Phase 2. Supports:
      - ``HalfSpace`` (1 row)       — single halfspace ``w^T y <= b``.
      - ``HalfSpace`` (k rows)      — AND of k halfspaces; loops over rows
                                      and union-bounds the scenario
                                      violation probability.
    OR-of-ANDs (``list[HalfSpace]``) is deferred to Phase 3.

    Args:
        flow_ode: trained FlowODE.
        threshold_q: calibrated conformal threshold (e.g., from
            :func:`n2v.probabilistic.flow.calibrate.calibrate`).
        spec: the spec to verify (VNN-LIB ``prop`` field shape).
        input_lb, input_ub: input-box bounds (passed to preimage search).
        network: target network for preimage search, or ``None`` to disable
            (then only 'verified' / 'unknown' verdicts are possible).
        alpha: conformal miscoverage level (reported as ``epsilon_1``).
        delta_1: conformal confidence (from Hashemi double-step).
        beta_2: scenario confidence-failure probability.
        n_samples: scenario sample size.
        t, n_ode_steps, preimage_*, output_shift: passed through to
            ``scenario_verify_halfspace``.

    Returns:
        Dict with keys:
          verdict: 'SAT' | 'UNSAT' | 'UNKNOWN'
            SAT    = at least one constraint falsified + real preimage found.
            UNSAT  = all constraints verified (joint probabilistic certificate).
            UNKNOWN = at least one constraint falsified in flow set but no
                     preimage found (hallucination) or preimage search was
                     disabled.
          epsilon_2: effective scenario bound after union-bound over K rows.
          delta_2: ``1 - beta_2``.
          n_samples_used: ``n_samples``.
          counterexample: (z, y, margin) of the worst-violating halfspace,
                         or None if verified.
          per_constraint_results: list of ScenarioResult, one per row.

    Raises:
        NotImplementedError: if ``spec`` is a list (OR-of-ANDs).
        TypeError: if ``spec`` has an unsupported type.
    """
    if isinstance(spec, list):
        raise NotImplementedError(
            "OR-of-ANDs spec structures are deferred to Phase 3; "
            f"got a list of {len(spec)} HalfSpace groups."
        )
    if not isinstance(spec, HalfSpace):
        raise TypeError(
            f"unsupported spec type: {type(spec).__name__} "
            "(expected HalfSpace or list[HalfSpace])"
        )

    G = spec.G  # (k, d)
    g = spec.g.flatten()  # (k,)
    k_rows = G.shape[0]

    per_row_results = []
    worst_outcome = 'verified'
    worst_counterexample = None

    for i in range(k_rows):
        # scenario_verify_halfspace treats its (w, b) as a SAFETY constraint
        # w^T y <= b and flags "violation" when w^T y > b. VNN-LIB encodes
        # the UNSAFE region as G y <= g; a point in that region is an SAT
        # witness. Bridge the two by negating: pass w = -G[i], b = -g[i]
        # so scenario_verify's "violation" (w y > b) corresponds to the
        # VNN-LIB unsafe-hit (G y <= g).
        w = -G[i]
        b = -float(g[i])
        result = scenario_verify_halfspace(
            flow_ode=flow_ode,
            threshold_q=threshold_q,
            w=w, b=b,
            n_samples=n_samples,
            beta_2=beta_2,
            t=t,
            n_ode_steps=n_ode_steps,
            target_fn=network,
            input_set_bounds=(
                (input_lb, input_ub) if network is not None else None
            ),
            preimage_n_restarts=preimage_n_restarts,
            preimage_n_steps=preimage_n_steps,
            preimage_lr=preimage_lr,
            preimage_tolerance=preimage_tolerance,
            output_shift=output_shift,
            ode_method=ode_method,
        )
        per_row_results.append(result)

        # AND semantics: the spec is falsified as soon as any row is falsified.
        # Track the "worst" outcome for the unified verdict, with severity
        # ordering falsified > unknown > verified.
        severity = {'verified': 0, 'unknown': 1, 'falsified': 2}
        if severity[result.outcome] > severity[worst_outcome]:
            worst_outcome = result.outcome
            worst_counterexample = result.counterexample

    # Union bound on epsilon_2 across the k rows (conservative; tighter
    # shared-samples bounds are a Phase 3 improvement if needed).
    import math as _math
    epsilon_2_single = -_math.log(beta_2) / n_samples
    epsilon_2_union = min(1.0, k_rows * epsilon_2_single)

    verdict = {
        'verified': 'UNSAT',
        'unknown': 'UNKNOWN',
        'falsified': 'SAT',
    }[worst_outcome]

    return {
        'verdict': verdict,
        'epsilon_2': epsilon_2_union,
        'delta_2': 1.0 - beta_2,
        'n_samples_used': n_samples,
        'counterexample': worst_counterexample,
        'per_constraint_results': per_row_results,
    }
