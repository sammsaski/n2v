"""
Violation LP, witness extraction, and per-neuron relaxation infidelity (Phase 1).

The violation LP is an *epigraph* form that finds, over a feasible set, the
point minimizing the worst spec-row margin:

    minimize   t
    subject to G_r (c_out + V_out alpha) - g_r <= t   for every spec row r
               C alpha <= d                            (only for the FAITHFUL set)
               pred_lb <= alpha <= pred_ub
               t free

Let ``t*`` be the optimum. The feasible set's image intersects the unsafe region
``{y : G y <= g}`` iff ``t* <= 0`` (some point makes every row margin <= 0). The
minimizer ``alpha*`` is the witness:

  * FAITHFUL  (``include_Cd=True``): minimized over the full predicate polytope P
    -> feasible in P by construction; this is the constraint-faithful witness and
    also the sound prune/UNSAT test (``t* > 0`` => safe).
  * BOX       (``include_Cd=False``): minimized over the predicate box only,
    ignoring the split constraints C/d -> the DRG-BaB box-corner analog, which
    may fall *outside* P.

``t`` is bounded by a problem-derived big-M so the LP stays backend-portable
(some HiGHS paths do not accept infinite variable bounds).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lpsolver import solve_lp
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.types import LinearSpec, NeuronMeta, Witness

logger = logging.getLogger(__name__)

# Prune tolerance: declare the node safe (UNSAT) only when the optimal worst
# margin is clearly positive. Conservative on the safety side => no false UNSAT.
PRUNE_TOL = 1e-7


def _big_M(S: Star, spec: LinearSpec) -> float:
    """A finite bound that cannot clip any achievable spec-row margin over the box."""
    c_out = S.V[:, 0]
    V_out = S.V[:, 1:]
    plb = S.predicate_lb.flatten()
    pub = S.predicate_ub.flatten()
    amax = np.maximum(np.abs(plb), np.abs(pub))            # bound on |alpha_j|
    GV = spec.G @ V_out                                    # (R, nVar)
    base = np.abs(spec.G @ c_out - spec.g)                 # (R,)
    spread = np.abs(GV) @ amax                             # (R,)
    return float(np.max(base + spread)) + 1.0


def violation_lp(
    S: Star,
    spec: LinearSpec,
    include_Cd: bool,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve the epigraph violation LP. Returns ``(alpha, t)`` or ``None`` if the
    feasible set is empty (predicate polytope infeasible).
    """
    nVar = S.nVar
    if nVar == 0:
        # Degenerate star: a single point. t = worst margin at that point.
        y = S.V[:, 0]
        t = float(np.max(spec.margins(y)))
        return np.zeros(0), t

    c_out = S.V[:, 0]
    V_out = S.V[:, 1:]
    GV = spec.G @ V_out                       # (R, nVar)
    rhs = spec.g - spec.G @ c_out             # (R,)
    R = GV.shape[0]

    # Variables: [alpha (nVar), t]. Margin rows: GV alpha - t <= rhs.
    A_rows = [np.hstack([GV, -np.ones((R, 1))])]
    b_rows = [rhs]
    if include_Cd and S.C.size > 0:
        A_rows.append(np.hstack([S.C, np.zeros((S.C.shape[0], 1))]))
        b_rows.append(S.d.flatten())
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)

    M = _big_M(S, spec)
    lb = np.concatenate([S.predicate_lb.flatten(), [-M]])
    ub = np.concatenate([S.predicate_ub.flatten(), [M]])

    f = np.zeros(nVar + 1)
    f[-1] = 1.0  # minimize t

    x, _, status, _ = solve_lp(
        f=f, A=A, b=b, lb=lb, ub=ub, lp_solver=lp_solver, minimize=True
    )
    if status not in ("optimal", "optimal_inaccurate") or x is None:
        return None  # predicate polytope empty => the set is empty (vacuously safe)

    x = np.asarray(x, dtype=np.float64).flatten()
    return x[:nVar], float(x[nVar])


def binding_row(S: Star, spec: LinearSpec, alpha: np.ndarray) -> int:
    """
    The active spec row at the epigraph optimum: ``argmax_r margin_r``. The
    epigraph drives ``t = max_r margin_r`` (the smallest t dominating every row),
    so the row achieving it -- the one obstructing safety -- is the argmax, NOT
    the argmin (which is the most-satisfied, least-relevant row).
    """
    y = S.V[:, 0] + S.V[:, 1:] @ alpha if S.nVar > 0 else S.V[:, 0]
    return int(np.argmax(spec.margins(y)))


def make_witness(
    S: Star, spec: LinearSpec, alpha: np.ndarray, t: float, mode: str
) -> Witness:
    """Wrap a solved ``(alpha, t)`` into a Witness, computing the binding row."""
    return Witness(alpha=alpha, t=t, binding_row=binding_row(S, spec, alpha), mode=mode)


def preactivation(nm: NeuronMeta, alpha: np.ndarray) -> float:
    """
    The relaxed neuron's pre-activation read at the witness:
    ``x_hat_j = preact_center + preact_gens . alpha[:k]`` (k = len(preact_gens)).

    The read is over the predicate *prefix* at the neuron's layer; because affine
    layers leave the predicate untouched and ReLU only appends variables, that
    prefix is preserved verbatim in the final ``alpha``, so the slice is correct.
    """
    k = len(nm.preact_gens)
    return nm.preact_center + float(nm.preact_gens @ alpha[:k])


def epsilon_vector(alpha: np.ndarray, meta: List[NeuronMeta]) -> np.ndarray:
    """
    Per-neuron relaxation infidelity ``eps_j = alpha_j^r - max(0, x_hat_j)`` at
    the witness. Non-negative when ``alpha`` is feasible in P (the triangle rows
    force ``alpha_j^r >= max(0, x_hat_j)``); it may go negative for a box-corner
    witness that violates those rows -- which is exactly the box arm's flaw.
    """
    eps = np.empty(len(meta), dtype=np.float64)
    for i, nm in enumerate(meta):
        x_hat = preactivation(nm, alpha)
        alpha_r = float(alpha[nm.pred_col])
        eps[i] = alpha_r - max(0.0, x_hat)
    return eps


def score_vector(
    alpha: np.ndarray,
    meta: List[NeuronMeta],
    S: Star,
    h: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Branching score ``score_j = eps_j * |h . V_out[:, p(j)]|`` (Def. 4) and the
    underlying ``eps`` vector. ``h`` is the spec direction (the binding row of G).
    """
    eps = epsilon_vector(alpha, meta)
    V_out = S.V[:, 1:]
    influence = np.array(
        [abs(float(h @ V_out[:, nm.pred_col])) for nm in meta], dtype=np.float64
    )
    return eps * influence, eps


def is_true_counterexample(
    alpha: np.ndarray,
    input_star: Star,
    model,
    spec: LinearSpec,
    tol: float = 0.0,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Map the witness's input components to a concrete input, run the EXACT model
    forward, and test membership in the unsafe region.

    Returns ``(is_ce, x_in, y)``. The input components are the predicate prefix
    ``alpha[:input_star.nVar]`` (the shared prefix property), so ``x_in`` lies in
    the input set whenever ``alpha`` is feasible in P.

    ``tol`` defaults to 0.0: a SAT claim requires the exact output to lie inside
    the (closed) unsafe region ``{margin <= 0}``. A positive tol would accept
    points just OUTSIDE the region as counterexamples (false SAT), which the
    zero-tolerance VNN-COMP witness grading would then reject.
    """
    import torch

    n_in = input_star.nVar
    x_in = input_star.V[:, 0] + input_star.V[:, 1:] @ alpha[:n_in]
    # Feed a batched (1, n_in) input so models with a leading nn.Flatten
    # (start_dim=1) forward correctly; squeeze the batch dim back off.
    with torch.no_grad():
        y = (
            model(torch.as_tensor(x_in, dtype=torch.float64).unsqueeze(0))
            .cpu()
            .numpy()
            .flatten()
        )
    return spec.is_unsafe(y, tol=tol), x_in, y
