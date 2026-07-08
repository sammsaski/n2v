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


def _epigraph_system(S: Star, spec: LinearSpec, include_Cd: bool = True):
    """Assemble the epigraph violation LP over variables ``[alpha, t]``:
    ``min t  s.t.  GV alpha - t <= rhs ; [C alpha <= d] ; box ; |t| <= big-M``.
    Returns ``(A, b, lb, ub, f)``. Single source of truth for the exact solver
    (``augmented_violation_lp``) and the GPU witness (``violation_witness_gpu``).
    Requires ``S.nVar > 0`` and finite predicate bounds.
    """
    c_out = S.V[:, 0]
    V_out = S.V[:, 1:]
    GV = spec.G @ V_out                       # (R, nVar)
    rhs = spec.g - spec.G @ c_out             # (R,)
    R = GV.shape[0]

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

    f = np.zeros(S.nVar + 1)
    f[-1] = 1.0  # minimize t
    return A, b, lb, ub, f


def augmented_violation_lp(
    S: Star,
    spec: LinearSpec,
    extra_A: Optional[np.ndarray] = None,
    extra_b: Optional[np.ndarray] = None,
    *,
    include_Cd: bool = True,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve the epigraph violation LP over P, optionally intersected with extra
    halfspaces ``extra_A @ alpha <= extra_b`` (each row over the ``nVar``
    predicate variables, no ``t`` column). Returns ``(alpha, t)`` or ``None`` if
    the feasible set is empty.

    This is the single source of truth for the epigraph LP. ``violation_lp`` is
    the no-extra-rows case; ``ExactStarSBSelector`` passes one split-cut row to
    score a candidate's exact post-split bound *on the existing output star* --
    no re-propagation (the cut is `c_i + V_i.alpha (>= | <=) 0` on the neuron's
    pre-activation, the one halfspace a star split adds to the shared polytope).
    """
    nVar = S.nVar
    if nVar == 0:
        # Degenerate star: a single point. Any extra cut reduces to 0 <= extra_b;
        # if some bound is negative the augmented set is empty (return None).
        if extra_b is not None and np.size(extra_b) > 0:
            if np.any(np.asarray(extra_b, dtype=np.float64).flatten() < 0.0):
                return None
        y = S.V[:, 0]
        t = float(np.max(spec.margins(y)))
        return np.zeros(0), t

    A, b, lb, ub, f = _epigraph_system(S, spec, include_Cd=include_Cd)
    if extra_A is not None and np.size(extra_A) > 0:
        extra_A = np.atleast_2d(np.asarray(extra_A, dtype=np.float64))
        extra_b = np.asarray(extra_b, dtype=np.float64).flatten()
        if extra_A.shape[1] != nVar:
            raise ValueError(
                f"extra_A has {extra_A.shape[1]} columns, expected nVar={nVar}"
            )
        A = np.vstack([A, np.hstack([extra_A, np.zeros((extra_A.shape[0], 1))])])
        b = np.concatenate([b, extra_b])

    x, _, status, _ = solve_lp(
        f=f, A=A, b=b, lb=lb, ub=ub, lp_solver=lp_solver, minimize=True
    )
    if status not in ("optimal", "optimal_inaccurate") or x is None:
        return None  # predicate polytope empty => the set is empty (vacuously safe)

    x = np.asarray(x, dtype=np.float64).flatten()
    return x[:nVar], float(x[nVar])


def violation_lp(
    S: Star,
    spec: LinearSpec,
    include_Cd: bool,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve the epigraph violation LP. Returns ``(alpha, t)`` or ``None`` if the
    feasible set is empty (predicate polytope infeasible). Thin wrapper over
    ``augmented_violation_lp`` with no extra split-cut rows.
    """
    return augmented_violation_lp(
        S, spec, include_Cd=include_Cd, lp_solver=lp_solver
    )


def certified_prune_wave(stars: List[Star], spec: LinearSpec,
                         max_iters: int = 300) -> List[bool]:
    """Batched :func:`certified_prune` over a frontier wave: one population
    GPU call for every star's per-row margin bounds (heterogeneous shapes
    zero-padded — padded rows/columns/variables are inert under NS). Same
    soundness contract as the scalar form; ``False`` = "could not prune".
    """
    from n2v.utils.lpsolver_gpu import solve_lp_population_gpu

    G = np.asarray(spec.G, dtype=np.float64)
    g = np.asarray(spec.g, dtype=np.float64).flatten()
    k = G.shape[0]
    out = [False] * len(stars)

    idxs, mats = [], []
    for i, S in enumerate(stars):
        if S.nVar == 0:
            # degenerate point: disjoint iff some unsafe row is violated at it
            out[i] = bool(np.any(G @ S.V[:, 0] > g))
            continue
        if S.predicate_lb is None or S.predicate_ub is None:
            continue                        # no finite box -> cannot certify here
        idxs.append(i)
        mats.append((S.C, S.d.flatten(), S.predicate_lb.flatten(),
                     S.predicate_ub.flatten(), (G @ S.V[:, 1:]).T,
                     g - G @ S.V[:, 0]))
    if not idxs:
        return out

    B = len(idxs)
    m_max = max(A.shape[0] for A, *_ in mats) or 1
    n_max = max(A.shape[1] if A.size else lbv.size
                for A, _, lbv, *_ in mats)
    Ap = np.zeros((B, m_max, n_max))
    bp = np.zeros((B, m_max))
    lbp = np.zeros((B, n_max))
    ubp = np.zeros((B, n_max))
    Cp = np.zeros((B, n_max, k))
    for j, (A, b, lbv, ubv, obj, _) in enumerate(mats):
        mi, ni = (A.shape if A.size else (0, lbv.size))
        if A.size:
            Ap[j, :mi, :ni] = A
            bp[j, :mi] = b
        lbp[j, :lbv.size] = lbv
        ubp[j, :ubv.size] = ubv
        Cp[j, :obj.shape[0], :] = obj
    lows = solve_lp_population_gpu(Ap, bp, lbp, ubp, Cp,
                                   minimize_flags=[True] * k,
                                   max_iters=max_iters, backend="grad")
    for j, i in enumerate(idxs):
        rhs = mats[j][5]
        out[i] = bool(np.any(lows[j] > rhs))
    return out


def violation_witness_gpu(
    S: Star, spec: LinearSpec, max_iters: int = 400,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[Tuple[np.ndarray, float]]:
    """Approximate faithful witness via GPU PDHG on the same epigraph LP.

    Drop-in for ``violation_lp(S, spec, include_Cd=True)`` on the BaB hot path.
    The returned ``alpha`` is box-feasible by construction and satisfies the
    ``C alpha <= d`` rows only up to the solver's primal residual (dual-informed,
    unlike a box corner); ``t`` is the EXACT max spec-row margin evaluated at
    ``alpha`` (not the LP objective), so ``t <= 0`` still means "this concrete
    point of the relaxation is in the unsafe region".

    Soundness is untouched: the witness only proposes counterexample candidates
    (always re-validated by an exact forward pass WITH an input-set membership
    check -- see ``is_true_counterexample``) and informs the split score. Two
    behavioral deltas vs the exact LP, both safe: (1) it cannot detect an empty
    predicate polytope (never returns ``None`` for infeasibility) -- an empty
    subtree may be expanded until its exact leaves, where ``verify_refine``'s
    exact backstop prunes it; (2) the split score reads slightly perturbed
    epsilons -- selection quality, never correctness. Falls back to the exact
    LP (on ``lp_solver``) for degenerate stars or unbounded predicate boxes.
    """
    if S.nVar == 0 or S.predicate_lb is None or S.predicate_ub is None:
        return violation_lp(S, spec, include_Cd=True, lp_solver=lp_solver)
    from n2v.utils.lpsolver_gpu import solve_lp_primal_gpu

    A, b, lb, ub, f = _epigraph_system(S, spec, include_Cd=True)
    x = solve_lp_primal_gpu(f, A, b, lb, ub, max_iters=max_iters)
    alpha = x[:S.nVar]
    y = S.V[:, 0] + S.V[:, 1:] @ alpha
    t = float(np.max(spec.margins(y)))
    return alpha, t


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
    ``alpha[:input_star.nVar]`` (the shared prefix property).

    Membership is CHECKED, not assumed: approximate witnesses
    (``violation_witness_gpu``) are only box-feasible, so a SAT claim
    additionally requires ``alpha``'s input components to satisfy the input
    star's own box AND ``C alpha <= d`` rows exactly (zero tolerance, matching
    the output test) -- an out-of-set "counterexample" is an invalid witness
    that VNN-COMP replay would reject. Exact-LP witnesses pass this trivially.

    ``tol`` defaults to 0.0: a SAT claim requires the exact output to lie inside
    the (closed) unsafe region ``{margin <= 0}``. A positive tol would accept
    points just OUTSIDE the region as counterexamples (false SAT), which the
    zero-tolerance VNN-COMP witness grading would then reject.
    """
    import torch

    n_in = input_star.nVar
    a_in = alpha[:n_in]
    in_set = True
    if input_star.predicate_lb is not None:
        in_set &= bool(np.all(a_in >= input_star.predicate_lb.flatten()))
    if input_star.predicate_ub is not None:
        in_set &= bool(np.all(a_in <= input_star.predicate_ub.flatten()))
    if input_star.C.size > 0:
        in_set &= bool(np.all(input_star.C @ a_in <= input_star.d.flatten()))
    x_in = input_star.V[:, 0] + input_star.V[:, 1:] @ a_in
    # Feed a batched (1, n_in) input so models with a leading nn.Flatten
    # (start_dim=1) forward correctly; squeeze the batch dim back off.
    with torch.no_grad():
        y = (
            model(torch.as_tensor(x_in, dtype=torch.float64).unsqueeze(0))
            .cpu()
            .numpy()
            .flatten()
        )
    return in_set and spec.is_unsafe(y, tol=tol), x_in, y


def certified_prune(S: Star, spec: LinearSpec, max_iters: int = 300) -> bool:
    """Sound GPU prune: certify ``[[S]] ∩ {G y <= g} = ∅`` without an exact LP.

    For each spec row ``i``, a Neumaier-Shcherbina-certified *lower* bound on
    ``min_{alpha in P} G_i (c + V alpha)`` (the same NS-dual gradient engine as
    ``bound_mode="dual_gpu"``) exceeding ``g_i`` proves every point of ``[[S]]``
    violates row ``i``, so the unsafe conjunction is unsatisfiable -> prune.

    Sound by construction (the certified bound is outward, so a prune implies
    the exact LP would also prove emptiness). For single-row specs -- the whole
    robustness family -- this is the exact emptiness condition up to bound
    tightness; for multi-row specs it is sufficient-only (the box/polytope may
    satisfy every row separately but not jointly), so a False here means
    "could not prune", never "intersects". Falls back to False (no prune) when
    the predicate box is unbounded (NS needs finite bounds) -- the caller's
    exact machinery still applies downstream.

    Thin wrapper over :func:`certified_prune_wave` (ONE implementation, so the
    scalar and wave paths cannot drift -- degenerate/point stars included).
    """
    return certified_prune_wave([S], spec, max_iters=max_iters)[0]
