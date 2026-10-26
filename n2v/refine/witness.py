"""
Witnesses for the refine search: the violation LP, certified pruning,
counterexample validation and branching guidance.

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
    ignoring the split constraints C/d, so the minimizer is a corner of the
    box and may fall *outside* P.

``t`` is bounded by a problem-derived big-M so the LP stays backend-portable
(some HiGHS paths do not accept infinite variable bounds).

The other roles of this module: approximate faithful witnesses for a whole
frontier wave in one population GPU call (``violation_witness_wave``); the
certified group prune, which proves a wave's stars disjoint from the unsafe
region with NS-certified GPU bounds (``certified_prune_wave``); zero-tolerance
counterexample validation, an exact float64 forward pass plus exact input
membership, with box corners snapped into the user's raw float64 box
(``is_true_counterexample``, ``clip_to_raw_box``); and branching guidance from
a witness (``make_witness``, ``preactivation``, ``epsilon_vector``,
``score_vector``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.refine.types import LinearSpec, NeuronMeta, Witness
from n2v.utils.lpsolver import (LPFailure, _solve_lp_careful,
                                confirm_infeasible, solve_lp)
from n2v.utils.ns_certificate import outward_spec_rows
from n2v.utils.lp_solver_enum import LPSolver


def clip_to_raw_box(x: np.ndarray, input_star) -> np.ndarray:
    """Snap a reconstructed input point into the ORIGINAL float64 box.

    A box-derived input star is parametrised as ``c + rad * alpha`` with
    ``alpha in [-1, 1]``; a corner reconstructed from ``alpha = +-1`` lands
    1 ulp outside the user's box on ~half the coordinates (measured on
    every corner witness of a 5-D box). VNN-COMP replays input membership
    at ZERO tolerance, so such a witness is graded invalid. Only applies
    when the star carries raw bounds and has no polytope rows (a plain box),
    where the clip is a no-op on membership; otherwise ``x`` is returned as is.
    """
    lb = getattr(input_star, "_raw_lb", None)
    ub = getattr(input_star, "_raw_ub", None)
    if lb is None or ub is None:
        return x
    # The attribute exists only on the object ``Box.to_star`` built, whose
    # C/d rows are exactly the box rows (2 * nVar of them); a star that gained
    # rows is a different object (every intersection/affine op returns a new
    # Star), but guard the row count anyway.
    C = getattr(input_star, "C", None)
    if C is not None and C.size and C.shape[0] > 2 * lb.size:
        return x
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != lb.size and x.size != lb.size:
        return x
    return np.clip(x, lb.reshape(x.shape), ub.reshape(x.shape))


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
    (``violation_lp``) and the batched GPU witness
    (``violation_witness_wave``). Requires ``S.nVar > 0`` and finite predicate
    bounds.
    """
    # rows G V' alpha - t <= g - G c, with the rhs inflated outward by the
    # rounding of G V' and g - G c (a touching set must not round into t* > 0)
    GV, rhs = outward_spec_rows(spec.G, spec.g, S.V,
                                S.predicate_lb, S.predicate_ub)   # (R, nVar), (R,)
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


def violation_lp(
    S: Star,
    spec: LinearSpec,
    include_Cd: bool,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve the epigraph violation LP (module docstring) over ``S``'s predicate
    polytope (``include_Cd=True``) or its predicate box only
    (``include_Cd=False``). Returns ``(alpha, t)``, or ``None`` if the feasible
    set is certified empty (predicate polytope infeasible). This is the single
    source of truth for the exact epigraph LP.

    A degenerate star (no predicate variables) is a single point: ``alpha`` is
    empty and ``t`` is that point's worst spec-row margin. A solver failure,
    or an 'infeasible' status that ``confirm_infeasible`` does not certify,
    raises ``LPFailure``: it is not evidence of emptiness, and the caller
    degrades to UNKNOWN.
    """
    nVar = S.nVar
    if nVar == 0:
        return np.zeros(0), float(np.max(spec.margins(S.V[:, 0])))

    A, b, lb, ub, f = _epigraph_system(S, spec, include_Cd=include_Cd)
    x, _, status, _ = solve_lp(
        f=f, A=A, b=b, lb=lb, ub=ub, lp_solver=lp_solver, minimize=True
    )
    # SOUNDNESS: ``None`` means "certified empty", and callers act on it by
    # pruning the branch as vacuously safe. Only an explicit ``infeasible``
    # status justifies that. A solver failure is not a certificate: returning
    # None for it would prune a branch that may well contain a counterexample
    # (a false UNSAT). Failures raise so the caller degrades to UNKNOWN.
    if status == "infeasible":
        # SOUNDNESS: confirm before treating as empty -- the default solve's
        # 'infeasible' is a presolve artefact on tiny-scale / near-degenerate
        # systems (measured false UNSAT at the root). If unconfirmed, re-solve
        # carefully; an optimum from that solve is used as the witness.
        if confirm_infeasible(A, b, lb, ub):
            return None  # certified-empty feasible set => vacuously safe
        # confirm_infeasible just refused to certify emptiness; a presolve-off
        # 'infeasible' is not a certificate either, so the caller degrades to
        # UNKNOWN. The re-solve uses scipy's HiGHS when highspy is absent.
        x, _, status, _ = _solve_lp_careful(f, A, b, lb, ub, minimize=True)
    if status not in ("optimal", "optimal_inaccurate") or x is None:
        raise LPFailure(
            f"violation LP did not decide (unconfirmed infeasibility or "
            f"solver status {status!r}; x is None: {x is None}); this is not "
            f"evidence of emptiness"
        )

    x = np.asarray(x, dtype=np.float64).flatten()
    return x[:nVar], float(x[nVar])


def certified_prune_wave(stars: List[Star], spec: LinearSpec,
                         max_iters: int = 300,
                         device: Optional[str] = None) -> List[bool]:
    """Sound GPU prune for a frontier wave: certify
    ``[[S]] ∩ {G y <= g} = ∅`` for every star without an exact LP, with ONE
    population GPU call for every star's per-row margin bounds.

    For each spec row ``i``, a Neumaier-Shcherbina-certified *lower* bound on
    ``min_{alpha in P} G_i (c + V alpha)`` (the same NS-dual gradient engine as
    ``bound_mode="dual_gpu"``) exceeding ``g_i`` proves every point of ``[[S]]``
    violates row ``i``, so the unsafe conjunction is unsatisfiable -> prune.

    Sound by construction (the certified bound is outward, so a prune implies
    the exact LP would also prove emptiness). For single-row specs -- the whole
    robustness family -- this is the exact emptiness condition up to bound
    tightness; for multi-row specs it is sufficient-only (the box/polytope may
    satisfy every row separately but not jointly), so ``False`` means "could
    not prune", never "intersects". A star without a finite predicate box gets
    ``False`` (no prune; NS needs finite bounds) -- the caller's exact
    machinery still applies downstream. Heterogeneous shapes are zero-padded
    (padded rows/columns/variables are inert under NS).
    """
    from n2v.utils.lpsolver_gpu import solve_lp_population_gpu

    G = np.asarray(spec.G, dtype=np.float64)
    g = np.asarray(spec.g, dtype=np.float64).flatten()
    k = G.shape[0]
    out = [False] * len(stars)

    idxs, mats = [], []
    for i, S in enumerate(stars):
        if S.nVar == 0:
            # degenerate point: disjoint iff some unsafe row is violated at
            # it, decided on the outward-inflated rhs (the same rounding
            # discipline as the general path) so a point exactly on the
            # boundary is never pruned
            _, rhs0 = outward_spec_rows(G, g, S.V)
            out[i] = bool(np.any(rhs0 < 0.0))
            continue
        if S.predicate_lb is None or S.predicate_ub is None:
            continue                        # no finite box -> cannot certify here
        idxs.append(i)
        plb = S.predicate_lb.flatten()
        pub = S.predicate_ub.flatten()
        # SOUNDNESS: the NS certificate is outward for the LP it is HANDED,
        # but the objective G V' and rhs g - G c are rounded products of the
        # star/spec data; outward_spec_rows charges that rounding to the rhs.
        # Without it, touching cases (exact margin 0) were pruned as disjoint
        # (measured: prune_mode='dual_gpu' said UNSAT where 'exact' said
        # UNKNOWN on the same float data).
        GV, rhs = outward_spec_rows(G, g, S.V, plb, pub)
        mats.append((S.C, S.d.flatten(), plb, pub, GV.T, rhs))
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
                                   max_iters=max_iters, device=device)
    for j, i in enumerate(idxs):
        rhs = mats[j][5]
        out[i] = bool(np.any(lows[j] > rhs))
    return out


def violation_witness_wave(
    stars: List[Star], spec: LinearSpec, max_iters: int = 400,
    lp_solver=LPSolver.DEFAULT, device: Optional[str] = None,
) -> List[Optional[Tuple[np.ndarray, float]]]:
    """Approximate faithful witnesses via GPU PDHG on the epigraph LP, batched
    over a frontier wave: assemble every star's epigraph LP (variables
    ``[alpha, t]``, minimize ``t``) and solve them all in ONE population PDHG
    call -- the sequential-iteration launch cost that made a per-branch GPU
    witness ~67ms is paid once per wave. Drop-in for per-star
    ``violation_lp(S, spec, include_Cd=True)`` on the BaB hot path.

    Per-star semantics: ``alpha`` is box-feasible by construction and
    satisfies the ``C alpha <= d`` rows only up to the solver's primal
    residual (dual-informed, unlike a box corner); ``t`` is the EXACT max
    spec-row margin evaluated at ``alpha`` (not the LP objective), so
    ``t <= 0`` still means "this concrete point of the relaxation is in the
    unsafe region".

    Soundness is untouched: the witness only proposes counterexample
    candidates (always re-validated by an exact forward pass WITH an input-set
    membership check -- see ``is_true_counterexample``) and informs the split
    score. Two behavioral deltas vs the exact LP, both safe: (1) the PDHG path
    cannot detect an empty predicate polytope (never returns ``None`` for
    infeasibility) -- an empty subtree may be expanded until its exact leaves,
    where ``verify_refine``'s exact backstop prunes it; (2) the split score
    reads slightly perturbed epsilons -- selection quality, never correctness.

    Degenerate stars (no variables / unbounded box) fall back to the exact LP
    (on ``lp_solver``). Heterogeneous shapes are zero-padded (inert: zero rows
    are always satisfied, padded variables are clamped to 0).
    """
    from n2v.utils.lpsolver_gpu import solve_lp_primal_population_gpu

    results: List[Optional[Tuple[np.ndarray, float]]] = [None] * len(stars)
    idxs, mats = [], []
    for i, S in enumerate(stars):
        if S.nVar == 0 or S.predicate_lb is None or S.predicate_ub is None:
            results[i] = violation_lp(S, spec, include_Cd=True,
                                      lp_solver=lp_solver)
            continue
        idxs.append(i)
        mats.append(_epigraph_system(S, spec, include_Cd=True))
    if not idxs:
        return results

    B = len(idxs)
    m_max = max(A.shape[0] for A, *_ in mats)
    n_max = max(A.shape[1] for A, *_ in mats)
    Ap = np.zeros((B, m_max, n_max))
    bp = np.zeros((B, m_max))
    lbp = np.zeros((B, n_max))
    ubp = np.zeros((B, n_max))
    fp = np.zeros((B, n_max))
    for j, (A, b, lb, ub, f) in enumerate(mats):
        mi, ni = A.shape
        Ap[j, :mi, :ni] = A
        bp[j, :mi] = b
        lbp[j, :ni] = lb
        ubp[j, :ni] = ub
        fp[j, :ni] = f
    X = solve_lp_primal_population_gpu(Ap, bp, lbp, ubp, fp,
                                       max_iters=max_iters, device=device)
    for j, i in enumerate(idxs):
        S = stars[i]
        alpha = X[j, :S.nVar]
        y = S.V[:, 0] + S.V[:, 1:] @ alpha
        results[i] = (alpha, float(np.max(spec.margins(y))))
    return results


def _is_input_member(a_in: np.ndarray, input_star: Star) -> bool:
    """Exact zero-tolerance membership of a candidate's input components in
    the input star's predicate set: box bounds (where present) AND every
    ``C alpha <= d`` row.

    This is the SAT-soundness gate of ``is_true_counterexample``: approximate
    witnesses are only box-feasible, so a SAT claim additionally requires
    exact input-set membership (zero tolerance, matching the output test) --
    an out-of-set "counterexample" is an invalid witness that VNN-COMP replay
    would reject.
    """
    in_set = True
    if input_star.predicate_lb is not None:
        in_set &= bool(np.all(a_in >= input_star.predicate_lb.flatten()))
    if input_star.predicate_ub is not None:
        in_set &= bool(np.all(a_in <= input_star.predicate_ub.flatten()))
    if input_star.C.size > 0:
        in_set &= bool(np.all(input_star.C @ a_in <= input_star.d.flatten()))
    return in_set


def binding_row(S: Star, spec: LinearSpec, alpha: np.ndarray) -> int:
    """
    The active spec row at the epigraph optimum: ``argmax_r margin_r``. The
    epigraph drives ``t = max_r margin_r`` (the smallest t dominating every row),
    so the row achieving it -- the one obstructing safety -- is the argmax, NOT
    the argmin (which is the most-satisfied, least-relevant row).
    """
    y = S.V[:, 0] + S.V[:, 1:] @ alpha if S.nVar > 0 else S.V[:, 0]
    return int(np.argmax(spec.margins(y)))


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
    (``violation_witness_wave``) are only box-feasible, so a SAT claim
    additionally requires ``alpha``'s input components to satisfy the input
    star's own box AND ``C alpha <= d`` rows exactly (``_is_input_member``:
    zero tolerance, matching the output test). Exact-LP witnesses pass this
    trivially.

    ``tol`` defaults to 0.0: a SAT claim requires the exact output to lie inside
    the (closed) unsafe region ``{margin <= 0}``. A positive tol would accept
    points just OUTSIDE the region as counterexamples (false SAT), which the
    zero-tolerance VNN-COMP witness grading would then reject.
    """
    import torch

    n_in = input_star.nVar
    a_in = alpha[:n_in]
    in_set = _is_input_member(a_in, input_star)
    x_in = input_star.V[:, 0] + input_star.V[:, 1:] @ a_in
    x_in = clip_to_raw_box(x_in, input_star)     # 1-ulp corner snap (see helper)
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
