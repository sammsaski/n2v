"""
Split-selection strategies (Phase 1).

Every selector returns the ``NeuronKey`` of the unstable neuron to split next.
The two strategies under test differ *only* in the witness feeding the
``eps * influence`` score:

  * ``FaithfulSelector`` scores at the constraint-faithful witness (minimizer
    over the full predicate polytope P) -- reuses the witness already solved by
    the BaB loop for pruning, so it adds no extra LP.
  * ``BoxCornerSelector`` scores at the box-corner witness (minimizer over the
    predicate box only, ignoring C/d) -- the DRG-BaB analog. It solves its own
    box LP, which may return a point outside P.

``RandomSelector`` and ``BoundWidthSelector`` are controls that use no witness.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.types import LinearSpec, NeuronKey, NeuronMeta, Witness
from n2v.refine.witness import (
    augmented_violation_lp,
    binding_row,
    score_vector,
    violation_lp,
)


class Selector:
    """Base class. ``name`` labels the strategy in experiment output."""

    name = "base"

    def choose(
        self,
        S_out: Star,
        spec: LinearSpec,
        meta: List[NeuronMeta],
        faithful_witness: Witness,
        lp_solver=LPSolver.DEFAULT,
    ) -> Optional[NeuronKey]:
        raise NotImplementedError


class FaithfulSelector(Selector):
    name = "faithful"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        h = spec.G[faithful_witness.binding_row]
        scores, _ = score_vector(faithful_witness.alpha, meta, S_out, h)
        return meta[int(np.argmax(scores))].key


class BoxCornerSelector(Selector):
    name = "box_corner"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        box = violation_lp(S_out, spec, include_Cd=False, lp_solver=lp_solver)
        if box is None:
            # Box LP infeasible (cannot happen when P is non-empty, since box >= P);
            # fall back to the faithful witness rather than failing.
            alpha, binding = faithful_witness.alpha, faithful_witness.binding_row
        else:
            alpha = box[0]
            binding = binding_row(S_out, spec, alpha)
        h = spec.G[binding]
        scores, _ = score_vector(alpha, meta, S_out, h)
        return meta[int(np.argmax(scores))].key


def _pin_rows(nm: NeuronMeta, nVar: int, active: bool):
    """
    The extra inequality rows that un-relax neuron ``nm`` to one phase *on the
    existing output star* -- the proposal's equality-pin discipline. Returns
    ``(A, b)`` with ``A @ alpha <= b``.

    Un-relaxing must PIN the relaxed output variable with an equality, not merely
    add the pre-activation halfspace: with the fresh var ``alpha_r`` (column
    ``pred_col``) left free in its triangle, the parent witness survives unchanged
    into one child and the score is degenerate. Reading
    ``g_i(alpha) = preact_center + preact_gens . alpha``:

      active   : g_i >= 0  AND  alpha_r = g_i   (exact identity branch)
      inactive : g_i <= 0  AND  alpha_r = 0      (exact zero branch)

    matching exactly what ``reach_relaxed`` adds for a forced neuron, but applied
    to ``S_out`` with no re-propagation. The equality is encoded as the pair of
    inequalities so the (inequality-only) epigraph LP accepts it.
    """
    g = np.zeros(nVar, dtype=np.float64)
    g[: len(nm.preact_gens)] = nm.preact_gens
    e = np.zeros(nVar, dtype=np.float64)
    e[nm.pred_col] = 1.0
    c = nm.preact_center
    if active:
        # g_i >= 0 ; alpha_r <= g_i ; alpha_r >= g_i
        A = np.vstack([-g, e - g, g - e])
        b = np.array([c, c, -c])
    else:
        # g_i <= 0 ; alpha_r <= 0 ; alpha_r >= 0
        A = np.vstack([g, e, -e])
        b = np.array([-c, 0.0, 0.0])
    return A, b


class ExactStarSBSelector(Selector):
    """
    Exact star strong branching: score each candidate by its *actual* post-split
    bound, computed as one LP per child *on the existing output star* -- no
    re-propagation (the structural insight: a star split un-relaxes one neuron by
    adding constraints to the shared predicate polytope and leaves every
    downstream affine map, and the output objective, unchanged).

    For candidate neuron ``i`` the two child worst-margins are

        m_i^+ = min t  over  P  with neuron i pinned ACTIVE   (g_i>=0, alpha_r=g_i)
        m_i^- = min t  over  P  with neuron i pinned INACTIVE (g_i<=0, alpha_r=0)

    (see ``_pin_rows`` -- the equality pin is the crux; the halfspace alone leaves
    the witness in one child and the score degenerate). The split is scored by the
    *improvement* over the parent worst-margin ``t_parent`` (the faithful witness
    ``t``, already solved by the BaB loop, so the parent costs no extra LP):

        Delta_i^pm = m_i^pm - t_parent   (>= 0; un-relaxing only shrinks the set)

    combined across the two children. The default combiner ``min`` is the FSB
    objective: maximize the *worst* child's bound improvement (UNSAT needs both
    children proven safe). An empty child (LP infeasible -> that phase is
    impossible over P, a free prune) scores ``+inf`` improvement.

    ``m_i^pm`` is the exact marginal effect with every *other* neuron's triangle
    held fixed -- a sound over-approximation of the *true* phase-restricted child
    (it lower-bounds the exact child's worst margin). It equals the re-propagated
    relaxed child *exactly* for a last-layer neuron (nothing downstream to
    re-tighten); for an earlier neuron it is a different, incomparable
    over-approximation (re-propagation rebuilds downstream triangles over a changed
    variable set -- neither bound dominates). Selection only needs the ranking, so
    this is sufficient.

    **Cost / the FSB filter.** Scoring every candidate is ``2 * |candidates|`` LPs
    per node -- ruinous on real nets (~250 unstable neurons on ACAS Xu => ~500 LPs
    per node). ``top_k`` applies the FSB pre-filter: rank candidates by the cheap
    ``eps * influence`` heuristic (reusing the already-solved faithful witness -- no
    extra LP) and exact-score only the top ``k``, cutting the cost to
    ``2 * min(k, |candidates|)`` LPs per node. ``top_k=None`` scores all (the pure
    look-ahead, for ablation). Warm-starting each candidate LP from the node's base
    basis is the next speedup (deferred).
    """

    name = "exact_sb"

    # Combiner of the two child improvements (Delta^+, Delta^-) into one score.
    # Both children are pinned, so both improvements are genuinely >= 0 and
    # discriminating (the degenerate halfspace-only form is gone). +inf (a pruned
    # child) flows through min/sum without NaN.
    _COMBINERS = {
        "min": min,                    # FSB worst-child improvement (default)
        "sum": lambda a, b: a + b,     # total improvement across both children
    }

    def __init__(self, combiner: str = "min", top_k: Optional[int] = 8):
        if combiner not in self._COMBINERS:
            raise ValueError(
                f"unknown combiner {combiner!r}; expected one of "
                f"{sorted(self._COMBINERS)}"
            )
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
        self.combiner = combiner
        self._combine = self._COMBINERS[combiner]
        self.top_k = top_k

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        nVar = S_out.nVar
        t_parent = faithful_witness.t

        # FSB pre-filter: keep the top_k candidates by the cheap eps*influence
        # heuristic (reuses the solved witness -> no extra LP), exact-score only
        # those. Skipped when scoring all (top_k None) or already small enough.
        cand = meta
        if self.top_k is not None and len(meta) > self.top_k:
            h = spec.G[faithful_witness.binding_row]
            hscores, _ = score_vector(faithful_witness.alpha, meta, S_out, h)
            top = np.argsort(hscores)[::-1][: self.top_k]
            cand = [meta[i] for i in top]

        best_key: Optional[NeuronKey] = None
        best_score = -np.inf
        for nm in cand:
            A_act, b_act = _pin_rows(nm, nVar, active=True)
            A_ina, b_ina = _pin_rows(nm, nVar, active=False)
            r_act = augmented_violation_lp(
                S_out, spec, A_act, b_act, include_Cd=True, lp_solver=lp_solver
            )
            r_ina = augmented_violation_lp(
                S_out, spec, A_ina, b_ina, include_Cd=True, lp_solver=lp_solver
            )
            m_act = np.inf if r_act is None else r_act[1]
            m_ina = np.inf if r_ina is None else r_ina[1]
            score = self._combine(m_act - t_parent, m_ina - t_parent)
            if score > best_score:
                best_score, best_key = score, nm.key
        return best_key


class RandomSelector(Selector):
    name = "random"

    def __init__(self, seed: int = 47):
        self._rng = np.random.default_rng(seed)

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        return meta[int(self._rng.integers(len(meta)))].key


class BoundWidthSelector(Selector):
    name = "bound_width"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        widths = np.array([nm.u - nm.l for nm in meta])
        return meta[int(np.argmax(widths))].key
