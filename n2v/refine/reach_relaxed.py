"""
Triangle-relaxation forward reach with fixed neuron phases and per-neuron
metadata (Phase 1).

This is a deliberately self-contained reach for fully-connected ReLU networks.
It is *not* a replacement for ``n2v.nn.reach`` -- it exists so the refinement
loop has full control over (a) forcing individual neurons ACTIVE/INACTIVE and
(b) recording, at the moment of relaxation, the metadata needed to compute the
per-neuron relaxation infidelity ``epsilon_j`` against a witness.

Soundness: an unfixed neuron is classified with ``Star.estimate_ranges`` (the
predicate-box interval, which over-approximates the true range, ignoring C/d).
``u <= 0`` => provably inactive (zeroed); ``l >= 0`` => provably active
(identity); otherwise the standard triangle over-approximation is applied. All
three are sound, so the output star is a sound over-approximation of the
fixed-phase reachable set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.refine.types import NeuronKey, NeuronMeta, Phase
from n2v.refine.tighten import neuron_bounds

logger = logging.getLogger(__name__)


@dataclass
class LinearLayer:
    W: np.ndarray  # (out, in)
    b: np.ndarray  # (out,)


@dataclass
class ReluLayer:
    pass


Layer = object  # LinearLayer | ReluLayer


def extract_layers(model) -> List[Layer]:
    """
    Flatten a simple FC ReLU ``torch.nn.Module`` into an ordered layer list.

    Supports ``nn.Sequential`` (or any module with an ordered ``.children()``)
    composed of ``Linear``, ``ReLU``, and ``Flatten`` (skipped: the star is
    already over a flat input vector). Anything else raises -- Phase 1 is FC
    ReLU only by design.
    """
    import torch.nn as nn

    children = list(model.children())
    if not children:
        children = [model]

    layers: List[Layer] = []
    for m in children:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().astype(np.float64)
            b = (
                m.bias.detach().cpu().numpy().astype(np.float64)
                if m.bias is not None
                else np.zeros(W.shape[0], dtype=np.float64)
            )
            layers.append(LinearLayer(W, b))
        elif isinstance(m, nn.ReLU):
            layers.append(ReluLayer())
        elif isinstance(m, nn.Flatten):
            continue
        else:
            raise TypeError(
                f"reach_relaxed supports only Linear/ReLU/Flatten in Phase 1, "
                f"got {type(m).__name__}"
            )
    return layers


def _relu_layer_relaxed(
    S: Star,
    layer_idx: int,
    fixed: Dict[NeuronKey, Phase],
    bound_mode: str = "box",
) -> Tuple[Star, List[NeuronMeta]]:
    """
    Apply ReLU to star ``S`` at ReLU-layer index ``layer_idx``.

    Fixed neurons are forced (INACTIVE -> zero row, ACTIVE -> identity, no
    relaxation). Unfixed neurons are classified by interval and either
    zeroed (inactive), passed through (active), or triangle-relaxed (unstable),
    in which case a fresh predicate variable is appended and a ``NeuronMeta`` is
    recorded.

    ``bound_mode`` selects how the pre-activation ranges are computed:
    ``"box"`` (predicate-box interval, cheap, Phase-1), or ``"lp_cpu"`` /
    ``"lp_gpu"`` (LP-over-P bound tightening, tighter, stabilises more neurons). ``box`` and
    ``lp_cpu`` trust the interval/HiGHS optimum as the rest of n2v's sound path
    does (``get_range``/``violation_lp``); ``lp_gpu`` is rigorously outward via
    Neumaier-Shcherbina. Tighter ranges only ever reduce the crossing set, never
    enlarge it.
    """
    if bound_mode == "box":
        lb, ub = S.estimate_ranges()
    else:
        lb, ub = neuron_bounds(S, backend=bound_mode)
    lb = lb.flatten()
    ub = ub.flatten()
    dim = S.dim
    nVar = S.nVar

    Vpre = S.V  # pre-ReLU basis (read pre-activations from here, before edits)
    V = S.V.copy()

    # Forcing a neuron's phase must also *constrain* the predicate to that phase's
    # subregion (x_hat_j >= 0 for ACTIVE, x_hat_j <= 0 for INACTIVE) -- exactly as
    # the exact ReLU split does. Without these rows the forced star can
    # under-approximate the true ReLU (output x_hat where truth is 0), which is
    # unsound. We collect them over the current predicate (nVar columns) and pad
    # for any appended variables below.
    phase_rows: List[np.ndarray] = []
    phase_d: List[float] = []

    crossing: List[Tuple[int, float, float]] = []
    for i in range(dim):
        key = NeuronKey(layer_idx, i)
        if key in fixed:
            center_i = float(Vpre[i, 0])
            gens_i = Vpre[i, 1 : nVar + 1].astype(np.float64)
            if fixed[key] == Phase.INACTIVE:
                V[i, :] = 0.0
                # x_hat_i <= 0  <=>  gens_i . alpha <= -center_i
                phase_rows.append(gens_i.copy())
                phase_d.append(-center_i)
            else:
                # ACTIVE: identity pass-through, restrict to x_hat_i >= 0
                # -gens_i . alpha <= center_i
                phase_rows.append(-gens_i)
                phase_d.append(center_i)
            continue
        li, ui = lb[i], ub[i]
        if ui <= 0.0:
            V[i, :] = 0.0          # provably inactive (sound, no constraint needed)
        elif li >= 0.0:
            pass                   # provably active
        else:
            crossing.append((i, li, ui))

    if not crossing:
        if phase_rows:
            extra = np.vstack(phase_rows)
            extra_d = np.array(phase_d, dtype=np.float64).reshape(-1, 1)
            if S.C.size > 0:
                newC = np.vstack([S.C, extra])
                newd = np.vstack([S.d, extra_d])
            else:
                newC, newd = extra, extra_d
            return Star(V, newC, newd, S.predicate_lb, S.predicate_ub), []
        return Star(V, S.C, S.d, S.predicate_lb, S.predicate_ub), []

    m = len(crossing)
    new_nVar = nVar + m

    # New basis: copy existing columns, append m zero columns; relaxed neuron
    # rows are rewritten to select their fresh predicate variable.
    newV = np.zeros((dim, new_nVar + 1), dtype=np.float64)
    newV[:, : nVar + 1] = V

    extra_rows: List[np.ndarray] = []
    extra_d: List[float] = []
    new_lb: List[float] = []
    new_ub: List[float] = []
    meta: List[NeuronMeta] = []

    for t, (i, li, ui) in enumerate(crossing):
        k = nVar + t          # fresh alpha index (0-based)
        col = k + 1           # its column in V
        center_i = float(Vpre[i, 0])
        gens_i = Vpre[i, 1 : nVar + 1].astype(np.float64)
        lam = ui / (ui - li)

        # neuron output becomes the fresh variable
        newV[i, :] = 0.0
        newV[i, col] = 1.0

        meta.append(
            NeuronMeta(
                key=NeuronKey(layer_idx, i),
                pred_col=k,
                preact_center=center_i,
                preact_gens=gens_i.copy(),
                l=li,
                u=ui,
            )
        )

        # (ii) alpha_new >= x_hat_i  <=>  gens_i . alpha + center_i - alpha_new <= 0
        row_ii = np.zeros(new_nVar, dtype=np.float64)
        row_ii[:nVar] = gens_i
        row_ii[k] = -1.0
        extra_rows.append(row_ii)
        extra_d.append(-center_i)

        # (iii) alpha_new <= lam (x_hat_i - l)
        #   <=>  -lam gens_i . alpha + alpha_new <= lam (center_i - l)
        row_iii = np.zeros(new_nVar, dtype=np.float64)
        row_iii[:nVar] = -lam * gens_i
        row_iii[k] = 1.0
        extra_rows.append(row_iii)
        extra_d.append(lam * (center_i - li))

        # (i) alpha_new >= 0 is enforced by the predicate box below.
        new_lb.append(0.0)
        new_ub.append(ui)

    # Assemble C, d (pad old + phase constraints with zero columns for new vars).
    blocks_C: List[np.ndarray] = []
    blocks_d: List[np.ndarray] = []
    if S.C.size > 0:
        blocks_C.append(np.hstack([S.C, np.zeros((S.C.shape[0], m), dtype=np.float64)]))
        blocks_d.append(S.d)
    if phase_rows:
        pr = np.vstack(phase_rows)
        blocks_C.append(np.hstack([pr, np.zeros((pr.shape[0], m), dtype=np.float64)]))
        blocks_d.append(np.array(phase_d, dtype=np.float64).reshape(-1, 1))
    blocks_C.append(np.vstack(extra_rows))
    blocks_d.append(np.array(extra_d, dtype=np.float64).reshape(-1, 1))
    newC = np.vstack(blocks_C)
    newd = np.vstack(blocks_d)

    plb = (
        np.vstack([S.predicate_lb, np.array(new_lb).reshape(-1, 1)])
        if S.predicate_lb is not None
        else None
    )
    pub = (
        np.vstack([S.predicate_ub, np.array(new_ub).reshape(-1, 1)])
        if S.predicate_ub is not None
        else None
    )

    return Star(newV, newC, newd, plb, pub), meta


def relaxed_reach(
    input_star: Star,
    layers: List[Layer],
    fixed: Optional[Dict[NeuronKey, Phase]] = None,
    bound_mode: str = "box",
) -> Tuple[Star, List[NeuronMeta]]:
    """
    Forward-propagate ``input_star`` through ``layers`` with triangle ReLU
    relaxation and the given fixed neuron phases.

    ``bound_mode`` ("box" | "lp_cpu" | "lp_gpu") selects Phase-1 box bounds or
    LP-over-P bound tightening for neuron classification.

    Returns the output star and the list of ``NeuronMeta`` for every relaxed
    (unfixed, unstable) neuron across all layers, in creation order.
    """
    if fixed is None:
        fixed = {}

    S = input_star
    meta: List[NeuronMeta] = []
    relu_idx = 0
    for layer in layers:
        if isinstance(layer, LinearLayer):
            S = S.affine_map(layer.W, layer.b)
        elif isinstance(layer, ReluLayer):
            S, layer_meta = _relu_layer_relaxed(S, relu_idx, fixed, bound_mode)
            meta.extend(layer_meta)
            relu_idx += 1
        else:
            raise TypeError(f"Unknown layer type {type(layer).__name__}")

    # Attach the relaxation metadata + search provenance to the output star so
    # the refine set-operations (refine/split) are self-describing. ``relax_meta``
    # is a declared Star field; ``fixed``/``bound_mode`` are refine-search
    # provenance. The tuple return is preserved for existing callers.
    if S is input_star:
        # No layer transformed the star (e.g. empty/Flatten-only ``layers``): wrap
        # a fresh Star so we attach provenance to it rather than mutating the
        # caller's input set (which is reused across calls).
        S = Star(S.V, S.C, S.d, S.predicate_lb, S.predicate_ub)
    S.relax_meta = meta
    S.fixed = dict(fixed)
    S.bound_mode = bound_mode
    return S, meta
