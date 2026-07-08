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


# Reduced dual_gpu iteration budget for child (resume) reaches: children need a
# marginal refresh of already-tight parent bounds, not from-scratch tightening
# (T1a: ~100 iterations recovers 93-99% of the box->exact gap). Root reaches use
# the tighten.py default (300).
RESUME_GPU_ITERS = 150


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
    zono_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    precomputed_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    gpu_iters: Optional[int] = None,
    warm_duals: Optional[dict] = None,
    collect_duals: bool = False,
) -> Tuple[Star, List[NeuronMeta], Tuple[np.ndarray, np.ndarray], Optional[dict]]:
    """
    Apply ReLU to star ``S`` at ReLU-layer index ``layer_idx``.

    Fixed neurons are forced (INACTIVE -> zero row, ACTIVE -> identity, no
    relaxation). Unfixed neurons are classified by interval and either
    zeroed (inactive), passed through (active), or triangle-relaxed (unstable),
    in which case a fresh predicate variable is appended and a ``NeuronMeta`` is
    recorded.

    ``bound_mode`` selects how the pre-activation ranges are computed:
    ``"box"`` (predicate-box interval, cheap, Phase-1); ``"lp_cpu"`` / ``"lp_gpu"``
    / ``"dual_gpu"`` (LP-over-P bound tightening, tighter, stabilises more
    neurons; ``dual_gpu`` = best-iterate gradient descent on the NS dual); ``"zono"``
    (intersect the box interval with a DeepZ outer-zonotope bound -- ``zono_bounds``
    -- which tracks correlations the predicate box loses, at propagation cost, no
    LP); or ``"zono_lp"`` (the nnenum recipe: use the cheap zono bound to
    pre-classify, then LP *only* the neurons zono leaves unstable). ``zono_lp`` is
    **result-identical to ``lp_cpu``** -- a zono-stable neuron is LP-stable with the
    same ReLU phase, and a stable neuron's bound value never reaches the output --
    so it is a pure speedup whenever zono stabilises more neurons than the LP would
    have to examine. All modes are sound; intersecting sound enclosures stays sound.
    """
    used_duals: Optional[dict] = None
    if precomputed_bounds is not None:
        # Caller-supplied bounds for THIS entry star (resume_reach at the split
        # layer: the child's entry star IS the parent's checkpoint, so the
        # parent's bounds are the identical LP's result -- exact reuse, the
        # whole OBBT batch of the split layer skipped).
        lb, ub = precomputed_bounds
    elif bound_mode in ("box", "zono", "zono_lp"):
        lb, ub = S.estimate_ranges()
        if bound_mode in ("zono", "zono_lp") and zono_bounds is not None:
            zlb, zub = zono_bounds
            lb = np.maximum(lb, zlb.reshape(lb.shape))
            ub = np.minimum(ub, zub.reshape(ub.shape))
        if bound_mode == "zono_lp":
            from n2v.refine.tighten import _lp_ranges
            lbf, ubf = lb.flatten(), ub.flatten()
            unstable = np.flatnonzero((lbf < 0.0) & (ubf > 0.0))
            if unstable.size > 0:
                lo, hi = _lp_ranges(S, "lp_cpu", unstable)   # LP only the zono-unstable
                lbf = lbf.copy(); ubf = ubf.copy()
                lbf[unstable] = np.maximum(lo, lbf[unstable])
                ubf[unstable] = np.minimum(hi, ubf[unstable])
                lb, ub = lbf.reshape(-1, 1), ubf.reshape(-1, 1)
    else:
        # dual_gpu: LP only the box-unstable dims (result-identical per the
        # neuron_bounds contract; the GPU batch is one tensor op either way, so
        # unlike lp_cpu the subset solve cannot regress).
        nb = neuron_bounds(S, backend=bound_mode,
                           prefilter=(bound_mode == "dual_gpu"),
                           gpu_iters=gpu_iters, warm_duals=warm_duals,
                           collect_duals=collect_duals)
        if collect_duals:
            lb, ub, used_duals = nb
        else:
            lb, ub = nb
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    used_bounds = (lb.copy(), ub.copy())   # recorded for exact reuse by children
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
            return (Star(V, newC, newd, S.predicate_lb, S.predicate_ub), [],
                    used_bounds, used_duals)
        return (Star(V, S.C, S.d, S.predicate_lb, S.predicate_ub), [],
                used_bounds, used_duals)

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

    return Star(newV, newC, newd, plb, pub), meta, used_bounds, used_duals


def relu_positions(layers: List[Layer]) -> List[int]:
    """Index into ``layers`` of each ``ReluLayer``, in order (relu_idx -> pos)."""
    return [i for i, l in enumerate(layers) if isinstance(l, ReluLayer)]


def _reach_from(
    start_star: Star,
    layers: List[Layer],
    fixed: Dict[NeuronKey, Phase],
    bound_mode: str,
    start_pos: int,
    start_relu_idx: int,
    start_zono=None,
    first_layer_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    gpu_iters: Optional[int] = None,
    parent_duals: Optional[list] = None,
    warm_rows: int = 0,
    collect_duals: bool = False,
) -> Tuple[Star, List[NeuronMeta], List[Star], list, list]:
    """
    Propagate ``start_star`` through ``layers[start_pos:]``, beginning at ReLU
    index ``start_relu_idx`` (the affine map preceding it is assumed already baked
    into ``start_star``).

    Returns ``(out_star, new_meta, new_checkpoints)`` where ``new_checkpoints[j]``
    is the pre-ReLU star entering ReLU index ``start_relu_idx + j`` -- the shared
    prefix a child splitting at that layer resumes from. This is the single reach
    core: ``relaxed_reach`` is the ``start_pos=0, start_relu_idx=0`` case and
    ``resume_reach`` restarts mid-network from a parent checkpoint.

    For ``bound_mode == "zono"`` an outer zonotope ``start_zono`` is propagated in
    lock-step (DeepZ ReLU overapprox) and its per-neuron pre-activation bounds
    refine the star's classification at each ReLU layer.
    """
    from n2v.nn.layer_ops.relu_reach import relu_zono_approx

    S = start_star
    Z = start_zono
    meta: List[NeuronMeta] = []
    checkpoints: List[Star] = []
    relu_bounds: list = []
    relu_duals: list = []
    relu_idx = start_relu_idx
    for layer in layers[start_pos:]:
        if isinstance(layer, LinearLayer):
            S = S.affine_map(layer.W, layer.b)
            if Z is not None:
                Z = Z.affine_map(layer.W, layer.b)
        elif isinstance(layer, ReluLayer):
            checkpoints.append(S)  # pre-ReLU star entering this layer
            zb = Z.get_bounds() if Z is not None else None
            pre = first_layer_bounds if relu_idx == start_relu_idx else None
            # Dual inheritance (resume only): warm-start this layer's descent
            # from the parent's duals on the shared constraint-row prefix (the
            # first ``warm_rows`` rows are identical between parent and child;
            # deeper rows differ and start at 0 -- sound for any values).
            warm = None
            if (parent_duals is not None and warm_rows > 0
                    and relu_idx >= start_relu_idx
                    and relu_idx < len(parent_duals)
                    and parent_duals[relu_idx]):
                warm = {d: (y[0][:warm_rows], y[1][:warm_rows])
                        for d, y in parent_duals[relu_idx].items()}
            S, layer_meta, used_bounds, used_duals = _relu_layer_relaxed(
                S, relu_idx, fixed, bound_mode, zono_bounds=zb,
                precomputed_bounds=pre, gpu_iters=gpu_iters,
                warm_duals=warm, collect_duals=collect_duals,
            )
            relu_bounds.append(used_bounds)
            relu_duals.append(used_duals)
            if Z is not None:
                Z = relu_zono_approx([Z])[0]
            meta.extend(layer_meta)
            relu_idx += 1
        else:
            raise TypeError(f"Unknown layer type {type(layer).__name__}")
    return S, meta, checkpoints, relu_bounds, relu_duals


def _attach_provenance(S, meta, fixed, bound_mode, checkpoints, relu_bounds=None,
                       relu_duals=None):
    """Attach the refine-search provenance fields to an output star (single
    source of truth for both ``relaxed_reach`` and ``resume_reach``, so the two
    paths can never drift)."""
    S.relax_meta = meta
    S.fixed = dict(fixed)
    S.bound_mode = bound_mode
    S.checkpoints = checkpoints
    S.relu_bounds = relu_bounds
    S.relu_duals = relu_duals
    return S


def relaxed_reach(
    input_star: Star,
    layers: List[Layer],
    fixed: Optional[Dict[NeuronKey, Phase]] = None,
    bound_mode: str = "box",
) -> Tuple[Star, List[NeuronMeta]]:
    """
    Forward-propagate ``input_star`` through ``layers`` with triangle ReLU
    relaxation and the given fixed neuron phases.

    ``bound_mode`` ("box" | "lp_cpu" | "lp_gpu" | "dual_gpu" | "zono") selects
    Phase-1 box bounds, LP-over-P bound tightening (exact CPU, PDHG+NS GPU, or
    best-iterate NS-dual-gradient GPU), or DeepZ outer-zonotope refinement (cheap
    correlation-aware bounds; best with input splitting, where the shrinking input
    box tightens the zonotope) for neuron classification.

    Returns the output star and the list of ``NeuronMeta`` for every relaxed
    (unfixed, unstable) neuron across all layers, in creation order. The output
    star also carries ``checkpoints`` (pre-ReLU star per ReLU index), enabling
    incremental shared-prefix reach for child nodes (see ``resume_reach``).
    """
    if fixed is None:
        fixed = {}

    start_zono = None
    if bound_mode in ("zono", "zono_lp"):
        from n2v.sets import Zono
        ilb, iub = input_star.estimate_ranges()
        start_zono = Zono.from_bounds(ilb.flatten(), iub.flatten())

    S, meta, checkpoints, relu_bounds, relu_duals = _reach_from(
        input_star, layers, fixed, bound_mode, 0, 0, start_zono=start_zono,
        collect_duals=(bound_mode == "dual_gpu"),
    )

    # Attach the relaxation metadata + search provenance to the output star so
    # the refine set-operations (refine/split) are self-describing. ``relax_meta``
    # is a declared Star field; ``fixed``/``bound_mode``/``checkpoints`` are
    # refine-search provenance. The tuple return is preserved for existing callers.
    if S is input_star:
        # No layer transformed the star (e.g. empty/Flatten-only ``layers``): wrap
        # a fresh Star so we attach provenance to it rather than mutating the
        # caller's input set (which is reused across calls).
        S = Star(S.V, S.C, S.d, S.predicate_lb, S.predicate_ub)
    _attach_provenance(S, meta, fixed, bound_mode, checkpoints, relu_bounds,
                       relu_duals)
    return S, meta


def resume_reach(
    parent: Star,
    layers: List[Layer],
    child_fixed: Dict[NeuronKey, Phase],
    bound_mode: str,
    split_layer: int,
) -> Star:
    """
    Incremental reach for a child that fixes one more neuron at ReLU index
    ``split_layer`` (``L``). Resumes from the parent's pre-ReLU checkpoint at
    ``L`` and reprocesses only layers ``>= L`` -- the prefix (affine maps and OBBT
    bound LPs of layers ``< L``) is reused unchanged.

    Correctness rests on the **shared-prefix precondition** (asserted below):
    ``child_fixed`` and the parent's ``fixed`` agree on every neuron fixed at a
    layer ``< L`` -- the bab ``split`` caller guarantees this (it only adds one
    fix, at ``L``). Then the reach up to entering ReLU ``L`` is identical, so the
    parent's checkpoint ``L`` is exactly the child's pre-ReLU star there; layers
    ``< L`` also add the same predicate variables in the same order, so the
    parent's layer-``<L`` meta (and ``pred_col`` indices) are valid verbatim.
    Reprocessing layers ``>= L`` with the *full* ``child_fixed`` re-applies any
    deeper fixes the parent already had, so non-monotonic split order is fine.
    Returns a star equal to ``relaxed_reach(input, layers, child_fixed,
    bound_mode)`` (see ``test_incremental_reach``); only provenance differs.

    Memory: prefix checkpoints are shared with the parent by reference, so the
    only new allocations are the layers actually recomputed. Live memory is
    bounded by the DFS frontier x depth (not the whole tree), strictly more than
    the old recompute-from-input path -- acceptable here, revisit if it spikes on
    deep/wide nets.
    """
    if parent.checkpoints is None:
        raise ValueError("parent star carries no checkpoints; run relaxed_reach first")
    n_relu = len(parent.checkpoints)
    if not (0 <= split_layer < n_relu):
        raise ValueError(f"split_layer {split_layer} out of range [0, {n_relu})")
    # Load-bearing precondition: layers < split_layer are reused from the parent,
    # so child_fixed must not introduce/contradict a fix below the split layer
    # (that fix would be silently dropped -> an unsound star).
    parent_fixed = parent.fixed or {}
    for k, ph in child_fixed.items():
        if k.layer < split_layer and parent_fixed.get(k) != ph:
            raise ValueError(
                f"resume_reach: child_fixed disagrees with parent below split "
                f"layer {split_layer} at {k}; shared-prefix reuse would be unsound"
            )

    start_pos = relu_positions(layers)[split_layer]
    start_star = parent.checkpoints[split_layer]

    # Exact bound reuse at the split layer: the child's entry star at L IS the
    # parent's checkpoint (identical object), and the layer-L bounds are
    # computed on the entry star BEFORE the phase row is added -- so the
    # parent's recorded bounds are the identical LP's result. This skips the
    # split layer's whole OBBT batch at zero approximation cost. Deeper layers
    # diverge (the phase row + changed relaxations alter the polytope) and are
    # recomputed -- at a reduced GPU iteration budget for dual_gpu (children
    # need marginal refresh, not from-scratch tightening; T1a: ~100 iters
    # recovers 93-99% of the gap).
    parent_rb = getattr(parent, "relu_bounds", None)
    first_bounds = None
    if parent_rb is not None and split_layer < len(parent_rb):
        first_bounds = parent_rb[split_layer]
    # Dual inheritance: the parent's duals index by ABSOLUTE relu_idx (its own
    # relu_duals list covers all layers); the shared row prefix between parent
    # and child at every layer > L is exactly the constraint rows of the
    # layer-L entry star (rows are only ever appended after that point).
    parent_duals = getattr(parent, "relu_duals", None)
    warm_rows = int(start_star.C.shape[0]) if parent_duals is not None else 0

    S, new_meta, new_ckpts, new_rb, new_rd = _reach_from(
        start_star, layers, child_fixed, bound_mode, start_pos, split_layer,
        first_layer_bounds=first_bounds,
        gpu_iters=RESUME_GPU_ITERS if bound_mode == "dual_gpu" else None,
        parent_duals=parent_duals, warm_rows=warm_rows,
        collect_duals=(bound_mode == "dual_gpu"),
    )
    prefix_meta = [m for m in (parent.relax_meta or []) if m.key.layer < split_layer]
    # Pad missing prefixes with None (NOT []): relu_bounds/relu_duals are
    # indexed by ABSOLUTE relu_idx everywhere, so a shortened list would make a
    # grandchild reuse a DEEPER layer's bounds as "exact" -- silently unsound.
    prefix_rb = (parent_rb[:split_layer] if parent_rb is not None
                 else [None] * split_layer)
    prefix_rd = (parent_duals[:split_layer] if parent_duals is not None
                 else [None] * split_layer)
    _attach_provenance(
        S, prefix_meta + new_meta, child_fixed, bound_mode,
        parent.checkpoints[:split_layer] + new_ckpts,
        prefix_rb + new_rb,
        prefix_rd + new_rd,
    )
    return S


def resume_reach_pair(
    parent: Star,
    layers: List[Layer],
    fixed_pair: List[Dict[NeuronKey, Phase]],
    bound_mode: str,
    split_layer: int,
) -> List[Star]:
    """Lockstep incremental reach for BOTH children of a split.

    ``dual_gpu`` fast path: the two children walk the layers together and each
    ReLU layer's OBBT batches merge into ONE population GPU call
    (``neuron_bounds_population``), halving kernel launches; classification and
    star construction stay per-child. The split layer itself reuses the
    parent's recorded bounds for both children (exact -- identical entry star),
    and deeper layers warm-start from the parent's duals on the shared row
    prefix. Bounds are NS-outward either way, so this is sound and
    verdict-equivalent to two ``resume_reach`` calls (values may differ within
    solver tolerance -- first-order bounds are batch-composition-sensitive).
    Non-``dual_gpu`` modes simply delegate to two ``resume_reach`` calls.
    """
    if bound_mode != "dual_gpu":
        return [resume_reach(parent, layers, f, bound_mode, split_layer)
                for f in fixed_pair]
    if parent.checkpoints is None:
        raise ValueError("parent star carries no checkpoints; run relaxed_reach first")
    n_relu = len(parent.checkpoints)
    if not (0 <= split_layer < n_relu):
        raise ValueError(f"split_layer {split_layer} out of range [0, {n_relu})")
    parent_fixed = parent.fixed or {}
    for child_fixed in fixed_pair:
        for kk, ph in child_fixed.items():
            if kk.layer < split_layer and parent_fixed.get(kk) != ph:
                raise ValueError(
                    f"resume_reach_pair: child_fixed disagrees with parent below "
                    f"split layer {split_layer} at {kk}")

    from n2v.refine.tighten import neuron_bounds_population

    start_pos = relu_positions(layers)[split_layer]
    start_star = parent.checkpoints[split_layer]
    parent_rb = getattr(parent, "relu_bounds", None)
    first_bounds = None
    if parent_rb is not None and split_layer < len(parent_rb):
        first_bounds = parent_rb[split_layer]
    parent_duals = getattr(parent, "relu_duals", None)
    warm_rows = int(start_star.C.shape[0]) if parent_duals is not None else 0

    states = [dict(S=start_star, meta=[], ckpts=[], rb=[], rd=[])
              for _ in fixed_pair]
    relu_idx = split_layer
    for layer in layers[start_pos:]:
        if isinstance(layer, LinearLayer):
            for st in states:
                st["S"] = st["S"].affine_map(layer.W, layer.b)
        elif isinstance(layer, ReluLayer):
            for st in states:
                st["ckpts"].append(st["S"])
            if relu_idx == split_layer and first_bounds is not None:
                pre_pair = [first_bounds] * len(states)
                duals_pair = [None] * len(states)
            else:
                warm_dict = None
                if (parent_duals is not None and warm_rows > 0
                        and relu_idx < len(parent_duals) and parent_duals[relu_idx]):
                    warm_dict = {d: (y[0][:warm_rows], y[1][:warm_rows])
                                 for d, y in parent_duals[relu_idx].items()}
                res = neuron_bounds_population(
                    [st["S"] for st in states], gpu_iters=RESUME_GPU_ITERS,
                    warm=[warm_dict] * len(states),
                )
                pre_pair = [(r[0], r[1]) for r in res]
                duals_pair = [r[2] for r in res]
            for st, pre, du, child_fixed in zip(states, pre_pair, duals_pair,
                                                fixed_pair):
                S2, lm, ub_used, _ = _relu_layer_relaxed(
                    st["S"], relu_idx, child_fixed, bound_mode,
                    precomputed_bounds=pre,
                )
                st["S"] = S2
                st["meta"].extend(lm)
                st["rb"].append(ub_used)
                st["rd"].append(du)
            relu_idx += 1
        else:
            raise TypeError(f"Unknown layer type {type(layer).__name__}")

    prefix_rb = (parent_rb[:split_layer] if parent_rb is not None
                 else [None] * split_layer)
    prefix_rd = (parent_duals[:split_layer] if parent_duals is not None
                 else [None] * split_layer)
    out = []
    for st, child_fixed in zip(states, fixed_pair):
        prefix_meta = [m for m in (parent.relax_meta or [])
                       if m.key.layer < split_layer]
        S = _attach_provenance(
            st["S"], prefix_meta + st["meta"], child_fixed, bound_mode,
            parent.checkpoints[:split_layer] + st["ckpts"],
            prefix_rb + st["rb"], prefix_rd + st["rd"],
        )
        out.append(S)
    return out
