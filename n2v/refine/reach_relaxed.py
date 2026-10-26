"""
Triangle-relaxation forward reach with fixed neuron phases and per-neuron
metadata.

This is a deliberately self-contained reach for fully-connected ReLU networks.
It is *not* a replacement for ``n2v.nn.reach`` -- it exists so the refinement
loop has full control over (a) forcing individual neurons ACTIVE/INACTIVE and
(b) recording, at the moment of relaxation, the metadata needed to compute the
per-neuron relaxation infidelity ``epsilon_j`` against a witness.

Soundness: an unfixed neuron is classified by an outward enclosure of its
pre-activation range, computed by the ``bound_mode``:

  * ``"box"``: ``Star.estimate_ranges`` (the predicate-box interval, which
    over-approximates the true range, ignoring C/d);
  * ``"lp_cpu"``: exact LP bounds over the predicate polytope (C/d rows and
    box), padded outward by the solver tolerance (``n2v.refine.tighten``);
  * ``"dual_gpu"``: NS-certified gradient-dual LP bounds over the predicate
    polytope (outward at any solver accuracy; CPU torch without CUDA).

``u <= 0`` => provably inactive (zeroed); ``l >= 0`` => provably active
(identity); otherwise the standard triangle over-approximation is applied. All
three cases are sound, so the output star is a sound over-approximation of the
fixed-phase reachable set.
"""

from __future__ import annotations

import copy
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
# (~100 iterations recover 93-99% of the box->exact gap). Root reaches use
# the tighten.py default (300).
RESUME_GPU_ITERS = 150

# Neuron-range modes the relaxed reach supports.
BOUND_MODES = ("box", "lp_cpu", "dual_gpu")


@dataclass
class ReluLayer:
    pass


def _bias_free(mod):
    """A copy of an affine image module with its additive term removed:
    ``lin(x) = mod(x) - mod(0)`` exactly, so generator columns never pass
    through a bias add-then-subtract (which rounds away up to ulp(|bias|)
    per entry)."""
    import torch
    import torch.nn as nn

    lin = copy.deepcopy(mod)
    if isinstance(lin, (nn.Conv2d, nn.ConvTranspose2d)):
        lin.bias = None
    elif isinstance(lin, nn.BatchNorm2d):
        if lin.running_mean is None:
            raise TypeError("BatchNorm2d(track_running_stats=False) is not "
                            "an affine map in eval mode")
        with torch.no_grad():
            lin.running_mean.zero_()
            if lin.bias is not None:
                lin.bias.zero_()
    # nn.AvgPool2d has no additive term
    return lin


@dataclass
class ConvLayer:
    """A convolutional (or other spatially-structured affine) layer, applied
    to the star's basis NATIVELY instead of via a materialized dense matrix.

    The star's predicate polytope (C, d, box) is untouched -- only the basis
    application differs (the ImageStar insight, Tran et al.: an affine map of
    a star is the map of its anchor and generator columns; for a convolution
    that is one batched conv over the columns, bias applied to the anchor
    only). The generator columns go through ``linear_module``, a bias-free
    copy of the module, so a bias is never added to them and subtracted
    again. Mathematically identical to LinearLayer with the conv's dense
    matrix; the dense matrix is simply never built.

    An ``nn.AvgPool2d`` whose border divisor torch and ONNX disagree on
    (``ceil_mode=True`` and ``count_include_pad=True`` with a window that
    overhangs the padded input) raises ``NotImplementedError`` at
    construction, the same refusal as the core reach
    (``n2v.nn.layer_ops.avgpool2d_reach._check_ceil_mode_divisor``).
    """
    module: object      # torch.nn Conv2d/AvgPool2d (float64, eval mode)
    in_shape: tuple     # (C, H, W) consumed by this layer
    out_shape: tuple    # shape produced (before flattening)
    linear_module: object = None  # bias-free copy of ``module`` (generators)

    def __post_init__(self):
        import torch
        import torch.nn as nn

        if isinstance(self.module, nn.AvgPool2d):
            # SOUNDNESS: with an overhanging last window, torch divides by the
            # window's part inside the padded input and ONNX AveragePool by
            # the full kernel area; the layer cannot tell which is meant, so
            # a verdict on either could be false for the other. The output
            # size comes from the module itself, not from ``out_shape``.
            from n2v.nn.layer_ops.avgpool2d_reach import \
                _check_ceil_mode_divisor
            with torch.no_grad():
                out = self.module(torch.zeros(1, *self.in_shape,
                                              dtype=torch.float64))
            _check_ceil_mode_divisor(self.module, self.in_shape[-2:],
                                     out.shape[-2:])
        if self.linear_module is None:
            self.linear_module = _bias_free(self.module)

    def apply(self, S: Star) -> Star:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # NEVER .to() the shared modules (in-place device move would break the
        # CPU exact-forward path that validates counterexamples): cache device
        # copies on first use instead.
        mod = getattr(self, "_dev_module", None)
        lin = getattr(self, "_dev_linear", None)
        if mod is None or lin is None:
            if dev != "cpu":
                mod = copy.deepcopy(self.module).to(dev)
                lin = copy.deepcopy(self.linear_module).to(dev)
            else:
                mod, lin = self.module, self.linear_module
            self._dev_module, self._dev_linear = mod, lin
        cols = torch.tensor(np.asarray(S.V, dtype=np.float64).T.copy(),
                            dtype=torch.float64, device=dev)
        cols = cols.reshape(cols.shape[0], *self.in_shape)
        with torch.no_grad():
            # the anchor keeps the additive term (the module's own bias add);
            # generator columns are mapped by the linear part only
            anchor = mod(cols[:1]).reshape(1, -1)
            gens = ([lin(c).reshape(c.shape[0], -1)
                     for c in torch.split(cols[1:], 1024)]
                    if cols.shape[0] > 1 else [])
            out = torch.cat([anchor] + gens)
        out = out.cpu().numpy().T
        return Star(out, S.C, S.d, S.predicate_lb, S.predicate_ub)


Layer = object  # LinearLayer | ReluLayer | ConvLayer


def apply_affine(S: Star, layer) -> Star:
    """Apply an affine layer to a star -- dense or conv, same mathematics."""
    if isinstance(layer, ConvLayer):
        return layer.apply(S)
    return S.affine_map(layer.W, layer.b)


def extract_layers(model, in_shape: Optional[tuple] = None) -> List[Layer]:
    """
    Flatten an ``nn.Sequential`` chain into an ordered layer list.

    Accepted models: an ``nn.Sequential`` (nested ``nn.Sequential`` allowed;
    a subclass only if it keeps ``nn.Sequential.forward``) of stock
    ``Linear``, ``ReLU``, ``Flatten``, ``Unflatten``, ``Dropout``,
    ``Conv2d``, ``ConvTranspose2d``, ``AvgPool2d`` and ``BatchNorm2d``
    modules, or one such module on its own. A subclass of these that
    overrides ``forward()`` is rejected, and so is any module with an
    instance-level ``forward`` or a forward (pre-)hook, and any registered
    global module forward hook: the walk reads ``.children()`` and cannot see
    what these compute, so the layer list would describe a different function
    from ``model(x)``, the function the counterexample checks run. The layers
    are the eval-mode function (``Dropout`` is skipped as the identity).

    ``Flatten`` must use its default dims (``start_dim=1, end_dim=-1``; it is
    skipped, since the star's basis is flat throughout), and ``Unflatten``
    must unflatten dim 1 of a flat ``(N, n)`` activation. ``BatchNorm2d``
    must track running statistics (without them it normalizes with the
    batch's own statistics even in eval mode, which is not an affine map).
    Conv-type modules need a (C, H, W) shape, from ``in_shape`` or a
    preceding ``nn.Unflatten``; each becomes a :class:`ConvLayer` whose affine
    map is applied natively to the basis columns (no dense matrix is
    materialized), built from a float64 eval-mode deep copy, so the caller's
    module keeps its dtype and training flag.

    Conv nets are supported in two ways: (a) a flat-input ``nn.Sequential``
    that reshapes with ``nn.Unflatten`` before its first conv; or (b)
    ``extract_layers(inner_seq, in_shape=(C, H, W))`` on the image-input
    ``nn.Sequential`` that a flat-input wrapper calls, passed to
    ``verify_refine`` as ``layers=`` together with the wrapper as ``model``.
    The wrapper itself is rejected here: its own ``forward()`` reshapes, and
    the walk cannot see that.

    Raises:
        TypeError: for any model or module outside the contract above.
        NotImplementedError: for an ``AvgPool2d`` whose border divisor torch
            and ONNX disagree on (see :class:`ConvLayer`), as in the core
            reach.
    """
    import torch
    import torch.nn as nn

    leaves = (nn.Linear, nn.ReLU, nn.Flatten, nn.Unflatten, nn.Dropout,
              nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.BatchNorm2d)

    def _stock_sequential(m):
        # nn.Sequential's own forward() applies its children in order; a
        # subclass that overrides forward() may not
        return (isinstance(m, nn.Sequential)
                and type(m).forward is nn.Sequential.forward)

    def _stock_leaf(m):
        for cls in leaves:
            if isinstance(m, cls):
                return type(m).forward is cls.forward
        return False

    def _flat_children(mod):
        for c in mod.children():
            if _stock_sequential(c):
                yield from _flat_children(c)
            else:
                yield c

    # SOUNDNESS: layers are read from .children(), which cannot see what a
    # custom forward() does (functional activations, skip connections,
    # reordering). Only accept models whose forward() IS the child chain.
    if _stock_sequential(model):
        children = list(_flat_children(model))
    elif not list(model.children()):
        children = [model]
    else:
        raise TypeError(
            f"extract_layers requires an nn.Sequential chain; "
            f"{type(model).__name__} defines its own forward(), which may "
            f"apply functional ops or skip connections that a walk over "
            f".children() cannot see (pass layers= explicitly)")

    # SOUNDNESS: the class checks above cannot see a forward() assigned on
    # the instance (an instance attribute wins over the class method) or a
    # registered forward / forward-pre hook, and either changes what
    # model(x) computes -- the function the counterexample check runs.
    from torch.nn.modules import module as _torch_module
    if (_torch_module._global_forward_hooks
            or _torch_module._global_forward_pre_hooks):
        raise TypeError(
            "extract_layers: global module forward hooks are registered; "
            "they change every module's forward() and the layer walk cannot "
            "see them (remove them, or pass layers= explicitly)")
    for mod in model.modules():
        if ("forward" in mod.__dict__ or mod._forward_hooks
                or mod._forward_pre_hooks):
            raise TypeError(
                f"extract_layers: {type(mod).__name__} has an instance-level "
                f"forward() or a forward hook, which the layer walk cannot "
                f"see (pass layers= explicitly)")

    layers: List[Layer] = []
    shape = tuple(in_shape) if in_shape is not None else None
    for m in children:
        if not _stock_leaf(m):
            raise TypeError(
                f"extract_layers supports only Linear/ReLU/Flatten/Unflatten/"
                f"Dropout/Conv2d/ConvTranspose2d/AvgPool2d/BatchNorm2d, got "
                f"{type(m).__name__} (or a subclass that overrides forward())")
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().astype(np.float64)
            b = (
                m.bias.detach().cpu().numpy().astype(np.float64)
                if m.bias is not None
                else np.zeros(W.shape[0], dtype=np.float64)
            )
            layers.append(LinearLayer(W, b))
            shape = (W.shape[0],)
        elif isinstance(m, nn.ReLU):
            layers.append(ReluLayer())
        elif isinstance(m, nn.Flatten):
            # the star basis is flat throughout, so only the default
            # (N, ...) -> (N, prod) flatten is the identity on it
            if m.start_dim != 1 or m.end_dim != -1:
                raise TypeError("extract_layers: only nn.Flatten(start_dim=1, "
                                "end_dim=-1) is supported")
            if shape is not None:
                shape = (int(np.prod(shape)),)
            continue
        elif isinstance(m, nn.Unflatten):
            # mid-network reshape into image form (e.g. Linear stem -> convs):
            # the flat basis is unchanged; only the conv shape bookkeeping
            # moves. It must reshape the flat feature vector.
            if m.dim not in (1, -1) or (shape is not None and len(shape) != 1):
                raise TypeError("extract_layers: nn.Unflatten must unflatten "
                                "dim 1 of a flat (N, n) activation")
            shape = tuple(int(v) for v in m.unflattened_size)
            continue
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.BatchNorm2d)):
            # BatchNorm2d in eval mode is affine (per-channel scale+shift):
            # same basis-application treatment as Conv2d. Without running
            # statistics it normalizes with the batch's own statistics even
            # in eval mode: not an affine map.
            if isinstance(m, nn.BatchNorm2d) and m.running_mean is None:
                raise TypeError(
                    "extract_layers: BatchNorm2d(track_running_stats=False) "
                    "is not affine in eval mode")
            if shape is None or len(shape) != 3:
                raise TypeError(
                    "extract_layers: Conv2d/ConvTranspose2d/AvgPool2d/"
                    "BatchNorm2d needs a (C, H, W) shape: pass in_shape for an "
                    "image-input model, or reshape a flat input with "
                    "nn.Unflatten before the first conv")
            # a float64 eval-mode COPY: the caller's module keeps its dtype and
            # training flag
            mod = copy.deepcopy(m).double().eval()
            with torch.no_grad():
                out = mod(torch.zeros(1, *shape, dtype=torch.float64))
            layers.append(ConvLayer(mod, shape, tuple(out.shape[1:])))
            shape = tuple(out.shape[1:])
        elif isinstance(m, nn.Dropout):
            continue  # identity in eval mode
    return layers


def _relu_layer_relaxed(
    S: Star,
    layer_idx: int,
    fixed: Dict[NeuronKey, Phase],
    bound_mode: str = "box",
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
    ``"box"`` (predicate-box interval, cheap); ``"lp_cpu"`` (exact LP over the
    predicate polytope P on every neuron, padded outward); or ``"dual_gpu"``
    (best-iterate gradient descent on the NS dual over P, NS-certified outward,
    on the box-unstable neurons only). The LP modes are tighter-or-equal to the
    box and stabilise more neurons. All modes are sound. ``precomputed_bounds``,
    when given, replace the computation for this entry star.
    """
    if bound_mode not in BOUND_MODES:
        raise ValueError(f"unknown bound_mode {bound_mode!r} "
                         f"(expected one of {BOUND_MODES})")
    used_duals: Optional[dict] = None
    if precomputed_bounds is not None:
        # Caller-supplied bounds for THIS entry star (resume_reach at the split
        # layer: the child's entry star IS the parent's checkpoint, so the
        # parent's bounds are the identical LP's result -- exact reuse, the
        # whole OBBT batch of the split layer skipped).
        lb, ub = precomputed_bounds
    elif bound_mode == "box":
        lb, ub = S.estimate_ranges()
    else:
        # lp_cpu / dual_gpu: LP bounds over P. Under dual_gpu, neuron_bounds
        # solves only the box-unstable dims (result-identical per its
        # contract; the GPU batch is one tensor op either way).
        nb = neuron_bounds(S, backend=bound_mode,
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
    # for any appended variables below. Fixed neurons are few (<= search depth),
    # so they keep a per-neuron loop; the unfixed classification is vectorized.
    phase_rows: List[np.ndarray] = []
    phase_d: List[float] = []

    fixed_here = {k.neuron: ph for k, ph in fixed.items() if k.layer == layer_idx}
    if fixed_here and not all(0 <= i < dim for i in fixed_here):
        # Neuron indices are used directly as row indices below: an index
        # >= dim would raise deep inside, and a NEGATIVE one would silently
        # wrap (numpy) and pin the WRONG neuron's phase -- an unsound
        # constraint. All in-tree producers derive keys from relax_meta (always
        # in range); this guards hand-built dicts from external callers.
        bad = sorted(i for i in fixed_here if not 0 <= i < dim)
        raise ValueError(
            f"forced-phase neuron index out of range for layer {layer_idx} "
            f"(dim {dim}): {bad}")
    for i in sorted(fixed_here):
        center_i = float(Vpre[i, 0])
        gens_i = Vpre[i, 1 : nVar + 1].astype(np.float64)
        if fixed_here[i] == Phase.INACTIVE:
            V[i, :] = 0.0
            # x_hat_i <= 0  <=>  gens_i . alpha <= -center_i
            phase_rows.append(gens_i.copy())
            phase_d.append(-center_i)
        else:
            # ACTIVE: identity pass-through, restrict to x_hat_i >= 0
            # -gens_i . alpha <= center_i
            phase_rows.append(-gens_i)
            phase_d.append(center_i)

    # SOUNDNESS: every classification below is a comparison, and NaN compares
    # False against everything -- so a NaN bound is neither "inactive"
    # (ub <= 0) nor "crossing" (l < 0 < u) and falls through as *stably
    # active*: identity pass-through with no phase constraint, which
    # under-approximates the true ReLU. Fail loudly instead; a NaN here means a
    # non-finite multiplier or predicate bound reached the certificate, which
    # is a bug in the caller, not a value to reason from.
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        bad = np.flatnonzero(~(np.isfinite(lb) & np.isfinite(ub)))
        raise ValueError(
            f"non-finite pre-activation bounds at ReLU layer {layer_idx} for "
            f"neuron indices {bad[:8].tolist()}"
            f"{'...' if bad.size > 8 else ''} (bound_mode={bound_mode!r}); "
            f"refusing to classify -- NaN/inf would silently read as "
            f"'stably active'")

    unfixed = np.ones(dim, dtype=bool)
    if fixed_here:
        unfixed[list(fixed_here)] = False
    inact = unfixed & (ub <= 0.0)
    if inact.any():
        V[inact, :] = 0.0          # provably inactive (sound, no constraint needed)
    cross_mask = unfixed & (lb < 0.0) & (ub > 0.0)
    crossing = [(int(i), lb[i], ub[i]) for i in np.flatnonzero(cross_mask)]

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

    # Vectorized triangle construction over all crossing neurons at once
    # (row semantics and ORDER identical to the former per-neuron loop:
    # interleaved [(ii); (iii)] pairs in neuron order).
    ci = np.fromiter((i for i, _, _ in crossing), dtype=int, count=m)
    cli = np.fromiter((li for _, li, _ in crossing), dtype=np.float64, count=m)
    cui = np.fromiter((ui for _, _, ui in crossing), dtype=np.float64, count=m)
    Gc = Vpre[ci, 1 : nVar + 1].astype(np.float64)     # (m, nVar) preact gens
    cc = Vpre[ci, 0].astype(np.float64)                # preact centers
    lam = cui / (cui - cli)
    ar = np.arange(m)

    # neuron outputs become the fresh variables
    newV[ci, :] = 0.0
    newV[ci, nVar + 1 + ar] = 1.0

    meta: List[NeuronMeta] = [
        NeuronMeta(
            key=NeuronKey(layer_idx, int(ci[t])),
            pred_col=nVar + t,
            preact_center=float(cc[t]),
            preact_gens=Gc[t].copy(),
            l=float(cli[t]),
            u=float(cui[t]),
        )
        for t in range(m)
    ]

    # (ii) alpha_new >= x_hat_i  <=>  gens_i . alpha + center_i - alpha_new <= 0
    A_ii = np.zeros((m, new_nVar), dtype=np.float64)
    A_ii[:, :nVar] = Gc
    A_ii[ar, nVar + ar] = -1.0
    # (iii) alpha_new <= lam (x_hat_i - l)
    #   <=>  -lam gens_i . alpha + alpha_new <= lam (center_i - l)
    A_iii = np.zeros((m, new_nVar), dtype=np.float64)
    A_iii[:, :nVar] = -lam[:, None] * Gc
    A_iii[ar, nVar + ar] = 1.0
    extra_rows = np.empty((2 * m, new_nVar), dtype=np.float64)
    extra_rows[0::2] = A_ii
    extra_rows[1::2] = A_iii
    extra_d = np.empty(2 * m, dtype=np.float64)
    extra_d[0::2] = -cc
    extra_d[1::2] = lam * (cc - cli)

    # (i) alpha_new >= 0 is enforced by the predicate box below.
    new_lb = np.zeros(m, dtype=np.float64)
    new_ub = cui

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
    blocks_C.append(extra_rows)
    blocks_d.append(extra_d.reshape(-1, 1))
    newC = np.vstack(blocks_C)
    newd = np.vstack(blocks_d)

    plb = (
        np.vstack([S.predicate_lb, new_lb.reshape(-1, 1)])
        if S.predicate_lb is not None
        else None
    )
    pub = (
        np.vstack([S.predicate_ub, new_ub.reshape(-1, 1)])
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
    """
    S = start_star
    meta: List[NeuronMeta] = []
    checkpoints: List[Star] = []
    relu_bounds: list = []
    relu_duals: list = []
    relu_idx = start_relu_idx
    for layer in layers[start_pos:]:
        if isinstance(layer, (LinearLayer, ConvLayer)):
            S = apply_affine(S, layer)
        elif isinstance(layer, ReluLayer):
            checkpoints.append(S)  # pre-ReLU star entering this layer
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
                S, relu_idx, fixed, bound_mode,
                precomputed_bounds=pre, gpu_iters=gpu_iters,
                warm_duals=warm, collect_duals=collect_duals,
            )
            relu_bounds.append(used_bounds)
            relu_duals.append(used_duals)
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
    *,
    gpu_iters: Optional[int] = None,
) -> Tuple[Star, List[NeuronMeta]]:
    """
    Forward-propagate ``input_star`` through ``layers`` with triangle ReLU
    relaxation and the given fixed neuron phases.

    ``bound_mode`` ("box" | "lp_cpu" | "dual_gpu") selects predicate-box
    interval bounds or LP-over-P bound tightening (exact CPU LP, or
    best-iterate NS-dual-gradient bounds on the GPU) for neuron classification.

    Returns the output star and the list of ``NeuronMeta`` for every relaxed
    (unfixed, unstable) neuron across all layers, in creation order. The output
    star also carries ``checkpoints`` (pre-ReLU star per ReLU index), enabling
    incremental shared-prefix reach for child nodes (see ``resume_reach``).

    ``gpu_iters`` (dual_gpu only): per-bound dual-iteration budget for
    this reach; ``None`` keeps the tighten default (300). Any budget is sound
    (every NS-certified iterate is a valid bound) -- it only trades tightness
    for time.
    """
    if fixed is None:
        fixed = {}

    S, meta, checkpoints, relu_bounds, relu_duals = _reach_from(
        input_star, layers, fixed, bound_mode, 0, 0,
        gpu_iters=gpu_iters,
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


def _check_shared_prefix(parent_fixed, child_fixed, split_layer, who):
    """Layers below ``split_layer`` are reused from the parent, so the parent
    and child fixed sets must agree EXACTLY there: a child fix the parent
    lacks would be silently dropped, and a parent fix the child lacks would
    silently restrict the child (its phase row lives in the reused
    checkpoint)."""
    def below(f):
        return {k: p for k, p in f.items() if k.layer < split_layer}
    if below(parent_fixed) != below(child_fixed):
        raise ValueError(
            f"{who}: child_fixed disagrees with parent below split layer "
            f"{split_layer}; shared-prefix reuse would be unsound")


def resume_reach(
    parent: Star,
    layers: List[Layer],
    child_fixed: Dict[NeuronKey, Phase],
    bound_mode: str,
    split_layer: int,
    *,
    gpu_iters: Optional[int] = None,
    warm_starts: bool = True,
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
    # so child_fixed must agree with the parent's fixes there exactly.
    _check_shared_prefix(parent.fixed or {}, child_fixed, split_layer,
                         "resume_reach")

    start_pos = relu_positions(layers)[split_layer]
    start_star = parent.checkpoints[split_layer]

    # Exact bound reuse at the split layer: the child's entry star at L IS the
    # parent's checkpoint (identical object), and the layer-L bounds are
    # computed on the entry star BEFORE the phase row is added -- so the
    # parent's recorded bounds are the identical LP's result. This skips the
    # split layer's whole OBBT batch at zero approximation cost. Deeper layers
    # diverge (the phase row + changed relaxations alter the polytope) and are
    # recomputed -- at a reduced GPU iteration budget for dual_gpu (children
    # need marginal refresh, not from-scratch tightening; ~100 iterations
    # recover 93-99% of the gap).
    parent_rb = getattr(parent, "relu_bounds", None)
    first_bounds = None
    if parent_rb is not None and split_layer < len(parent_rb):
        first_bounds = parent_rb[split_layer]
    # Dual inheritance: the parent's duals index by ABSOLUTE relu_idx (its own
    # relu_duals list covers all layers); the shared row prefix between parent
    # and child at every layer > L is exactly the constraint rows of the
    # layer-L entry star (rows are only ever appended after that point).
    # ``warm_starts=False`` (ablation knob): drop dual inheritance only --
    # checkpoint/prefix reuse and the split layer's exact bound reuse stay on
    # (those are the ``incremental`` knob's job). Sound either way: warm starts
    # change the ascent's starting point, never the certificate.
    parent_duals = getattr(parent, "relu_duals", None) if warm_starts else None
    warm_rows = int(start_star.C.shape[0]) if parent_duals is not None else 0

    S, new_meta, new_ckpts, new_rb, new_rd = _reach_from(
        start_star, layers, child_fixed, bound_mode, start_pos, split_layer,
        first_layer_bounds=first_bounds,
        gpu_iters=((gpu_iters if gpu_iters is not None else RESUME_GPU_ITERS)
                   if bound_mode == "dual_gpu" else None),
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
    *,
    gpu_iters: Optional[int] = None,
    warm_starts: bool = True,
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
        return [resume_reach(parent, layers, f, bound_mode, split_layer,
                             gpu_iters=gpu_iters, warm_starts=warm_starts)
                for f in fixed_pair]
    if parent.checkpoints is None:
        raise ValueError("parent star carries no checkpoints; run relaxed_reach first")
    n_relu = len(parent.checkpoints)
    if not (0 <= split_layer < n_relu):
        raise ValueError(f"split_layer {split_layer} out of range [0, {n_relu})")
    for child_fixed in fixed_pair:
        _check_shared_prefix(parent.fixed or {}, child_fixed, split_layer,
                             "resume_reach_pair")

    from n2v.refine.tighten import neuron_bounds_population

    start_pos = relu_positions(layers)[split_layer]
    start_star = parent.checkpoints[split_layer]
    parent_rb = getattr(parent, "relu_bounds", None)
    first_bounds = None
    if parent_rb is not None and split_layer < len(parent_rb):
        first_bounds = parent_rb[split_layer]
    parent_duals = getattr(parent, "relu_duals", None) if warm_starts else None
    warm_rows = int(start_star.C.shape[0]) if parent_duals is not None else 0

    states = [dict(S=start_star, meta=[], ckpts=[], rb=[], rd=[])
              for _ in fixed_pair]
    relu_idx = split_layer
    for layer in layers[start_pos:]:
        if isinstance(layer, (LinearLayer, ConvLayer)):
            for st in states:
                st["S"] = apply_affine(st["S"], layer)
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
                    [st["S"] for st in states],
                    gpu_iters=(gpu_iters if gpu_iters is not None
                               else RESUME_GPU_ITERS),
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
