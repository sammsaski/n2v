"""Batched CROWN bounds for FC-ReLU layer lists (the low-dim input-split engine).

Computes, for a BATCH of input boxes at once, sound linear lower/upper bounding
functions of every layer's pre-activations and of spec-row margins, by the
standard CROWN backward substitution (adaptive lower slope, chord upper slope).
This is the "cheap bounds in bulk" primitive the low-input-dimension regime
rewards: one batched pass bounds thousands of input subdomains in milliseconds,
where LP-grade bound refresh (exact or GPU-dual) costs that much per SINGLE
subdomain. It deliberately trades per-subdomain tightness for volume — the
input-split BaB recovers tightness by splitting, which in low dimension is
cheap (each bisection tightens every neuron at once).

Spec margins use LAST-LAYER ELISION: the final backward pass is seeded with the
spec matrix ``G`` itself rather than projecting identity output forms through
``G`` afterwards — provably never looser (sign cancellation across output
neurons is kept) and strictly tighter in practice.

Soundness: standard CROWN inequalities evaluated in float64 round-to-nearest —
the same arithmetic contract as the existing zono / IBP-precompute intersection
modes in this package (not bit-level rigorous like the NS-certified LP path;
per-op error ~1e-16 relative on 50-wide nets). Verdicts additionally pass
through the engine's exact CE checks (SAT) and, where used, exact LP prunes.
The ReLU relaxation coefficients mirror ``n2v/nn/crown_reach.py`` (the op-DAG
CROWN); the tie at ``u == -l`` picks slope 1 here vs 0 there — both sound,
equal-area choices.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from n2v.refine.reach_relaxed import Layer, LinearLayer, ReluLayer


def prepare_layers(layers: List[Layer], device: str = "cpu") -> list:
    """Convert a LinearLayer/ReluLayer list to device tensors ONCE per
    verification call (CrownBatch runs O(L^2) backward passes per wave; per-wave
    conversion would repeat this work, and on CUDA each conversion is a blocking
    H2D upload). Also validates the structural precondition that every ReLU
    directly follows a Linear (CrownBatch reads the preceding layer's width).
    """
    prepared: list = []
    dt = torch.float64
    for idx, layer in enumerate(layers):
        if isinstance(layer, LinearLayer):
            prepared.append((
                torch.as_tensor(layer.W, dtype=dt, device=device),
                torch.as_tensor(layer.b, dtype=dt, device=device),
            ))
        elif isinstance(layer, ReluLayer):
            if idx == 0 or not isinstance(layers[idx - 1], LinearLayer):
                raise ValueError(
                    "CrownBatch requires every ReLU to follow a Linear layer "
                    f"(ReLU at layer index {idx})")
            prepared.append(None)
        else:
            raise TypeError(f"Unknown layer type {type(layer).__name__}")
    return prepared


def _relu_relax(l: torch.Tensor, u: torch.Tensor):
    """Per-neuron CROWN ReLU relaxation coefficients over pre-act bounds [l, u].

    Returns ``(sl, su, bu)``: sound bounds ``sl*x <= relu(x) <= su*x + bu`` on
    ``[l, u]`` — lower slope adaptive (1 if u >= -l else 0, exact for stable
    neurons), upper the chord ``u/(u-l)`` with intercept ``-u*l/(u-l)``.
    """
    pos = l >= 0
    neg = u <= 0
    cross = ~(pos | neg)
    sl = torch.where(pos, torch.ones_like(l),
                     torch.where(neg, torch.zeros_like(l),
                                 (u >= -l).to(l.dtype)))
    denom = torch.where(cross, u - l, torch.ones_like(l))
    su = torch.where(pos, torch.ones_like(l),
                     torch.where(neg, torch.zeros_like(l), u / denom))
    bu = torch.where(cross, -u * l / denom, torch.zeros_like(l))
    return sl, su, bu


def _backward(prepared: list, upto: int,
              relax: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
              C: torch.Tensor):
    """Backward-substitute the form ``C`` from layer index ``upto`` (exclusive)
    to the input, through the relaxations recorded in ``relax``.

    ``C`` is ``(B, rows, n_upto)``. Returns lower and upper linear forms in the
    INPUT: ``(Al, bl, Au, bu)`` with ``Al/Au (B, rows, n_in)``, ``bl/bu
    (B, rows)``, such that for every x in the box,
    ``Al x + bl <= C * z_upto(x) <= Au x + bu``.
    """
    B, rows, _ = C.shape
    Al = C
    Au = C
    bl = C.new_zeros((B, rows))
    bu = C.new_zeros((B, rows))
    zero = C.new_zeros(())
    for idx in range(upto - 1, -1, -1):
        entry = prepared[idx]
        if entry is not None:                          # Linear layer
            W, b = entry
            bl = bl + Al @ b
            bu = bu + Au @ b
            Al = Al @ W
            Au = Au @ W
        else:                                          # ReLU layer
            sl, su, sb = relax[idx]                    # (B, n) each
            sl_ = sl.unsqueeze(1)                      # (B, 1, n)
            su_ = su.unsqueeze(1)
            sb_ = sb.unsqueeze(1)
            # lower form: positive coefficients take the LOWER relu bound
            # (sl*x), negative take the UPPER (su*x + sb); mirrored above.
            # Fused (single where/mul) — bit-identical to the clamp split.
            bl = bl + (torch.minimum(Al, zero) * sb_).sum(-1)
            bu = bu + (torch.maximum(Au, zero) * sb_).sum(-1)
            Al = Al * torch.where(Al >= 0, sl_, su_)
            Au = Au * torch.where(Au >= 0, su_, sl_)
    return Al, bl, Au, bu


def _lower_concretize(A, b, lb, ub):
    """Closed-form minimum of the lower linear forms over the input box.
    ``lb/ub (B, n_in)``; returns ``(B, rows)``."""
    lb_ = lb.unsqueeze(1)
    ub_ = ub.unsqueeze(1)
    return b + (A.clamp_min(0) * lb_ + A.clamp_max(0) * ub_).sum(-1)


class CrownBatch:
    """Batched CROWN pass over ``B`` input boxes for one FC-ReLU layer list.

    Runs the progressive backward substitution once (O(L^2) batched matmuls):
    pre-activation bounds of each ReLU layer are computed from the input box
    and used to relax that layer for all deeper passes. ``prepared`` comes from
    :func:`prepare_layers` (weights converted once per verification call).
    """

    def __init__(self, prepared: list, lb: np.ndarray, ub: np.ndarray,
                 device: str = "cpu"):
        self.prepared = prepared
        dt = torch.float64
        self.lb = torch.as_tensor(lb, dtype=dt, device=device)   # (B, n_in)
        self.ub = torch.as_tensor(ub, dtype=dt, device=device)
        B = self.lb.shape[0]

        self._relax: List[Optional[Tuple]] = [None] * len(prepared)
        self.pre_bounds: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, entry in enumerate(prepared):
            if entry is not None:
                continue                                # Linear: nothing to do
            n = prepared[idx - 1][0].shape[0]           # preceding Linear width
            C = torch.eye(n, dtype=dt, device=device).expand(B, n, n)
            Al, bl_, Au, bu_ = _backward(prepared, idx, self._relax, C)
            lo = _lower_concretize(Al, bl_, self.lb, self.ub)
            hi = -_lower_concretize(-Au, -bu_, self.lb, self.ub)
            self.pre_bounds.append((lo, hi))
            self._relax[idx] = _relu_relax(lo, hi)

    def margin_report(self, G: np.ndarray, g: np.ndarray,
                      row_rule: str = "argmax"):
        """One-pass per-box spec analysis over the unsafe region ``{G y <= g}``,
        with last-layer elision (the backward pass is seeded with ``G``).

        Returns ``(margin_lo, vertices, smear)`` as numpy arrays:
          * ``margin_lo (B, R)`` — sound lower bound of each row margin
            ``G_r y - g_r`` over the box; a row with ``margin_lo > 0`` is
            violated everywhere -> the box is provably SAFE (prunable).
          * ``vertices (B, n_in)`` — the box vertex minimizing the driving
            row's lower linear form: a cheap counterexample candidate for the
            caller's exact checks.
          * ``smear (B, n_in)`` — split scores for the driving row:
            ``|A_d| * width_d`` (sensitivity x range).

        ``row_rule`` picks the driving row per box: "argmax" (the row closest
        to being provably positive — the obstruction, cf. ``witness.binding_row``)
        or "argmin" (the loosest row). Heuristic only; never affects soundness.
        """
        B = self.lb.shape[0]
        Gt = torch.as_tensor(G, dtype=self.lb.dtype, device=self.lb.device)
        gt = torch.as_tensor(np.asarray(g).flatten(), dtype=self.lb.dtype,
                             device=self.lb.device)
        C = Gt.expand(B, *Gt.shape)
        Al, bl, _, _ = _backward(self.prepared, len(self.prepared),
                                 self._relax, C)
        margin_lo = _lower_concretize(Al, bl, self.lb, self.ub) - gt

        r_star = (margin_lo.argmax(dim=1) if row_rule == "argmax"
                  else margin_lo.argmin(dim=1))
        Ar = Al[torch.arange(B, device=Al.device), r_star]       # (B, n_in)
        vertices = torch.where(Ar >= 0, self.lb, self.ub)
        smear = Ar.abs() * (self.ub - self.lb)
        return (margin_lo.cpu().numpy(), vertices.cpu().numpy(),
                smear.cpu().numpy())

    def output_forms(self):
        """Identity-seeded output linear forms ``(Al, bl, Au, bu)`` — for
        enclosure tests and callers needing raw output bounds."""
        n_out = next(e[0].shape[0] for e in reversed(self.prepared)
                     if e is not None)
        B = self.lb.shape[0]
        C = torch.eye(n_out, dtype=self.lb.dtype,
                      device=self.lb.device).expand(B, n_out, n_out)
        return _backward(self.prepared, len(self.prepared), self._relax, C)
