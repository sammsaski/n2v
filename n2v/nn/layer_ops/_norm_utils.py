"""Shared helpers for normalisation reachability (LayerNorm / RMSNorm / GroupNorm / GRN).

A normalised layer y = (x - mu) / sqrt(var + eps) * gamma + beta is
*not* affine in x because ``mu`` and ``var`` depend on x. For sound
reachability we bound the per-feature scale factor
``s = 1 / sqrt(var + eps)`` from below and above using interval
estimates of var, then over-approximate the output as an axis-aligned
box centred at the IBP image with a width that accounts for both the
elementwise translation by ``-mu`` and the scaling by ``s``.

This is intentionally simple and sound; it matches the conservative
fallback nnVLA uses when CROWN slopes for normalisation cannot be
sharpened.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def interval_mean_var(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interval bounds on ``mean(x)`` and ``var(x)`` over the box [lb, ub].

    Returns
    -------
    mu_lb, mu_ub, var_lb, var_ub
        Each shaped ``(K,)`` where K is the number of feature groups
        (1 for LayerNorm/RMSNorm/GRN over the last axis when ``lb`` and
        ``ub`` are 1-D vectors).
    """
    lb = lb.reshape(-1).astype(np.float64)
    ub = ub.reshape(-1).astype(np.float64)
    n = lb.size
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    mu_lb = lb.mean()
    mu_ub = ub.mean()

    # Worst-case variance: each x_i deviates as far as possible from the
    # achievable mean. A conservative interval bound is:
    #   max |x_i - mu| <= max(|ub_i - mu_lb|, |lb_i - mu_ub|)
    devs = np.maximum(np.abs(ub - mu_lb), np.abs(lb - mu_ub))
    var_ub = (devs ** 2).mean()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # T0-4 / T0-5 booby-trap NOTE (audit C2/C3/C4 + workflow critical):
    # var_lb is hardcoded to zero. For eps=1e-5 this gives sigma_ub ≈ 316
    # per dim and the predicate-preserving Star norm reach's per-feature
    # slack absorbs the GLOBAL-mean centring error in the LayerNorm /
    # GroupNorm Star paths (the "masked" unsoundness). Until Commit 7
    # (PR12_FIX_LIST T1-1) lands the per-group mean + sigma intervals, do
    # NOT tighten var_lb -- doing so removes the masking and flips
    # LayerNorm/GroupNorm Star reach from latently-unsound to actively-
    # unsound (excluded true outputs, SAT/UNSAT inversion). A real
    # var_lb computed from the centred-square lower bound is a follow-up
    # to Commit 7, not before.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    var_lb = 0.0  # See booby-trap NOTE above.

    return (
        np.array([mu_lb]),
        np.array([mu_ub]),
        np.array([var_lb]),
        np.array([var_ub]),
    )


def normalised_interval(lb: np.ndarray, ub: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Sound interval over y = (x - mean(x)) / sqrt(var(x) + eps) on [lb, ub]."""
    mu_lb, mu_ub, var_lb, var_ub = interval_mean_var(lb, ub)
    s_lb = 1.0 / float(np.sqrt(np.asarray(var_ub).item() + eps))
    s_ub = 1.0 / float(np.sqrt(np.asarray(var_lb).item() + eps))

    diff_lb = lb.reshape(-1) - float(np.asarray(mu_ub).item())
    diff_ub = ub.reshape(-1) - float(np.asarray(mu_lb).item())

    # Product with positive-range scale: each entry's image is
    #   s * (x - mu) ∈ [min(s_lb, s_ub) * min(diff_lb, ...), ...]
    candidates_lb = np.stack([
        s_lb * diff_lb, s_lb * diff_ub, s_ub * diff_lb, s_ub * diff_ub
    ])
    candidates_ub = candidates_lb.copy()

    out_lb = candidates_lb.min(axis=0)
    out_ub = candidates_ub.max(axis=0)
    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)


def affine_after_norm(
    norm_lb: np.ndarray,
    norm_ub: np.ndarray,
    weight: np.ndarray | None,
    bias: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply optional elementwise ``y = gamma * y + beta`` to an interval."""
    out_lb = norm_lb.copy()
    out_ub = norm_ub.copy()
    if weight is not None:
        w = weight.reshape(-1, 1)
        pos = np.where(w >= 0, w, 0.0)
        neg = np.where(w < 0, w, 0.0)
        new_lb = pos * out_lb + neg * out_ub
        new_ub = pos * out_ub + neg * out_lb
        out_lb, out_ub = new_lb, new_ub
    if bias is not None:
        b = bias.reshape(-1, 1)
        out_lb = out_lb + b
        out_ub = out_ub + b
    return out_lb, out_ub
