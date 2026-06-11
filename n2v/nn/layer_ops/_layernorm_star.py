"""Predicate-preserving Star reach for LayerNorm-style normalisation.

The plain "box-lift" implementation in :func:`layernorm_star_approx`
discards the input Star's predicate basis: it computes interval bounds
on the output and calls ``Star.from_bounds`` on them, producing a Star
whose predicates are *disconnected* from the input's. After three or
four such layers in a transformer encoder the bounds explode because
cross-neuron correlations are lost at every step.

This helper computes a tighter Star reach that *preserves* the input
predicates and only adds a small fresh predicate per output dimension
to cover the slack from the scale factor's interval. The mean
subtraction step is performed exactly (linear in the input
predicates), so a downstream linear/attention layer can still tighten
via the input correlations.

The implementation is sound but uses an axis-aligned slack model for
the scale (no per-dimension McCormick envelope), so it is looser than
nnVLA's ``layernorm/methods/star.py``. Tightening that further is
tracked as a follow-up.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from n2v.sets import Star


def _mean_along_features(V: np.ndarray) -> np.ndarray:
    """Mean along feature axis (axis 0). Result shape ``(1, nVar+1)``."""
    return V.mean(axis=0, keepdims=True)


def _bounds_for_z(
    V_centred: np.ndarray,
    pred_lb: Optional[np.ndarray],
    pred_ub: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Box bound on ``z = V_centred @ [1, alpha]^T`` via interval arithmetic."""
    n_var = V_centred.shape[1] - 1
    if pred_lb is None or pred_ub is None:
        pred_lb = -np.ones((n_var, 1))
        pred_ub = np.ones((n_var, 1))
    center = V_centred[:, :1]
    gens = V_centred[:, 1:]
    pos = np.where(gens >= 0, gens, 0.0)
    neg = np.where(gens < 0, gens, 0.0)
    z_lb = center + pos @ pred_lb + neg @ pred_ub
    z_ub = center + pos @ pred_ub + neg @ pred_lb
    return z_lb, z_ub


def predicate_preserving_norm_star(
    base: Star,
    sigma_bounds: Tuple[float, float],
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    subtract_mean: bool,
) -> Star:
    """Construct a Star that approximates ``y = (x - mu) / sigma * gamma + beta``.

    Parameters
    ----------
    base
        Input Star (must already be in 2-D form; ImageStar callers should
        flatten then re-wrap).
    sigma_bounds
        ``(sigma_lb, sigma_ub)`` — interval bound on the normalisation
        denominator. For LayerNorm this comes from a worst-case variance
        estimate; for RMSNorm directly from the RMS interval.
    weight
        Optional per-feature ``gamma`` (shape ``(N,)``). ``None`` ⇒ ones.
    bias
        Optional per-feature ``beta`` (shape ``(N,)``). ``None`` ⇒ zeros.
    subtract_mean
        True for LayerNorm-style normalisation; False for RMSNorm.

    Returns
    -------
    Star
        Output Star with the same input predicates plus ``N`` fresh slack
        predicates (one per output dimension) bounded to cover the scale
        interval.
    """
    V = base.V.astype(np.float64).copy()
    N, n_var_plus_1 = V.shape
    n_var = n_var_plus_1 - 1

    # Step 1: exact mean subtraction (linear in alpha, preserves predicates).
    if subtract_mean:
        row_mean = _mean_along_features(V)
        V_centred = V - row_mean
    else:
        V_centred = V

    # Step 2: bound z = x (- mean) elementwise.
    z_lb, z_ub = _bounds_for_z(V_centred, base.predicate_lb, base.predicate_ub)
    z_max_abs = np.maximum(np.abs(z_lb), np.abs(z_ub))  # (N, 1)

    # Step 3: scale interval.
    sigma_lb, sigma_ub = sigma_bounds
    s_lb = 1.0 / max(sigma_ub, 1e-12)
    s_ub = 1.0 / max(sigma_lb, 1e-12)
    s_mid = 0.5 * (s_lb + s_ub)
    s_half_range = 0.5 * (s_ub - s_lb)

    # Step 4: build the affine part y_lin = gamma * s_mid * z + beta.
    if weight is None:
        gamma = np.ones((N, 1))
    else:
        gamma = weight.reshape(-1, 1).astype(np.float64)
    if bias is None:
        beta = np.zeros((N, 1))
    else:
        beta = bias.reshape(-1, 1).astype(np.float64)

    scaled = gamma * s_mid  # (N, 1)
    V_lin = V_centred * scaled  # broadcasts scaled across columns
    V_lin[:, :1] = V_lin[:, :1] + beta

    # Step 5: per-dim slack predicate covering |gamma * s_half_range * z|.
    # slack_radius[i] = |gamma_i| * s_half_range * z_max_abs[i]
    slack_radius = np.abs(gamma) * s_half_range * z_max_abs  # (N, 1)
    V_slack = np.zeros((N, N))
    np.fill_diagonal(V_slack, slack_radius.reshape(-1))

    # New basis: [center, alpha_gens, slack_gens]
    new_V = np.hstack([V_lin, V_slack])

    # Constraints: keep old C @ alpha <= d, extend with zero on slack vars.
    if base.C is not None and base.C.size > 0:
        old_C = base.C
        C0 = np.hstack([old_C, np.zeros((old_C.shape[0], N))])
        d0 = base.d
    else:
        C0 = np.zeros((0, n_var + N))
        d0 = np.zeros((0, 1))

    # Predicate bounds: existing alpha + slack ∈ [-1, 1].
    if base.predicate_lb is not None:
        new_pred_lb = np.vstack([base.predicate_lb, -np.ones((N, 1))])
        new_pred_ub = np.vstack([base.predicate_ub, np.ones((N, 1))])
    else:
        new_pred_lb = np.vstack([-np.ones((n_var, 1)), -np.ones((N, 1))])
        new_pred_ub = np.vstack([np.ones((n_var, 1)), np.ones((N, 1))])

    return Star(new_V, C0, d0, new_pred_lb, new_pred_ub)
