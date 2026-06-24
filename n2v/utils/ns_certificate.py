"""Neumaier-Shcherbina sound LP-bound certificate (CPU / NumPy).

Turns an *approximate* dual ``y`` -- produced by any solver, including a
first-order GPU PDLP whose primal/dual are only accurate to ~1e-6 -- into a
*guaranteed* bound on the LP optimum via weak duality. This is the soundness
keystone of the batched-GPU LP path (PT-2): the GPU solver is only ever a
heuristic for producing ``y``; the value returned to the rest of n2v is the
certified bound, which over-/under-estimates the true optimum for *any*
``y >= 0`` regardless of solver error. A sloppy dual merely loosens the bound.

The LP is the per-coordinate Star bound problem::

    max / min   c^T x      s.t.   A x <= b,   lb <= x <= ub

(``A = C``, ``b = d``, ``lb/ub = predicate_lb/ub``). For an upper bound on the
maximum, weak duality gives, for any ``y >= 0``::

    r        = c - A^T y                         # reduced cost
    ub_bound = b^T y + sum_j max(r_j lb_j, r_j ub_j)

because ``c^T x = b^T y + r^T x - y^T(b - A x) <= b^T y + r^T x`` (the last term
is >= 0), and ``r^T x`` over the box is maximized coordinate-wise. The lower
bound on the minimum is obtained by the symmetric reduction
``min c^T x = -max (-c)^T x``.

FP-soundness
------------
The certificate is computed in round-to-nearest float64, then inflated
*outward* by a rigorous bound on the accumulated rounding error (Higham-style
``gamma_k = k u / (1 - k u)`` forward-error bounds, ``u = 2^-53``). The returned
value is therefore guaranteed outward of the exactly-rounded certificate, which
is itself outward of the true optimum. No GPU/FPU rounding-mode control is
needed -- the inflation is an explicit directed error term, as the PT-2 plan
prescribes. The inflation is ~1e-13 relative, negligible for verdict tightness.

The box bounds are handled analytically (the ``max(r lb, r ub)`` term), so only
the ``A x <= b`` rows need a dual; the box never needs one. Stars carry finite
``predicate_lb/ub``, so the "-inf vacuous bound" degeneracy that strikes free
variables does not arise.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

# Unit roundoff for IEEE-754 binary64 (round to nearest). One ULP of relative
# error per elementary operation is bounded by ``u``.
_U = 2.0 ** -53
# Extra safety multiplier on the (already conservative) error bound. The
# inflation is ~1e-13 relative; a small constant factor costs nothing in
# tightness but absorbs any slack in the hand-derived bound.
_SAFETY = 4.0


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` forward-error constant (k ops)."""
    ku = k * _U
    # Guard against the (astronomically unlikely) k u >= 1 for huge k.
    return ku / (1.0 - ku) if ku < 1.0 else np.inf


def _ns_upper(
    c: np.ndarray,
    y: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
) -> float:
    """Sound upper bound on ``max c^T x`` s.t. ``A x <= b, lb <= x <= ub``.

    ``c, lb, ub`` are length-n; ``y, b`` are length-m (m may be 0). ``y`` is
    clamped to ``>= 0`` internally (weak duality requires nonnegative duals).
    """
    n = c.shape[0]
    m = 0 if b is None else b.shape[0]

    # A^T y and the d^T y term. With no general constraints these vanish and the
    # certificate reduces to the exact (sound) box bound.
    if m > 0:
        yc = np.clip(y, 0.0, None)
        Aty = A.T @ yc                      # (n,)  reduced-cost contribution
        absAty = np.abs(A).T @ np.abs(yc)   # (n,)  magnitude bound on the m-term sums
        dy = float(b @ yc)                  # scalar
        abs_dy_terms = float(np.abs(b) @ np.abs(yc))
        err_Aty = _gamma(m) * absAty        # |fl(A^T y) - exact| per coordinate
        err_dy = _gamma(m) * abs_dy_terms   # |fl(b^T y) - exact|
    else:
        Aty = np.zeros(n)
        dy = 0.0
        err_Aty = np.zeros(n)
        err_dy = 0.0

    # Reduced cost r = c - A^T y (round-to-nearest), with its FP error bound.
    r = c - Aty
    err_r = err_Aty + _U * np.abs(r)        # subtraction adds <= u|r|

    # Box term: per coordinate pick the box endpoint that maximizes r_j x_j.
    # ``np.where`` (not r_pos*ub + r_neg*lb) keeps 0 * inf -> 0 if an endpoint
    # is infinite while r_j == 0 (defensive; stars carry finite bounds).
    Bsel = np.where(r >= 0.0, ub, lb)       # endpoint chosen by computed sign of r
    box = r * Bsel                          # (n,)
    box = np.where(r == 0.0, 0.0, box)      # kill 0*inf -> nan
    Mbox = np.maximum(np.abs(lb), np.abs(ub))
    # Per-coordinate box-term error: rounding of the product (u|box|) plus the
    # propagated error of r scaled by the endpoint magnitude. Factor 2 covers
    # the rare case where FP error flips the sign of a near-zero r_j (then the
    # analytic argmax endpoint differs from the computed one).
    err_box = _U * np.abs(box) + 2.0 * Mbox * err_r

    # Final accumulation b^T y + sum_j box_j  (an (n+1)-term sum).
    bound = dy + float(np.sum(box))
    err_sum = _gamma(n + 1) * (abs(dy) + float(np.sum(np.abs(box))))

    err = err_dy + float(np.sum(err_box)) + err_sum
    return bound + _SAFETY * err


def ns_bound(
    c: np.ndarray,
    y: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize: bool = False,
) -> float:
    """Certified bound on ``min``/``max c^T x`` s.t. ``A x <= b, lb <= x <= ub``.

    Args:
        c: Objective vector, shape ``(n,)`` or ``(n, 1)``.
        y: Approximate dual for the ``A x <= b`` rows, shape ``(m,)`` or
            ``(m, 1)`` (length 0 / ``None`` when there are no general
            constraints). Clamped to ``>= 0`` internally.
        A, b: Inequality constraints (``A x <= b``). ``None`` => box-only LP.
        lb, ub: Finite box bounds, shape ``(n,)`` or ``(n, 1)``.
        minimize: If True return a sound *lower* bound on the minimum; else a
            sound *upper* bound on the maximum.

    Returns:
        A float that is guaranteed to enclose the true optimum (>= max, resp.
        <= min) for any ``y``, accounting for float64 rounding error.
    """
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    n = c.shape[0]
    lb = (np.full(n, -np.inf) if lb is None
          else np.asarray(lb, dtype=np.float64).reshape(-1))
    ub = (np.full(n, np.inf) if ub is None
          else np.asarray(ub, dtype=np.float64).reshape(-1))

    if A is not None and b is not None and np.size(b) > 0:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
    else:
        A = None
        b = None
        y = np.zeros(0)

    if minimize:
        # min c^T x = -max (-c)^T x.
        return -_ns_upper(-c, y, A, b, lb, ub)
    return _ns_upper(c, y, A, b, lb, ub)


def ns_bounds_batch(
    objectives: Sequence[np.ndarray],
    duals: Sequence[np.ndarray],
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize_flags: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Vectorized :func:`ns_bound` over a batch sharing ``A, b, lb, ub``.

    Args:
        objectives: ``k`` objective vectors, each length ``n``.
        duals: ``k`` approximate duals, each length ``m`` (one per objective).
        A, b, lb, ub: Shared constraints/box (see :func:`ns_bound`).
        minimize_flags: ``k`` booleans (True => lower bound on min). Defaults to
            all-maximize.

    Returns:
        ``np.ndarray`` of ``k`` certified bounds, matching the order of
        ``objectives``.
    """
    k = len(objectives)
    if minimize_flags is None:
        minimize_flags = [False] * k
    return np.array([
        ns_bound(objectives[i], duals[i], A=A, b=b, lb=lb, ub=ub,
                 minimize=minimize_flags[i])
        for i in range(k)
    ], dtype=np.float64)


def ns_bounds_population(
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    minimize_flags: Sequence[bool],
) -> np.ndarray:
    """Vectorized NS bounds for a *population* of stars in one array op.

    Faithful batched form of :func:`ns_bound` (the scalar version is the audited
    reference; ``TestNSBatch`` pins their equivalence). Used by the
    cross-population GPU path so the per-(star, objective) certificate is not a
    Python loop. Padded constraint rows (all-zero ``A`` row + zero ``b``)
    contribute nothing and may be passed freely.

    Args:
        C: Objective coefficients, shape ``(B, n, k)`` (``B`` stars, ``n`` vars,
            ``k`` objectives per star).
        Y: Approximate duals, shape ``(B, m, k)``.
        A: Constraints, shape ``(B, m, n)``; ``b`` shape ``(B, m)``.
        lb, ub: Box bounds, shape ``(B, n)``.
        minimize_flags: length-``k`` booleans (shared across stars), True =>
            lower bound on the min for that objective column.

    Returns:
        ``(B, k)`` array of certified bounds.
    """
    B, n, k = C.shape
    m = A.shape[1]
    # sign: +1 maximize, -1 minimize. min c^T x = -max (-c)^T x, so certify the
    # upper bound of the sign-flipped objective and flip the result back.
    sign = np.where(np.asarray(minimize_flags, dtype=bool), -1.0, 1.0)  # (k,)
    G = C * sign[None, None, :]                                        # (B, n, k)

    Yc = np.clip(Y, 0.0, None)                                         # (B, m, k)
    Aty = np.matmul(np.transpose(A, (0, 2, 1)), Yc)                   # (B, n, k)
    absAty = np.matmul(np.transpose(np.abs(A), (0, 2, 1)), np.abs(Yc))
    dy = np.einsum("bm,bmk->bk", b, Yc)                               # (B, k)
    abs_dy = np.einsum("bm,bmk->bk", np.abs(b), np.abs(Yc))

    g_m = _gamma(m)
    err_Aty = g_m * absAty
    err_dy = g_m * abs_dy

    r = G - Aty
    err_r = err_Aty + _U * np.abs(r)

    lb3 = lb[:, :, None]
    ub3 = ub[:, :, None]
    Bsel = np.where(r >= 0.0, ub3, lb3)
    box = np.where(r == 0.0, 0.0, r * Bsel)
    Mbox = np.maximum(np.abs(lb3), np.abs(ub3))
    err_box = _U * np.abs(box) + 2.0 * Mbox * err_r

    bound = dy + np.sum(box, axis=1)                                   # (B, k)
    err_sum = _gamma(n + 1) * (np.abs(dy) + np.sum(np.abs(box), axis=1))
    err = err_dy + np.sum(err_box, axis=1) + err_sum

    upper = bound + _SAFETY * err                                      # (B, k)
    return upper * sign[None, :]


__all__ = ["ns_bound", "ns_bounds_batch", "ns_bounds_population"]
