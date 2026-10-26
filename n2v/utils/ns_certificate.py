"""Rigorous outward floating-point bounds for LP and interval computations
over star sets (CPU / NumPy).

Each function accounts for float64 rounding with an explicit, rigorous error
bound applied OUTWARD, so a certificate built on its result holds for the set
as defined rather than for its rounded image:

- ``outward_interval_bounds``: the interval bounds of ``c + V alpha`` over
  the predicate box, each direction inflated only by the rounding of the
  terms it actually sums (``Star``/``ImageStar`` estimate ranges and
  ``Zono.get_bounds``).
- ``outward_interval_slack``: a single symmetric slack for a caller that
  computes the round-to-nearest interval itself (the row slack of
  ``lpsolver``'s certified emptiness check).
- ``outward_spec_rows``: spec rows ``H y <= g`` pulled back to the predicate
  variables with an outward right-hand side (``Star.intersect_half_space``
  and the violation LP).
- ``ns_bound``: the Neumaier-Shcherbina certificate. It turns an
  *approximate* dual ``y`` from any solver into a *guaranteed* bound on the
  LP optimum via weak duality, which over-/under-estimates the true optimum
  for *any* ``y >= 0`` regardless of solver error; a sloppy dual merely
  loosens the bound. Used by ``confirm_infeasible`` (Farkas check) and by the
  refine engine's GPU bounds, where the solver only produces ``y`` and the
  value returned is the certified bound.
- ``ns_bounds_population``: the vectorized form of ``ns_bound`` over a
  population of stars.
- ``_add_dir`` / ``_mul_dir`` / ``_div_dir`` / ``_widen_sum`` /
  ``_widen_relative``: directed rounding for elementwise interval arithmetic
  (the IBP bound pre-pass and ``Box`` reach).

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
needed -- the inflation is an explicit directed error term. The inflation is
~1e-13 relative, negligible for verdict tightness.

The box bounds are handled analytically (the ``max(r lb, r ub)`` term), so only
the ``A x <= b`` rows need a dual; the box never needs one. Stars carry finite
``predicate_lb/ub``, so the "-inf vacuous bound" degeneracy that strikes free
variables does not arise.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

# Unit roundoff for IEEE-754 binary64 (round to nearest). One ULP of relative
# error per elementary operation is bounded by ``u``.
_U = 2.0 ** -53
# Extra safety multiplier on the (already conservative) error bound. The
# inflation is ~1e-13 relative; a small constant factor costs nothing in
# tightness but absorbs any slack in the hand-derived bound.
_SAFETY = 4.0
# Smallest positive subnormal binary64. With gradual underflow a product that
# lands in the subnormal range carries an absolute error of at most half of
# this on top of the relative ``u`` (sums carry none), so ``gamma_k`` alone
# does not bound a sum of tiny products.
_ETA = 2.0 ** -1074
# Row block of the interval kernels, in matrix elements (16 MB of float64):
# bounds their temporaries independently of the size of the set.
_ROW_BLOCK_ELEMS = 1 << 21
# Column 0 of the (rows, 2) interval-kernel arrays is the lower bound (moved
# toward -inf), column 1 the upper bound (moved toward +inf).
_DIRECTION = np.array([-1.0, 1.0])


def _gamma(k: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` forward-error constant (k ops)."""
    ku = k * _U
    # Guard against the (astronomically unlikely) k u >= 1 for huge k.
    return ku / (1.0 - ku) if ku < 1.0 else np.inf


def _gamma_rows(k):
    """Vectorized ``gamma_{k+1}`` for per-row counts ``k`` of nonzero terms."""
    ku = (k + 1) * _U
    return np.where(ku < 1.0, ku / (1.0 - ku), np.inf)


def _row_blocks(G, width):
    """Yield ``(row_slice, block)`` over the rows of ``G`` (shape
    ``(..., n)``) flattened to ``(-1, n)``, sized so that ``(block, width)``
    temporaries stay near ``_ROW_BLOCK_ELEMS`` elements.

    The blocks are views when the leading axes of ``G`` merge without a copy
    (a C-ordered basis). Otherwise (e.g. a transposed ImageStar basis) each
    block is gathered on its own, so the whole tensor is never copied.
    """
    lead = G.shape[:-1]
    rows = int(np.prod(lead))
    step = max(1, _ROW_BLOCK_ELEMS // max(width, 1))
    dims = [(d, st) for d, st in zip(lead, G.strides[:-1]) if d != 1]
    mergeable = all(s0 == s1 * d1 for (_, s0), (d1, s1) in zip(dims, dims[1:]))
    G2 = G.reshape(rows, G.shape[-1]) if mergeable else None
    for start in range(0, rows, step):
        sl = slice(start, min(start + step, rows))
        if G2 is not None:
            yield sl, G2[sl]
        else:
            yield sl, G[np.unravel_index(np.arange(sl.start, sl.stop), lead)]


def outward_interval_bounds(center, generators, lb, ub):
    """Rigorous interval enclosure of ``c + V alpha`` over the box
    ``lb <= alpha <= ub``, computed in round-to-nearest float64.

    Per output coordinate the lower bound is the sum
    ``c + sum_{g_j > 0} g_j lb_j + sum_{g_j < 0} g_j ub_j`` (the upper bound
    swaps the endpoints), and each direction is moved outward only by the
    rounding error of the terms that direction actually sums:

    - A product with a zero factor is an exact zero and adds no error. A
      direction whose products are all exact zeros is the exact center and is
      returned unchanged: a row with no nonzero generator, or the lower bound
      of a ReLU relaxation variable on ``[0, u]``.
    - A row with at most one nonzero generator whose chosen endpoint is ``0``
      or ``+-1`` is ``c + g`` with one rounding (the product is exact). The
      exact error of that addition is recovered with Knuth's TwoSum, and the
      bound moves to the adjacent float only when the error points inward.
      An exactly computed bound therefore stays exact, e.g. the lower bound 0
      of ``[0, 1]`` written as ``0.5 + 0.5 alpha``.
    - Otherwise a sum of ``k + 1`` terms (``k`` nonzero generators) is off by
      at most ``gamma_{k+1} * sum |terms|`` in any summation order (Higham's
      inner-product bound), plus half a subnormal per product that
      underflows. The bound moves by ``SAFETY`` times that.

    Without the outward move the interval can sit a few ulps INSIDE the true
    range, enough to classify a neuron whose true upper bound is +1e-17 as
    stably inactive (its output zeroed), which is an under-approximation. A
    move where the bound is exact costs no soundness but turns a neuron whose
    bound is exactly 0 into a crossing one (an extra split in exact reach).

    The enclosure is rigorous for ``c`` and ``V`` as given. It does not
    cover rounding committed by whatever computed them (e.g. the affine map
    of a Linear layer, done in round-to-nearest), so an exact bound here can
    still sit an ulp inside the range of the real-number network.

    Infinite box entries are handled: a zero generator times an infinite
    endpoint contributes 0 (not NaN); a nonzero one makes that direction
    infinite. Rows are processed in blocks, so the temporaries stay bounded
    however large the set is.

    Args:
        center: Centers, shape ``(...)``.
        generators: Generators, shape ``(..., nVar)``.
        lb, ub: Predicate box, ``nVar`` entries each.

    Returns:
        ``(lo, hi)``, float64 arrays of shape ``(...)`` with
        ``lo <= c + V alpha <= hi`` for every ``alpha`` in the box.
    """
    G = np.asarray(generators, dtype=np.float64)
    out_shape = G.shape[:-1]
    n = G.shape[-1]
    rows = int(np.prod(out_shape))
    c = np.broadcast_to(np.asarray(center, dtype=np.float64),
                        out_shape).reshape(-1)
    if n == 0 or rows == 0:
        return c.reshape(out_shape).copy(), c.reshape(out_shape).copy()

    # Row 0 holds lb, row 1 ub; every quantity below is such a pair.
    B = np.vstack([np.asarray(lb, dtype=np.float64).reshape(-1),
                   np.asarray(ub, dtype=np.float64).reshape(-1)])
    finite = np.isfinite(B)
    has_inf = not finite.all()
    if has_inf:
        B = np.where(finite, B, 0.0)
    absB = np.abs(B)
    # One matrix product of the positive and one of the negative generator
    # parts against these column pairs gives every per-row quantity at once.
    # A sum of nonnegative indicator-weighted generators is > 0 exactly when
    # some generator of that sign meets a flagged endpoint.
    #   pair 0  endpoint values (infinite entries zeroed, flagged by pair 4)
    #   pair 1  endpoint magnitudes (term magnitudes)
    #   pair 2  endpoint != 0 (the product is not an exact zero)
    #   pair 3  endpoint not in {0, +-1} (the product may round)
    #   pair 4  endpoint infinite (only when the box has one)
    nonzero = (B != 0.0) | ~finite
    cols = [B, absB, nonzero, nonzero & (absB != 1.0) & finite]
    if has_inf:
        cols.append(~finite)
    M = np.vstack(cols).T.astype(np.float64)

    bounds = np.empty((rows, 2))
    # Overflow is not an error here: an overflowing sum or magnitude makes
    # that direction infinite (see ``_outward_both``).
    with np.errstate(over="ignore", invalid="ignore"):
        for sl, blk in _row_blocks(G, max(n, M.shape[1])):
            m = blk.shape[0]
            # (m, pair, side): side 0 feeds the lower bound, side 1 the upper.
            # A positive generator meets lb in the lower bound, a negative one
            # ub, so the negative part's sides are swapped.
            P = (np.maximum(blk, 0.0) @ M).reshape(m, -1, 2)
            N = (np.minimum(blk, 0.0) @ M).reshape(m, -1, 2)[:, :, ::-1]
            bounds[sl] = _outward_both(
                c[sl][:, None], P[:, 0] + N[:, 0], P[:, 1] - N[:, 1],
                (P[:, 2] > 0.0) | (N[:, 2] < 0.0),
                (P[:, 3] > 0.0) | (N[:, 3] < 0.0),
                ((P[:, 4] > 0.0) | (N[:, 4] < 0.0)) if has_inf else None,
                np.count_nonzero(blk, axis=1)[:, None])
    return (bounds[:, 0].reshape(out_shape).copy(),
            bounds[:, 1].reshape(out_shape).copy())


def _outward_both(c, part, gen_mag, live, may_round, unbounded, k):
    """Both directions of :func:`outward_interval_bounds` for a block of rows.

    ``part``, ``gen_mag``, ``live``, ``may_round`` and ``unbounded`` are
    ``(rows, 2)`` (lower, upper): the generator part of the sum, its sum of
    term magnitudes, a product that is not an exact zero, a product whose
    endpoint is not in ``{0, +-1}``, and a nonzero generator on an infinite
    endpoint. ``c`` is the center and ``k`` the nonzero generators, ``(rows,
    1)``.
    """
    val = c + part
    err = _SAFETY * (_gamma_rows(k) * (np.abs(c) + gen_mag) + k * _ETA)
    slack = np.where(live, err, 0.0)
    out = val + _DIRECTION * slack
    # Overflowing or infinite terms: the only honest bound is infinite.
    bad = np.isinf(slack) | np.isnan(out)
    # At most one nonzero generator and an exact product: ``part`` is exact,
    # so the only rounding is ``c + part`` and TwoSum gives its exact error.
    single = (k <= 1) & ~may_round & np.isfinite(val)
    if single.any():
        rows, side = np.nonzero(single)
        cs, ps, vs = c[rows, 0], part[rows, side], val[rows, side]
        d = _DIRECTION[side]
        bb = vs - cs
        rounding = (cs - (vs - bb)) + (ps - bb)    # exact: (cs + ps) - vs
        out[rows, side] = np.where(d * rounding > 0.0,
                                   np.nextafter(vs, d * np.inf), vs)
        # TwoSum is error-free unless an intermediate overflows.
        bad[rows, side] = ~np.isfinite(rounding)
    if unbounded is not None:
        bad |= unbounded
    return np.where(bad, _DIRECTION * np.inf, out)


def outward_interval_slack(center_abs, generators, lb, ub):
    """Rigorous symmetric outward slack for an interval bound of
    ``c + V alpha`` over the box ``lb <= alpha <= ub`` that the CALLER
    computes in round-to-nearest float64.

    A row with ``k`` nonzero generators is a ``(k + 1)``-term sum; its
    forward error is at most ``gamma_{k+1}`` times the sum of the absolute
    values of its terms (plus the underflow term, see ``_ETA``). Subtracting
    the returned slack from the lower bound and adding it to the upper bound
    makes the interval a rigorous enclosure. A row with no nonzero generator
    is the exact center and gets zero slack. :func:`outward_interval_bounds`
    is tighter (it charges each direction only for its own terms and keeps
    exactly computed bounds exact) and is what the set classes use. The
    certified emptiness check in ``n2v.utils.lpsolver`` (``_careful_emptiness``,
    behind ``confirm_infeasible`` and ``check_feasibility``) slackens its rows
    with this function.

    Shapes: ``center_abs`` (...,), ``generators`` (..., nVar), ``lb``/``ub``
    (nVar,). Infinite box entries are handled: a zero generator times an
    infinite bound contributes 0 (not NaN); a nonzero one yields an infinite
    slack, i.e. an unbounded coordinate, which is the truthful answer.
    """
    G = np.asarray(generators, dtype=np.float64)
    out_shape = G.shape[:-1]
    n = G.shape[-1]
    rows = int(np.prod(out_shape))
    absmax = np.maximum(np.abs(np.asarray(lb, dtype=np.float64).ravel()),
                        np.abs(np.asarray(ub, dtype=np.float64).ravel()))
    finite = np.isfinite(absmax)
    cols = [np.where(finite, absmax, 0.0)]
    if not finite.all():
        cols.append(~finite)
    M = np.stack(cols, axis=1).astype(np.float64)
    mag = np.zeros(rows)
    unbounded = np.zeros(rows, dtype=bool)
    k = np.zeros(rows, dtype=np.int64)
    c_abs = np.broadcast_to(np.asarray(center_abs, dtype=np.float64),
                            out_shape).reshape(-1)
    # Overflow yields an infinite slack, which is the honest answer.
    with np.errstate(over="ignore", invalid="ignore"):
        # Row blocks keep the |V| temporary bounded (no full-size copies).
        for sl, blk in _row_blocks(G, max(n, M.shape[1])):
            R = np.abs(blk) @ M
            mag[sl] = R[:, 0]
            if M.shape[1] > 1:
                unbounded[sl] = R[:, 1] > 0.0
            k[sl] = np.count_nonzero(blk, axis=1)
        magnitude = c_abs + mag
        magnitude[unbounded] = np.inf
        slack = np.where(k == 0, 0.0,
                         _SAFETY * (_gamma_rows(k) * magnitude + k * _ETA))
    return slack.reshape(out_shape)


def outward_spec_rows(H, g, V, pred_lb=None, pred_ub=None):
    """Rows of ``H (c + V' alpha) <= g`` over the predicate variables, with the
    right-hand side inflated OUTWARD by a rigorous bound on the rounding of
    ``H V'`` and ``g - H c``.

    Returns ``(H_alpha, rhs)`` with ``H_alpha = H @ V'`` (k, nVar) and
    ``rhs = g - H c + slack`` (k,). Any emptiness / disjointness certificate
    that is rigorous for the returned rows is then valid for the star and
    halfspace AS DEFINED: without the slack, a set that touches the halfspace
    at exact margin 0 is routinely rounded 1e-17 into "disjoint", and a
    rigorous certificate (Farkas/NS) then proves the wrong system empty --
    a false UNSAT (measured: the fc662_s8_touch golden case).

    The slack is ``SAFETY * (gamma_{dim+nVar+2} * (|H||c| + |g| + (|H||V'|)|alpha|_max)
    + (dim+nVar+2) * eta * (1 + sum |alpha|_max))`` per row. The ``eta``
    term is the absolute error of products that underflow (see ``_ETA``),
    which the relative term does not bound: with subnormal products it
    rounds to zero, and a touching set was then declared disjoint. The
    ``|alpha|_max`` parts need a finite predicate box and are omitted when
    none is given (best effort: still the rounding of the constant part).
    """
    H = np.asarray(H, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64)
    c = V[:, 0]
    Vg = V[:, 1:]
    H_alpha = H @ Vg
    rhs = g - H @ c
    absH = np.abs(H)
    magnitude = absH @ np.abs(c) + np.abs(g)
    # Each entry of H c and of H V' sums ``dim`` products; an underflowed
    # product is off by up to eta/2 absolutely, and the H V' part is scaled
    # by |alpha| <= |alpha|_max.
    underflow_scale = 1.0
    if pred_lb is not None and pred_ub is not None and Vg.shape[1]:
        absmax = np.maximum(np.abs(np.asarray(pred_lb, dtype=np.float64).ravel()),
                            np.abs(np.asarray(pred_ub, dtype=np.float64).ravel()))
        if np.all(np.isfinite(absmax)):
            magnitude = magnitude + (absH @ np.abs(Vg)) @ absmax
            underflow_scale = 1.0 + float(np.sum(absmax))
    k_ops = V.shape[0] + Vg.shape[1] + 2
    return H_alpha, rhs + _SAFETY * (_gamma(k_ops) * magnitude
                                     + k_ops * _ETA * underflow_scale)


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
        # A non-finite multiplier is not a proof: ``inf`` propagates to
        # ``0 * inf = NaN`` in the reduced cost, and a NaN bound is silently
        # read as "stably active" by the ReLU classifier. Zero is always a
        # valid multiplier, so degrade to it (the bound falls back to the box).
        yc = np.nan_to_num(np.clip(y, 0.0, None), nan=0.0, posinf=0.0,
                           neginf=0.0)
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
    # ``Mbox`` is infinite when a predicate bound is infinite. Where the
    # propagated error is exactly zero the product must be 0, not ``inf * 0 =
    # NaN``; where it is positive an infinite error term is the honest answer
    # (an infinite bound carries no information, but it is sound).
    err_box = _U * np.abs(box) + np.where(err_r == 0.0, 0.0, 2.0 * Mbox * err_r)

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

    # Non-finite multipliers degrade to zero (see ``_ns_upper``): a valid
    # multiplier that costs tightness, never a NaN that costs soundness.
    Yc = np.nan_to_num(np.clip(Y, 0.0, None), nan=0.0, posinf=0.0,
                       neginf=0.0)                                     # (B, m, k)
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
    err_box = _U * np.abs(box) + np.where(err_r == 0.0, 0.0,
                                          2.0 * Mbox * err_r)

    bound = dy + np.sum(box, axis=1)                                   # (B, k)
    err_sum = _gamma(n + 1) * (np.abs(dy) + np.sum(np.abs(box), axis=1))
    err = err_dy + np.sum(err_box, axis=1) + err_sum

    upper = bound + _SAFETY * err                                      # (B, k)
    return upper * sign[None, :]


# ---------------------------------------------------------------------------
# Directed rounding for elementwise interval arithmetic (the IBP bound
# pre-pass and Box reach). Each helper returns a bound moved toward -inf
# (d = -1) or +inf (d = +1) past the round-to-nearest result, and leaves a
# result that is provably exact where it is. The operands are converted to
# float64 first (exact for float32 values), so the one rounding is a float64
# one and a one-ulp move covers it.
#
# A finite operation that overflows to +-inf is not exact: its exact value
# lies just beyond the largest finite double. ``nextafter`` toward the other
# side gives that largest double, a valid bound; the side of the overflow
# stays infinite.
#
# An operation with no value (``inf - inf``, ``inf / inf``, a zero divisor or
# a NaN operand) returns ``d * inf``, the bound that claims nothing, never NaN:
# NaN fails every comparison, so a NaN bound can read as a decided one.
# ---------------------------------------------------------------------------

def _add_dir(a: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """``a + b`` rounded toward ``d * inf``. TwoSum gives the exact error of
    the one rounding, so the result moves one ulp only when that error points
    past it (an exact sum stays exact)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        s = a + b
        bb = s - a
        err = (a - (s - bb)) + (b - bb)    # exact: (a + b) - s
        out = np.where(d * err > 0.0, np.nextafter(s, d * np.inf), s)
        finite_in = np.isfinite(a) & np.isfinite(b)
        # A finite sum that overflowed (see above).
        out = np.where(finite_in & ~np.isfinite(s),
                       np.nextafter(s, d * np.inf), out)
    # An infinite operand gives an infinite (exact) sum; a TwoSum
    # intermediate that overflowed leaves the error unknown, and ``inf - inf``
    # has no value, so those bounds become infinite.
    return np.where((finite_in & np.isfinite(s) & ~np.isfinite(err))
                    | np.isnan(s), d * np.inf, out)


def _mul_dir(a: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """``a * b`` rounded toward ``d * inf``: a product with a zero or +-1
    factor, or with an infinite factor, is exact; any other moves one ulp
    (one rounding, including underflow to 0 and overflow to +-inf). A zero
    times an infinite factor is 0, not NaN, as in ``outward_interval_bounds``:
    an infinite interval endpoint stands for unbounded reals, and 0 times
    any real is 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        aa, ab = np.abs(a), np.abs(b)
        zero_inf = ((aa == 0.0) & np.isinf(b)) | (np.isinf(a) & (ab == 0.0))
        p = np.where(zero_inf, 0.0, a * b)
        exact = ((aa == 0.0) | (ab == 0.0) | (aa == 1.0) | (ab == 1.0)
                 | ~(np.isfinite(a) & np.isfinite(b)))
        out = np.where(exact, p, np.nextafter(p, d * np.inf))
    return np.where(np.isnan(out), d * np.inf, out)


def _div_dir(a: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """``a / b`` rounded toward ``d * inf`` (one rounding, including overflow;
    exact for a zero numerator, a divisor of +-1, or an infinite operand).
    A zero divisor has no quotient (``1 / 0.0`` is ``+inf`` and ``1 / -0.0``
    ``-inf``, by the sign of the zero alone), and neither has ``inf / inf``:
    both give ``d * inf``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = a / b
        exact = ((a == 0.0) | (np.abs(b) == 1.0)
                 | ~(np.isfinite(a) & np.isfinite(b)))
        out = np.where(exact, q, np.nextafter(q, d * np.inf))
    return np.where((b == 0.0) | np.isnan(out), d * np.inf, out)


def outward_center_radius(lb: np.ndarray,
                          ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center ``c`` and half-width ``r`` of the interval ``[lb, ub]`` such
    that ``[c - r, c + r]`` contains ``[lb, ub]`` in exact arithmetic: the
    zonotope (``c + r alpha``, ``alpha in [-1, 1]``) of a box.

    The round-to-nearest ``(lb + ub) / 2`` and ``(ub - lb) / 2`` miss an
    endpoint by an ulp in about half of all intervals. Here ``c`` is the
    rounded midpoint and ``r`` the larger of ``ub - c`` and ``c - lb``, each
    rounded up, so an exact midpoint and half-width (e.g. ``[0, 1]``) are
    unchanged. Each bound is halved before the sum, so the midpoint of finite
    bounds never overflows. A zonotope has no unbounded form: an interval
    with an infinite bound gets center 0 and an infinite half-width.

    Returns:
        ``(c, r)``, float64 arrays with the shape of ``lb`` and ``ub``.
    """
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        c = 0.5 * lb + 0.5 * ub
    c = np.where(np.isfinite(c), c, 0.0)
    r = np.maximum(_add_dir(ub, -c, 1.0), _add_dir(c, -lb, 1.0))
    return c, r


def _widen_sum(lo: np.ndarray, hi: np.ndarray, mag: np.ndarray,
               count: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Move ``lo`` / ``hi``, each computed in round-to-nearest as a sum of at
    most ``count`` nonzero products plus one added constant (in any order),
    outward by a rigorous bound on that rounding: ``gamma_{count+2}`` times
    ``mag``, an upper bound on the sum of the terms' magnitudes, plus half a
    subnormal per product for underflow (Higham's inner-product bound),
    times ``_SAFETY``. Where ``count`` is 0 every product is an exact zero, the
    sum is the constant, and the bounds are left exact."""
    count = np.asarray(count, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        err = _SAFETY * (_gamma_rows(count + 1) * mag + count * _ETA)
        err = np.where(count > 0, err, 0.0)
        # An overflowing magnitude or bound leaves only the infinite bound.
        new_lo = np.where(np.isfinite(err) & np.isfinite(lo), lo - err, -np.inf)
        new_hi = np.where(np.isfinite(err) & np.isfinite(hi), hi + err, np.inf)
    return new_lo, new_hi


def _widen_relative(lo: np.ndarray, hi: np.ndarray, x_lo: np.ndarray,
                    x_hi: np.ndarray, ops: int) -> Tuple[np.ndarray, np.ndarray]:
    """Move bounds computed by an elementwise function of accuracy
    ``gamma_ops`` (relative) outward, leaving values at an input of exactly 0
    unchanged (``tanh(0)`` and ``sigmoid(0)`` are exact)."""
    pad_lo = _SAFETY * _gamma(ops) * np.abs(lo) + _ETA
    pad_hi = _SAFETY * _gamma(ops) * np.abs(hi) + _ETA
    return (np.where(x_lo == 0.0, lo, lo - pad_lo),
            np.where(x_hi == 0.0, hi, hi + pad_hi))


__all__ = ["outward_interval_bounds", "outward_interval_slack",
           "outward_spec_rows", "outward_center_radius", "ns_bound",
           "ns_bounds_population"]
