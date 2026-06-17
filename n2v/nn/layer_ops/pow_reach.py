"""
Pow(x, p) reachability for a constant positive-integer exponent p.

All Pow occurrences in the VNN-COMP corpus are ``x ** c`` with a constant
scalar exponent (p = 2 for ml4acopf / collins / smart_turn, p = 3 for
nn4sys pensieve); the base is always a computed set. There is no
``base ** x`` (exponential) form.

Soundness (per neuron, LP bounds [l, u], one fresh predicate z):

  * f(x) = x^p convex on [l, u]  (p even, or p odd with l >= 0):
      lower: two tangent lines at l and u   (f convex => above its tangents)
      upper: the secant from (l, f(l)) to (u, f(u))
  * f concave on [l, u]  (p odd with u <= 0):
      the mirror — tangents above, secant below
  * p odd with l < 0 < u  (inflection at 0, neither convex nor concave):
      sound interval box  z in [l^p, u^p]  with no affine link (looser;
      tightening by splitting at 0 is future work)
  * constant neuron (l == u): z = l^p exactly.

The fresh predicate is always bounded by the exact range of x^p over
[l, u], which for even p with a sign-spanning interval is [0, max].
"""

import numpy as np
from typing import List, Optional

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def _validate_p(p) -> int:
    pf = float(p)
    if not pf.is_integer() or pf < 0:
        raise NotImplementedError(
            f"Pow exponent {p} not supported (only non-negative integers)")
    return int(pf)


def _range_of_power(l: float, u: float, p: int):
    """Exact [min, max] of x^p over [l, u]."""
    fl, fu = l ** p, u ** p
    if p % 2 == 0:
        if l < 0.0 < u:
            return 0.0, max(fl, fu)
        return min(fl, fu), max(fl, fu)
    # odd p is monotonic increasing
    return fl, fu


# --------------------------------------------------------------------------
# Star
# --------------------------------------------------------------------------

def pow_star(input_stars: List, p, lp_solver: str = 'default') -> List:
    p = _validate_p(p)
    out = []
    for s in input_stars:
        star = s.to_star() if isinstance(s, ImageStar) else s
        res = _pow_single_star(star, p, lp_solver)
        if isinstance(s, ImageStar):
            res = res.to_image_star(s.height, s.width, s.num_channels)
        out.append(res)
    return out


def _pow_single_star(I: Star, p: int, lp_solver: str) -> Optional[Star]:
    if I is None or I.dim == 0:
        return I
    N, n = I.dim, I.nVar

    lbs = np.zeros(N)
    ubs = np.zeros(N)
    for i in range(N):
        lo, hi = I.get_range(i, lp_solver)
        if lo is None or hi is None:
            raise ValueError(
                f"LP returned None for dim {i} of Pow input (infeasible?)")
        lbs[i], ubs[i] = lo, hi

    const_mask = np.abs(ubs - lbs) < 1e-12
    vary = np.where(~const_mask)[0]
    m = len(vary)

    V1 = np.zeros((N, n + 1))
    V1[const_mask, 0] = lbs[const_mask] ** p
    V2 = np.zeros((N, m))
    V2[vary, np.arange(m)] = 1.0
    new_V = np.hstack([V1, V2])

    C_old = (np.asarray(I.C, dtype=np.float64).reshape(-1, n)
             if np.asarray(I.C).size else np.zeros((0, n)))
    d_old = (np.asarray(I.d, dtype=np.float64).reshape(-1, 1)
             if np.asarray(I.d).size else np.zeros((0, 1)))
    rows, rhs = [], []
    if C_old.shape[0]:
        rows.append(np.hstack([C_old, np.zeros((C_old.shape[0], m))]))
        rhs.append(d_old)

    z_lb = np.zeros(m)
    z_ub = np.zeros(m)
    even = (p % 2 == 0)

    for j, idx in enumerate(vary):
        l, u = lbs[idx], ubs[idx]
        Vrow = I.V[idx, 1:n + 1]
        cval = I.V[idx, 0]
        fl, fu = l ** p, u ** p
        dfl, dfu = p * l ** (p - 1), p * u ** (p - 1)
        secant = (fu - fl) / (u - l)
        z_lb[j], z_ub[j] = _range_of_power(l, u, p)

        if even or l >= 0:            # convex on [l, u]
            for a, fa, dfa in ((l, fl, dfl), (u, fu, dfu)):
                # z >= f(a) + f'(a)(x - a)
                r = np.zeros(n + m)
                r[:n] = dfa * Vrow
                r[n + j] = -1.0
                rows_append(rows, rhs, r, dfa * a - fa - dfa * cval)
            # z <= secant
            r = np.zeros(n + m)
            r[:n] = -secant * Vrow
            r[n + j] = 1.0
            rows_append(rows, rhs, r, fl - secant * l + secant * cval)
        elif u <= 0:                  # concave on [l, u]
            for a, fa, dfa in ((l, fl, dfl), (u, fu, dfu)):
                # z <= f(a) + f'(a)(x - a)
                r = np.zeros(n + m)
                r[:n] = -dfa * Vrow
                r[n + j] = 1.0
                rows_append(rows, rhs, r, fa - dfa * a + dfa * cval)
            # z >= secant
            r = np.zeros(n + m)
            r[:n] = secant * Vrow
            r[n + j] = -1.0
            rows_append(rows, rhs, r, secant * l - fl - secant * cval)
        # else: odd p, l < 0 < u -> box only (z bounded below by z_lb/z_ub)

    new_C = np.vstack(rows) if rows else np.zeros((0, n + m))
    new_d = np.vstack(rhs) if rhs else np.zeros((0, 1))

    if I.predicate_lb is not None:
        plb = np.asarray(I.predicate_lb, dtype=np.float64).reshape(-1, 1)
        pub = np.asarray(I.predicate_ub, dtype=np.float64).reshape(-1, 1)
    else:
        plb = np.full((n, 1), -np.inf)
        pub = np.full((n, 1), np.inf)
    new_plb = np.vstack([plb, z_lb.reshape(-1, 1)])
    new_pub = np.vstack([pub, z_ub.reshape(-1, 1)])
    return Star(new_V, new_C, new_d, new_plb, new_pub)


def rows_append(rows, rhs, row, val):
    rows.append(row.reshape(1, -1))
    rhs.append(np.array([[val]], dtype=np.float64))


# --------------------------------------------------------------------------
# Zono / Box — sound interval enclosure
# --------------------------------------------------------------------------

def pow_zono(input_zonos: List, p) -> List:
    p = _validate_p(p)
    out = []
    for z in input_zonos:
        lo, hi = z.get_ranges()
        lo = np.asarray(lo, dtype=np.float64).flatten()
        hi = np.asarray(hi, dtype=np.float64).flatten()
        zlo = np.empty_like(lo)
        zhi = np.empty_like(hi)
        for i in range(lo.size):
            zlo[i], zhi[i] = _range_of_power(lo[i], hi[i], p)
        c = ((zlo + zhi) / 2.0).reshape(-1, 1)
        rad = np.diag((zhi - zlo) / 2.0)
        if isinstance(z, ImageZono):
            out.append(ImageZono(c, rad, z.height, z.width, z.num_channels))
        else:
            out.append(Zono(c, rad))
    return out


def pow_box(input_boxes: List, p) -> List:
    p = _validate_p(p)
    out = []
    for b in input_boxes:
        lo = np.asarray(b.lb, dtype=np.float64).flatten()
        hi = np.asarray(b.ub, dtype=np.float64).flatten()
        zlo = np.empty_like(lo)
        zhi = np.empty_like(hi)
        for i in range(lo.size):
            zlo[i], zhi[i] = _range_of_power(lo[i], hi[i], p)
        out.append(Box(zlo, zhi))
    return out
