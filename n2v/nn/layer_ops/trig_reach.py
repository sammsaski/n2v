"""
Sin / Cos reachability (sound relaxation over an arbitrary interval).

Both appear in ml4acopf as unary activations on bus phase-angle
differences. They are bounded in [-1, 1], smooth, and 2*pi-periodic, so
a relaxation must account for the interior extrema.

Per neuron with LP bounds [l, u] (fresh predicate z):

  * exact range [zmin, zmax] over [l, u] is found from the endpoints
    PLUS any interior critical point (sin: pi/2 + k*pi; cos: k*pi),
    where the function touches +/-1. This is what keeps the box sound
    across extrema.
  * if [l, u] contains no interior critical point AND no interior
    inflection point (sin inflects at k*pi; cos at pi/2 + k*pi), the
    function is monotonic and single-curvature there:
        f'' = -f  for both sin and cos, so it is convex where f < 0 and
        concave where f > 0. Use two tangent bounds + one secant bound
        (tight).
  * otherwise: the sound interval box (fresh z in [zmin, zmax], no
    affine link). Tightening by splitting at the interior critical
    points is future work.
  * constant neuron (l == u): z = f(l) exactly.
"""

import numpy as np
from typing import List, Optional

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

_PI = np.pi
_TOL = 1e-12


def _grid_in(l: float, u: float, offset: float, period: float = _PI):
    """Interior points  offset + k*period  strictly within (l, u)."""
    kmin = int(np.ceil((l - offset) / period))
    kmax = int(np.floor((u - offset) / period))
    pts = [offset + k * period for k in range(kmin, kmax + 1)]
    return [p for p in pts if l + _TOL < p < u - _TOL]


def _spec(kind):
    if kind == 'sin':
        return np.sin, np.cos, _PI / 2.0, 0.0      # func, dfunc, crit, infl
    return np.cos, (lambda x: -np.sin(x)), 0.0, _PI / 2.0


def _trig_range(func, l, u, crit_off):
    cps = _grid_in(l, u, crit_off)
    cand = [func(l), func(u)] + [func(c) for c in cps]
    return float(min(cand)), float(max(cand))


# --------------------------------------------------------------------------
# Star
# --------------------------------------------------------------------------

def trig_star(input_stars: List, kind: str, lp_solver: str = 'default') -> List:
    out = []
    for s in input_stars:
        star = s.to_star() if isinstance(s, ImageStar) else s
        res = _trig_single_star(star, kind, lp_solver)
        if isinstance(s, ImageStar):
            res = res.to_image_star(s.height, s.width, s.num_channels)
        out.append(res)
    return out


def _trig_single_star(I: Star, kind: str, lp_solver: str) -> Optional[Star]:
    if I is None or I.dim == 0:
        return I
    func, dfunc, crit_off, infl_off = _spec(kind)
    N, n = I.dim, I.nVar

    lbs = np.zeros(N)
    ubs = np.zeros(N)
    for i in range(N):
        lo, hi = I.get_range(i, lp_solver)
        if lo is None or hi is None:
            raise ValueError(
                f"LP returned None for dim {i} of {kind} input (infeasible?)")
        lbs[i], ubs[i] = lo, hi

    const_mask = np.abs(ubs - lbs) < 1e-12
    vary = np.where(~const_mask)[0]
    m = len(vary)

    V1 = np.zeros((N, n + 1))
    V1[const_mask, 0] = func(lbs[const_mask])
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

    for j, idx in enumerate(vary):
        l, u = lbs[idx], ubs[idx]
        Vrow = I.V[idx, 1:n + 1]
        cval = I.V[idx, 0]
        z_lb[j], z_ub[j] = _trig_range(func, l, u, crit_off)

        crit_inside = bool(_grid_in(l, u, crit_off))
        infl_inside = bool(_grid_in(l, u, infl_off))
        if crit_inside or infl_inside:
            continue   # box-only: z bounded by the exact range above

        fl, fu = func(l), func(u)
        dfl, dfu = dfunc(l), dfunc(u)
        secant = (fu - fl) / (u - l)
        # f'' = -f ; convex where f(mid) < 0
        convex = func((l + u) / 2.0) < 0.0

        if convex:
            for a, fa, dfa in ((l, fl, dfl), (u, fu, dfu)):
                r = np.zeros(n + m)          # z >= f(a) + f'(a)(x-a)
                r[:n] = dfa * Vrow
                r[n + j] = -1.0
                _row(rows, rhs, r, dfa * a - fa - dfa * cval)
            r = np.zeros(n + m)              # z <= secant
            r[:n] = -secant * Vrow
            r[n + j] = 1.0
            _row(rows, rhs, r, fl - secant * l + secant * cval)
        else:                               # concave
            for a, fa, dfa in ((l, fl, dfl), (u, fu, dfu)):
                r = np.zeros(n + m)          # z <= f(a) + f'(a)(x-a)
                r[:n] = -dfa * Vrow
                r[n + j] = 1.0
                _row(rows, rhs, r, fa - dfa * a + dfa * cval)
            r = np.zeros(n + m)              # z >= secant
            r[:n] = secant * Vrow
            r[n + j] = -1.0
            _row(rows, rhs, r, secant * l - fl - secant * cval)

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


def _row(rows, rhs, row, val):
    rows.append(row.reshape(1, -1))
    rhs.append(np.array([[val]], dtype=np.float64))


# --------------------------------------------------------------------------
# Zono / Box — sound interval enclosure
# --------------------------------------------------------------------------

def trig_zono(input_zonos: List, kind: str) -> List:
    func, _, crit_off, _ = _spec(kind)
    out = []
    for z in input_zonos:
        lo, hi = z.get_ranges()
        lo = np.asarray(lo, dtype=np.float64).flatten()
        hi = np.asarray(hi, dtype=np.float64).flatten()
        zlo = np.empty_like(lo)
        zhi = np.empty_like(hi)
        for i in range(lo.size):
            zlo[i], zhi[i] = _trig_range(func, lo[i], hi[i], crit_off)
        c = ((zlo + zhi) / 2.0).reshape(-1, 1)
        rad = np.diag((zhi - zlo) / 2.0)
        if isinstance(z, ImageZono):
            out.append(ImageZono(c, rad, z.height, z.width, z.num_channels))
        else:
            out.append(Zono(c, rad))
    return out


def trig_box(input_boxes: List, kind: str) -> List:
    func, _, crit_off, _ = _spec(kind)
    out = []
    for b in input_boxes:
        lo = np.asarray(b.lb, dtype=np.float64).flatten()
        hi = np.asarray(b.ub, dtype=np.float64).flatten()
        zlo = np.empty_like(lo)
        zhi = np.empty_like(hi)
        for i in range(lo.size):
            zlo[i], zhi[i] = _trig_range(func, lo[i], hi[i], crit_off)
        out.append(Box(zlo, zhi))
    return out
