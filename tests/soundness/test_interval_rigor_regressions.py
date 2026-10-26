"""Regressions for the interval enclosures of Star / ImageStar / Zono.

  * Every interval bound (``estimate_ranges``, ``estimate_range``,
    ``Zono.get_bounds`` / ``get_box`` / ``get_ranges`` / ``get_range``)
    encloses the exact rational interval, including infinite predicate boxes
    (no ``0 * inf = NaN``) and products that underflow.
  * Each direction is charged only for the rounding of the terms it sums, so
    an exactly computed bound stays exact: ReLU on a ``[0, 1]^n`` box sees a
    lower bound of exactly 0, not -1e-15 (which made every such neuron
    crossing and split exact reach into 2^n stars).
  * ``Zono.order_reduction_box`` rounds its interval-hull radius outward, so
    the reduced Zono contains the original.
  * ``estimate_ranges`` keeps its temporaries bounded (it runs on every ReLU
    layer; the full-size slack temporaries cost ~5x the generator matrix).
  * A Star / ImageStar built without C/d stores a ``(0, nVar)`` constraint
    matrix, so reach and ``verify_specification`` can stack rows onto it.
"""

import tracemalloc
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.box import Box
from n2v.sets.halfspace import HalfSpace
from n2v.sets.image_star import ImageStar
from n2v.sets.star import Star
from n2v.sets.zono import Zono
from n2v.utils.verify_specification import verify_specification


def _exact_interval(c, G, lb, ub):
    """Exact rational interval of ``c + G alpha`` over ``[lb, ub]``; an
    infinite endpoint met by a nonzero generator gives an infinite bound."""
    lo, hi = [], []
    for i in range(G.shape[0]):
        lo_i = hi_i = Fraction(float(c[i]))
        lo_inf = hi_inf = False
        for j in range(G.shape[1]):
            g = float(G[i, j])
            if g == 0.0:
                continue
            b_lo, b_hi = (lb[j], ub[j]) if g > 0 else (ub[j], lb[j])
            if np.isinf(b_lo):
                lo_inf = True
            else:
                lo_i += Fraction(g) * Fraction(float(b_lo))
            if np.isinf(b_hi):
                hi_inf = True
            else:
                hi_i += Fraction(g) * Fraction(float(b_hi))
        lo.append(None if lo_inf else lo_i)
        hi.append(None if hi_inf else hi_i)
    return lo, hi


def _assert_encloses(flo, fhi, elo, ehi, tag):
    for i, (a, b) in enumerate(zip(np.ravel(flo), np.ravel(fhi))):
        assert not (np.isnan(a) or np.isnan(b)), (tag, i)
        if elo[i] is None:
            assert a == -np.inf, (tag, i)
        else:
            assert a == -np.inf or Fraction(float(a)) <= elo[i], (tag, i, a)
        if ehi[i] is None:
            assert b == np.inf, (tag, i)
        else:
            assert b == np.inf or Fraction(float(b)) >= ehi[i], (tag, i, b)


def _random_case(rng, kind):
    n = int(rng.integers(1, 9))
    dim = int(rng.integers(1, 6))
    G = rng.standard_normal((dim, n)) * 10.0 ** rng.integers(-8, 9, (dim, n))
    G[rng.random((dim, n)) < 0.4] = 0.0
    c = rng.standard_normal(dim) * 10.0 ** rng.integers(-3, 4, dim)
    c[rng.random(dim) < 0.3] = 0.0
    if kind == "unit":
        lb, ub = -np.ones(n), np.ones(n)
    elif kind == "relax":
        lb, ub = np.zeros(n), rng.uniform(0.1, 3.0, n)
    elif kind == "general":
        lb, ub = -rng.uniform(0.1, 3.0, n), rng.uniform(0.1, 3.0, n)
    else:  # "inf": some unbounded predicates, some generators on them zero
        lb, ub = -rng.uniform(0.1, 3.0, n), rng.uniform(0.1, 3.0, n)
        lb[rng.random(n) < 0.3] = -np.inf
        ub[rng.random(n) < 0.3] = np.inf
    return c, G, lb, ub


# ---------------------------------------------------------------------------
# Rigor: enclosure of the exact interval
# ---------------------------------------------------------------------------

def _special_rows():
    """Rows aimed at the single-rounding and underflow paths."""
    return [
        # c + g rounds to c: the exact lower bound is 1 - 1e-17 < 1.
        (np.array([1.0]), np.array([[1e-17]]), -np.ones(1), np.ones(1)),
        # single inexact addition of exact products
        (np.array([0.1, 0.3]), np.array([[0.2], [-0.7]]), -np.ones(1),
         np.ones(1)),
        # products that underflow: the computed sum is 0, the exact one is not
        (np.array([0.0]), np.array([[1e-200, 1e-200]]),
         np.array([-1e-200, -3e-200]), np.array([1e-200, 3e-200])),
        (np.array([0.0]), np.array([[1e-200]]), np.array([-1e-200]),
         np.array([1e-200])),
        # heavy cancellation
        (np.array([1e16]), np.array([[1.0, -1e16, 3.0]]), -np.ones(3),
         np.ones(3)),
    ]


@pytest.mark.parametrize("kind", ["unit", "relax", "general", "inf"])
def test_star_and_imagestar_interval_encloses_exact(kind):
    rng = np.random.default_rng({"unit": 1, "relax": 2, "general": 3,
                                 "inf": 4}[kind])
    cases = [_random_case(rng, kind) for _ in range(40)]
    if kind == "general":
        cases += _special_rows()
    for t, (c, G, lb, ub) in enumerate(cases):
        V = np.hstack([c.reshape(-1, 1), G])
        n, dim = G.shape[1], G.shape[0]
        elo, ehi = _exact_interval(c, G, lb, ub)
        S = Star(V, np.zeros((0, n)), np.zeros((0, 1)), lb, ub)
        flo, fhi = S.estimate_ranges()
        _assert_encloses(flo, fhi, elo, ehi, (kind, t, "Star"))
        single = [S.estimate_range(i) for i in range(dim)]
        _assert_encloses([a for a, _ in single], [b for _, b in single],
                         elo, ehi, (kind, t, "Star.estimate_range"))
        I = ImageStar(V.reshape(1, 1, dim, n + 1), np.zeros((0, n)),
                      np.zeros((0, 1)), lb, ub)
        flo, fhi = I.estimate_ranges()
        _assert_encloses(flo, fhi, elo, ehi, (kind, t, "ImageStar"))
        single = [I.estimate_range(i) for i in range(dim)]
        _assert_encloses([a for a, _ in single], [b for _, b in single],
                         elo, ehi, (kind, t, "ImageStar.estimate_range"))


def test_zono_bounds_box_ranges_enclose_exact():
    """get_box / get_ranges / get_range used to skip the outward rounding
    that get_bounds had (they feed the trig and pow reach)."""
    rng = np.random.default_rng(5)
    cases = [_random_case(rng, "unit") for _ in range(60)]
    cases += [(c, G, lb, ub) for c, G, lb, ub in _special_rows()
              if np.all(np.abs(lb) == 1)]
    for t, (c, G, _, _) in enumerate(cases):
        n = G.shape[1]
        elo, ehi = _exact_interval(c, G, -np.ones(n), np.ones(n))
        Z = Zono(c.reshape(-1, 1), G)
        _assert_encloses(*Z.get_bounds(), elo, ehi, (t, "get_bounds"))
        box = Z.get_box()
        _assert_encloses(box.lb, box.ub, elo, ehi, (t, "get_box"))
        _assert_encloses(*Z.get_ranges(), elo, ehi, (t, "get_ranges"))
        single = [Z.get_range(i) for i in range(G.shape[0])]
        _assert_encloses([a for a, _ in single], [b for _, b in single],
                         elo, ehi, (t, "get_range"))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_zono_order_reduction_hull_encloses_removed_generators(dtype):
    """The hull radius was a round-to-nearest sum of |removed generators|,
    below the exact sum in about half the rows."""
    rng = np.random.default_rng(6)
    for t in range(300):
        V = rng.standard_normal((2, 12)) * 10.0 ** rng.integers(-3, 4, (2, 12))
        Z = Zono(rng.standard_normal((2, 1)).astype(dtype), V.astype(dtype))
        R = Z.order_reduction_box(3)
        assert R.V.dtype == dtype
        order = np.argsort(-np.linalg.norm(Z.V, axis=0))
        np.testing.assert_array_equal(R.V[:, :3], Z.V[:, order[:3]])
        removed = order[3:]
        radius = np.abs(R.V[:, 3:]).sum(axis=1)   # one nonzero per row
        for i in range(2):
            exact = sum(abs(Fraction(float(Z.V[i, j]))) for j in removed)
            assert Fraction(float(radius[i])) >= exact, (t, i)


def test_unbounded_predicate_does_not_poison_bounded_rows():
    """A zero generator on an infinite predicate contributes 0, not NaN, in
    every estimate (the single-row Star path and the ImageStar path used to
    return NaN, which a ReLU reads as stably active)."""
    V = np.array([[0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
    lb, ub = np.array([-1.0, -np.inf]), np.array([1.0, np.inf])
    S = Star(V, np.zeros((0, 2)), np.zeros((0, 1)), lb, ub)
    assert S.estimate_range(0) == (-0.5, 1.5)
    assert S.estimate_range(1) == (-np.inf, np.inf)
    I = ImageStar(V.reshape(1, 1, 2, 3), np.zeros((0, 2)), np.zeros((0, 1)),
                  lb, ub)
    ilb, iub = I.estimate_ranges()
    np.testing.assert_array_equal(ilb.ravel(), [-0.5, -np.inf])
    np.testing.assert_array_equal(iub.ravel(), [1.5, np.inf])
    assert I.estimate_range(0) == (-0.5, 1.5)


# ---------------------------------------------------------------------------
# Per-direction slack: exactly computed bounds stay exact
# ---------------------------------------------------------------------------

def test_unit_box_star_bounds_are_exact():
    """[0, 1]^n is 0.5 + 0.5 alpha: each bound is one exact addition, so it
    must come back as exactly 0 and 1 (a -1e-15 lower bound made every ReLU
    on it crossing)."""
    S = Star.from_bounds(np.zeros((6, 1)), np.ones((6, 1)))
    lb, ub = S.estimate_ranges()
    np.testing.assert_array_equal(lb, np.zeros((6, 1)))
    np.testing.assert_array_equal(ub, np.ones((6, 1)))
    lo = np.array([-1.0, 0.0, -2.0, 0.0, -0.75])
    hi = np.array([0.0, 1.0, 0.0, 3.0, 0.25])
    lb, ub = Box(lo, hi).to_star().estimate_ranges()
    np.testing.assert_array_equal(lb.ravel(), lo)
    np.testing.assert_array_equal(ub.ravel(), hi)
    I = ImageStar.from_bounds(np.zeros((2, 2, 1)), np.ones((2, 2, 1)), 2, 2, 1)
    lb, ub = I.estimate_ranges()
    np.testing.assert_array_equal(lb, np.zeros((4, 1)))
    np.testing.assert_array_equal(ub, np.ones((4, 1)))
    z_lb, z_ub = Zono.from_bounds(np.zeros(3), np.ones(3)).get_bounds()
    np.testing.assert_array_equal(z_lb, np.zeros((3, 1)))
    np.testing.assert_array_equal(z_ub, np.ones((3, 1)))


def test_each_direction_charged_only_for_its_own_terms():
    """Relaxation variables on [0, u]: every lower-bound product is an exact
    zero, so the lower bound is exactly the center even though the upper
    bound sums inexact products and is inflated."""
    V = np.array([[0.0, 0.3, 0.7, 0.0],
                  [0.25, 0.0, 0.0, 0.0],      # no generator: exact center
                  [0.0, 0.0, 0.0, 1.0]])      # identity on a relax variable
    ub_pred = np.array([0.3, 0.9, 0.7])
    S = Star(V, np.zeros((0, 3)), np.zeros((0, 1)), np.zeros(3), ub_pred)
    lb, ub = S.estimate_ranges()
    np.testing.assert_array_equal(lb.ravel(), [0.0, 0.25, 0.0])
    assert ub[0, 0] > 0.3 * 0.3 + 0.7 * 0.9          # inflated upward only
    assert ub[0, 0] - (0.3 * 0.3 + 0.7 * 0.9) < 1e-14
    assert ub[1, 0] == 0.25 and 0.7 <= ub[2, 0] < 0.7 + 1e-14


def test_relu_on_nonneg_box_is_one_star_in_approx_reach():
    """Two stacked ReLUs on a [0, 1] box: with exact zero lower bounds no
    neuron is crossing, so approx reach adds no relaxation variable."""
    model = nn.Sequential(nn.ReLU(), nn.ReLU()).double()
    S = Star.from_bounds(np.zeros((4, 1)), np.ones((4, 1)))
    out = reach_pytorch_model(model, S, "approx")
    assert len(out) == 1 and out[0].nVar == S.nVar


# ---------------------------------------------------------------------------
# Bounded temporaries
# ---------------------------------------------------------------------------

def test_estimate_ranges_temporaries_are_bounded():
    rng = np.random.default_rng(0)
    h, w, ch, n = 16, 16, 32, 2047
    V = rng.standard_normal((h, w, ch, n + 1))
    V[..., 1:][rng.random((h, w, ch, n)) < 0.3] = 0.0
    lb, ub = -np.ones(n), np.ones(n)
    I = ImageStar(V, np.zeros((0, n)), np.zeros((0, 1)), lb, ub)
    S = Star(V.reshape(-1, n + 1), np.zeros((0, n)), np.zeros((0, 1)), lb, ub)
    for X in (I, S):
        tracemalloc.start()
        try:
            flo, fhi = X.estimate_ranges()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The generator matrix is 134 MB; the positive/negative parts plus
        # the old full-size slack temporaries peaked at about 5x that.
        assert peak < 0.5 * V.nbytes, (type(X).__name__, peak, V.nbytes)
    # same enclosure as the plain round-to-nearest interval, gamma_k out
    G2, c = V[..., 1:].reshape(-1, n), V[..., 0].reshape(-1)
    rad = np.abs(G2) @ np.ones(n)
    assert np.all(flo.ravel() <= c - rad) and np.all(fhi.ravel() >= c + rad)
    assert np.max((c - rad) - flo.ravel()) < 1e-8


def test_transposed_imagestar_basis_matches_contiguous():
    """A basis whose leading axes do not merge into rows (a transposed
    ImageStar) is read block by block and gives the same bounds."""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((6, 5, 3, 9)).transpose(1, 0, 2, 3)
    lb, ub = -rng.uniform(0.1, 2.0, 8), rng.uniform(0.1, 2.0, 8)
    I = ImageStar(V, np.zeros((0, 8)), np.zeros((0, 1)), lb, ub)
    J = ImageStar(np.ascontiguousarray(V), np.zeros((0, 8)),
                  np.zeros((0, 1)), lb, ub)
    for a, b in zip(I.estimate_ranges(), J.estimate_ranges()):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Stars without C / d
# ---------------------------------------------------------------------------

def _small_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).double()


def test_star_without_constraints_supports_reach_and_verify():
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    S = Star(V, None, None, -np.ones(2), np.ones(2))
    assert S.C.shape == (0, 2) and S.d.shape == (0, 1)
    H = S.intersect_half_space(np.array([[1.0, 0.0]]), np.array([0.5]))
    assert H.C.shape == (1, 2)
    model = _small_net()
    ref = Star(V, np.zeros((0, 2)), np.zeros((0, 1)), -np.ones(2), np.ones(2))
    for method in ("exact", "approx"):
        out = reach_pytorch_model(model, S, method)
        assert len(out) == len(reach_pytorch_model(model, ref, method))
    spec = HalfSpace(np.array([[1.0, 0.0]]), np.array([-2.0]))
    assert (verify_specification([S], spec).verdict
            == verify_specification([ref], spec).verdict)
    # An empty 1-D C is normalized the same way.
    assert Star(V, np.array([]), None, -np.ones(2), np.ones(2)).C.shape == (0, 2)


def test_point_star_zero_column_constraints_are_kept():
    """A point star (nVar = 0) with live rows ``0 <= d_i`` keeps them."""
    V = np.array([[2.0], [5.0]])
    S = Star(V, np.empty((1, 0)), np.array([[-1.0]]),
             np.empty((0, 1)), np.empty((0, 1)))
    assert S.C.shape == (1, 0) and S.is_empty_set()


def test_imagestar_without_constraints_stores_nvar_columns():
    V = np.zeros((2, 2, 1, 3))
    V[..., 1] = 1.0
    I = ImageStar(V, None, None, -np.ones(2), np.ones(2))
    assert I.C.shape == (0, 2)
    S = I.to_star()
    assert S.intersect_half_space(np.ones((1, 4)), np.array([1.0])).C.shape == (1, 2)
