"""Soundness regressions for the last round of cleanup fixes.

- ``ImageStar.from_bounds`` and ``Zono.from_bounds`` contain the input box
  exactly (the round-to-nearest center and half-width missed an endpoint by an
  ulp in about half of all intervals).
- ``ImageStar.estimate_ranges`` / ``estimate_range`` without a predicate box
  return a rigorous enclosure instead of raw LP values or None, checked
  against the exact range of every entry in fractions.
- A Box reach through Linear -> BatchNorm keeps the BatchNorm unfused, so the
  output box encloses the real-number output (the float fold of a fused layer
  escaped the Box's outward rounding).
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, ImageStar, Zono
from tests._oracles.rational import (Membership, dot, matrix, star, vector,
                                    vertices)


def _random_intervals(rng, n):
    lb = rng.uniform(-10, 10, n) * 10.0 ** rng.integers(-8, 3, n)
    width = rng.uniform(0, 2, n) * 10.0 ** rng.integers(-12, 1, n)
    ub = lb + width
    return lb, ub


def _contains_exactly(c, r, lb, ub):
    for ci, ri, li, ui in zip(c, r, lb, ub):
        if Fraction(ci) - Fraction(ri) > Fraction(li):
            return False
        if Fraction(ci) + Fraction(ri) < Fraction(ui):
            return False
    return True


# Two seeds: each of the five seeds first used failed under each measured
# reverse (round-to-nearest centre and half-width, wrong radius, layout).
@pytest.mark.parametrize("seed", range(2))
def test_imagestar_from_bounds_contains_the_box(seed):
    """``ImageStar.from_bounds`` of a 2x3x4 box (bounds given flat in the
    stored height, width, channel order) over eleven orders of magnitude,
    every fifth pixel a point, holds the box exactly: each centre plus or
    minus its rounded-outward half-width covers the pixel's interval in
    fractions, and the two extreme corners of the box are members of the set
    by its own predicate box and rows, decided exactly. A point pixel stays
    its value with no generator. Catches the round-to-nearest centre and
    half-width, a half-width or centre read for another pixel, a basis
    reshaped in another layout and a predicate box or rows that do not span
    [-1, 1]."""
    rng = np.random.default_rng(seed)
    lb, ub = _random_intervals(rng, 2 * 3 * 4)
    lb[::5] = ub[::5]  # point dimensions stay constants
    S = ImageStar.from_bounds(lb, ub, 2, 3, 4)
    V = S.V.reshape(-1, S.V.shape[3])
    c = V[:, 0]
    r = np.abs(V[:, 1:]).sum(axis=1)  # one generator per perturbed pixel
    assert _contains_exactly(c, r, lb, ub)
    point = lb == ub
    assert np.array_equal(c[point], lb[point])
    assert np.all(r[point] == 0)
    assert S.nVar == int((~point).sum())
    member = Membership(star(S))
    assert member(lb) and member(ub)


@pytest.mark.parametrize("seed", range(2))
def test_zono_from_bounds_contains_the_box(seed):
    """``Zono.from_bounds`` of 40 intervals over eleven orders of magnitude
    holds the box exactly: centre plus or minus the rounded-outward
    half-width covers each interval in fractions. Catches the round-to-nearest
    centre and half-width and a half-width read for another entry."""
    rng = np.random.default_rng(seed)
    lb, ub = _random_intervals(rng, 40)
    Z = Zono.from_bounds(lb, ub)
    c = Z.c.reshape(-1)
    r = np.abs(Z.V).sum(axis=1)
    assert _contains_exactly(c, r, lb, ub)


def test_imagestar_estimate_ranges_without_predicate_box_is_an_enclosure():
    """``ImageStar.estimate_ranges`` and ``estimate_range`` of an image star
    given by its rows alone (no predicate box) return an enclosure: padded
    LP bounds over a finite box computed from the rows
    (``Star._lp_enclosure``), never raw LP values. Against the exact range
    of every stored entry (the vertices of the predicate polytope in
    fractions, each stored float read as the rational it is) every bound
    holds with zero slack, also from ``estimate_range`` for each entry. Two
    stars: a 1x2x1 image over the unit square, and a 1x2x2 image (two
    channels, stored height, width, channel) whose polytope has a coupling
    row, with an entry far above 1, a negative range and a constant entry.
    Catches the raw LP values (with the centre added in round to nearest,
    both bounds of pixel 1 of the first star lie 1 ulp inside), an upper
    bound read from the minimisation, a pad applied inward and a one-entry
    query that solves the LP of another entry."""
    # 1x2x1 over alpha in the unit square given only as C/d rows
    V = np.zeros((1, 2, 1, 3))
    V[0, 0, 0] = [0.1, 1.0, 0.0]
    V[0, 1, 0] = [-0.3, 1.0, -2.0]
    C = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
    d = np.array([[1.0], [0.0], [1.0], [0.0]])
    # 1x2x2 (stored h, w, c) over 0 <= a0 <= 3, -1 <= a1 <= 2, a0 + a1 <= 2.5
    V2 = np.zeros((1, 2, 2, 3))
    V2[0, 0, 0] = [0.75, 2.0, -0.5]
    V2[0, 0, 1] = [-1.25, 0.0, -3.0]
    V2[0, 1, 0] = [0.1, 1000.0, 1.0]
    V2[0, 1, 1] = [7.0, 0.0, 0.0]
    C2 = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0], [1.0, 1.0]])
    d2 = np.array([[3.0], [0.0], [2.0], [1.0], [2.5]])
    for Vs, Cs, ds, (h, w, c) in ((V, C, d, (1, 2, 1)),
                                  (V2, C2, d2, (1, 2, 2))):
        S = ImageStar(Vs, Cs, ds, None, None, h, w, c)
        verts = vertices(matrix(Cs), vector(np.ravel(ds)))
        lb, ub = S.estimate_ranges()
        assert lb.shape == (h * w * c, 1)
        assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
        for i, row in enumerate(Vs.reshape(-1, Vs.shape[-1])):
            values = [Fraction(float(row[0])) + dot(vector(row[1:]), v)
                      for v in verts]
            lo, hi = S.estimate_range(i)
            assert isinstance(lo, float) and isinstance(hi, float)
            for low, high in ((lb[i, 0], ub[i, 0]), (lo, hi)):
                assert Fraction(float(low)) <= min(values), (h, w, c, i)
                assert Fraction(float(high)) >= max(values), (h, w, c, i)


def test_box_reach_through_batchnorm_encloses_the_real_output():
    """A Box reach through Linear and BatchNorm1d keeps BatchNorm unfused
    (the float fold of a fused layer escapes the Box's outward rounding), so
    the output box holds the real-number output of the stored layers,
    computed in 200-bit arithmetic, at a zero point input (no rounding margin
    in the Box affine map) and 30 sampled points, with a pruned neuron, a
    negative BatchNorm scale and a small running variance."""
    import mpmath
    from n2v.nn import NeuralNetwork

    torch.manual_seed(0)
    lin = nn.Linear(3, 2)
    bn = nn.BatchNorm1d(2)
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([0.37, -1.21]))
        bn.running_var.copy_(torch.tensor([0.013, 3.7]))
        bn.weight.copy_(torch.tensor([1.9, -0.61]))
        bn.bias.copy_(torch.tensor([0.05, 2.3]))
    model = nn.Sequential(lin, bn).eval()

    W = lin.weight.detach().double().numpy()
    b = lin.bias.detach().double().numpy()
    mu = bn.running_mean.double().numpy()
    var = bn.running_var.double().numpy()
    g = bn.weight.detach().double().numpy()
    beta = bn.bias.detach().double().numpy()
    eps = bn.eps

    # A zero point input (and a pruned all-zero weight row) leaves the Box
    # affine map no rounding margin, so any rounding in pre-fused parameters
    # would show up as a miss.
    with torch.no_grad():
        lin.weight[1].zero_()
    W = lin.weight.detach().double().numpy()
    rng = np.random.default_rng(1)
    points = [np.zeros(3)] + [rng.uniform(-1, 1, 3) for _ in range(30)]
    misses = 0
    # a local precision: the global mpmath setting is shared with other tests
    with mpmath.workprec(200):
        for x in points:
            R = NeuralNetwork(model).reach(Box(x, x), method='approx')
            out_lb = np.asarray(R[0].lb, dtype=np.float64).reshape(-1)
            out_ub = np.asarray(R[0].ub, dtype=np.float64).reshape(-1)
            for i in range(2):
                z = sum(mpmath.mpf(float(W[i, j])) * mpmath.mpf(float(x[j]))
                        for j in range(3)) + mpmath.mpf(float(b[i]))
                y = (mpmath.mpf(float(g[i])) * (z - mpmath.mpf(float(mu[i])))
                     / mpmath.sqrt(mpmath.mpf(float(var[i]))
                                   + mpmath.mpf(eps))
                     + mpmath.mpf(float(beta[i])))
                if not (mpmath.mpf(float(out_lb[i])) <= y
                        <= mpmath.mpf(float(out_ub[i]))):
                    misses += 1
    assert misses == 0
