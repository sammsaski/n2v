"""Soundness regressions for the last round of cleanup fixes.

- ``ImageStar.from_bounds`` and ``Zono.from_bounds`` contain the input box
  exactly (the round-to-nearest center and half-width missed an endpoint by an
  ulp in about half of all intervals).
- ``ImageStar.estimate_ranges`` / ``estimate_range`` without a predicate box
  return a rigorous enclosure instead of raw LP values or None.
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


@pytest.mark.parametrize("seed", range(5))
def test_imagestar_from_bounds_contains_the_box(seed):
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


@pytest.mark.parametrize("seed", range(5))
def test_zono_from_bounds_contains_the_box(seed):
    rng = np.random.default_rng(seed)
    lb, ub = _random_intervals(rng, 40)
    Z = Zono.from_bounds(lb, ub)
    c = Z.c.reshape(-1)
    r = np.abs(Z.V).sum(axis=1)
    assert _contains_exactly(c, r, lb, ub)


def test_imagestar_estimate_ranges_without_predicate_box_is_an_enclosure():
    # A 1x2x1 image over alpha in the unit square given only as C/d rows.
    V = np.zeros((1, 2, 1, 3))
    V[0, 0, 0] = [0.1, 1.0, 0.0]
    V[0, 1, 0] = [-0.3, 1.0, -2.0]
    C = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
    d = np.array([[1.0], [0.0], [1.0], [0.0]])
    S = ImageStar(V, C, d, None, None, 1, 2, 1)
    lb, ub = S.estimate_ranges()
    assert lb.shape == (2, 1) and np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
    # true ranges: pixel 0 in [0.1, 1.1], pixel 1 in [-2.3, 0.7]
    assert lb[0, 0] <= 0.1 <= 1.1 <= ub[0, 0]
    assert lb[1, 0] <= -2.3 and 0.7 <= ub[1, 0]
    lo, hi = S.estimate_range(1)
    assert isinstance(lo, float) and lo <= -2.3 and 0.7 <= hi


def test_box_reach_through_batchnorm_encloses_the_real_output():
    import mpmath
    from n2v.nn import NeuralNetwork

    mpmath.mp.prec = 200
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
    for x in points:
        R = NeuralNetwork(model).reach(Box(x, x), method='approx')
        out_lb = np.asarray(R[0].lb, dtype=np.float64).reshape(-1)
        out_ub = np.asarray(R[0].ub, dtype=np.float64).reshape(-1)
        for i in range(2):
            z = sum(mpmath.mpf(float(W[i, j])) * mpmath.mpf(float(x[j]))
                    for j in range(3)) + mpmath.mpf(float(b[i]))
            y = (mpmath.mpf(float(g[i])) * (z - mpmath.mpf(float(mu[i])))
                 / mpmath.sqrt(mpmath.mpf(float(var[i])) + mpmath.mpf(eps))
                 + mpmath.mpf(float(beta[i])))
            if not (mpmath.mpf(float(out_lb[i])) <= y <= mpmath.mpf(float(out_ub[i]))):
                misses += 1
    assert misses == 0
