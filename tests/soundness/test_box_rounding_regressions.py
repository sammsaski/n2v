"""
Regression tests for the outward rounding of Box reach and of the shared
directed-rounding helpers (n2v/utils/ns_certificate.py).

Each test is derived from a repro in which a Box missed the exact output of
the real-number network, or a bound was lost for no reason:

- ONNX Div by a constant scaled a Box by the reciprocal of the constant.
  The reciprocal was rounded to nearest and only the product was rounded
  outward, so the Box could miss the exact quotient (a false UNSAT).
- BatchNorm on an ``(N, C, ...)`` flat layout sent a Box through
  ``affine_map(diag(scale), shift)``, which is rigorous for the rounded
  scale and shift but not for the rounding committed while folding them.
  When ``beta`` is close to ``scale * mean`` that rounding is much larger
  than the map's own (a false UNSAT).
- The Box ``set @ set`` MatMul used the midpoint-radius interval matmul in
  round-to-nearest with no rounding margin.
- ``_add_dir`` / ``_mul_dir`` / ``_div_dir`` returned a finite operation that
  overflowed as an exact infinity, so a lower bound came back as ``+inf``;
  with two float32 operands they rounded in float32 but moved one float64
  ulp.
- BatchNorm and ReduceSum/Mean on a Box charged both bounds for the larger
  endpoint magnitude, so a half-infinite Box lost its finite side and exact
  zero bounds moved.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.layer_ops import batchnorm_reach, reduce_reach
from n2v.nn.layer_ops.bilinear_matmul_reach import bilinear_matmul_box
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, HalfSpace
from n2v.utils.ns_certificate import _add_dir, _div_dir, _mul_dir
from n2v.utils.verify_specification import verify_specification


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node(op, inputs, outputs, **attrs):
    from onnx import helper
    return helper.make_node(op, inputs, outputs, **attrs)


def _onnx_graph_module(nodes, in_shape, out_shape, initializers=()):
    """Build an ONNX model from nodes and convert it with onnx2torch."""
    import onnx
    onnx2torch = pytest.importorskip('onnx2torch')
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph(
        nodes, 'g',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(out_shape))],
        initializer=[numpy_helper.from_array(np.asarray(a, dtype=np.float32), n)
                     for n, a in initializers])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model)
    return onnx2torch.convert(model).eval()


def _touching_halfspace(dim, i, exact, lo, hi):
    """A halfspace on output ``i`` that contains ``exact`` on the side where
    the Box ``[lo, hi]`` misses it (or on the upper side if it does not)."""
    G = np.zeros((1, dim))
    if exact > Fraction(hi):
        t = float(exact)
        if Fraction(t) > exact:
            t = float(np.nextafter(t, -np.inf))
        G[0, i] = -1.0                      # y_i >= t
        return HalfSpace(G, np.array([[-t]]))
    t = float(exact)
    if Fraction(t) < exact:
        t = float(np.nextafter(t, np.inf))
    G[0, i] = 1.0                           # y_i <= t
    return HalfSpace(G, np.array([[t]]))


def _bn_exact(x, mu, var, gamma, beta, eps):
    """The real-number BatchNorm output, to 60 significant digits."""
    with localcontext() as ctx:
        ctx.prec = 60
        s = Decimal(gamma) / (Decimal(var) + Decimal(eps)).sqrt()
        y = s * (Decimal(x) - Decimal(mu)) + Decimal(beta)
    return Fraction(y)


class _BN1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return self.bn(x)


# ---------------------------------------------------------------------------
# ONNX Div by a constant
# ---------------------------------------------------------------------------

def test_box_div_by_constant_does_not_give_a_false_unsat():
    # y = x / c - d with d the float32 nearest x / c: the old Box was
    # x * fl(1/c) rounded outward and missed the exact output by ~1e-18.
    x, c = 1.391450047492981, 7.420313358306885
    assert float(np.float32(x)) == x and float(np.float32(c)) == c
    exact_q = Fraction(x) / Fraction(c)
    d = float(np.float32(float(exact_q)))
    model = _onnx_graph_module(
        [_node('Div', ['X', 'c'], ['Q']), _node('Sub', ['Q', 'd'], ['Y'])],
        [1, 1], [1, 1], initializers=[('c', [c]), ('d', [d])])
    out = reach_pytorch_model(model, Box(np.array([x]), np.array([x])),
                              method='approx')
    exact = exact_q - Fraction(d)
    lo, hi = float(out[0].lb[0, 0]), float(out[0].ub[0, 0])
    assert Fraction(lo) <= exact <= Fraction(hi)
    spec = _touching_halfspace(1, 0, exact, lo, hi)
    assert verify_specification(out, spec).verdict != "UNSAT"


def test_box_div_by_constant_encloses_every_exact_quotient():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(-2.0, 2.0, n).astype(np.float32).astype(np.float64)
    c = (rng.uniform(1.0, 20.0, n) * rng.choice([-1.0, 1.0], n)) \
        .astype(np.float32).astype(np.float64)
    model = _onnx_graph_module([_node('Div', ['X', 'c'], ['Y'])],
                               [1, n], [1, n], initializers=[('c', c)])
    out = reach_pytorch_model(model, Box(x, x), method='approx')[0]
    lo, hi = out.lb.ravel(), out.ub.ravel()
    misses = sum(not (Fraction(float(lo[i])) <= Fraction(float(x[i])) / Fraction(float(c[i]))
                      <= Fraction(float(hi[i])))
                 for i in range(n))
    assert misses == 0


# ---------------------------------------------------------------------------
# BatchNorm on an (N, C, ...) layout
# ---------------------------------------------------------------------------

def _bn_model(mu, var, gamma, beta):
    m = _BN1d(len(mu)).eval()
    with torch.no_grad():
        m.bn.running_mean.copy_(torch.tensor(mu, dtype=torch.float32))
        m.bn.running_var.copy_(torch.tensor(var, dtype=torch.float32))
        m.bn.weight.copy_(torch.tensor(gamma, dtype=torch.float32))
        m.bn.bias.copy_(torch.tensor(beta, dtype=torch.float32))
    return m


def _bn_params(m):
    return [t.detach().double().numpy() for t in
            (m.bn.running_mean, m.bn.running_var, m.bn.weight, m.bn.bias)]


def test_box_batchnorm_on_nchw_layout_does_not_give_a_false_unsat():
    m = _bn_model([500.0, 300.0], [3.0, 5.0], [1.5, 0.75],
                  [433.0127, 100.6231])
    mu, var, gamma, beta = _bn_params(m)
    x = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = reach_pytorch_model(m, Box(x, x), method='approx',
                              input_shape=(1, 2, 3))
    exact = _bn_exact(x[0], mu[0], var[0], gamma[0], beta[0], m.bn.eps)
    lo, hi = float(out[0].lb[0, 0]), float(out[0].ub[0, 0])
    assert Fraction(lo) <= exact <= Fraction(hi)
    spec = _touching_halfspace(6, 0, exact, lo, hi)
    assert verify_specification(out, spec).verdict != "UNSAT"


def test_box_batchnorm_on_nchw_layout_encloses_the_real_layer():
    # beta = scale * mean: the folded shift cancels, and its rounding is far
    # larger than the rounding of the map itself.
    rng = np.random.default_rng(0)
    C, L = 2, 3
    misses = total = 0
    for _ in range(20):
        mu = rng.uniform(100, 1000, C)
        var = rng.uniform(0.1, 10, C)
        gamma = rng.uniform(0.5, 2, C)
        m = _bn_model(mu, var, gamma, np.zeros(C))
        mu, var, gamma, _ = _bn_params(m)
        with torch.no_grad():
            m.bn.bias.copy_(torch.tensor(
                gamma * mu / np.sqrt(var + m.bn.eps), dtype=torch.float32))
        beta = m.bn.bias.detach().double().numpy()
        x = rng.uniform(-1e-3, 1e-3, C * L)
        out = reach_pytorch_model(m, Box(x, x), method='approx',
                                  input_shape=(1, C, L))[0]
        for i in range(C * L):
            ch = i // L                      # row-major (1, C, L)
            exact = _bn_exact(x[i], mu[ch], var[ch], gamma[ch], beta[ch],
                              m.bn.eps)
            total += 1
            misses += not (Fraction(float(out.lb[i, 0])) <= exact
                           <= Fraction(float(out.ub[i, 0])))
    assert total == 120 and misses == 0


# ---------------------------------------------------------------------------
# Box set @ set MatMul
# ---------------------------------------------------------------------------

def test_bilinear_matmul_box_encloses_the_exact_interval_product():
    # Point case: fl(2^53 + 1 + 1) = 2^53, the exact product is 2^53 + 2.
    out = bilinear_matmul_box([Box([2.0 ** 53, 1, 1], [2.0 ** 53, 1, 1])],
                              [Box([1.0, 1, 1], [1.0, 1, 1])], (1, 3), (3, 1))[0]
    assert Fraction(float(out.lb[0, 0])) <= 2 ** 53 + 2 <= Fraction(float(out.ub[0, 0]))

    rng = np.random.default_rng(0)
    misses = 0
    for _ in range(300):
        k = int(rng.integers(2, 8))
        A = rng.standard_normal(k) * 10.0 ** rng.integers(-3, 4, k)
        B = rng.standard_normal(k)
        wA = np.abs(rng.standard_normal(k)) * 1e-3 * (rng.random(k) < 0.5)
        wB = np.abs(rng.standard_normal(k)) * 1e-3 * (rng.random(k) < 0.5)
        La, Ua, Lb, Ub = A - wA, A + wA, B - wB, B + wB
        out = bilinear_matmul_box([Box(La, Ua)], [Box(Lb, Ub)],
                                  (1, k), (k, 1))[0]
        prods = [[Fraction(float(a)) * Fraction(float(b))
                  for a in (La[j], Ua[j]) for b in (Lb[j], Ub[j])]
                 for j in range(k)]
        lo = sum(min(p) for p in prods)
        hi = sum(max(p) for p in prods)
        misses += not (Fraction(float(out.lb[0, 0])) <= lo
                       and Fraction(float(out.ub[0, 0])) >= hi)
    assert misses == 0


# ---------------------------------------------------------------------------
# Directed-rounding helpers
# ---------------------------------------------------------------------------

def _below(v, exact):
    """``v <= exact`` for a float bound that may be infinite."""
    return v == -np.inf or (v != np.inf and Fraction(v) <= exact)


def _above(v, exact):
    """``v >= exact`` for a float bound that may be infinite."""
    return v == np.inf or (v != -np.inf and Fraction(v) >= exact)


@pytest.mark.parametrize("op, a, b, exact", [
    (_add_dir, 1e308, 1e308, Fraction(1e308) * 2),
    (_add_dir, -1e308, -1e308, Fraction(-1e308) * 2),
    (_mul_dir, 1e200, 1e200, Fraction(1e200) ** 2),
    (_mul_dir, -1e200, 1e200, -Fraction(1e200) ** 2),
    (_div_dir, 1e300, 1e-100, Fraction(1e300) / Fraction(1e-100)),
])
def test_directed_rounding_of_an_overflow_is_an_outward_bound(op, a, b, exact):
    lo = float(op(np.array([a]), np.array([b]), -1.0)[0])
    hi = float(op(np.array([a]), np.array([b]), 1.0)[0])
    assert _below(lo, exact) and _above(hi, exact)
    # The side away from the overflow keeps the largest finite double.
    assert np.isfinite(lo) or np.isfinite(hi)


def test_box_sum_that_overflows_keeps_an_outward_lower_bound():
    s = Box([1e308], [1e308]).minkowski_sum(Box([1e308], [1e308]))
    exact = Fraction(1e308) * 2
    assert _below(float(s.lb[0, 0]), exact) and s.ub[0, 0] == np.inf


def test_directed_rounding_of_float32_operands_is_outward():
    a32, b32 = np.float32(1.0), np.float32(2.0 ** -30)
    exact = Fraction(1) + Fraction(2) ** -30
    assert Fraction(float(_add_dir(a32, b32, 1.0))) >= exact
    assert Fraction(float(_add_dir(a32, -b32, -1.0))) <= 1 - Fraction(2) ** -30
    x32, y32 = np.float32(1.1), np.float32(3.3)
    prod = Fraction(float(x32)) * Fraction(float(y32))
    assert Fraction(float(_mul_dir(x32, y32, -1.0))) <= prod \
        <= Fraction(float(_mul_dir(x32, y32, 1.0)))
    quot = Fraction(float(x32)) / Fraction(float(y32))
    assert Fraction(float(_div_dir(x32, y32, -1.0))) <= quot \
        <= Fraction(float(_div_dir(x32, y32, 1.0)))


# ---------------------------------------------------------------------------
# Half-infinite boxes and exact zeros through BatchNorm and Reduce
# ---------------------------------------------------------------------------

def test_half_infinite_box_keeps_its_finite_side_through_batchnorm():
    box = Box([0.0, 0.0], [np.inf, 1.0])
    out = batchnorm_reach.batchnorm_box(nn.BatchNorm1d(2).eval(), [box])[0]
    # Default BN (mean 0, beta 0): the lower bounds are exact zeros.
    assert np.array_equal(out.lb.ravel(), [0.0, 0.0])
    assert out.ub[0, 0] == np.inf
    assert Fraction(float(out.ub[1, 0])) >= _bn_exact(1.0, 0.0, 1.0, 1.0, 0.0, 1e-5)

    bn = nn.BatchNorm1d(1).eval()
    with torch.no_grad():
        bn.running_mean.fill_(0.5)
        bn.bias.fill_(1.0)
    out = batchnorm_reach.batchnorm_box(bn, [Box([-np.inf], [2.0])])[0]
    assert out.lb[0, 0] == -np.inf
    assert Fraction(float(out.ub[0, 0])) >= _bn_exact(2.0, 0.5, 1.0, 1.0, 1.0, bn.eps)


@pytest.mark.parametrize("op", ['sum', 'mean'])
def test_half_infinite_box_keeps_its_finite_side_through_reduce(op):
    out = reduce_reach._reduce_flat_box(Box([0.0, 0.0], [np.inf, 1.0]),
                                        None, op, True)
    assert out.lb[0, 0] == 0.0 and out.ub[0, 0] == np.inf
    out = reduce_reach._reduce_flat_box(Box([-np.inf, -0.1], [0.0, 0.0]),
                                        None, op, True)
    assert out.lb[0, 0] == -np.inf and out.ub[0, 0] == 0.0
