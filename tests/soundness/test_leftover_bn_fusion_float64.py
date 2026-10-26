"""Regressions for BatchNorm fusion before sound reach
(n2v/utils/model_preprocessing.py).

reach_pytorch_model fuses every Conv2d/Linear + BatchNorm pair before a sound
reach. The folded weight and bias were rounded to the layer's float32, which
moved the fused network about 6e-8 relative from the original: a Box reach of
Linear -> BatchNorm1d from a point missed the real-number output in every
coordinate, by up to 1.5e-6, and the Star, Zono and ImageStar reaches missed
it by the same amount. The fold is now computed and kept in float64.

The checks compare the reach bounds of a point input with the exact output of
the original float32 network (Decimal at 60 digits; the only inexact step is
the square root). Box reach is rounded outward for the parameters it is
given, not for the rounding of a fold, so reach_pytorch_model keeps BatchNorm
unfused for a Box input (batchnorm_box bounds its own fold rounding; see the
pruned-neuron test). The Star and Zono
affine maps round to nearest in float64 by design, so for them the check
allows float64-level error, far below the old miss.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_preprocessing import fuse_batchnorm


def _randomize(bn, rng):
    with torch.no_grad():
        n = bn.num_features
        bn.running_mean.copy_(torch.tensor(rng.uniform(-5, 5, n)))
        bn.running_var.copy_(torch.tensor(rng.uniform(0.1, 5, n)))
        bn.weight.copy_(torch.tensor(rng.uniform(0.5, 2, n)))
        bn.bias.copy_(torch.tensor(rng.uniform(-1, 1, n)))
    return bn


def _exact_affine_bn(W, b, bn, x):
    """Exact ``BN(W x + b)`` of the float32 parameters, as Fractions."""
    p = lambda t: [Decimal(float(v)) for v in t.detach().double().reshape(-1)]  # noqa: E731
    W = np.asarray(W.detach().double(), dtype=np.float64).reshape(len(b), -1)
    b, g, be, mu, var = p(b), p(bn.weight), p(bn.bias), p(bn.running_mean), p(bn.running_var)
    out = []
    with localcontext() as ctx:
        ctx.prec = 60
        for i in range(len(b)):
            h = sum(Decimal(float(W[i, j])) * Decimal(float(x[j])) for j in range(W.shape[1]))
            s = g[i] / (var[i] + Decimal(bn.eps)).sqrt()
            out.append(Fraction(s * (h + b[i] - mu[i]) + be[i]))
    return out


def _worst_miss(lb, ub, y):
    lb, ub = np.asarray(lb).reshape(-1), np.asarray(ub).reshape(-1)
    return max(float(max(Fraction(float(lb[i])) - yi, yi - Fraction(float(ub[i])), 0))
               for i, yi in enumerate(y))


def test_fused_linear_bn_box_contains_the_real_output():
    # The reviewer's repro: every coordinate was missed, by up to 1.5e-6.
    rng = np.random.default_rng(0)
    for _ in range(40):
        lin, bn = nn.Linear(3, 2), _randomize(nn.BatchNorm1d(2), rng)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(rng.uniform(-1, 1, (2, 3))))
            lin.bias.copy_(torch.tensor(rng.uniform(-1, 1, 2)))
        model = nn.Sequential(lin, bn).eval()
        x = rng.uniform(-1, 1, 3)
        out = reach_pytorch_model(model, Box(x.reshape(-1, 1), x.reshape(-1, 1)),
                                  method='approx')[0]
        y = _exact_affine_bn(lin.weight, lin.bias, bn, x)
        assert _worst_miss(out.lb, out.ub, y) == 0


def test_fused_star_zono_imagestar_are_float64_close():
    rng = np.random.default_rng(1)
    for _ in range(10):
        lin, bn1 = nn.Linear(3, 2), _randomize(nn.BatchNorm1d(2), rng)
        flat = nn.Sequential(lin, bn1).eval()
        x = rng.uniform(-1, 1, 3)
        y = _exact_affine_bn(lin.weight, lin.bias, bn1, x)
        xc = x.reshape(-1, 1)
        for s in (Star.from_bounds(xc, xc), Zono(xc, np.zeros((3, 0)))):
            out = reach_pytorch_model(flat, s, method='approx')[0]
            lb, ub = out.get_bounds() if isinstance(out, Zono) else out.estimate_ranges()
            assert _worst_miss(lb, ub, y) < 1e-12, type(s).__name__

        # A 3x3 conv on a 3x3 image is one dot product per output channel.
        conv, bn2 = nn.Conv2d(1, 2, 3), _randomize(nn.BatchNorm2d(2), rng)
        model = nn.Sequential(conv, bn2, nn.Flatten()).eval()
        img = rng.uniform(-1, 1, (3, 3))
        S = ImageStar.from_bounds(img[:, :, None], img[:, :, None],
                                  height=3, width=3, num_channels=1)
        out = reach_pytorch_model(model, S, method='approx')[0]
        y = _exact_affine_bn(conv.weight, conv.bias, bn2, img.reshape(-1))
        assert _worst_miss(*out.estimate_ranges(), y) < 1e-12


def test_fused_model_keeps_the_callers_dtype_and_model():
    # The float64 fused layers still take and return the network's float32,
    # and fusing leaves the caller's model untouched.
    rng = np.random.default_rng(2)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), _randomize(nn.BatchNorm2d(2), rng),
                          nn.ReLU(), nn.Flatten(), nn.Linear(18, 3),
                          _randomize(nn.BatchNorm1d(3), rng)).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    fused = fuse_batchnorm(model)
    assert fused[0].weight.dtype == torch.float64
    x = torch.randn(4, 1, 5, 5)
    with torch.no_grad():
        assert fused(x).dtype == torch.float32
        torch.testing.assert_close(fused(x), model(x))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_fused_box_contains_a_pruned_neurons_real_output():
    # An all-zero weight row (a pruned neuron) before a BatchNorm: a fused
    # layer would map any Box to its rounded bias as a single point, which
    # cannot contain the irrational real value. reach_pytorch_model therefore
    # keeps BatchNorm unfused for a Box input (batchnorm_box bounds its own
    # fold rounding).
    rng = np.random.default_rng(3)
    lin, bn = nn.Linear(4, 3), _randomize(nn.BatchNorm1d(3), rng)
    with torch.no_grad():
        lin.weight[0].zero_()
    model = nn.Sequential(lin, bn).eval()
    c = rng.uniform(-1, 1, (4, 1))
    out = reach_pytorch_model(model, Box(c - 0.1, c + 0.1), method='approx')[0]
    y = _exact_affine_bn(lin.weight, lin.bias, bn, c.reshape(-1))
    assert _worst_miss(out.lb[:1], out.ub[:1], y[:1]) == 0


def test_fused_model_scripts_and_float_drops_the_casts():
    # fuse_batchnorm is public: its result must pass torch.jit.script (the
    # cast hooks need type annotations), and after .float() a trace records
    # no casts, so a torch or ONNX use of the model runs in float32.
    rng = np.random.default_rng(4)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), _randomize(nn.BatchNorm2d(2), rng),
                          nn.ReLU()).eval()
    x = torch.randn(2, 1, 5, 5)
    fused = fuse_batchnorm(model)
    with torch.no_grad():
        torch.testing.assert_close(torch.jit.script(fused)(x), model(x))
        traced = torch.jit.trace(fused.float(), x)
    assert fused[0].weight.dtype == torch.float32
    assert not any(n.kind() == 'aten::to' for n in traced.graph.nodes())
