"""Zonotope bound pre-pass: the recorded bounds enclose the real-number range.

With precompute_bounds='zono' the pre-pass recorded the bounds of zonotopes
computed in round-to-nearest, which could sit an ulp inside the exact range
of a ReLU input (about 1 in 8 random Linear -> ReLU cases whose range ends
near 0). The pass now carries a rigorous bound on the distance between its
zonotopes and the real-number sets through Linear, Conv2d, ReLU and Flatten
layers and moves the recorded bounds out by it. The exact ranges below are
computed in rational arithmetic. Each single-component case below fails
when that component of the bound is removed; the deep-network case checks
that the bound stays small where the real network zeroes a neuron.
"""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from n2v.sets import Box, ImageStar, Star
from n2v.utils.bounds_precomputation import (
    _convert_to_zono, _zono_sequential, compute_intermediate_bounds,
)


def _q(v):
    return F(float(v))


def _dot(w, x):
    return sum(_q(a) * b for a, b in zip(w, x))


def _last_relu_bounds(model, input_set, **kwargs):
    """Recorded (lo, hi) of the last ReLU input, as rationals."""
    pre = compute_intermediate_bounds(model, input_set, method='zono',
                                      **kwargs)
    lo, hi = pre[list(pre)[-1]]
    return _q(lo[0, 0]), _q(hi[0, 0])


def _set_params(layer, weight, bias):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(np.asarray(weight, dtype=np.float64)))
        layer.bias.copy_(torch.tensor(np.asarray(bias, dtype=np.float64)))


@pytest.mark.parametrize('traced', [False, True])
def test_linear_relu_bounds_enclose_exact_range(traced):
    """Linear(n, 1) -> ReLU over a box, bias set so the exact range ends
    within an ulp of 0: the recorded bounds contain it and stay tight."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 5))
        lb = np.round(rng.uniform(-1, 1, n), int(rng.integers(1, 4)))
        ub = lb + np.round(rng.uniform(0.01, 1, n), int(rng.integers(1, 4)))
        w = np.round(rng.uniform(-2, 2, n), int(rng.integers(1, 4)))
        hi_t = sum(max(_q(a) * _q(l), _q(a) * _q(u)) for a, l, u in zip(w, lb, ub))
        lo_t = sum(min(_q(a) * _q(l), _q(a) * _q(u)) for a, l, u in zip(w, lb, ub))
        b = -float(hi_t) if rng.integers(0, 2) == 0 else -float(lo_t)
        net = nn.Sequential(nn.Linear(n, 1), nn.ReLU()).double()
        _set_params(net[0], [w], [b])
        model = fx.symbolic_trace(net) if traced else net
        lo, hi = _last_relu_bounds(
            model, Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)))
        lo_t, hi_t = lo_t + _q(b), hi_t + _q(b)
        assert lo <= lo_t and hi >= hi_t, (lb, ub, w, b)
        assert hi - hi_t <= 1e-12 and lo_t - lo <= 1e-12


# Linear -> ReLU -> Linear -> ReLU with the hidden neurons stably active
# (bias 5), so the second pre-activation is affine in x and its exact range
# is attained at box corners; the second bias puts one end within an ulp
# of 0. Each case missed its exact range before the fix.
_MLP_CASES = [
    dict(lb=[-0.968], ub=[-0.16799999999999993],
         W1=[[1.72], [-1.74], [1.37], [-1.73]],
         W2=[[-0.623, -0.279, 1.864, 0.249]], b2=-5.79681592),
    dict(lb=[-0.77, -0.19], ub=[-0.77, 0.51],
         W1=[[-1.444, 0.815], [1.284, 1.927]], W2=[[1.4, -0.3]],
         b2=-7.246284999999999),
    dict(lb=[0.65], ub=[1.237], W1=[[-1.8], [0.8], [0.3]],
         W2=[[0.129, 1.253, 1.988]], b2=-18.540484199999998),
]


@pytest.mark.parametrize('traced', [False, True])
@pytest.mark.parametrize('case', range(len(_MLP_CASES)))
def test_two_layer_bounds_enclose_exact_range(case, traced):
    c = _MLP_CASES[case]
    W1, W2 = np.array(c['W1']), np.array(c['W2'])
    b1 = np.full(W1.shape[0], 5.0)
    vals = []
    for x in itertools.product(*zip(map(_q, c['lb']), map(_q, c['ub']))):
        h = [max(_dot(row, x) + _q(bi), F(0)) for row, bi in zip(W1, b1)]
        vals.append(_dot(W2[0], h) + _q(c['b2']))
    net = nn.Sequential(nn.Linear(W1.shape[1], W1.shape[0]), nn.ReLU(),
                        nn.Linear(W1.shape[0], 1), nn.ReLU()).double()
    _set_params(net[0], W1, b1)
    _set_params(net[2], W2, [c['b2']])
    model = fx.symbolic_trace(net) if traced else net
    lb, ub = np.array(c['lb']), np.array(c['ub'])
    lo, hi = _last_relu_bounds(
        model, Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)))
    assert lo <= min(vals) and hi >= max(vals)


# Conv2d(1 -> 2, 2x2) on a 2x2 image -> ReLU -> Flatten -> Linear(2, 1) ->
# ReLU, conv channels stably active (bias 9), ImageStar input. Each case
# missed its exact range before the fix.
_CNN_CASES = [
    dict(lb=[0.53, -0.58, -0.79, 0.09],
         ub=[0.94, -0.29999999999999993, -0.76, 0.11],
         K=[-0.638, -0.507, -1.824, -0.622, 1.709, 0.045, -0.639, 1.057],
         W3=[1.039, 0.419], b3=-15.15695918),
    dict(lb=[-0.82, -0.76, -0.86, -0.04],
         ub=[-0.58, 0.020000000000000018, -0.8, 0.37],
         K=[-0.2, -0.1, 0.2, -0.4, 1.0, 0.2, -0.8, 0.4],
         W3=[1.6, 0.7], b3=-20.4468),
]


@pytest.mark.parametrize('case', range(len(_CNN_CASES)))
def test_conv_flatten_bounds_enclose_exact_range(case):
    c = _CNN_CASES[case]
    K = np.array(c['K']).reshape(2, 1, 2, 2)
    kb = np.full(2, 9.0)
    vals = []
    for x in itertools.product(*zip(map(_q, c['lb']), map(_q, c['ub']))):
        h = [max(_dot(K[o].ravel(), x) + _q(kb[o]), F(0)) for o in range(2)]
        vals.append(_dot(c['W3'], h) + _q(c['b3']))
    net = nn.Sequential(nn.Conv2d(1, 2, 2), nn.ReLU(), nn.Flatten(),
                        nn.Linear(2, 1), nn.ReLU()).double()
    _set_params(net[0], K, kb)
    _set_params(net[3], [c['W3']], [c['b3']])
    lb = np.array(c['lb']).reshape(2, 2, 1)
    ub = np.array(c['ub']).reshape(2, 2, 1)
    S = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)
    lo, hi = _last_relu_bounds(fx.symbolic_trace(net), S,
                               input_shape=(1, 2, 2))
    assert lo <= min(vals) and hi >= max(vals)


def _scalar_chain(params):
    """Sequential of Linear(1, 1) (for each ``(w, b)``, None for a ReLU)
    and a final ReLU, in float64."""
    layers = []
    for p in params:
        if p is None:
            layers.append(nn.ReLU())
        else:
            lin = nn.Linear(1, 1, dtype=torch.float64)
            _set_params(lin, [[p[0]]], [p[1]])
            layers.append(lin)
    return nn.Sequential(*layers, nn.ReLU()).double()


def _chain_range(params, lb, ub):
    """Exact range of the input of the final ReLU of :func:`_scalar_chain`
    over ``[lb, ub]`` (every op is monotone on a scalar)."""
    lo, hi = _q(lb), _q(ub)
    for p in params:
        if p is None:
            lo, hi = max(lo, F(0)), max(hi, F(0))
        else:
            w, b = _q(p[0]), _q(p[1])
            lo, hi = min(w * lo, w * hi) + b, max(w * lo, w * hi) + b
    return lo, hi


# (params, lb, ub): each misses its exact range when one part of the
# carried error is dropped. ReLU on the input: the rounding of the input's
# conversion to a zonotope. A cancelling 1e8 weight: the error carried
# through a Linear. A dyadic box (converted exactly): the Linear's own
# rounding (the exact minimum is -3.5e-18; without that term the recorded
# lower bound is +2.5e-17). A hidden ReLU the computed set calls inactive
# while its real input reaches +1e-17, times a 1.8e9 weight: the part of a
# zeroed row's error that the real input can exceed 0 by.
_CHAIN_CASES = [
    ([], 0.08, 1.01),
    ([(1e8, -1.3e8), (1.0, -70.00000000090267)], 1.3, 1.3000007),
    ([(-0.736, -0.0345)], -0.90625, -0.046875),
    ([(-0.41, -0.35464999999999997), None,
      (1773139205.3048937, -1.76581623451899e-08)],
     -0.865, -0.04200000000000004),
]


@pytest.mark.parametrize('traced', [False, True])
@pytest.mark.parametrize('case', range(len(_CHAIN_CASES)))
def test_scalar_chain_bounds_enclose_exact_range(case, traced):
    params, lb, ub = _CHAIN_CASES[case]
    net = _scalar_chain(params)
    model = fx.symbolic_trace(net) if traced else net
    lo_t, hi_t = _chain_range(params, lb, ub)
    lo, hi = _last_relu_bounds(model, Star.from_bounds(np.array([[lb]]),
                                                       np.array([[ub]])))
    assert lo <= lo_t and hi >= hi_t


_ETA = 2.0 ** -1074


@pytest.mark.parametrize('kind', ['box', 'star', 'image'])
def test_subnormal_products_are_enclosed(kind):
    """0.3 times an input in [-eta, eta] (eta the smallest subnormal)
    underflows to 0 in the reach; the exact range is +-0.3 eta."""
    if kind == 'image':
        net = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU()).double()
        _set_params(net[0], [[[[0.3]]]], [0.0])
        lb, ub = np.full((1, 1, 1), -_ETA), np.full((1, 1, 1), _ETA)
        S = ImageStar.from_bounds(lb, ub, height=1, width=1, num_channels=1)
        lo, hi = _last_relu_bounds(fx.symbolic_trace(net), S,
                                   input_shape=(1, 1, 1))
    else:
        net = nn.Sequential(nn.Linear(1, 1), nn.ReLU()).double()
        _set_params(net[0], [[0.3]], [0.0])
        lb, ub = np.array([[-_ETA]]), np.array([[_ETA]])
        S = (Box(lb.ravel(), ub.ravel()) if kind == 'box'
             else Star.from_bounds(lb, ub))
        lo, hi = _last_relu_bounds(net, S)
    assert lo <= -F(3, 10) * F(_ETA) and hi >= F(3, 10) * F(_ETA)


def test_deep_net_margin_stays_small():
    """A 20-layer random MLP whose neurons are all stable on a small box.
    The error bound must not grow on rows the ReLU zeroes: before, |W|
    multiplied it at every layer, the margin reached 8e-2 and 12 stable
    neurons became crossing. The recorded bounds must stay within 1e-5 of
    the computed zonotopes' own bounds and keep every stable neuron."""
    g = torch.Generator().manual_seed(0)
    layers = [nn.Linear(20, 64), nn.ReLU()]
    for _ in range(19):
        layers += [nn.Linear(64, 64), nn.ReLU()]
    net = nn.Sequential(*layers).double()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.normal_(0, 0.5 * (2 / m.in_features) ** 0.5,
                                 generator=g)
                m.bias.normal_(0, 0.3, generator=g)
    x0 = np.linspace(0, 1, 20).reshape(-1, 1)
    S = Star.from_bounds(x0 - 1e-3, x0 + 1e-3)
    pre = compute_intermediate_bounds(net, S, method='zono')
    raw = _zono_sequential(net, _convert_to_zono(S)[0], None)
    assert pre.keys() == raw.keys()
    for k in pre:
        assert np.all(pre[k][0] <= raw[k][0]) and np.all(pre[k][1] >= raw[k][1])
        assert np.max(raw[k][0] - pre[k][0]) <= 1e-5
        assert np.max(pre[k][1] - raw[k][1]) <= 1e-5
        stable = (raw[k][0] >= 0) | (raw[k][1] <= 0)
        assert np.all((pre[k][0] >= 0) | (pre[k][1] <= 0) | ~stable)
