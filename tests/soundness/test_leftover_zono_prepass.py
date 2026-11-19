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

The second half checks every recorded neuron on inputs where exactly one
share of the allowance is needed: the input centre and the absolute value
of a weight in a Linear, the pass-through of Identity and Dropout, one
half-subnormal per underflowing product, the (height, width, channel) row
order of the convolved allowance, and the move of the rows through a
Flatten (torch module, function, ONNX), on the sequential walker, the
traced graph walker and ONNX files. A recorded bound moves one ulp
outward for any positive allowance, so a share only shows where the row
cancels to about 0 (or is a point whose allowance would be exactly 0).
Known defect marked here: ROUND-11 on routes its registry row does not
list (a traced ``x.flatten(1)`` or ``x.view``, and the ``nn.Sequential``
that ``load_onnx`` builds for a Conv with unequal pads).
"""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from n2v.sets import Box, ImageStar, Star, Zono
from n2v.utils.bounds_precomputation import (
    _convert_to_zono, _zono_sequential, compute_intermediate_bounds,
)
from tests.known_bugs import known_bug


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
    within an ulp of 0: the recorded bounds contain it (one-sided, no
    slack; how close they stay is not checked, tightness is not a promise
    of the suite). Catches an input zonotope or row bounds that miss an
    end of the box, a Linear allowance with a signed weight, and bounds
    recorded inward of the zonotope's own range."""
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


# --------------------------------------------------------------------------- #
# Rounding allowance of single terms, every recorded neuron, both walkers      #
# --------------------------------------------------------------------------- #
# a = 1 + 2^-30: a * a = 1 + 2^-29 + 2^-60 is not a float64 (it rounds to
# 1 + 2^-29), and every other number below is exact in float64.
_A = 1.0 + 2.0 ** -30
_E60 = 2.0 ** -60


def _target(net, walker):
    return fx.symbolic_trace(net) if walker == 'traced' else net


def _assert_recorded_hold(pre, exact, what):
    """``exact`` maps a recorded key to the exact values (Fractions) its
    layer input takes, one list per member point, in the recorded order;
    every recorded interval holds every value, with no slack."""
    assert set(pre) == set(exact), (what, sorted(map(str, pre)))
    for key, points in exact.items():
        lo, hi = pre[key]
        lo = [_q(v) for v in lo.ravel()]
        hi = [_q(v) for v in hi.ravel()]
        for values in points:
            out = [i for i, v in enumerate(values)
                   if not lo[i] <= v <= hi[i]]
            assert not out, (
                f"{what}: layer {key}, entries {out}: real "
                f"{[float(values[i]) for i in out]}, recorded "
                f"{[(float(lo[i]), float(hi[i])) for i in out]}")


@pytest.mark.parametrize('walker', ['sequential', 'traced'])
@pytest.mark.parametrize('kind', ['Box', 'Star', 'Zono'])
def test_centre_and_signed_shares_of_a_linear_layer_are_enclosed(kind,
                                                                  walker):
    """Linear(2, 2), Identity, Dropout (eval: the identity), ReLU on x0 = a
    fixed and x1 in [-1/4, 1/4]. Row 0 is -a x0 with no bias: its real
    value -(1 + 2^-29 + 2^-60) is not a float64 and the row has no
    generator, so the computed row is a point and only an allowance that
    counts the input's centre, with the absolute value of the negative
    weight, moves the recorded bounds off it. Row 1 is x1 - 1, exact,
    whose range [-5/4, -3/4] the recorded interval must keep holding: the
    bias is large next to the terms. The two pass-through layers must
    carry the allowance to the ReLU.

    Catches an allowance that leaves out the input centre, that takes the
    signed weight instead of its absolute value, or the signed bias (a
    negative weight or bias then moves the bounds inward), and one that is
    dropped at a layer that passes its input sets through."""
    lin = nn.Linear(2, 2, dtype=torch.float64)
    _set_params(lin, [[-_A, 0], [0, 1]], [0, -1])
    net = nn.Sequential(lin, nn.Identity(), nn.Dropout(0.5),
                        nn.ReLU()).double().eval()
    lo, hi = np.array([_A, -0.25]), np.array([_A, 0.25])
    S = {'Box': Box(lo, hi),
         'Star': Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1)),
         'Zono': Zono.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))}[kind]
    pre = compute_intermediate_bounds(_target(net, walker), S, method='zono')
    key = 3 if walker == 'sequential' else '_3'
    exact = [[-_q(_A) * _q(_A), x1 - 1] for x1 in (F(-1, 4), F(1, 4))]
    _assert_recorded_hold(pre, {key: exact}, f"{kind}, {walker} walker")


@pytest.mark.parametrize('walker', ['sequential', 'traced'])
@pytest.mark.parametrize('kind', ['Box', 'Star'])
def test_underflowing_products_of_every_generator_are_enclosed(kind,
                                                               walker):
    """Linear(16, 32) of ones, then Linear(32, 1) of weights 2^-538, then
    ReLU, on the box [-2^-537, 2^-537]^16. In the second layer every
    product of an input generator with a weight is 2^-1075, half the
    smallest subnormal, and rounds to 0: each of the 16 generator rows of
    the zonotope loses 32 such products and comes out 0, while the real
    pre-activation reaches +-16 * 32 * 2^-1075 = +-256 subnormals. (With
    one half-subnormal per inner product the allowance is 4 * 17 plus the
    relative share of about 4 * 32, 196 subnormals; fewer than nine
    generators would hide that.)

    Catches an underflow allowance of one half-subnormal per inner product
    instead of one per product that can underflow."""
    l1 = nn.Linear(16, 32, bias=False, dtype=torch.float64)
    l2 = nn.Linear(32, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        l1.weight.fill_(1.0)
        l2.weight.fill_(2.0 ** -538)
    net = nn.Sequential(l1, l2, nn.ReLU()).double()
    lo, hi = np.full(16, -2.0 ** -537), np.full(16, 2.0 ** -537)
    S = (Box(lo, hi) if kind == 'Box'
         else Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1)))
    target = _target(net, walker)
    pre = compute_intermediate_bounds(target, S, method='zono')
    key = 2 if walker == 'sequential' else '_2'
    reach = 16 * 32 * F(2) ** -1075
    _assert_recorded_hold(pre, {key: [[-reach], [reach]]},
                          f"{kind}, {walker} walker")


def _cancel_net(flatten=None):
    """Conv2d(2 -> 2, 1x1, no bias) on a (2, 1, 2) image, then (``flatten``
    given) a Flatten spelled as an ``nn.Flatten`` module (``'module'``),
    ``torch.flatten`` (``'function'``), ``x.flatten(1)`` (``'method'``) or
    ``x.view(x.size(0), -1)`` (``'view'``), then ReLU. Output channel 1 is
    ``-a u + (1 + 2^-29) v``; at pixel 0 the input is fixed at (u, v) =
    (a, 1), where the float product a * a drops 2^-60 and the row computes
    exactly 0 while the real value is -2^-60. At pixel 1, u is free in
    [-1, 1] and v is 0. Output channel 0 has weights 2^-40, so its
    allowance is negligible. In (height, width, channel) order the
    cancelling row is row 1 and in (channel, height, width) order row 2;
    row 1 of the second order and row 2 of the first are channel 0 at pixel
    1: an allowance read in the wrong order misses. Without a Flatten or
    with the module it is an ``nn.Sequential``, so that the sequential
    walker, which walks the children, walks the model's own function."""
    conv = nn.Conv2d(2, 2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(
            _CANCEL_K, dtype=torch.float64).reshape(2, 2, 1, 1))
    if flatten in (None, 'module'):
        middle = [nn.Flatten()] if flatten else []
        return nn.Sequential(conv, *middle, nn.ReLU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.act = conv, nn.ReLU()

        def forward(self, x):
            h = self.conv(x)
            if flatten == 'function':
                h = torch.flatten(h, 1)
            elif flatten == 'method':
                h = h.flatten(1)
            elif flatten == 'view':
                h = h.view(h.size(0), -1)
            return self.act(h)
    return Net()


_CANCEL_K = [[2.0 ** -40, 2.0 ** -40], [-_A, 1.0 + 2.0 ** -29]]
# (C, H, W) bounds of the input image: pixel 0 fixed at (a, 1), pixel 1
# with u in [-1, 1] and v = 0.
_CANCEL_LO = np.array([[[_A, -1.0]], [[1.0, 0.0]]])
_CANCEL_HI = np.array([[[_A, 1.0]], [[1.0, 0.0]]])


def _cancel_exact(flatten):
    """The exact ReLU input at u = -1, 0, 1 of pixel 1, in the recorded
    order: (height, width, channel) without a Flatten, (channel, height,
    width) with one."""
    K = [[_q(w) for w in row] for row in _CANCEL_K]
    out = []
    for u in (F(-1), F(0), F(1)):
        pixels = [(_q(_A), F(1)), (u, F(0))]
        chw = [[K[k][0] * x[0] + K[k][1] * x[1] for x in pixels]
               for k in range(2)]
        out.append([chw[k][j] for k in range(2) for j in range(2)]
                   if flatten else
                   [chw[k][j] for j in range(2) for k in range(2)])
    return out


def _cancel_input(kind):
    from n2v.sets.image_zono import ImageZono
    cls = ImageStar if kind == 'ImageStar' else ImageZono
    return cls.from_bounds(_CANCEL_LO.transpose(1, 2, 0),
                           _CANCEL_HI.transpose(1, 2, 0),
                           height=1, width=2, num_channels=2)


def _cancel_onnx(tmp_path, flatten, pads=(0, 0, 0, 0)):
    """The model of :func:`_cancel_net` as a float64 ONNX file (Conv,
    optionally Flatten, Relu). The Flatten has axis 2, which ``load_onnx``
    builds as an ``OnnxFlatten`` (axis 1 becomes a plain ``nn.Flatten``);
    on the (1, 2, 1, 2) tensor it gives (2, 2), the same row order as a
    flatten from axis 1. Unequal ``pads`` make ``load_onnx`` build an
    ``nn.Sequential`` of a pad and the convolution."""
    from onnx import TensorProto, helper
    from n2v.utils.model_loader import load_onnx
    from tests.cell_support import save_onnx
    nodes = [helper.make_node('Conv', ['X', 'K'], ['c'], kernel_shape=[1, 1],
                              pads=list(pads))]
    prev = 'c'
    if flatten:
        nodes.append(helper.make_node('Flatten', ['c'], ['f'], axis=2))
        prev = 'f'
    nodes.append(helper.make_node('Relu', [prev], ['Y']))
    width = 2 + pads[1] + pads[3]
    out_shape = [2, width] if flatten else [1, 2, 1, width]
    path = save_onnx(tmp_path, nodes, [1, 2, 1, 2], out_shape,
                     {'K': np.array(_CANCEL_K).reshape(2, 2, 1, 1)},
                     in_type=TensorProto.DOUBLE, out_type=TensorProto.DOUBLE)
    return load_onnx(path)


def _relu_key(model):
    """The node name of the one ReLU of a graph."""
    named = dict(model.named_modules())
    keys = [n.name for n in model.graph.nodes if n.op == 'call_module'
            and isinstance(named[n.target], nn.ReLU)]
    assert len(keys) == 1, keys
    return keys[0]


@pytest.mark.parametrize('walker', ['sequential', 'traced', 'onnx'])
@pytest.mark.parametrize('flatten', [None, 'module'])
def test_conv_and_flatten_allowances_follow_the_rows(flatten, walker,
                                                     tmp_path):
    """The model of :func:`_cancel_net` without and with an ``nn.Flatten``
    (for ``onnx`` the ONNX Conv, Flatten with axis 2 and Relu, loaded as
    Conv2d, OnnxFlatten and ReLU), on an ImageStar and an ImageZono: the ReLU's
    recorded interval holds the real value of every row at every member
    point, in (height, width, channel) order behind the convolution and
    (channel, height, width) order behind the Flatten. The cancelling row
    computes 0 where the real value is -2^-60, so only its own allowance
    covers it.

    Catches the convolved allowance read in (channel, height, width) order
    for the zonotope's (height, width, channel) rows, an allowance through
    a Flatten kept per row while the rows move, an OnnxFlatten that is not
    recognised as a Flatten, and a convolution allowance without the input
    centre."""
    for kind in ('ImageStar', 'ImageZono'):
        if walker == 'onnx':
            target = _cancel_onnx(tmp_path, flatten)
            key = _relu_key(target)
        else:
            target = _target(_cancel_net(flatten), walker)
            key = 2 if flatten else 1
            key = key if walker == 'sequential' else f'_{key}'
        pre = compute_intermediate_bounds(target, _cancel_input(kind),
                                          method='zono',
                                          input_shape=(2, 1, 2))
        _assert_recorded_hold(pre, {key: _cancel_exact(flatten)},
                              f"Flatten {flatten}, {walker}, {kind}")


def _round11(route):
    return known_bug(
        7, f"ROUND-11: the zonotope pre-pass carries no rounding allowance "
           f"through {route}, which the graph walk runs without reporting "
           f"it as a layer, and records the ReLU behind it unwidened: the "
           f"real input -2^-60 lies below the recorded [0, 0]")


@pytest.mark.parametrize('flatten', [
    pytest.param('function', id='torch.flatten'),
    pytest.param('method', id='x.flatten', marks=_round11(
        "a traced tensor method x.flatten(1)")),
    pytest.param('view', id='x.view', marks=_round11(
        "a traced x.view(x.size(0), -1)")),
])
def test_traced_flatten_spellings_carry_the_allowance(flatten):
    """:func:`_cancel_net` traced, its Flatten spelled as ``torch.flatten``
    (a function node the walk reports as a Flatten; the passing neighbour,
    with the ``nn.Flatten`` module of the cell above), as the tensor method
    ``x.flatten(1)`` or as ``x.view(x.size(0), -1)``: the ReLU's recorded
    interval holds the real value of every row. The module docstring
    promises the allowance along Flatten layers.

    Catches a Flatten that the walk runs without carrying the allowance
    (registry ROUND-11 for the method spellings)."""
    target = fx.symbolic_trace(_cancel_net(flatten))
    pre = compute_intermediate_bounds(target, _cancel_input('ImageStar'),
                                      method='zono', input_shape=(2, 1, 2))
    if 'act' not in pre:
        pytest.fail(f"{flatten}: nothing recorded for the ReLU")
    _assert_recorded_hold(pre, {'act': _cancel_exact(flatten)},
                          f"Flatten spelled {flatten}")


@pytest.mark.parametrize('pads', [
    pytest.param((0, 0, 0, 0), id='equal-pads-conv'),
    pytest.param((0, 0, 0, 1), id='unequal-pads-sequential',
                 marks=_round11("the nn.Sequential (pad, Conv2d) that "
                                "load_onnx builds for a Conv with unequal "
                                "pads")),
])
def test_onnx_conv_with_padding_carries_the_allowance(pads, tmp_path):
    """The convolution of :func:`_cancel_net` as an ONNX file, then Relu:
    with equal pads it loads as a Conv2d node (the passing neighbour); with
    unequal pads as an ``nn.Sequential`` of a pad and the Conv2d, a node
    the error step has no rule for. The recorded interval of the cancelling
    row (pixel 0, channel 1; the pad adds a zero pixel at the end, so the
    row keeps its index) holds its real value -2^-60.

    Catches the allowance lost behind a container node (registry
    ROUND-11)."""
    model = _cancel_onnx(tmp_path, None, pads)
    pre = compute_intermediate_bounds(model, _cancel_input('ImageStar'),
                                      method='zono', input_shape=(2, 1, 2))
    key = _relu_key(model)
    if key not in pre:
        pytest.fail("nothing recorded for the Relu")
    lo, hi = pre[key]
    real = _cancel_exact(None)[0][1]
    assert _q(lo[1, 0]) <= real <= _q(hi[1, 0]), (
        pads, float(lo[1, 0]), float(hi[1, 0]), float(real))
