"""The bounds pre-pass: are the recorded neuron bounds outward, at every layer,
in the layout the reach reads them in, and is reach with the pre-pass on still
sound?

The pre-pass (``n2v.utils.bounds_precomputation.compute_intermediate_bounds``,
switched on in reach by ``precompute_bounds='ibp'``, ``'zono'`` or ``True``)
records, for every nonlinear layer (ReLU, LeakyReLU, Sigmoid, Tanh), an
interval per neuron of the layer's input. The approximate ReLU and LeakyReLU
kernels use those intervals as given, and the exact ones use them to skip
neurons they call stable, so an interval that misses one real pre-activation
of one member point is a lost piece or a cut-off output (a false UNSAT).

Two kinds of cell:

- Direct: call the pre-pass on a model and an input set, then for member
  points (corners, the quarter grid, points on a binding non-box row)
  compute the real value of every recorded neuron and check that it lies in
  the recorded interval. The model walkers checked are the plain
  ``nn.Sequential`` walker (direct calls only), the traced ``GraphModule``
  walker (what reach uses for a torch model) and the ONNX ``GraphModule``
  walker (``load_onnx``).
- End to end: reach with the pre-pass on, every true output a member of the
  returned sets, for each input set type and for exact and approx.

Sections: 1 and 2 the direct cells on torch models and ONNX graphs (1 also
on input sets that are not boxes); 3 the rounding cells (inputs where
round-to-nearest lands inside the real value); 4 the zonotope pass behind
layers whose rounding it does not bound (registry ROUND-11, marked); 5 the
graph walk and the sequential walk (trust, operations without a rule,
functional layers, bare modules, Conv1d, a flat input in front of a Conv2d,
the walk order of a module that is not a GraphModule, ``getitem`` of a
tensor, a broadcasting constant, refusals); 6 end to end through
``reach_pytorch_model`` and ``load_onnx``; 7 the ReLU and LeakyReLU kernels
given bounds through ``reach_layer``. Known defects marked here: AFFINE-3 (a
Cast to an integer type passed through), ROUND-11 (a float32 Zono in the
zonotope pass too), SHAPE-10 (the Conv2d shape guess of the interval pass),
REACH-7 (Hexatope and Octatope crash the option), and four
that show only through a direct call (the sequential walkers walk a module
that is not a Sequential in the order its children were registered; the
interval walker crashes on a BatchNorm without running statistics; a
``getitem`` of a tensor copies its source's bounds; a constant that
broadcasts to a larger tensor is applied entry by entry). A Sequential that
holds one module object at two places is marked in
``tests/soundness/test_soundness_prepass_handover.py``, not here.

Oracles, none of them n2v: for torch models an exact evaluator in
``fractions.Fraction`` written here (Linear, Conv2d, pooling, BatchNorm with
an exact square root, Flatten, ReLU, LeakyReLU), with Sigmoid and Tanh
enclosed by ``tests._oracles.interval_proof.value`` (mpmath interval
arithmetic at 100 bits); for ONNX graphs onnxruntime in float64 with the
pre-activation tensors added as graph outputs, on dyadic data where float64
is exact; the LP membership oracle (``assert_covered``) for end-to-end
results; exact rational membership where an escape is at rounding scale;
the exact image of a polytope (``tests._oracles.rational``) for the kernel
cells. Comparisons are one-sided with zero slack against exact references
(a pad of 1e-12 only against the float64 torch value behind a Softmax).

Layout: a recorded interval array belongs to a layer whose input is either an
image tensor (batch, C, H, W) or a flat tensor. For an image it is in
(height, width, channel) order, the row order of the ImageStar the ReLU
kernel reads (``membership.hwc``); for a flat tensor it is in the tensor's own
row-major order (channel, height, width after a Flatten). The reference is
always computed in the network's order and converted explicitly.
"""

import math
from fractions import Fraction

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.bounds_precomputation import compute_intermediate_bounds
from tests._oracles import rational
from tests._oracles.interval_proof import SIGMOID, TANH, chain, value
from tests._oracles.membership import hwc
from tests.cell_support import (
    assert_covered, constrained_image_star, constrained_star, flat_set,
    hwc_index, image_set, image_split_cover, reach_model, replay_onnx,
    replay_torch, samples, split_cover,
)
from tests.known_bugs import known_bug

F = Fraction
N_POINTS = 128


# --------------------------------------------------------------------------- #
# Exact evaluation of a torch Sequential, in Fractions                         #
# --------------------------------------------------------------------------- #
def _q(v):
    """The rational a float is, exactly."""
    return F(float(v))


def _qarr(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.empty(a.shape, dtype=object)
    for idx in np.ndindex(a.shape):
        out[idx] = F(float(a[idx]))
    return out


def _exact_sqrt(q):
    """The square root of a rational that is the square of one; anything else
    is a mistake in the test's own data."""
    q = F(q)
    n, d = math.isqrt(q.numerator), math.isqrt(q.denominator)
    if F(n * n, d * d) != q:
        pytest.fail(f"test data: {q} is not the square of a rational")
    return F(n, d)


def _affine(W, b, lo, hi):
    """Interval of ``W x + b`` over the box ``[lo, hi]`` (exact for a point)."""
    out_lo = np.empty(W.shape[0], dtype=object)
    out_hi = np.empty(W.shape[0], dtype=object)
    for i in range(W.shape[0]):
        lo_i = hi_i = (b[i] if b is not None else F(0))
        for j in range(W.shape[1]):
            w = W[i, j]
            if w > 0:
                lo_i += w * lo[j]
                hi_i += w * hi[j]
            elif w < 0:
                lo_i += w * hi[j]
                hi_i += w * lo[j]
        out_lo[i], out_hi[i] = lo_i, hi_i
    return out_lo, out_hi


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(int(t) for t in v)


_NUMPY_PAD = {"replicate": "edge", "reflect": "reflect", "circular": "wrap"}


def _conv2d(layer, lo, hi):
    """Exact Conv2d with stride, dilation, groups and numeric padding: zero
    padding, or the border entries copied the way torch's ``replicate``,
    ``reflect`` and ``circular`` modes copy them (numpy's ``edge``,
    ``reflect`` and ``wrap``), after which the convolution has no padding.
    On a point (``lo`` equal to ``hi``) the result is the exact value; on a
    thin enclosure (behind a Sigmoid or Tanh) it is an enclosure."""
    if isinstance(layer.padding, str):
        pytest.fail("test data: the exact Conv2d handles numeric padding")
    Wt = _qarr(layer.weight.detach().cpu().double().numpy())
    b = (None if layer.bias is None
         else _qarr(layer.bias.detach().cpu().double().numpy()))
    (sh, sw), (ph, pw), (dh, dw) = (_pair(layer.stride), _pair(layer.padding),
                                    _pair(layer.dilation))
    if layer.padding_mode != "zeros" and (ph or pw):
        width = ((0, 0), (ph, ph), (pw, pw))
        lo = np.pad(lo, width, mode=_NUMPY_PAD[layer.padding_mode])
        hi = np.pad(hi, width, mode=_NUMPY_PAD[layer.padding_mode])
        ph = pw = 0
    groups = layer.groups
    c_in, h_in, w_in = lo.shape
    c_out, cpg, kh, kw = Wt.shape
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out_lo = np.empty((c_out, h_out, w_out), dtype=object)
    out_hi = np.empty((c_out, h_out, w_out), dtype=object)
    opg = c_out // groups
    for o in range(c_out):
        g = o // opg
        for y in range(h_out):
            for x in range(w_out):
                l = h = (b[o] if b is not None else F(0))
                for c in range(cpg):
                    ci = g * cpg + c
                    for ky in range(kh):
                        iy = y * sh - ph + ky * dh
                        if not 0 <= iy < h_in:
                            continue
                        for kx in range(kw):
                            ix = x * sw - pw + kx * dw
                            if not 0 <= ix < w_in:
                                continue
                            w = Wt[o, c, ky, kx]
                            if w > 0:
                                l += w * lo[ci, iy, ix]
                                h += w * hi[ci, iy, ix]
                            elif w < 0:
                                l += w * hi[ci, iy, ix]
                                h += w * lo[ci, iy, ix]
                out_lo[o, y, x], out_hi[o, y, x] = l, h
    return out_lo, out_hi


def _pool_windows(layer, h_in, w_in):
    if getattr(layer, "ceil_mode", False):
        pytest.fail("test data: the exact pooling handles ceil_mode=False")
    (kh, kw), (ph, pw) = _pair(layer.kernel_size), _pair(layer.padding)
    sh, sw = _pair(layer.stride if layer.stride is not None
                   else layer.kernel_size)
    dh, dw = _pair(getattr(layer, "dilation", 1))
    h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    for y in range(h_out):
        for x in range(w_out):
            cells, area = [], kh * kw
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = y * sh - ph + ky * dh, x * sw - pw + kx * dw
                    if 0 <= iy < h_in and 0 <= ix < w_in:
                        cells.append((iy, ix))
            yield y, x, cells, area, (h_out, w_out)


def _pool(layer, lo, hi):
    c, h_in, w_in = lo.shape
    windows = list(_pool_windows(layer, h_in, w_in))
    h_out, w_out = windows[0][4]
    out_lo = np.empty((c, h_out, w_out), dtype=object)
    out_hi = np.empty((c, h_out, w_out), dtype=object)
    is_max = isinstance(layer, nn.MaxPool2d)
    for ch in range(c):
        for y, x, cells, area, _ in windows:
            if is_max:
                # zero padding of a max pool is -inf: only in-image cells
                out_lo[ch, y, x] = max(lo[ch, i, j] for i, j in cells)
                out_hi[ch, y, x] = max(hi[ch, i, j] for i, j in cells)
            else:
                div = area if layer.count_include_pad else len(cells)
                out_lo[ch, y, x] = sum(lo[ch, i, j] for i, j in cells) / div
                out_hi[ch, y, x] = sum(hi[ch, i, j] for i, j in cells) / div
    return out_lo, out_hi


def _batchnorm(layer, lo, hi):
    mean = _qarr(layer.running_mean.detach().double().numpy())
    var = _qarr(layer.running_var.detach().double().numpy())
    eps = _q(layer.eps)
    n = mean.size
    gamma = (_qarr(layer.weight.detach().double().numpy())
             if layer.weight is not None else [F(1)] * n)
    beta = (_qarr(layer.bias.detach().double().numpy())
            if layer.bias is not None else [F(0)] * n)
    out_lo = np.empty(lo.shape, dtype=object)
    out_hi = np.empty(lo.shape, dtype=object)
    for idx in np.ndindex(lo.shape):
        ch = idx[0]
        scale = gamma[ch] / _exact_sqrt(var[ch] + eps)
        a = scale * (lo[idx] - mean[ch]) + beta[ch]
        z = scale * (hi[idx] - mean[ch]) + beta[ch]
        out_lo[idx], out_hi[idx] = min(a, z), max(a, z)
    return out_lo, out_hi


def _monotone(stage, lo, hi):
    """A monotone increasing smooth function on every entry, enclosed by
    mpmath interval arithmetic (``interval_proof.value``)."""
    flat_lo, flat_hi = list(lo.ravel()), list(hi.ravel())
    low = value(chain(stage), flat_lo)[0]
    high = value(chain(stage), flat_hi)[1]
    return (np.array(low, dtype=object).reshape(lo.shape),
            np.array(high, dtype=object).reshape(lo.shape))


def _exact_layer(layer, lo, hi):
    """The layer on the box ``[lo, hi]`` (a point when ``lo is hi``); arrays
    carry the tensor shape without the batch axis."""
    if isinstance(layer, nn.Linear):
        W = _qarr(layer.weight.detach().cpu().double().numpy())
        b = (None if layer.bias is None
             else _qarr(layer.bias.detach().cpu().double().numpy()))
        return _affine(W, b, lo.ravel(), hi.ravel())
    if isinstance(layer, nn.Conv2d):
        return _conv2d(layer, lo, hi)
    if isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
        return _pool(layer, lo, hi)
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return _batchnorm(layer, lo, hi)
    if isinstance(layer, nn.Flatten):
        return lo.reshape(-1), hi.reshape(-1)
    if isinstance(layer, nn.ReLU):
        f = np.vectorize(lambda t: max(t, F(0)), otypes=[object])
        return f(lo), f(hi)
    if isinstance(layer, nn.LeakyReLU):
        s = _q(layer.negative_slope)
        f = np.vectorize(lambda t: t if t >= 0 else s * t, otypes=[object])
        a, z = f(lo), f(hi)
        lo_out = np.vectorize(min, otypes=[object])(a, z)
        hi_out = np.vectorize(max, otypes=[object])(a, z)
        # a crossing entry reaches f(0) = 0 inside
        cross = np.vectorize(lambda l, h: l < 0 < h, otypes=[bool])(lo, hi)
        if cross.any():
            lo_out = np.where(cross, np.vectorize(lambda t: min(t, F(0)),
                                                  otypes=[object])(lo_out),
                              lo_out)
            hi_out = np.where(cross, np.vectorize(lambda t: max(t, F(0)),
                                                  otypes=[object])(hi_out),
                              hi_out)
        return lo_out, hi_out
    if isinstance(layer, nn.Sigmoid):
        return _monotone(SIGMOID, lo, hi)
    if isinstance(layer, nn.Tanh):
        return _monotone(TANH, lo, hi)
    if isinstance(layer, (nn.Identity, nn.Dropout)):
        return lo, hi
    pytest.fail(f"test data: no exact rule for {type(layer).__name__}")


def _layer_inputs(model, x):
    """``[(lo, hi)]``: the exact input of every layer of the Sequential
    ``model`` at the point ``x`` (network order, no batch axis); entry ``i`` is
    the input of layer ``i``. Smooth layers make the value an enclosure."""
    v = _qarr(x)
    lo, hi = v, v
    out = []
    for layer in model:
        out.append((lo, hi))
        lo, hi = _exact_layer(layer, lo, hi)
    return out


def _consumer_order(a):
    """A layer input as the recorded array lists it: an image tensor (C, H, W)
    in (H, W, C) order, a flat tensor as it is."""
    if a.ndim == 3:
        c, h, w = a.shape
        return list(a.transpose(1, 2, 0).reshape(-1))
    return list(a.reshape(-1))


def _q_bound(v, what):
    """A recorded bound as a rational, or as the float itself when it is
    infinite (an infinite bound claims nothing and compares correctly with
    every rational); a NaN bound is an error, since it fails every
    comparison and could read as a decided one."""
    v = float(v)
    if math.isnan(v):
        raise AssertionError(f"{what}: a recorded bound is NaN")
    return v if math.isinf(v) else F(v)


def _check_recorded(bounds, refs, what):
    """Every recorded interval holds the real value of its neuron at every
    point. ``refs`` maps a recorded key to a list (one per point) of
    ``(lo, hi)`` arrays in network order: the real value, or an enclosure of
    it a ~1e-30 wide. A recorded -inf lower or +inf upper bound holds."""
    for key, (rec_lo, rec_hi) in bounds.items():
        if key not in refs:
            raise AssertionError(
                f"{what}: bounds recorded under {key!r}, which is not a "
                f"nonlinear layer of the model (recorded keys "
                f"{sorted(map(str, bounds))})")
        rec_lo = [_q_bound(v, what) for v in np.ravel(rec_lo)]
        rec_hi = [_q_bound(v, what) for v in np.ravel(rec_hi)]
        for p, (lo, hi) in enumerate(refs[key]):
            lo, hi = _consumer_order(lo), _consumer_order(hi)
            if len(lo) != len(rec_lo) or len(lo) != len(rec_hi):
                raise AssertionError(
                    f"{what}: layer {key!r} has {len(lo)} neurons, the "
                    f"recorded bounds have {len(rec_lo)}/{len(rec_hi)}")
            for j in range(len(lo)):
                if rec_lo[j] > lo[j] or hi[j] > rec_hi[j]:
                    raise AssertionError(
                        f"{what}: neuron {j} of layer {key!r} takes the real "
                        f"value [{float(lo[j])!r}, {float(hi[j])!r}] at "
                        f"member point {p}, outside the recorded "
                        f"[{float(rec_lo[j])!r}, {float(rec_hi[j])!r}] "
                        f"(below by {float(rec_lo[j] - lo[j]):.3g}, above "
                        f"by {float(hi[j] - rec_hi[j]):.3g})")


_NONLINEAR = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def _sequential_refs(model, points, keyed):
    """The reference of every nonlinear layer of a Sequential, keyed the way
    the walker records it: by index for ``keyed='index'``, by the node name
    of the traced module for a ``GraphModule``."""
    traces = [_layer_inputs(model, x) for x in points]
    refs = {}
    for i, layer in enumerate(model):
        if isinstance(layer, _NONLINEAR):
            refs[i] = [t[i] for t in traces]
    if keyed == "index":
        return refs
    by_name = {}
    for node in keyed.graph.nodes:
        if node.op == "call_module" and node.target.isdigit():
            i = int(node.target)
            if i in refs:
                by_name[node.name] = refs[i]
    return by_name


# --------------------------------------------------------------------------- #
# Models with dyadic weights                                                    #
# --------------------------------------------------------------------------- #
def _set(layer, W, b=None):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(np.asarray(W, dtype=np.float64))
                           .reshape(layer.weight.shape))
        if b is not None:
            layer.bias.copy_(torch.tensor(np.asarray(b, dtype=np.float64)))
        elif layer.bias is not None:
            layer.bias.zero_()
    return layer


def _dyadic(shape, seed, denom=4, span=4):
    rng = np.random.default_rng(seed)
    return rng.integers(-span, span + 1, size=shape) / denom


def _set_bn(bn, mean, var, gamma, beta):
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor(mean, dtype=torch.float64))
        bn.running_var.copy_(torch.tensor(var, dtype=torch.float64))
        bn.weight.copy_(torch.tensor(gamma, dtype=torch.float64))
        bn.bias.copy_(torch.tensor(beta, dtype=torch.float64))
    return bn


# The flat input box: x0 and x3 straddle 0, x1 is positive, x2 negative.
FLAT_LO = np.array([-0.75, 0.25, -1.5, -0.5])
FLAT_HI = np.array([0.5, 1.0, -0.5, 0.875])


def _flat_relu():
    """Three ReLU layers. Layer 1: h0, h3, h4 cross 0, h1 is stably inactive
    (its real output is 0), h2 stably active. Layer 2: g0 = 2 r1 + r2 - 5/4
    reads the inactive neuron with a positive weight, so a ReLU rule that
    keeps its negative upper bound lowers g0's bound below its real maximum
    1/4; every hidden neuron reads one or two inputs, so interval
    propagation is tight and a bound moved inward is visible."""
    l1 = _set(nn.Linear(4, 5).double(),
              [[1, -0.5, 0, 0], [0, 0, 1, 0], [0, 0.5, 0, 0],
               [0.25, 0, 0, 0.5], [-1, 0, 0, 0.25]],
              [0.25, -0.25, 1, 0, -0.125])
    l2 = _set(nn.Linear(5, 4).double(),
              [[0, 2, 1, 0, 0], [1, 0, 0, 0, -1], [0.5, 0, 0, 1, 0],
               [0, 0.5, -1, 0, 0]],
              [-1.25, 0.125, -0.25, 1.25])
    l3 = _set(nn.Linear(4, 3).double(),
              [[1, -0.5, 0, 0.25], [0, 1, 1, 0], [-0.5, 0, 0.75, 1]],
              [-0.125, -0.25, 0.0625])
    l4 = _set(nn.Linear(3, 2).double(), [[1, -1, 0.5], [0.25, 0.5, -1]],
              [0, 0.5])
    return nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3, nn.ReLU(),
                         l4).double().eval()


def _flat_leaky():
    """LeakyReLU of slope 1/4, then of slope -1/2 (a non-monotone function,
    legal in ONNX), then a ReLU; the second layer reads negative LeakyReLU
    outputs with positive weights, so a ReLU rule used for a LeakyReLU (its
    outputs clipped at 0) raises the next lower bound above the real
    minimum. The inputs of the slope -1/2 layer are one all-negative neuron
    (g0 in [-1.09, -0.75], whose image [0.375, 0.55] an endpoint-to-endpoint
    map turns upside down), one crossing neuron whose negative side
    dominates (g1 in [-1.03, 0.125]: its maximum 0.52 is f(lower end), and
    its minimum 0 is f(0) inside), and one ordinary crossing neuron; the
    layer after it reads g0 alone and g1 with g2, so a wrong interval for
    either is visible in the next recorded bounds."""
    l1 = _set(nn.Linear(4, 4).double(),
              [[1, 0, 0, 0], [0, 0, 1, 0.5], [0, 1, 0, 0], [0.5, 0, 0, -1]],
              [0, 0.5, -0.5, 0.125])
    l2 = _set(nn.Linear(4, 3).double(),
              [[0.5, 0, 0, 0], [0, 0, 0, 1], [-0.5, 0, 1, 0.25]],
              [-1, -0.75, 0])
    l3 = _set(nn.Linear(3, 2).double(), [[1, 0, 0], [0, 1, 0.5]],
              [-0.5, -0.25])
    l4 = _set(nn.Linear(2, 2).double(), [[1, -1], [0.5, 1]], [0, 0])
    return nn.Sequential(l1, nn.LeakyReLU(0.25), l2, nn.LeakyReLU(-0.5), l3,
                         nn.ReLU(), l4).double().eval()


def _flat_smooth():
    """Sigmoid, then Tanh, then a ReLU; the layers after each smooth layer
    read it with positive and negative weights on one or two neurons, so a
    smooth-layer bound taken from the wrong end moves the next interval
    inward."""
    l1 = _set(nn.Linear(4, 3).double(),
              [[2, 0, 0, 0], [0, -1, 0, 0.5], [0, 0, 1, -1]], [0, 0.5, 0.25])
    l2 = _set(nn.Linear(3, 3).double(),
              [[2, 0, 0], [0, -1.5, 1], [1, 0, -2]], [-1, 0.25, 0.5])
    l3 = _set(nn.Linear(3, 2).double(), [[2, -1, 0], [0, 1.5, 1]],
              [-0.25, -0.5])
    l4 = _set(nn.Linear(2, 1).double(), [[1, -1]], [0])
    return nn.Sequential(l1, nn.Sigmoid(), l2, nn.Tanh(), l3, nn.ReLU(),
                         l4).double().eval()


def _flat_misc():
    """BatchNorm1d (a negative scale, an exact square root: eps 0 and
    variances 1/4, 1, 4, 1/16), a ReLU, Dropout and Identity (both
    pass-throughs at inference), then a Linear and a ReLU."""
    l1 = _set(nn.Linear(4, 4).double(),
              [[1, 0.5, 0, 0], [0, 0, -1, 0.25], [0.5, 0, 0, 1], [0, 1, 0.5, 0]],
              [0.125, -1, 0, -0.5])
    bn = _set_bn(nn.BatchNorm1d(4, eps=0.0).double(),
                 [0.25, -0.5, 0, 1], [0.25, 1, 4, 0.0625],
                 [1, -0.5, 2, 0.25], [0, 0.25, -0.5, 0.125])
    l2 = _set(nn.Linear(4, 3).double(),
              [[1, -1, 0.5, 0], [0, 0.5, 1, -1], [0.25, 0, -0.5, 1]],
              [0, -0.25, 0.125])
    return nn.Sequential(l1, bn, nn.ReLU(), nn.Dropout(0.5), nn.Identity(),
                         l2, nn.ReLU()).double().eval()


IMG_CHW = (2, 3, 5)


def _image_box():
    """A different dyadic interval per pixel of a (2, 3, 5) image; some
    pixels straddle 0."""
    rng = np.random.default_rng(11)
    lo = rng.integers(-6, 5, size=IMG_CHW) / 4.0
    return lo, lo + rng.integers(1, 5, size=IMG_CHW) / 4.0


def _img_conv():
    """Conv2d 2 -> 3 channels, kernel (2, 3), padding (1, 0): output
    (3, 4, 3); ReLU; Conv2d 3 -> 2, kernel (2, 2), stride (2, 1): output
    (2, 2, 2); Flatten; a ReLU directly on the flattened image (its bounds
    are in (channel, height, width) order, not the ImageStar's); Linear and
    ReLU. The biases set the channels apart (1.5, -1, 1/4 and 3, -3); the
    second convolution's weights are at most 1/16, so its two channels have
    disjoint ranges (about [1.8, 3.3] and [-3.6, -2.7]) and a bound of the
    flattened image laid out as an image (another channel's entry) misses
    the real value."""
    c1 = _set(nn.Conv2d(2, 3, (2, 3), padding=(1, 0)).double(),
              _dyadic((3, 2, 2, 3), 1), [1.5, -1, 0.25])
    c2 = _set(nn.Conv2d(3, 2, (2, 2), stride=(2, 1)).double(),
              _dyadic((2, 3, 2, 2), 3, denom=64), [3, -3])
    l = _set(nn.Linear(8, 3).double(), _dyadic((3, 8), 5),
             _dyadic(3, 6, denom=8))
    return nn.Sequential(c1, nn.ReLU(), c2, nn.Flatten(), nn.ReLU(), l,
                         nn.ReLU()).double().eval()


def _img_pool():
    """Conv2d 2 -> 2, kernel (1, 2): (2, 3, 4); BatchNorm2d (negative scale,
    eps 0, variances 1/4 and 4); ReLU; AvgPool2d kernel (2, 2), stride 1,
    padding (1, 0) without the padding in the divisor: (2, 4, 3), border
    windows divided by 2; LeakyReLU 1/4; MaxPool2d kernel (2, 1): (2, 2, 3);
    Flatten; Linear; ReLU."""
    c1 = _set(nn.Conv2d(2, 2, (1, 2)).double(), _dyadic((2, 2, 1, 2), 7),
              [0.25, -0.5])
    bn = _set_bn(nn.BatchNorm2d(2, eps=0.0).double(), [0.5, -0.25],
                 [0.25, 4], [-1, 0.5], [0.25, 0])
    pool = nn.AvgPool2d((2, 2), stride=1, padding=(1, 0),
                        count_include_pad=False)
    l = _set(nn.Linear(12, 2).double(), _dyadic((2, 12), 8),
             [0.125, -0.25])
    return nn.Sequential(c1, bn, nn.ReLU(), pool, nn.LeakyReLU(0.25),
                         nn.MaxPool2d((2, 1)), nn.Flatten(), l,
                         nn.ReLU()).double().eval()


def _img_smooth():
    """Conv2d 2 -> 2, kernel (2, 2): (2, 2, 4); Sigmoid; Conv2d kernel
    (1, 3): (2, 2, 2); Tanh; Flatten; Linear; ReLU."""
    c1 = _set(nn.Conv2d(2, 2, (2, 2)).double(), _dyadic((2, 2, 2, 2), 9),
              [0, 0.25])
    c2 = _set(nn.Conv2d(2, 2, (1, 3)).double(), _dyadic((2, 2, 1, 3), 10),
              [-0.5, 0.125])
    l = _set(nn.Linear(8, 2).double(), _dyadic((2, 8), 12), [0, 0.5])
    return nn.Sequential(c1, nn.Sigmoid(), c2, nn.Tanh(), nn.Flatten(), l,
                         nn.ReLU()).double().eval()


def _img_padmodes():
    """Three convolutions whose padding copies border entries instead of
    adding zeros: Conv2d 2 -> 2, kernel (2, 3), padding (1, 1), mode
    ``replicate``: (2, 4, 5); ReLU; kernel (3, 2), padding (1, 1), mode
    ``reflect``: (2, 4, 6); ReLU; kernel (1, 2), padding (0, 1), mode
    ``circular``: (2, 4, 7); Flatten; Linear; ReLU. A bound computed with
    zero padding misses the real value of an output whose window holds a
    copied entry that is not 0."""
    c1 = _set(nn.Conv2d(2, 2, (2, 3), padding=(1, 1),
                        padding_mode="replicate").double(),
              _dyadic((2, 2, 2, 3), 13), [0.5, -0.25])
    c2 = _set(nn.Conv2d(2, 2, (3, 2), padding=(1, 1),
                        padding_mode="reflect").double(),
              _dyadic((2, 2, 3, 2), 14, denom=8), [-0.25, 0.5])
    c3 = _set(nn.Conv2d(2, 2, (1, 2), padding=(0, 1),
                        padding_mode="circular").double(),
              _dyadic((2, 2, 1, 2), 15), [0.125, -0.125])
    l = _set(nn.Linear(56, 2).double(), _dyadic((2, 56), 16, denom=16),
             [0.25, -0.25])
    return nn.Sequential(c1, nn.ReLU(), c2, nn.ReLU(), c3, nn.Flatten(), l,
                         nn.ReLU()).double().eval()


def _img_dilated():
    """Conv2d 2 -> 4, kernel (2, 2), dilation (2, 1), groups 2, padding
    (1, 0): (4, 3, 4), each output channel reading one input channel at rows
    two apart; LeakyReLU 1/2; MaxPool2d kernel 2, stride 1, padding 1:
    (4, 4, 5), whose border windows hold one to two image entries; the
    weights are at most 1/4 in size and channel 0's bias is -3, so channel 0
    is below 0 everywhere and the max of its border windows is negative (a
    padding read as 0 would be the max); BatchNorm2d with eps 1/16 and no
    affine parameters (gamma 1, beta 0), variances 3/16, 15/16, 63/16, 3/16
    (var + eps is 1/4, 1, 4, 1/4: the scale is exact, and wrong without
    eps); ReLU; Flatten; Linear; ReLU."""
    c1 = _set(nn.Conv2d(2, 4, (2, 2), dilation=(2, 1), groups=2,
                        padding=(1, 0)).double(),
              _dyadic((4, 1, 2, 2), 17, denom=16), [-3, 0.5, -0.25, 1])
    bn = nn.BatchNorm2d(4, eps=1.0 / 16, affine=False).double()
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([-1.5, 0.25, -0.5, 0.75],
                                           dtype=torch.float64))
        bn.running_var.copy_(torch.tensor([3 / 16, 15 / 16, 63 / 16, 3 / 16],
                                          dtype=torch.float64))
    l = _set(nn.Linear(80, 2).double(), _dyadic((2, 80), 18, denom=16),
             [0.25, -0.5])
    return nn.Sequential(c1, nn.LeakyReLU(0.5),
                         nn.MaxPool2d(2, stride=1, padding=1), bn, nn.ReLU(),
                         nn.Flatten(), l, nn.ReLU()).double().eval()


FLAT_MODELS = {"relu": _flat_relu, "leaky": _flat_leaky,
               "smooth": _flat_smooth, "misc": _flat_misc}
IMAGE_MODELS = {"conv": _img_conv, "pool": _img_pool, "smooth": _img_smooth,
                "padmodes": _img_padmodes, "dilated": _img_dilated}


def _flat_input(kind, n=N_POINTS):
    """``(set, member points)`` over FLAT_LO..FLAT_HI."""
    if kind == "Star+row":
        return constrained_star(FLAT_LO, FLAT_HI, n=n, seed=3)
    return flat_set(kind, FLAT_LO, FLAT_HI), samples(FLAT_LO, FLAT_HI, n)


def _image_input(kind, n=N_POINTS // 2):
    lo, hi = _image_box()
    if kind == "ImageStar+row":
        return constrained_image_star(lo, hi, n=n, seed=4)
    return image_set(kind, lo, hi), samples(lo, hi, n)


_REF_CACHE = {}


def _refs_for(family, name, kind, model, points, walker):
    """The exact references, computed once per (model, input kind) and keyed
    for the walker."""
    key = (family, name, kind)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _sequential_refs(model, points, "index")
    by_index = _REF_CACHE[key]
    if walker == "sequential":
        return by_index
    names = {}
    for node in walker.graph.nodes:
        if node.op == "call_module" and node.target.isdigit():
            i = int(node.target)
            if i in by_index:
                names[node.name] = by_index[i]
    return names


def _nonlinear_count(model):
    return sum(isinstance(m, _NONLINEAR) for m in model)


# --------------------------------------------------------------------------- #
# 1. Direct cells on torch models: both walkers, both passes, every set type    #
# --------------------------------------------------------------------------- #
def _model_methods(models):
    """(model, pass) pairs. The zonotope pass on the smooth networks is left
    out: there it reaches Sigmoid and Tanh with the zonotope kernel, whose
    interval [f(lb), f(ub)] is not rounded outward (registry ROUND-2, the
    kernel's own cell), and the bounds it records miss real values by
    about 1e-17 for that reason; the pass's own handling behind Sigmoid and
    Tanh is the ROUND-11 cell below."""
    return [(name, method) for name in sorted(models)
            for method in ("ibp", "zono")
            if not (name == "smooth" and method == "zono")]


@pytest.mark.parametrize("walker", ["sequential", "traced"])
@pytest.mark.parametrize("kind", ["Box", "Star", "Star+row", "Zono"])
@pytest.mark.parametrize("name, method", _model_methods(FLAT_MODELS))
def test_flat_recorded_bounds_hold_every_member_point(name, method, kind,
                                                       walker):
    """Every interval the pre-pass records for a nonlinear layer of a flat
    network holds the real pre-activation of every member point (corners,
    the quarter grid, points on a binding non-box row), checked in exact
    arithmetic.

    Catches an input set read with its lower bound as both ends or with the
    two ends swapped (Box, Star, Zono branches), a Linear without its bias
    or with a sign slip, a ReLU rule that keeps the negative upper bound of a
    stably inactive neuron, a LeakyReLU sent to the ReLU rule or mapped
    endpoint to endpoint (a negative slope), a Sigmoid or Tanh bound taken
    from the wrong end, a BatchNorm scale with the wrong sign, a walker that
    records a layer's bounds after propagating through it, and bounds
    recorded under another layer's key."""
    model = FLAT_MODELS[name]()
    input_set, points = _flat_input(kind)
    target = model if walker == "sequential" else fx.symbolic_trace(model)
    bounds = compute_intermediate_bounds(target, input_set, method=method)
    refs = _refs_for("flat", name, kind, model, points,
                     "sequential" if walker == "sequential" else target)
    what = f"{method} pre-pass, {walker} walker, {name} net, {kind}"
    if method == "ibp" and len(bounds) != _nonlinear_count(model):
        pytest.fail(f"{what}: the interval pass recorded {len(bounds)} of "
                    f"{_nonlinear_count(model)} nonlinear layers")
    if not bounds:
        pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
    _check_recorded(bounds, refs, what)


@pytest.mark.parametrize("walker", ["sequential", "traced"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageStar+row", "ImageZono"])
@pytest.mark.parametrize("name, method", _model_methods(IMAGE_MODELS))
def test_image_recorded_bounds_hold_every_member_point(name, method, kind,
                                                        walker):
    """Every interval the pre-pass records for a nonlinear layer of a
    convolutional network on a (2, 3, 5) image holds the real pre-activation
    of every member point, in the order the reach reads it: (height, width,
    channel) for an image, (channel, height, width) after a Flatten.

    Catches the image input read in its stored (height, width, channel)
    order as if it were (channel, height, width), height and width swapped
    in the entry shape, a convolution bound paired with the wrong weight
    signs or a bias added on the wrong axis, an average pool divided by the
    full kernel area at the border or pooled from one end, a max pool, a
    BatchNorm (a scale with the wrong sign; on the 'dilated' net one
    without eps or with a shift of 1 where it has no affine parameters), a
    Flatten that keeps the image shape (the ReLU after it would be
    reordered as an image), the recording that reorders an image layer
    wrongly or takes one end twice, a convolution without its dilation or
    groups, and a padded max pool whose padding is read as 0."""
    model = IMAGE_MODELS[name]()
    input_set, points = _image_input(kind)
    target = model if walker == "sequential" else fx.symbolic_trace(model)
    bounds = compute_intermediate_bounds(target, input_set, method=method,
                                         input_shape=IMG_CHW)
    refs = _refs_for("image", name, kind, model, points,
                     "sequential" if walker == "sequential" else target)
    what = f"{method} pre-pass, {walker} walker, {name} net, {kind}"
    if method == "ibp" and len(bounds) != _nonlinear_count(model):
        pytest.fail(f"{what}: the interval pass recorded {len(bounds)} of "
                    f"{_nonlinear_count(model)} nonlinear layers")
    if not bounds:
        pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
    _check_recorded(bounds, refs, what)


def _not_a_box(kind, seed=19):
    """``(set, member points)`` of a set that is not a box, in network order
    (flat, or (C, H, W) for an image of IMG_CHW). 'Zono' and 'ImageZono':
    generators that mix entries, more of them than entries. 'Star' and
    'ImageStar': a basis that is not diagonal (unit diagonal, entries below
    it), the predicate box [-2, 1/2] x [0, 1] x [-1/2, 3/2] x [-1, 1/4]
    (repeated), not [-1, 1], and the row a0 + a1 <= 1/2, which cuts off the
    corner (1/2, 1). Points are c + V a for corners and half-grid points a
    of the predicate domain; all data dyadic, so the points are exact."""
    rng = np.random.default_rng(seed)
    image = kind.startswith("Image")
    n = int(np.prod(IMG_CHW)) if image else 4
    c = rng.integers(-4, 5, n) / 8.0
    if kind.endswith("Zono"):
        m = n + 2
        V = rng.integers(-4, 5, (n, m)) / 8.0
        alphas = ([rng.choice([-1.0, 1.0], m) for _ in range(24)]
                  + [rng.integers(-2, 3, m) / 2.0 for _ in range(8)])
        if image:
            s = ImageZono(c, V, IMG_CHW[1], IMG_CHW[2], IMG_CHW[0])
        else:
            s = Zono(c.reshape(-1, 1), V)
    else:
        m = n
        V = np.eye(n) + np.tril(rng.integers(-2, 3, (n, n)) / 4.0, -1)
        plo = np.resize([-2.0, 0.0, -0.5, -1.0], n)
        pup = np.resize([0.5, 1.0, 1.5, 0.25], n)
        row = np.zeros(n)
        row[:2] = 1.0
        C = np.vstack([np.eye(n), -np.eye(n), row])
        d = np.concatenate([pup, -plo, [0.5]]).reshape(-1, 1)
        basis = np.hstack([c[:, None], V])
        if image:
            s = ImageStar(basis, C, d, plo.reshape(-1, 1), pup.reshape(-1, 1),
                          height=IMG_CHW[1], width=IMG_CHW[2],
                          num_channels=IMG_CHW[0])
        else:
            s = Star(basis, C, d, plo.reshape(-1, 1), pup.reshape(-1, 1))
        alphas = []
        while len(alphas) < 32:
            pick = (rng.integers(0, 2, n) if len(alphas) % 2
                    else rng.integers(0, 5, n) / 4.0)
            a = plo + (pup - plo) * pick
            if row @ a <= 0.5:
                alphas.append(a)
    stored = [c + V @ a for a in alphas]
    if not image:
        return s, stored
    order = hwc_index(*IMG_CHW)
    points = []
    for v in stored:                    # stored (H, W, C) -> network (C, H, W)
        x = np.empty(n)
        x[order] = v
        points.append(x.reshape(IMG_CHW))
    return s, points


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["Zono", "Star", "ImageZono", "ImageStar"])
def test_recorded_bounds_hold_on_input_sets_that_are_not_boxes(kind, method):
    """Every interval the pre-pass records (traced walker, the reach's)
    holds the real pre-activation of every member point when the input set
    is not a box: a zonotope whose generators mix entries and outnumber
    them, and a star with a basis that is not diagonal, a predicate box
    other than [-1, 1] and a binding extra row (flat: the three-ReLU net;
    image: the convolutional net). Exact rationals, no slack.

    Catches an input set read as if its predicate box were [-1, 1] or its
    basis diagonal (a Star taken as c +- sum |V|, a Zono radius summed over
    the wrong axis), and an image set's generators reordered with the
    entries of a box."""
    image = kind.startswith("Image")
    model = (IMAGE_MODELS["conv"] if image else FLAT_MODELS["relu"])()
    input_set, points = _not_a_box(kind)
    traced = fx.symbolic_trace(model)
    bounds = compute_intermediate_bounds(
        traced, input_set, method=method,
        input_shape=IMG_CHW if image else None)
    what = f"{method} pre-pass, traced walker, {kind} that is not a box"
    if method == "ibp" and len(bounds) != _nonlinear_count(model):
        pytest.fail(f"{what}: the interval pass recorded {len(bounds)} of "
                    f"{_nonlinear_count(model)} nonlinear layers")
    if not bounds:
        pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
    _check_recorded(bounds, _sequential_refs(model, points, traced), what)


# --------------------------------------------------------------------------- #
# 2. Direct cells on ONNX graphs (the GraphModule walker after load_onnx)       #
# --------------------------------------------------------------------------- #
def _node(op, inputs, outputs, name=None, **attrs):
    return helper.make_node(op, inputs, outputs, name=name, **attrs)


def _save_graph(path, nodes, in_shape, outputs, inits,
                elem=TensorProto.DOUBLE):
    """An ONNX file of ``nodes`` on an input of element type ``elem`` (float64
    unless onnxruntime has no float64 kernel for an op, as for Conv), with
    the graph outputs ``outputs``, their shapes from ONNX shape inference.
    Float initializers are stored in the input's precision."""
    dtype = np.float64 if elem == TensorProto.DOUBLE else np.float32

    def store(a):
        a = np.asarray(a)
        return a.astype(dtype) if a.dtype.kind == "f" else a

    def build(outs):
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("X", elem, list(in_shape))],
            outs,
            initializer=[numpy_helper.from_array(store(a), k)
                         for k, a in inits.items()])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        return model
    inferred = onnx.shape_inference.infer_shapes(build([]))
    infos = {v.name: v for v in inferred.graph.value_info}
    missing = [o for o in outputs if o not in infos]
    if missing:
        pytest.fail(f"test data: no inferred shape for {missing}")
    model = build([infos[o] for o in outputs])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _ort_tensors(path, points, in_shape):
    """onnxruntime (float64, one thread) on every point: a dict from graph
    output name to the output array, batch axis kept."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    feed = session.get_inputs()[0]
    dtype = np.float64 if feed.type == "tensor(double)" else np.float32
    out = []
    for x in points:
        point = np.asarray(x, np.float64).reshape(in_shape)
        if not np.array_equal(point.astype(dtype).astype(np.float64), point):
            pytest.fail("test data: a member point is not exact in the "
                        "graph's input type")
        vals = session.run(None, {feed.name: point.astype(dtype)})
        out.append(dict(zip(names, vals)))
    return out


def _net_order(t):
    """An onnxruntime tensor without its batch axis, as Fractions: (C, H, W)
    for a rank-4 tensor, flat for a rank-2 one."""
    t = np.asarray(t, dtype=np.float64)
    if t.ndim == 4:
        a = _qarr(t[0])
    elif t.ndim == 2:
        a = _qarr(t[0])
    else:
        pytest.fail(f"test data: a pre-activation of rank {t.ndim}")
    return a


def _w(shape, seed, denom=4, span=4):
    return _dyadic(shape, seed, denom, span).astype(np.float64)


def _graph_flat():
    """``X (1, 4)``: MatMul by a 4 x 5 weight and Add of a bias, Relu
    ``act1``; MatMul by a square weight that is not symmetric, Relu
    ``act2``; Sub of a constant and Mul by a constant with negative
    entries, Relu ``act3``; Div by a constant, Neg and Add of a constant,
    Relu ``act4``; MatMul to two outputs. Each Relu reads a different graph
    operation, so each operation's interval rule is checked by the bounds
    of the Relu after it."""
    W2 = np.array([[1, 0.5, 0, 0, -0.25], [0, 1, 0, 0.5, 0],
                   [-0.5, 0, 1, 0, 0], [0, 0, 0.25, 1, 0],
                   [0, -1, 0, 0, 0.75]])
    inits = {"W1": np.array([[1, 0, 0.5, 0, -0.5], [0, 1, 0, -0.5, 0],
                             [0.5, 0, 0, 1, 0], [0, -0.5, 1, 0, 0.25]]),
             "B1": np.array([0.25, -0.5, 0.125, 0, -0.25]),
             "W2": W2,
             "C3": np.array([0.25, 0, -0.5, 0.125, 0]),
             "C4": np.array([2, -1, 0.5, -0.5, 1]),
             "C5": np.array([2.0]),
             "C6": np.array([0.5, 0.25, -0.125, 0, 0.75]),
             "W7": np.array([[1, -1], [0.5, 0], [0, 1], [-0.5, 0.25],
                             [1, 0.5]])}
    nodes = [_node("MatMul", ["X", "W1"], ["m1"]),
             _node("Add", ["m1", "B1"], ["p1"]),
             _node("Relu", ["p1"], ["r1"], name="act1"),
             _node("MatMul", ["r1", "W2"], ["p2"]),
             _node("Relu", ["p2"], ["r2"], name="act2"),
             _node("Sub", ["r2", "C3"], ["s3"]),
             _node("Mul", ["s3", "C4"], ["p3"]),
             _node("Relu", ["p3"], ["r3"], name="act3"),
             _node("Div", ["r3", "C5"], ["d4"]),
             _node("Neg", ["d4"], ["n4"]),
             _node("Add", ["n4", "C6"], ["p4"]),
             _node("Relu", ["p4"], ["r4"], name="act4"),
             _node("MatMul", ["r4", "W7"], ["Y"])]
    return nodes, inits, [1, 4], {"act1": "p1", "act2": "p2", "act3": "p3",
                                  "act4": "p4"}, TensorProto.DOUBLE


def _graph_residual():
    """A residual block: ``a = Relu(X A + bA)``, ``c = Relu(a + a B)`` (an Add
    of two computed branches, B nonnegative, so the interval sum is the real
    maximum), ``d = Relu(c - a D)`` (a Sub of two branches). Every neuron of
    ``a`` reads its own input, so the interval of each branch is tight and a
    sum built from the wrong end of the second branch misses a real value."""
    inits = {"A": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0.5],
                            [0, 0, 0]]),
             "bA": np.array([0.25, 0.5, 0.125]),
             "B": np.array([[0.5, 0, 0.25], [0, 1, 0], [0, 0.5, 1]]),
             "D": np.array([[1, 0, 0], [0.25, 0.5, 0], [0, 0, 1]]),
             "E": np.array([[1, -1], [0.5, 1], [-0.25, 0.5]])}
    nodes = [_node("MatMul", ["X", "A"], ["ma"]),
             _node("Add", ["ma", "bA"], ["pa"]),
             _node("Relu", ["pa"], ["a"], name="act_a"),
             _node("MatMul", ["a", "B"], ["b"]),
             _node("Add", ["a", "b"], ["pc"]),
             _node("Relu", ["pc"], ["c"], name="act_c"),
             _node("MatMul", ["a", "D"], ["e"]),
             _node("Sub", ["c", "e"], ["pd"]),
             _node("Relu", ["pd"], ["d"], name="act_d"),
             _node("MatMul", ["d", "E"], ["Y"])]
    return nodes, inits, [1, 4], {"act_a": "pa", "act_c": "pc",
                                  "act_d": "pd"}, TensorProto.DOUBLE


def _graph_product():
    """A Mul of two computed branches: ``Relu((X P) * (X Q))``, each factor
    crossing 0, so the four end products are all needed."""
    inits = {"P": np.array([[1, 0], [0, 0.5], [0, 0], [0.5, 0]]),
             "Q": np.array([[0, 1], [0, 0], [1, 0], [0, -0.5]])}
    nodes = [_node("MatMul", ["X", "P"], ["u"]),
             _node("MatMul", ["X", "Q"], ["v"]),
             _node("Mul", ["u", "v"], ["p"]),
             _node("Relu", ["p"], ["Y"], name="act_p")]
    return nodes, inits, [1, 4], {"act_p": "p"}, TensorProto.DOUBLE


def _graph_image():
    """``X (1, 2, 3, 5)``: Conv kernel (2, 3) to 3 channels: (1, 3, 2, 3);
    Transpose perm (0, 1, 3, 2): (1, 3, 3, 2), Relu ``act1``; Transpose perm
    (0, 2, 3, 1) (a cyclic perm): (1, 3, 2, 3), Relu ``act2``; Flatten, Relu
    ``act3`` (flat order); MatMul. The channel biases (1, -1, 1/4) set the
    channels apart, so bounds of a permuted image laid out in the wrong
    order miss real values. Float32 (onnxruntime has no float64 Conv); the
    data are dyadic with few bits, so float32 is exact."""
    inits = {"K": _w((3, 2, 2, 3), 21), "Kb": np.array([1, -1, 0.25]),
             "W": _w((18, 2), 22)}
    nodes = [_node("Conv", ["X", "K", "Kb"], ["c"], kernel_shape=[2, 3]),
             _node("Transpose", ["c"], ["t1"], perm=[0, 1, 3, 2]),
             _node("Relu", ["t1"], ["r1"], name="act1"),
             _node("Transpose", ["r1"], ["t2"], perm=[0, 2, 3, 1]),
             _node("Relu", ["t2"], ["r2"], name="act2"),
             _node("Flatten", ["r2"], ["f"], axis=1),
             _node("Relu", ["f"], ["r3"], name="act3"),
             _node("MatMul", ["r3", "W"], ["Y"])]
    return nodes, inits, [1, 2, 3, 5], {"act1": "t1", "act2": "t2",
                                        "act3": "f"}, TensorProto.FLOAT


def _graph_image_residual():
    """``X (1, 2, 3, 5)``: Conv 1 x 1 to 2 channels with the biases (1, -1),
    ``p = c + X`` (an Add of two image branches), Relu ``act``; Flatten,
    MatMul. The pixels differ in their intervals and the biases set the two
    channels apart, so bounds of the sum laid out in (channel, height,
    width) order miss real values in the (height, width, channel) order an
    image set reads them in. Float32, dyadic data with few bits, exact."""
    inits = {"K": _w((2, 2, 1, 1), 23), "Kb": np.array([1.0, -1.0]),
             "W": _w((30, 2), 24)}
    nodes = [_node("Conv", ["X", "K", "Kb"], ["c"], kernel_shape=[1, 1]),
             _node("Add", ["c", "X"], ["p"]),
             _node("Relu", ["p"], ["r"], name="act"),
             _node("Flatten", ["r"], ["f"], axis=1),
             _node("MatMul", ["f", "W"], ["Y"])]
    return nodes, inits, [1, 2, 3, 5], {"act": "p"}, TensorProto.FLOAT


def _graph_cast(target):
    """``Relu(Cast(Cast(X, target), DOUBLE))``: for an integer target the
    cast truncates, so the Relu input is not X."""
    nodes = [_node("Cast", ["X"], ["ci"], to=target),
             _node("Cast", ["ci"], ["cf"], to=TensorProto.DOUBLE),
             _node("Relu", ["cf"], ["Y"], name="act")]
    return nodes, {}, [1, 4], {"act": "cf"}, TensorProto.DOUBLE


ONNX_FLAT = {"flat": _graph_flat, "residual": _graph_residual,
             "product": _graph_product}
ONNX_IMAGE = {"transposes": _graph_image, "residual": _graph_image_residual}


def _load_graph(tmp_path, build):
    """``(n2v model, onnxruntime file, input shape, Relu name -> tensor)``:
    the graph saved once with the model output only, for n2v, and once with
    every Relu input as an extra output, for the reference."""
    from n2v.utils.model_loader import load_onnx
    nodes, inits, in_shape, pre, elem = build()
    model_path = _save_graph(tmp_path / "model.onnx", nodes, in_shape, ["Y"],
                             inits, elem)
    ref_path = _save_graph(tmp_path / "ref.onnx", nodes, in_shape,
                           ["Y"] + sorted(set(pre.values())), inits, elem)
    return load_onnx(model_path), ref_path, in_shape, pre


def _onnx_refs(ref_path, in_shape, pre, points):
    tensors = _ort_tensors(ref_path, points, in_shape)
    refs = {}
    for act, tensor in pre.items():
        vals = [_net_order(t[tensor]) for t in tensors]
        refs[act] = [(v, v) for v in vals]
    return refs


def _assert_onnx_bounds(model, ref_path, in_shape, pre, input_set, points,
                        method, what, expect_all=True, allow_none=False):
    """Every interval recorded for a Relu holds its real input. With
    ``allow_none`` a pass that records nothing (it stops trusting the graph,
    which only costs LPs) is a right answer too; otherwise recording nothing
    means the cell checked nothing, a mistake in the test."""
    bounds = compute_intermediate_bounds(model, input_set, method=method,
                                         input_shape=tuple(in_shape[1:]))
    if (method == "ibp" and expect_all and not allow_none
            and set(bounds) != set(pre)):
        pytest.fail(f"{what}: the interval pass recorded {sorted(bounds)}, "
                    f"the graph's Relu nodes are {sorted(pre)}")
    if not bounds and not allow_none:
        pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
    _check_recorded(bounds, _onnx_refs(ref_path, in_shape, pre, points),
                    what)


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["Box", "Star", "Star+row", "Zono"])
@pytest.mark.parametrize("graph", sorted(ONNX_FLAT))
def test_onnx_flat_recorded_bounds_hold_every_member_point(graph, kind,
                                                            method, tmp_path):
    """Every interval the pre-pass records for a Relu of an ONNX graph holds
    the real pre-activation (onnxruntime in float64 on dyadic data, exact)
    of every member point.

    Catches a MatMul bounded from one end of its input, or multiplied by W
    instead of its transpose (a square W that is not symmetric), an Add,
    Sub, Mul or Div by a constant with the wrong sign or end, a Neg that
    keeps the ends in place, an Add or Sub of two computed branches built
    from the wrong end of the second branch, a Mul of two branches that
    leaves out an end product, and bounds of one Relu recorded under
    another's name."""
    model, ref_path, in_shape, pre = _load_graph(tmp_path, ONNX_FLAT[graph])
    input_set, points = _flat_input(kind)
    _assert_onnx_bounds(model, ref_path, in_shape, pre, input_set, points,
                        method, f"{method} pre-pass, ONNX {graph}, {kind}")


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageStar+row", "ImageZono"])
@pytest.mark.parametrize("graph", sorted(ONNX_IMAGE))
def test_onnx_image_recorded_bounds_hold_every_member_point(graph, kind,
                                                             method, tmp_path):
    """Every interval the pre-pass records for a Relu of an ONNX image graph
    holds the real pre-activation of every member point, in (height, width,
    channel) order for an image and flat order after a Flatten: a graph
    with two batch-preserving Transposes and a Flatten, and a residual
    graph whose Relu reads the sum of two image branches.

    Catches a Transpose whose bounds are not moved with the entries (a
    batch-preserving perm treated as the identity), moved by the inverse
    perm, or taken from one end, a Flatten that keeps the image layout, and
    a sum of two image branches recorded without its image layout."""
    model, ref_path, in_shape, pre = _load_graph(tmp_path, ONNX_IMAGE[graph])
    input_set, points = _image_input(kind)
    _assert_onnx_bounds(model, ref_path, in_shape, pre, input_set, points,
                        method, f"{method} pre-pass, ONNX image {graph}, "
                                f"{kind}")


@pytest.mark.parametrize("kind", ["Box", "Star"])
@pytest.mark.parametrize("target", [
    pytest.param(TensorProto.DOUBLE, id="to-double"),
    pytest.param(TensorProto.INT64, id="to-int64", marks=known_bug(
        7, "AFFINE-3: the interval pre-pass treats an ONNX Cast as the "
           "identity whatever the target, so behind a Cast to INT64 it "
           "records the bounds of the untruncated input and a real "
           "pre-activation (0 for x in [0.25, 0.75]) lies outside them")),
])
def test_onnx_cast_recorded_bounds_hold(target, kind, tmp_path):
    """The interval pre-pass behind ``Cast(X, target)`` then ``Cast(.,
    DOUBLE)`` records intervals that hold the real Relu input: X itself for
    a float64 target, X truncated toward zero for INT64. Behind the INT64
    cast, recording nothing for the Relu (the pass stops trusting the graph
    there) is a right answer too; behind the float64 cast the Relu must be
    recorded, or the passing neighbour would check nothing.

    Catches a Cast to an integer type passed through as the identity
    (AFFINE-3). The zonotope pass is not part of the cell: it runs the
    reach's own Cast rule on a zonotope, the identity of registry AFFINE-1,
    which ``test_soundness_cast.py`` marks."""
    model, ref_path, in_shape, pre = _load_graph(
        tmp_path, lambda: _graph_cast(target))
    lo = np.array([0.25, -0.75, 1.25, -2.0])
    hi = np.array([0.75, 0.5, 2.75, -1.25])
    input_set = flat_set(kind, lo, hi)
    _assert_onnx_bounds(model, ref_path, in_shape, pre, input_set,
                        samples(lo, hi, 32), "ibp",
                        f"ibp pre-pass, Cast, {kind}",
                        allow_none=target != TensorProto.DOUBLE)


# --------------------------------------------------------------------------- #
# 3. Rounding: the recorded bounds hold the real value where float64 rounds    #
# --------------------------------------------------------------------------- #
# Each case is built so that round-to-nearest and outward rounding differ:
# a product or sum whose real value is not a float64 and whose float lies
# on the wrong side of it, with the terms of the other rows so small that a
# rounding allowance taken from the wrong row, or one that leaves a term
# out, does not cover it. The reference is the exact rational value of the
# network on the float64 weights (the evaluator above, or the case's own
# exact rule); there is no slack.
E30, E29, E60 = 2.0 ** -30, 2.0 ** -29, 2.0 ** -60
C1 = 1.0 + E30


def _box_case(lo, hi, kinds=("Box", "Star", "Zono")):
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    return {kind: flat_set(kind, lo, hi) for kind in kinds}, (lo, hi)


def _case_centre_product(sign):
    """``Linear(2, 2)`` without bias, row 0 = ``(1 + 2^-30) x0`` with x0 fixed
    at ``1 +- 2^-30``: the real product ``1 +- 2^-60`` (dropping the
    ``2^-29`` term) is not a float64, the float is one rounding away on the
    inner side, and the row has no generator: only a rounding allowance
    that counts the centre of the input covers it."""
    x0 = 1.0 + sign * E30
    model = nn.Sequential(_set(nn.Linear(2, 2, bias=False).double(),
                               [[C1, 0], [0, 1]]), nn.ReLU()).double().eval()
    sets, box = _box_case([x0, -1.0], [x0, 1.0])
    return model, sets, box, None


def _case_underflow():
    """``Linear(31, 32)`` of ones, then ``Linear(32, 1)`` of weights 2^-538 on
    inputs of size 2^-537: every product of the second layer is half the
    smallest subnormal and rounds to 0, in the centre and in each of the 31
    generators, while the real pre-activation reaches 512 subnormals."""
    l1 = _set(nn.Linear(31, 32, bias=False).double(), np.ones((32, 31)))
    l2 = _set(nn.Linear(32, 1, bias=False).double(),
              np.full((1, 32), 2.0 ** -538))
    model = nn.Sequential(l1, l2, nn.ReLU()).double().eval()
    lo, hi = np.full(31, -2.0 ** -537), np.full(31, 2.0 ** -537)
    lo[0], hi[0] = 0.0, 2.0 ** -536
    sets, box = _box_case(lo, hi, ("Box", "Star"))
    return model, sets, box, 24


def _image_case(lo_chw, hi_chw):
    lo_chw = np.asarray(lo_chw, np.float64)
    hi_chw = np.asarray(hi_chw, np.float64)
    return ({kind: image_set(kind, lo_chw, hi_chw)
             for kind in ("ImageStar", "ImageZono")}, (lo_chw, hi_chw))


def _case_conv_cancel(width, pixel, channel):
    """``Conv2d(2, 2, 1)`` without bias on a (2, 1, width) image. Output
    channel ``channel`` is ``-(1 + 2^-30) a + (1 + 2^-29) b``; at pixel
    ``pixel`` the input is fixed at (a, b) = (1 + 2^-30, 1), where the float
    product drops 2^-60 and the row computes exactly 0 while the real value
    is -2^-60. Elsewhere a is free in [-1, 1] and b is 0. The other output
    channel has weights 2^-40, so its rounding allowance is negligible: an
    allowance read from the wrong (channel, pixel) entry misses."""
    K = np.full((2, 2, 1, 1), 2.0 ** -40)
    K[channel, :, 0, 0] = [-C1, 1.0 + E29]
    model = nn.Sequential(_set(nn.Conv2d(2, 2, 1, bias=False).double(), K),
                          nn.ReLU()).double().eval()
    lo = np.zeros((2, 1, width))
    hi = np.zeros((2, 1, width))
    lo[0], hi[0] = -1.0, 1.0
    lo[:, 0, pixel] = hi[:, 0, pixel] = [C1, 1.0]
    sets, box = _image_case(lo, hi)
    return model, sets, box, None


def _case_conv_flatten_cancel(width, pixel):
    """``Conv2d(1, 2, 1)`` then Flatten then ReLU on a (1, 1, width) image.
    Output channel 1 is ``-(1 + 2^-30) x + (1 + 2^-29)``; at pixel ``pixel``
    x is fixed at 1 + 2^-30, so the row computes 0 where the real value is
    -2^-60; channel 0 has weight 2^-40 and no bias. Flattening moves the
    rows from (height, width, channel) to (channel, height, width) order, so
    the allowance carried through the Flatten must hold for every row."""
    conv = _set(nn.Conv2d(1, 2, 1).double(), [2.0 ** -40, -C1],
                [0.0, 1.0 + E29])
    model = nn.Sequential(conv, nn.Flatten(), nn.ReLU()).double().eval()
    lo, hi = np.full((1, 1, width), -1.0), np.full((1, 1, width), 1.0)
    lo[0, 0, pixel] = hi[0, 0, pixel] = C1
    sets, box = _image_case(lo, hi)
    return model, sets, box, None


def _case_conv_bias_sum(sign):
    """``Conv2d(1, 1, 1)`` with weight 2^-30 and bias ``sign`` (1 or -1),
    then ReLU, on a (1, 1, 2) image: pixel 0 is fixed at ``sign`` 2^-30,
    where the real pre-activation ``sign (1 + 2^-60)`` is not a float64 and
    the window sum plus the bias rounds to ``sign``, on the inner side of
    it; pixel 1 is free in [-1, 1]. The product is tiny next to the bias,
    so only a rounding allowance that counts the magnitude of the bias,
    whatever its sign, among the terms covers the miss."""
    conv = _set(nn.Conv2d(1, 1, 1).double(), [[[[E30]]]], [float(sign)])
    model = nn.Sequential(conv, nn.ReLU()).double().eval()
    sets, box = _image_case([[[sign * E30, -1.0]]], [[[sign * E30, 1.0]]])
    return model, sets, box, None


def _case_conv_nonpositive():
    """``Conv2d(1, 1, 1)`` with weight 1 + 2^-30 and no bias, then ReLU, on a
    (1, 1, 2) image whose pixels lie in [-(1 + 2^-30), 0]: at the lower end
    the real pre-activation -(1 + 2^-29 + 2^-60) is not a float64 and the
    product rounds to -(1 + 2^-29), above it. The upper end of every input
    is 0, so only a rounding allowance taken from the larger of the two
    ends' magnitudes covers the miss."""
    conv = _set(nn.Conv2d(1, 1, 1, bias=False).double(), [[[[C1]]]])
    model = nn.Sequential(conv, nn.ReLU()).double().eval()
    sets, box = _image_case([[[-C1, -C1]]], [[[0.0, 0.0]]])
    return model, sets, box, None


def _case_sigmoid_underflow():
    """``Sigmoid``, ``Linear(1, 1)`` of weight 2^1020, ``ReLU`` on x in
    [-740, -712]: float64 ``expit`` returns exactly 0 below about -709.78,
    the real sigmoid is about e^x > 0, and the large weight turns it into a
    pre-activation between about 1e-14 and 0.04."""
    model = nn.Sequential(nn.Sigmoid(),
                          _set(nn.Linear(1, 1).double(), [[2.0 ** 1020]],
                               [0.0]), nn.ReLU()).double().eval()
    sets, box = _box_case([-740.0], [-712.0], ("Box", "Star"))
    return model, sets, box, None


def _case_avgpool_third():
    """``AvgPool2d((1, 3))`` then ReLU on a (1, 1, 3) image: the division by 3
    rounds."""
    model = nn.Sequential(nn.AvgPool2d((1, 3)), nn.ReLU()).double().eval()
    sets, box = _image_case([[[0.1, -0.7, 1.3]]], [[[0.35, -0.2, 1.9]]])
    return model, sets, box, None


def _case_batchnorm_shift():
    """``BatchNorm1d`` with variance 1 (eps 0) and a mean and scale that
    are not dyadic: the folded shift ``beta - gamma mean`` rounds."""
    bn = _set_bn(nn.BatchNorm1d(2, eps=0.0).double(), [0.1, -0.3], [1.0, 1.0],
                 [3.0, -0.7], [-0.3, 0.21])
    model = nn.Sequential(bn, nn.ReLU()).double().eval()
    sets, box = _box_case([0.05, -0.5], [0.15, 0.2], ("Box", "Star"))
    return model, sets, box, None


def _case_relu_inactive_error():
    """``Linear`` (the centre product of :func:`_case_centre_product`, minus
    1 + 2^-29), ``ReLU``, ``Linear`` of weight 2^50 and bias -2^-11,
    ``ReLU``. The first row computes exactly 0 (the ReLU zeroes it as
    inactive) while its real value is 2^-60, so the real input of the second
    ReLU is 2^50 2^-60 - 2^-11 = 2^-11, above the computed -2^-11: the
    rounding allowance of a zeroed row must carry what the real input can
    exceed 0 by."""
    l1 = _set(nn.Linear(2, 2).double(), [[C1, 0], [0, 1]], [-(1.0 + E29), 0])
    l2 = _set(nn.Linear(2, 2).double(), [[2.0 ** 50, 0], [0, 1]],
              [-2.0 ** -11, 0])
    model = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU()).double().eval()
    sets, box = _box_case([C1, -1.0], [C1, 1.0])
    return model, sets, box, None


# Interval ends at which float64 ``expit``, ``tanh`` and the product 0.1 x
# land on the inner side of the real value (found with mpmath at 200 bits;
# the cell's reference decides again): the recorded interval is right only
# if the pass moves its ends outward.
_SIGMOID_ENDS = ([-3.0, -1.640625, -1.015625, -0.375],
                 [-2.984375, -1.625, -1.0, -0.359375])
_TANH_ENDS = ([-3.03125, -1.609375, -0.96875, -0.328125],
              [-3.0, -1.59375, -0.953125, -0.3125])


def _case_smooth_ends(layer, ends):
    """``layer`` (Sigmoid or Tanh) then ``ReLU`` on a box whose every end is
    one where the float64 function rounds inward."""
    model = nn.Sequential(layer, nn.ReLU()).double().eval()
    sets, box = _box_case(*ends, kinds=("Box", "Star"))
    return model, sets, box, None


def _case_leaky_slope():
    """``LeakyReLU(0.1)`` then ``ReLU``: the float slope 0.1 times the lower
    ends -3.984375 and -3.75 rounds up, above the real product."""
    model = nn.Sequential(nn.LeakyReLU(0.1), nn.ReLU()).double().eval()
    sets, box = _box_case([-3.984375, -3.75], [0.5, -0.5],
                          kinds=("Box", "Star"))
    return model, sets, box, None


# (id, builder, passes): the zonotope pass bounds its rounding along
# Linear, Conv2d, ReLU and Flatten only, so cases through other layers run
# the interval pass alone (behind the others see ROUND-11 below).
ROUNDING_CASES = [
    ("centre-product-up", lambda: _case_centre_product(+1), ("ibp", "zono")),
    ("centre-product-down", lambda: _case_centre_product(-1),
     ("ibp", "zono")),
    ("underflow-31", _case_underflow, ("ibp", "zono")),
    ("conv-cancel-w2-p0-c1", lambda: _case_conv_cancel(2, 0, 1),
     ("ibp", "zono")),
    ("conv-cancel-w3-p1-c0", lambda: _case_conv_cancel(3, 1, 0),
     ("ibp", "zono")),
    ("conv-cancel-w3-p2-c1", lambda: _case_conv_cancel(3, 2, 1),
     ("ibp", "zono")),
    ("conv-flatten-w2-p0", lambda: _case_conv_flatten_cancel(2, 0),
     ("ibp", "zono")),
    ("conv-flatten-w3-p1", lambda: _case_conv_flatten_cancel(3, 1),
     ("ibp", "zono")),
    ("conv-bias-sum", lambda: _case_conv_bias_sum(+1), ("ibp", "zono")),
    ("conv-negative-bias-sum", lambda: _case_conv_bias_sum(-1),
     ("ibp", "zono")),
    ("conv-nonpositive-input", _case_conv_nonpositive, ("ibp", "zono")),
    ("sigmoid-underflow", _case_sigmoid_underflow, ("ibp",)),
    ("avgpool-third", _case_avgpool_third, ("ibp",)),
    ("batchnorm-shift", _case_batchnorm_shift, ("ibp",)),
    ("relu-inactive-error", _case_relu_inactive_error, ("ibp", "zono")),
    ("sigmoid-ends", lambda: _case_smooth_ends(nn.Sigmoid(), _SIGMOID_ENDS),
     ("ibp",)),
    ("tanh-ends", lambda: _case_smooth_ends(nn.Tanh(), _TANH_ENDS),
     ("ibp",)),
    ("leaky-slope", _case_leaky_slope, ("ibp",)),
]


def _rounding_params():
    out = []
    for name, build, passes in ROUNDING_CASES:
        for method in passes:
            for walker in ("sequential", "traced"):
                out.append(pytest.param(name, build, method, walker,
                                        id=f"{name}-{method}-{walker}"))
    return out


@pytest.mark.parametrize("name, build, method, walker", _rounding_params())
def test_recorded_bounds_hold_at_rounding_scale(name, build, method, walker):
    """The recorded bounds hold the real (rational) pre-activation on inputs
    where round-to-nearest lands on the inner side of it, for every input
    set type the case allows; no slack.

    Catches a rounding allowance of the zonotope pass that leaves the input
    centre out of the magnitude, counts one underflowing product per inner
    product instead of every one, reads the convolved allowance in
    (channel, height, width) order for an image zonotope stored in (height,
    width, channel) order, or keeps it per row through a Flatten; and an
    interval pass whose Sigmoid upper bound is not raised off 0 where
    ``expit`` underflows, or whose pooling or BatchNorm arithmetic is not
    rounded outward, or whose convolution margin leaves the bias out of the
    magnitude of the terms or reads an input's magnitude from one end of
    its interval, and a convolution allowance of the zonotope pass that
    takes a negative bias with its sign. Also a zeroed (inactive) ReLU row
    that carries no allowance although its real input is above 0, a
    Sigmoid, Tanh or LeakyReLU slope product taken as float64 computes it,
    and, on the traced walker, an allowance not carried from the layer that
    produced the ReLU's input."""
    model, sets, (lo, hi), n_points = build()
    points = samples(lo, hi, n_points or 64)
    target = model if walker == "sequential" else fx.symbolic_trace(model)
    refs = _sequential_refs(model, points, "index" if walker == "sequential"
                            else target)
    for kind, input_set in sets.items():
        what = f"{method} pre-pass, {walker} walker, {name}, {kind}"
        bounds = compute_intermediate_bounds(target, input_set, method=method)
        if not bounds:
            pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
        _check_recorded(bounds, refs, what)


def _const_graph(op, const):
    nodes = [_node(op, ["X", "C"], ["p"]),
             _node("Relu", ["p"], ["Y"], name="act")]
    return nodes, {"C": np.asarray(const, np.float64)}, [1, 3], {"act": "p"}, \
        TensorProto.DOUBLE


_EXACT_OPS = {"Add": lambda x, c: x + c, "Sub": lambda x, c: x - c,
              "Mul": lambda x, c: x * c, "Div": lambda x, c: x / c}


@pytest.mark.parametrize("op, const", [
    ("Add", [-E60, 2.0 ** -55, -3 * 2.0 ** -56]),
    ("Sub", [E60, -2.0 ** -55, 3 * 2.0 ** -56]),
    ("Mul", [1.0 + 2.0 ** -52, 1.0 - 2.0 ** -53, -(1.0 + 3 * 2.0 ** -52)]),
    ("Div", [3.0, -7.0, 1.0 + 2.0 ** -52]),
])
@pytest.mark.parametrize("kind", ["Box", "Star"])
def test_onnx_constant_operand_bounds_round_outward(op, const, kind,
                                                    tmp_path):
    """The interval pre-pass behind an ONNX Add, Sub, Mul or Div by a
    constant records bounds that hold the exact rational result (computed
    here, not by onnxruntime, which rounds the same way n2v would): e.g.
    ``1 + (-2^-60)`` rounds to 1.0 in float64, above the real value.

    Catches the constant operation computed in round-to-nearest on either
    end instead of rounded down for the lower bound and up for the
    upper."""
    model, _, _, pre = _load_graph(tmp_path, lambda: _const_graph(op, const))
    lo, hi = np.array([1.0, 0.75, -2.0]), np.array([2.0, 1.5, -1.0])
    points = samples(lo, hi, 32)
    c = [_q(v) for v in const]
    refs = {"act": []}
    for x in points:
        v = np.array([_EXACT_OPS[op](_q(xi), ci) for xi, ci in zip(x, c)],
                     dtype=object)
        refs["act"].append((v, v))
    bounds = compute_intermediate_bounds(model, flat_set(kind, lo, hi),
                                         method="ibp", input_shape=(3,))
    if set(bounds) != {"act"}:
        pytest.fail(f"test data: recorded {sorted(bounds)}")
    _check_recorded(bounds, refs, f"ibp pre-pass, ONNX {op} constant, {kind}")


# (op, A, B): Relu(op(X A, X B)) with X = (1, x1), x1 in [-1, 1]. Every
# product of the two MatMuls has the end 1 or +-1 and is exact, so both
# branches are exact intervals, while entry 0 of the result is not a
# float64: 1 + 2^-60 (Add), 1 - 2^-60 (Sub), (1 + 2^-30)^2 = 1 + 2^-29 +
# 2^-60 (Mul).
_BRANCH_CASES = {
    "Add": ([[1.0, 0.0], [0.0, 1.0]], [[E60, 0.0], [0.0, 1.0]]),
    "Sub": ([[1.0, 0.0], [0.0, 1.0]], [[E60, 0.0], [0.0, 1.0]]),
    "Mul": ([[C1, 0.0], [0.0, 1.0]], [[C1, 0.0], [0.0, 1.0]]),
}


@pytest.mark.parametrize("op", sorted(_BRANCH_CASES))
def test_onnx_two_branch_operations_round_outward(op, tmp_path):
    """The interval pre-pass behind an ONNX Add, Sub or Mul of two computed
    branches (``X A`` and ``X B``) records bounds that hold the exact
    rational result, where one rounding to nearest lands on the inner side
    of it (1 + 2^-60 rounds to 1.0). Reference: the exact evaluation here,
    not onnxruntime (which rounds the same way n2v would).

    Catches a sum, difference or end product of two branches computed in
    round-to-nearest instead of rounded down for the lower bound and up for
    the upper."""
    A, B = _BRANCH_CASES[op]
    nodes = [_node("MatMul", ["X", "A"], ["a"]), _node("MatMul", ["X", "B"], ["b"]),
             _node(op, ["a", "b"], ["p"]), _node("Relu", ["p"], ["Y"], name="act")]
    path = _save_graph(tmp_path / "branches.onnx", nodes, [1, 2], ["Y"],
                       {"A": np.array(A), "B": np.array(B)})
    from n2v.utils.model_loader import load_onnx
    lo, hi = np.array([1.0, -1.0]), np.array([1.0, 1.0])
    Aq, Bq = _qarr(np.array(A)), _qarr(np.array(B))
    exact = {"Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
             "Mul": lambda a, b: a * b}[op]
    refs = {"act": []}
    for x in samples(lo, hi, 16):
        xq = _qarr(x)
        a = [xq[0] * Aq[0, j] + xq[1] * Aq[1, j] for j in range(2)]
        b = [xq[0] * Bq[0, j] + xq[1] * Bq[1, j] for j in range(2)]
        v = np.array([exact(a[j], b[j]) for j in range(2)], dtype=object)
        refs["act"].append((v, v))
    bounds = compute_intermediate_bounds(load_onnx(path), flat_set("Star", lo, hi),
                                         method="ibp", input_shape=(2,))
    if set(bounds) != {"act"}:
        pytest.fail(f"test data: recorded {sorted(bounds)}")
    _check_recorded(bounds, refs, f"ibp pre-pass, ONNX {op} of two branches")


# --------------------------------------------------------------------------- #
# 4. The zonotope pass behind a layer whose rounding it does not bound          #
# --------------------------------------------------------------------------- #
# The zonotope pass runs the approx zonotope reach in round-to-nearest and
# carries a bound on its own rounding only along Linear, Conv2d, ReLU,
# Flatten and pass-through layers from the input. Behind any other layer or
# ONNX graph operation that bound is unknown, and the bounds it records are
# the row bounds of the computed zonotope with no widening (registry
# ROUND-11). Each cell below puts one such layer in front of the same two
# affine maps: the first makes an entry exactly 1 + 2^-30, the second
# computes (1 + 2^-30)^2 - (1 + 2^-29), whose real value is 2^-60 while one
# rounding of the product gives exactly 0. The recorded upper bound of that
# ReLU input must hold 2^-60 (reference: the exact evaluator above, in
# Fractions). Passing neighbours: the same maps behind a ReLU, Identity and
# Dropout, a Conv2d, or nothing, where the pass tracks its rounding, and the
# interval pass on every case.
_R11_CANCEL = [[C1, 0.0], [0.0, 1.0]], [-(1.0 + E29), 0.0]


def _r11_bn_identity():
    """BatchNorm1d that is the identity on its input: mean 0, variance 1,
    eps 0, scale 1, shift 0 (exact in every arithmetic)."""
    return _set_bn(nn.BatchNorm1d(2, eps=0.0).double(), [0.0, 0.0],
                   [1.0, 1.0], [1.0, 1.0], [0.0, 0.0])


def _r11_avg_conv():
    """Conv2d 2 -> 2, kernel (1, 2), each output channel the mean of the
    two pixels of its own input channel."""
    K = np.zeros((2, 2, 1, 2))
    K[0, 0, 0, :] = K[1, 1, 0, :] = 0.5
    return _set(nn.Conv2d(2, 2, (1, 2)).double(), K, [0.0, 0.0])


# name -> (front layers, image input?, x0, first affine map (W, b)).
# Flat input (x0, x1): x0 fixed, x1 in [-1, 1]. Image input (2, 1, 2):
# channel 0 fixed at x0, channel 1 free in [-1, 1]; pooled to (2, 1, 1) and
# flattened. Sigmoid(0) = 1/2 and Tanh(0) = 0 exactly, so the first map
# still makes 1 + 2^-30 from them.
_R11_FRONTS = {
    "leakyrelu": (lambda: [nn.LeakyReLU(0.25)], False, 1.0,
                  ([[1, 0], [0, 1]], [E30, 0])),
    "sigmoid": (lambda: [nn.Sigmoid()], False, 0.0,
                ([[2, 0], [0, 0]], [E30, 0.25])),
    "tanh": (lambda: [nn.Tanh()], False, 0.0,
             ([[1, 0], [0, 0]], [C1, 0.25])),
    "batchnorm": (lambda: [_r11_bn_identity()], False, 1.0,
                  ([[1, 0], [0, 1]], [E30, 0])),
    "avgpool": (lambda: [nn.AvgPool2d((1, 2)), nn.Flatten()], True, 1.0,
                ([[1, 0], [0, 1]], [E30, 0])),
    "maxpool": (lambda: [nn.MaxPool2d((1, 2)), nn.Flatten()], True, 1.0,
                ([[1, 0], [0, 1]], [E30, 0])),
    # neighbours: the pass bounds its rounding along these
    "linear-only": (lambda: [], False, 1.0, ([[1, 0], [0, 1]], [E30, 0])),
    "relu": (lambda: [nn.ReLU()], False, 1.0, ([[1, 0], [0, 1]], [E30, 0])),
    "identity-dropout": (lambda: [nn.Identity(), nn.Dropout(0.5)], False,
                         1.0, ([[1, 0], [0, 1]], [E30, 0])),
    "conv-flatten": (lambda: [_r11_avg_conv(), nn.Flatten()], True, 1.0,
                     ([[1, 0], [0, 1]], [E30, 0])),
}
_R11_UNBOUNDED = ("leakyrelu", "sigmoid", "tanh", "batchnorm", "avgpool",
                  "maxpool")


def _r11_model(name):
    front, image, x0, (W1, b1) = _R11_FRONTS[name]
    W2, b2 = _R11_CANCEL
    layers = front() + [_set(nn.Linear(2, 2).double(), W1, b1),
                        _set(nn.Linear(2, 2).double(), W2, b2), nn.ReLU()]
    model = nn.Sequential(*layers).double().eval()
    if image:
        lo = np.array([[[x0, x0]], [[-1.0, -1.0]]])
        hi = np.array([[[x0, x0]], [[1.0, 1.0]]])
    else:
        lo, hi = np.array([x0, -1.0]), np.array([x0, 1.0])
    return model, image, lo, hi


def _r11_known_bug(name):
    return known_bug(
        7, f"ROUND-11: the zonotope pre-pass does not bound its own rounding "
           f"behind {name}, and records the row bounds of the computed "
           f"zonotope unwidened: the ReLU input (1 + 2^-30)^2 - (1 + 2^-29) "
           f"= 2^-60 is recorded as [0, 0]")


def _r11_params():
    out = []
    for name in _R11_FRONTS:
        for walker in ("sequential", "traced"):
            for method in ("zono", "ibp"):
                marks = (_r11_known_bug(name)
                         if method == "zono" and name in _R11_UNBOUNDED
                         else ())
                out.append(pytest.param(name, walker, method, marks=marks,
                                        id=f"{name}-{walker}-{method}"))
    return out


@pytest.mark.parametrize("name, walker, method", _r11_params())
def test_bounds_behind_every_layer_hold_at_rounding_scale(name, walker,
                                                          method):
    """Every interval the pre-pass records holds the real (rational)
    pre-activation behind a LeakyReLU, Sigmoid, Tanh, BatchNorm, AvgPool or
    MaxPool, for every input set type; a cancellation of the next affine map
    rounds to 0 where the real value is 2^-60.

    Catches the zonotope pass recording, behind a layer whose rounding it
    does not bound, the row bounds of the round-to-nearest zonotope without
    a margin (ROUND-11, marked on those layers), and on the neighbours a
    rounding bound that is lost through a ReLU, Identity, Dropout, Conv2d or
    Flatten, or not added to the recorded bounds. Behind a marked layer a
    pass that records nothing for the last ReLU is a right answer too; on a
    neighbour the last ReLU must be recorded."""
    model, image, lo, hi = _r11_model(name)
    target = model if walker == "sequential" else fx.symbolic_trace(model)
    kinds = ("ImageStar", "ImageZono") if image else ("Box", "Star", "Zono")
    points = samples(lo, hi, 16)
    refs = _sequential_refs(model, points, "index" if walker == "sequential"
                            else target)
    last = list(refs)[-1]                       # the ReLU at the end
    for kind in kinds:
        input_set = (image_set(kind, lo, hi) if image
                     else flat_set(kind, lo, hi))
        what = f"{method} pre-pass, {walker} walker, behind {name}, {kind}"
        bounds = compute_intermediate_bounds(target, input_set,
                                             method=method)
        if last not in bounds and not (method == "zono"
                                       and name in _R11_UNBOUNDED):
            pytest.fail(f"{what}: the last ReLU ({last!r}) was not recorded "
                        f"(recorded {sorted(map(str, bounds))})")
        _check_recorded(bounds, refs, what)


def _r11_onnx_graph(front):
    """``X (1, 2)`` double: an optional front operation, then the two affine
    maps of the cells above as ``MatMul`` and ``Add`` (ONNX graph operations
    the zonotope pass runs itself) or as ``Gemm`` (a torch Linear after
    loading, whose rounding the pass bounds), then a Relu ``act``."""
    W1, b1 = np.eye(2), np.array([E30, 0.0])
    W2, b2 = (np.array(v) for v in _R11_CANCEL)
    if front == "neg":
        W1 = -W1
    inits = {"W1": W1, "B1": b1, "W2": W2.T.copy(), "B2": b2}
    nodes, src = [], "X"
    if front == "reshape":
        inits["S"] = np.array([1, 2], dtype=np.int64)
        nodes.append(_node("Reshape", ["X", "S"], ["r"]))
        src = "r"
    elif front == "neg":
        nodes.append(_node("Neg", ["X"], ["r"]))
        src = "r"
    elif front == "mul-const":
        inits["M"] = np.array([1.0, 1.0])
        nodes.append(_node("Mul", ["X", "M"], ["r"]))
        src = "r"
    if front == "matmul-add":
        inits["W1"], inits["W2"] = W1, W2.T.copy()
        nodes += [_node("MatMul", [src, "W1"], ["m1"]),
                  _node("Add", ["m1", "B1"], ["h"]),
                  _node("MatMul", ["h", "W2"], ["m2"]),
                  _node("Add", ["m2", "B2"], ["p"])]
    else:
        nodes += [_node("Gemm", [src, "W1", "B1"], ["h"]),
                  _node("Gemm", ["h", "W2", "B2"], ["p"])]
    nodes.append(_node("Relu", ["p"], ["Y"], name="act"))
    return nodes, inits, [1, 2], {"act": "p"}, TensorProto.DOUBLE


def _r11_onnx_exact(front, x):
    """The real Relu input of :func:`_r11_onnx_graph` at the point ``x``."""
    x = [_q(v) for v in x]
    if front == "neg":
        x = [-v for v in x]
        h = [-x[0] + _q(E30), -x[1]]
    else:
        h = [x[0] + _q(E30), x[1]]
    p0 = _q(C1) * h[0] - _q(1.0 + E29)
    return np.array([p0, h[1]], dtype=object)


_R11_ONNX = ("matmul-add", "reshape", "neg", "mul-const")


def _r11_onnx_params():
    out = []
    for front in _R11_ONNX + ("gemm",):
        for method in ("zono", "ibp"):
            marks = ()
            if front in _R11_ONNX and method == "zono":
                marks = known_bug(
                    7, f"ROUND-11: behind the ONNX graph operation of "
                       f"'{front}' the zonotope pre-pass does not bound its "
                       f"own rounding and records the unwidened row bounds "
                       f"[0, 0] of a Relu input that is 2^-60")
            out.append(pytest.param(front, method, marks=marks,
                                    id=f"{front}-{method}"))
    return out


@pytest.mark.parametrize("front, method", _r11_onnx_params())
def test_onnx_bounds_behind_graph_operations_hold_at_rounding_scale(
        front, method, tmp_path):
    """Every interval the pre-pass records for the Relu of an ONNX graph
    holds the real (rational) pre-activation, 2^-60, which one rounding of
    the cancellation ``(1 + 2^-30)^2 - (1 + 2^-29)`` turns into 0.

    Catches the zonotope pass recording unwidened row bounds behind an ONNX
    MatMul, Add, Reshape, Neg or Mul by a constant, which it runs without
    bounding their rounding (ROUND-11, marked); the Gemm neighbour (two torch
    Linear layers after loading) passes with the pass's rounding bound, and
    the interval pass passes on every graph. The interval pass, which has no
    rule for Reshape, may record nothing behind it."""
    model, _, _, pre = _load_graph(tmp_path, lambda: _r11_onnx_graph(front))
    lo, hi = np.array([1.0, -1.0]), np.array([1.0, 1.0])
    points = samples(lo, hi, 16)
    refs = {"act": [(_r11_onnx_exact(front, x),) * 2 for x in points]}
    for kind in ("Box", "Star", "Zono"):
        what = f"{method} pre-pass, ONNX {front}, {kind}"
        bounds = compute_intermediate_bounds(
            model, flat_set(kind, lo, hi), method=method, input_shape=(2,))
        may_skip = (front in _R11_ONNX and method == "zono") or (
            front == "reshape" and method == "ibp")
        if "act" not in bounds and not may_skip:
            pytest.fail(f"{what}: the Relu was not recorded")
        _check_recorded(bounds, refs, what)


# The registry's end-to-end case: y = Relu(x W + B) with W = [w, K],
# B = [-(w + 2^-30), -K], w = 1 + 2^-23, K = 2^50, on the Star
# x = 1 + 2^-30 a, a in [0, 2]. The star's own affine map is exact here; the
# zonotope the pre-pass builds from the star's range has a centre whose
# product with w rounds up, so the recorded lower bound of the first neuron
# is about 1.1e-16 above its real minimum -2^-30 (at x = 1), and the
# triangle row of the approx ReLU cuts off the true output (0, 0) of x = 1.
# The violated row has coefficients near 2^-30 and a right-hand side near
# -5.6e-17, below the LP feasibility tolerance, so membership is decided in
# exact arithmetic (``rational.member``).
_R11_W, _R11_K = 1.0 + 2.0 ** -23, 2.0 ** 50
_R11_B = (-(_R11_W + E30), -_R11_K)


def _r11_e2e_star():
    return Star(np.array([[1.0, E30]]), np.zeros((0, 1)), np.zeros((0, 1)),
                np.array([[0.0]]), np.array([[2.0]]))


def _r11_e2e_model(graph, tmp_path):
    if graph == "onnx-matmul-add":
        from n2v.utils.model_loader import load_onnx
        nodes = [_node("MatMul", ["X", "W"], ["m"]),
                 _node("Add", ["m", "B"], ["p"]),
                 _node("Relu", ["p"], ["Y"], name="act")]
        path = _save_graph(tmp_path / "e2e.onnx", nodes, [1, 1], ["Y"],
                           {"W": np.array([[_R11_W, _R11_K]]),
                            "B": np.array(_R11_B)})
        return load_onnx(path)
    lin = _set(nn.Linear(1, 2).double(), [[_R11_W], [_R11_K]], list(_R11_B))
    return nn.Sequential(nn.LeakyReLU(0.5), lin, nn.ReLU()).double().eval()


def _r11_e2e_params():
    out = []
    for graph in ("onnx-matmul-add", "torch-leakyrelu"):
        for pre in (None, "ibp", "zono"):
            marks = () if pre != "zono" else known_bug(
                7, "ROUND-11: the zonotope pre-pass records a lower bound "
                   "1.1e-16 above the real minimum of a ReLU input (no "
                   "margin for its own rounding behind an ONNX MatMul/Add or "
                   "a LeakyReLU), and the approx ReLU's triangle row cuts "
                   "off the true output (0, 0) of the member point x = 1")
            out.append(pytest.param(graph, pre, marks=marks,
                                    id=f"{graph}-{pre}"))
    return out


@pytest.mark.parametrize("graph, pre", _r11_e2e_params())
def test_reach_with_zonotope_bounds_holds_outputs_at_rounding_scale(
        graph, pre, tmp_path):
    """Approx reach of ``Relu(x W + B)`` (an ONNX MatMul and Add, or a torch
    LeakyReLU then Linear) on the Star x = 1 + 2^-30 a, a in [0, 2], holds
    the exact output of every checked member point, decided in exact
    arithmetic, with the pre-pass off, on the interval pass and on the
    zonotope pass.

    Catches a recorded bound that sits inside the real range by a rounding
    error and is then used as given by the approx ReLU (ROUND-11 on the
    zonotope pass, marked); the interval pass and no pre-pass are the
    passing neighbours with the same reference."""
    model = _r11_e2e_model(graph, tmp_path)
    options = {} if pre is None else {"precompute_bounds": pre}
    sets = reach_model(model, _r11_e2e_star(), "approx",
                       f"{graph}, precompute_bounds={pre}", **options)
    computed = [rational.Membership(rational.star(s)) for s in sets]
    for a in (F(0), F(1, 4), F(1), F(2)):
        x = F(1) + _q(E30) * a
        y = [max(_q(_R11_W) * x + _q(_R11_B[0]), F(0)),
             max(_q(_R11_K) * x + _q(_R11_B[1]), F(0))]
        if not any(m(y) for m in computed):
            raise AssertionError(
                f"{graph}, precompute_bounds={pre}: the exact output "
                f"{[float(v) for v in y]} of the member point x = 1 + "
                f"2^-30 * {a} is in none of the {len(sets)} returned sets")


# --------------------------------------------------------------------------- #
# 5. The graph walk: trust, operations without a rule, functional layers       #
# --------------------------------------------------------------------------- #
# The interval pass trusts a node only when every operation on its path was
# propagated by a rule; a nonlinear layer behind an untrusted node gets no
# bounds (reach then computes its own). A pass that trusts too much records
# the bounds of another node, which the cells below see as a real value
# outside the recorded interval or as the wrong number of entries.
def _graph_untrusted():
    """``m = X A`` (A 4 x 3, m1 in [-1.19, 0.56], so |m1| exceeds every
    bound of m1); Relu ``act1`` on m (trusted); ``u = Abs(m)`` (no interval
    rule: an ``OnnxFunction`` after loading), Relu ``act2`` on u; an Add of
    the trusted branch ``act1`` and the untrusted ``u``, Relu ``act3``; a
    Split of that, Relu ``act4`` on the second part (a ``getitem`` of an
    untrusted multi-output node). Only ``act1`` may be recorded; recording
    ``act2`` to ``act4`` with the bounds the pass carries (those of m, or of
    the whole Split input) misses real values. The zonotope pass stops at
    the Abs (the zonotope reach has no rule for it) and keeps ``act1``."""
    inits = {"A": np.array([[1, 0, 0.5], [0, -1, 0], [0.5, 0, 0],
                            [0, 0.25, -1]]),
             "S": np.array([1, 2], dtype=np.int64)}
    nodes = [_node("MatMul", ["X", "A"], ["m"]),
             _node("Relu", ["m"], ["r1"], name="act1"),
             _node("Abs", ["m"], ["u"]),
             _node("Relu", ["u"], ["r2"], name="act2"),
             _node("Add", ["r1", "u"], ["s"]),
             _node("Relu", ["s"], ["r3"], name="act3"),
             _node("Split", ["r3", "S"], ["sa", "sb"], axis=1),
             _node("Relu", ["sb"], ["Y"], name="act4")]
    return nodes, inits, [1, 4], {"act1": "m", "act2": "u", "act3": "s",
                                  "act4": "sb"}, TensorProto.DOUBLE


def _graph_unpropagated():
    """Operations the interval pass cannot propagate, each in front of a
    Relu: a MatMul with the constant on the left (``C X``, C = [[-1/2]]),
    an Add and a Sub with the constant first (``B + X``, ``B - X``; the Sub
    with its operands swapped is ``X - B``), an Add of two computed
    branches of different sizes (broadcast), a MatMul by a 3-D constant, a
    Div of two computed branches, and a MatMul of two computed branches
    (``w`` times the transpose of ``X G``). None of their Relus may be
    recorded with the bounds of another node; ``act0`` (a plain MatMul) is
    recorded."""
    inits = {"W": np.array([[1, 0.5], [0, -1], [0.5, 0], [0, 0.25]]),
             "C": np.array([[-0.5]]), "B": np.array([0.5, -1, 0.25, 0]),
             "V": np.array([[1], [0.5], [0], [-0.5]]),
             "T": np.array([[[1, 0], [0, 1], [0, 0], [0.5, 0]],
                            [[0, 0.5], [1, 0], [0, -1], [0, 0]]]),
             "Q": np.array([[0, 0.25], [0, 0], [0.5, 0], [0, 0]]),
             "Qb": np.array([1.0, 2.0]),
             "G": np.array([[0.5, 0], [0, 1], [-1, 0], [0, 0.25]])}
    nodes = [_node("MatMul", ["X", "W"], ["w"]),
             _node("Relu", ["w"], ["r0"], name="act0"),
             _node("MatMul", ["C", "X"], ["cx"]),
             _node("Relu", ["cx"], ["r1"], name="act1"),
             _node("Add", ["B", "X"], ["bx"]),
             _node("Relu", ["bx"], ["r2"], name="act2"),
             _node("Sub", ["B", "X"], ["xb"]),
             _node("Relu", ["xb"], ["r7"], name="act7"),
             _node("MatMul", ["X", "V"], ["v"]),
             _node("Add", ["w", "v"], ["wv"]),
             _node("Relu", ["wv"], ["r3"], name="act3"),
             _node("MatMul", ["X", "T"], ["t"]),
             _node("Relu", ["t"], ["r4"], name="act4"),
             _node("MatMul", ["X", "Q"], ["q0"]),
             _node("Add", ["q0", "Qb"], ["q"]),
             _node("Div", ["w", "q"], ["dq"]),
             _node("Relu", ["dq"], ["r5"], name="act5"),
             _node("MatMul", ["X", "G"], ["g"]),
             _node("Transpose", ["g"], ["gt"], perm=[1, 0]),
             _node("MatMul", ["w", "gt"], ["wg"]),
             _node("Relu", ["wg"], ["r6"], name="act6"),
             _node("Add", ["r0", "r5"], ["Y"])]
    return nodes, inits, [1, 4], {"act0": "w", "act1": "cx", "act2": "bx",
                                  "act3": "wv", "act4": "t", "act5": "dq",
                                  "act6": "wg",
                                  "act7": "xb"}, TensorProto.DOUBLE


def _onnx_refs_any(ref_path, in_shape, pre, points):
    """:func:`_onnx_refs` for Relu inputs of any rank, flattened in the
    tensor's own row-major order (the order a flat set keeps)."""
    tensors = _ort_tensors(ref_path, points, in_shape)
    refs = {}
    for act, tensor in pre.items():
        vals = [_qarr(np.asarray(t[tensor], np.float64).reshape(-1))
                for t in tensors]
        refs[act] = [(v, v) for v in vals]
    return refs


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("graph, must_record", [
    ("untrusted", {"act1"}), ("unpropagated", {"act0"})])
def test_onnx_untrusted_nodes_record_no_wrong_bounds(graph, must_record,
                                                     kind, method, tmp_path):
    """Every interval the pre-pass records for a Relu of an ONNX graph holds
    the real pre-activation (onnxruntime, float64, exact on these dyadic
    data), when the graph has operations without an interval rule.

    Catches a graph walk that keeps the trust of a layer's input when the
    layer has no rule (the Abs), records a nonlinear layer whose input is
    untrusted, trusts an Add of a trusted and an untrusted branch, trusts
    the output of a multi-output node read through ``getitem``, falls back
    to the network input box as if it were trusted, or propagates an
    operation it cannot (a MatMul with the constant on the left or of rank
    3, an Add with the constant first or of two branches of different
    sizes, a Sub with the constant first read with its operands swapped, a
    Div of two branches). The plain Relus in front must be recorded, or the
    cell would check nothing."""
    build = {"untrusted": _graph_untrusted,
             "unpropagated": _graph_unpropagated}[graph]
    model, ref_path, in_shape, pre = _load_graph(tmp_path, build)
    lo = np.array([-1.0, -0.5, 0.25, -0.75])
    hi = np.array([0.5, 1.0, 0.75, 0.25])
    what = f"{method} pre-pass, ONNX {graph}, {kind}"
    bounds = compute_intermediate_bounds(model, flat_set(kind, lo, hi),
                                         method=method, input_shape=(4,))
    if not must_record <= set(bounds):
        pytest.fail(f"{what}: {sorted(must_record)} not recorded "
                    f"(recorded {sorted(bounds)})")
    _check_recorded(bounds, _onnx_refs_any(ref_path, in_shape, pre,
                                           samples(lo, hi, 32)), what)


class _Functional(nn.Module):
    """Functional activations and tensor methods, traced by torch.fx:
    Conv2d on a (2, 3, 5) image, ``contiguous`` (the tensor stays an
    image), ``F.relu`` (an image), ``contiguous`` again,
    ``flatten(1)``, Linear, ``F.leaky_relu`` with the slope written as the
    second argument (torch.fx records it as the keyword ``negative_slope``,
    so the converter's branch for a positional slope is not reached by a
    traced module), Linear, ``torch.sigmoid``, Linear, ``torch.tanh``,
    ``torch.flatten``,
    then two producers the interval pass has no rule for, a ``view`` and an
    ``operator.add`` of two branches, each read by ``F.relu``: those two
    Relus may get no bounds, and bounds borrowed from the network input box
    have the wrong number of entries."""

    def __init__(self):
        super().__init__()
        self.conv = _set(nn.Conv2d(2, 2, (2, 3)).double(),
                         _dyadic((2, 2, 2, 3), 31), [0.5, -0.5])
        self.l1 = _set(nn.Linear(12, 4).double(), _dyadic((4, 12), 32, 8),
                       [0.25, -0.25, 0, 0.5])
        self.l2 = _set(nn.Linear(4, 3).double(), _dyadic((3, 4), 33),
                       [0, 0.5, -0.25])
        self.l3 = _set(nn.Linear(3, 3).double(), _dyadic((3, 3), 34),
                       [0.25, 0, -0.5])
        self.l4 = _set(nn.Linear(3, 2).double(), _dyadic((2, 3), 35), [0, 0])

    def forward(self, x):
        h = torch.nn.functional.relu(self.conv(x).contiguous())
        h = h.contiguous().flatten(1)
        h = torch.nn.functional.leaky_relu(self.l1(h), 0.25)
        h = torch.sigmoid(self.l2(h))
        h = torch.flatten(torch.tanh(self.l3(h)), 1)
        a = torch.nn.functional.relu(h.view(1, -1))
        b = torch.nn.functional.relu(self.l4(h) + self.l4(h))
        return a.sum() + b.sum()


class _FunctionalTight(_Functional):
    """``_Functional`` with the two Linear layers around ``F.leaky_relu``
    made tight: each of the four leaky inputs is one ReLU output of the
    convolution, chosen among the outputs whose real range crosses 0, minus
    1/2, so it reaches -1/2; each Sigmoid input reads one leaky output with
    weight 1. The Sigmoid inputs' lower bounds are then exactly 1/4 times
    -1/2, and a slope converted wrongly moves them."""

    def __init__(self):
        super().__init__()
        lo, hi = _image_box()
        c_lo, c_hi = _conv2d(self.conv, _qarr(lo), _qarr(hi))
        crossing = [k for k, (a, z) in enumerate(zip(c_lo.ravel(),
                                                     c_hi.ravel()))
                    if a < 0 < z][:4]
        if len(crossing) < 4:
            pytest.fail(f"test data: crossing convolution outputs {crossing}")
        w1 = np.zeros((4, 12))
        w1[np.arange(4), crossing] = 1.0
        _set(self.l1, w1, [-0.5] * 4)
        _set(self.l2, np.eye(3, 4), [0, 0, 0])


def _functional_refs(model, traced, points):
    """The exact input of every functional activation, keyed by the node
    name of the traced module, in the order of the forward pass."""
    order = []
    for node in traced.graph.nodes:
        if node.op == "call_function" and node.target in (
                torch.nn.functional.relu, torch.nn.functional.leaky_relu,
                torch.sigmoid, torch.tanh):
            order.append(node.name)
    if len(order) != 6:
        pytest.fail(f"test data: functional activations {order}")
    refs = {name: [] for name in order}
    for x in points:
        v = _qarr(x)
        lo, hi = _exact_layer(model.conv, v, v)
        steps = [(lo, hi)]
        lo, hi = _exact_layer(nn.ReLU(), lo, hi)
        lo, hi = _exact_layer(model.l1, lo.reshape(-1), hi.reshape(-1))
        steps.append((lo, hi))
        lo, hi = _exact_layer(nn.LeakyReLU(0.25), lo, hi)
        lo, hi = _exact_layer(model.l2, lo, hi)
        steps.append((lo, hi))
        lo, hi = _exact_layer(nn.Sigmoid(), lo, hi)
        lo, hi = _exact_layer(model.l3, lo, hi)
        steps.append((lo, hi))
        lo, hi = _exact_layer(nn.Tanh(), lo, hi)
        steps.append((lo, hi))                          # the view
        a, z = _exact_layer(model.l4, lo, hi)
        steps.append((2 * a, 2 * z))                    # l4(h) + l4(h)
        for name, step in zip(order, steps):
            refs[name].append(step)
    return refs


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageStar+row", "ImageZono"])
def test_traced_functional_layers_record_bounds_that_hold(kind, method):
    """Every interval the pre-pass records for a functional activation of a
    traced torch module (``F.relu`` on an image, ``F.leaky_relu``,
    ``torch.sigmoid``, ``torch.tanh``) holds its real input, with the image
    entries in (height, width, channel) order; the four in front of the
    ``view`` must be recorded by the interval pass.

    Catches the call_function branch of the graph walk converting a
    functional activation wrongly (a LeakyReLU slope lost), ``contiguous``
    or ``flatten`` methods and ``torch.flatten`` that move or drop the
    bounds, and a consumer of a node the pass has no rule for (``view``,
    ``operator.add``) recorded with the network input box. The tight
    variant (``_FunctionalTight``) makes a wrong LeakyReLU slope visible in
    the bounds recorded for the Sigmoid behind it."""
    input_set, points = _image_input(kind, n=32)
    for model in (_Functional().eval(), _FunctionalTight().eval()):
        traced = fx.symbolic_trace(model)
        refs = _functional_refs(model, traced, points)
        bounds = compute_intermediate_bounds(traced, input_set, method=method,
                                             input_shape=IMG_CHW)
        what = (f"{method} pre-pass, traced {type(model).__name__} module, "
                f"{kind}")
        needed = set(list(refs)[:4])
        if method == "ibp" and not needed <= set(bounds):
            pytest.fail(f"{what}: recorded {sorted(bounds)}, expected at "
                        f"least {sorted(needed)}")
        if not bounds:
            pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
        _check_recorded(bounds, refs, what)


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("layer", [nn.LeakyReLU(-0.5), nn.Sigmoid(),
                                   nn.ReLU()], ids=lambda m: type(m).__name__)
def test_a_bare_nonlinear_module_records_its_input(layer, kind, method):
    """A model that is a single nonlinear module (no children) gets the
    bounds of its input recorded under index 0, and they hold every member
    point.

    Catches the walkers' fallback for a model without children dropping the
    module or recording something other than the input."""
    lo, hi = FLAT_LO, FLAT_HI
    bounds = compute_intermediate_bounds(layer, flat_set(kind, lo, hi),
                                         method=method)
    what = f"{method} pre-pass, bare {type(layer).__name__}, {kind}"
    if set(bounds) != {0}:
        raise AssertionError(f"{what}: recorded {sorted(bounds)}, the "
                             f"module is layer 0")
    points = samples(lo, hi, 32)
    _check_recorded(bounds, {0: [(_qarr(x), _qarr(x)) for x in points]},
                    what)


def _graph_layout_ops():
    """``X (1, 2, 3, 5)``: Add of a scalar, Relu ``act1``; Mul by a constant
    of the full image shape (a different factor per entry, signs mixed),
    Relu ``act2``; Transpose by the identity perm, Relu ``act3``; Add of a
    per-channel constant (1, 2, 1, 1), which the interval pass cannot apply
    to its flat vector, Relu ``act4`` (may get no bounds). Float32 (exact on
    these dyadic data)."""
    rng = np.random.default_rng(41)
    full = (rng.integers(-4, 5, size=(1, 2, 3, 5)) / 4).astype(np.float64)
    inits = {"S": np.array([0.25]), "M": full,
             "P": np.array([0.5, -0.75]).reshape(1, 2, 1, 1)}
    nodes = [_node("Add", ["X", "S"], ["a"]),
             _node("Relu", ["a"], ["r1"], name="act1"),
             _node("Mul", ["r1", "M"], ["m"]),
             _node("Relu", ["m"], ["r2"], name="act2"),
             _node("Transpose", ["r2"], ["t"], perm=[0, 1, 2, 3]),
             _node("Relu", ["t"], ["r3"], name="act3"),
             _node("Add", ["r3", "P"], ["p"]),
             _node("Relu", ["p"], ["Y"], name="act4")]
    return nodes, inits, [1, 2, 3, 5], {"act1": "a", "act2": "m",
                                        "act3": "t", "act4": "p"}, \
        TensorProto.FLOAT


def _graph_flat_transposes():
    """``X (1, 4)``: Transpose by the identity perm (0, 1), Relu ``act1``;
    Transpose (1, 0) to (4, 1), which the interval pass cannot follow on a
    flat vector, Relu ``act2`` (may get no bounds)."""
    nodes = [_node("Transpose", ["X"], ["t1"], perm=[0, 1]),
             _node("Relu", ["t1"], ["r1"], name="act1"),
             _node("Transpose", ["r1"], ["t2"], perm=[1, 0]),
             _node("Relu", ["t2"], ["Y"], name="act2")]
    return nodes, {}, [1, 4], {"act1": "t1", "act2": "t2"}, \
        TensorProto.DOUBLE


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("graph, kinds, must", [
    ("image", ("ImageStar", "ImageStar+row", "ImageZono"),
     {"act1", "act2", "act3"}),
    ("flat", ("Box", "Star", "Zono"), {"act1"})])
def test_onnx_constants_and_transposes_record_bounds_that_hold(
        graph, kinds, must, method, tmp_path):
    """Every interval the pre-pass records behind an Add of a scalar, a Mul
    by a full-size constant, an identity Transpose (image and flat) and the
    operations it cannot follow (a per-channel constant on a flat vector, a
    flat Transpose that is not the identity) holds the real Relu input, in
    (height, width, channel) order for an image.

    Catches a full-size constant applied in the stored (height, width,
    channel) order instead of the pass's (channel, height, width), a scalar
    broadcast wrongly, an identity perm that moves the bounds, and a
    per-channel constant or a flat transpose propagated as if it fitted.
    The interval pass must record the Relus in front of them."""
    build = {"image": _graph_layout_ops, "flat": _graph_flat_transposes}
    model, ref_path, in_shape, pre = _load_graph(tmp_path, build[graph])
    for kind in kinds:
        input_set, points = (_image_input(kind, n=32) if graph == "image"
                             else _flat_input(kind, n=32))
        what = f"{method} pre-pass, ONNX {graph} layout ops, {kind}"
        bounds = compute_intermediate_bounds(
            model, input_set, method=method, input_shape=tuple(in_shape[1:]))
        if method == "ibp" and not must <= set(bounds):
            pytest.fail(f"{what}: recorded {sorted(bounds)}, expected at "
                        f"least {sorted(must)}")
        if not bounds:
            pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
        refs = (_onnx_refs if graph == "image" else _onnx_refs_any)(
            ref_path, in_shape, pre, points)
        _check_recorded(bounds, refs, what)


def _conv1d(layer, lo, hi):
    """Exact Conv1d with zero padding, stride, dilation and groups on a
    (C, L) array."""
    Wt = _qarr(layer.weight.detach().cpu().double().numpy())
    b = (None if layer.bias is None
         else _qarr(layer.bias.detach().cpu().double().numpy()))
    (s,), (p,), (dl,) = layer.stride, layer.padding, layer.dilation
    c_in, n_in = lo.shape
    c_out, cpg, k = Wt.shape
    n_out = (n_in + 2 * p - dl * (k - 1) - 1) // s + 1
    opg = c_out // layer.groups
    out_lo = np.empty((c_out, n_out), dtype=object)
    out_hi = np.empty((c_out, n_out), dtype=object)
    for o in range(c_out):
        g = o // opg
        for t in range(n_out):
            l = h = (b[o] if b is not None else F(0))
            for c in range(cpg):
                ci = g * cpg + c
                for j in range(k):
                    i = t * s - p + j * dl
                    if not 0 <= i < n_in:
                        continue
                    w = Wt[o, c, j]
                    if w > 0:
                        l += w * lo[ci, i]
                        h += w * hi[ci, i]
                    elif w < 0:
                        l += w * hi[ci, i]
                        h += w * lo[ci, i]
            out_lo[o, t], out_hi[o, t] = l, h
    return out_lo, out_hi


def _conv1d_model():
    """Conv1d 2 -> 3, kernel 3, padding 1: (3, 6); ReLU; Conv1d 3 -> 2,
    kernel 2, stride 2: (2, 3); Flatten; ReLU; Linear; ReLU. Channel biases
    1, -1, 1/2 and 2, -2 set the channels apart."""
    c1 = _set(nn.Conv1d(2, 3, 3, padding=1).double(), _dyadic((3, 2, 3), 51),
              [1, -1, 0.5])
    c2 = _set(nn.Conv1d(3, 2, 2, stride=2).double(),
              _dyadic((2, 3, 2), 52, denom=16), [2, -2])
    l = _set(nn.Linear(6, 2).double(), _dyadic((2, 6), 53), [0.25, -0.25])
    return nn.Sequential(c1, nn.ReLU(), c2, nn.Flatten(), nn.ReLU(), l,
                         nn.ReLU()).double().eval()


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["Box", "Star", "Star+row", "Zono"])
def test_conv1d_recorded_bounds_hold_every_member_point(kind, method):
    """Every interval the pre-pass records for a network of Conv1d layers on
    a flat input of 2 channels by 6 holds the real pre-activation, in the
    (channel, length) order of the flat set.

    Catches the interval pass inferring the (channels, length) shape of a
    flat input wrongly, or a Conv1d bound built from the wrong weight signs,
    stride or padding."""
    model = _conv1d_model()
    lo = (np.arange(12) % 5 - 2).reshape(2, 6) / 4.0
    hi = lo + (np.arange(12) % 3 + 1).reshape(2, 6) / 4.0
    if kind == "Star+row":
        input_set, points = constrained_star(lo.reshape(-1), hi.reshape(-1),
                                             n=48, seed=5)
    else:
        input_set = flat_set(kind, lo, hi)
        points = samples(lo.reshape(-1), hi.reshape(-1), 48)
    refs = {}
    for x in points:
        v = _qarr(np.asarray(x).reshape(2, 6))
        a, z = _conv1d(model[0], v, v)
        refs.setdefault(1, []).append((a, z))
        a, z = _exact_layer(nn.ReLU(), a, z)
        a, z = _conv1d(model[2], a, z)
        a, z = a.reshape(-1), z.reshape(-1)
        refs.setdefault(4, []).append((a, z))
        a, z = _exact_layer(model[5], *_exact_layer(nn.ReLU(), a, z))
        refs.setdefault(6, []).append((a, z))
    refs = {k: [(np.asarray(a).reshape(-1), np.asarray(z).reshape(-1))
                for a, z in v] for k, v in refs.items()}
    what = f"{method} pre-pass, Conv1d net, {kind}"
    bounds = compute_intermediate_bounds(model, input_set, method=method)
    if method == "ibp" and set(bounds) != {1, 4, 6}:
        pytest.fail(f"{what}: recorded {sorted(bounds)}")
    if not bounds:
        pytest.fail(f"{what}: nothing recorded, the cell checks nothing")
    _check_recorded(bounds, refs, what)


def _conv_on_flat_model():
    """Conv2d 1 -> 1, kernel 2 x 2, stride 2, then ReLU: one channel, so the
    (height, width, channel) and (channel, height, width) orders of the
    output agree and the only question is which pixels share a window."""
    conv = _set(nn.Conv2d(1, 1, 2, stride=2, bias=False).double(),
                [[[[1.0, -1.0], [0.5, 0.25]]]])
    return nn.Sequential(conv, nn.ReLU()).double().eval()


@pytest.mark.parametrize("kind", ["Box", "Star"])
@pytest.mark.parametrize("chw", [
    pytest.param((1, 4, 4), id="square-4x4"),
    pytest.param((1, 2, 8), id="2x8", marks=known_bug(
        12, "SHAPE-10: the interval pre-pass ignores input_shape for a flat "
            "set in front of a Conv2d and guesses a square image (4 x 4 for "
            "16 pixels of a 2 x 8 image), so the convolution windows are the "
            "wrong pixels and a recorded ReLU bound misses the real "
            "pre-activation (neuron 0 recorded as [-0.72, -0.03], real "
            "range [-0.34, 0.34])")),
])
def test_conv_on_a_flat_input_set_uses_the_given_image_shape(chw, kind):
    """``compute_intermediate_bounds`` on a Conv2d network given a flat input
    set of an image and ``input_shape=(C, H, W)`` records bounds that hold
    the real pre-activation of every member point, or refuses with
    ``NotImplementedError``.

    Catches the interval pass's shape inference for a convolution whose
    input has no image shape: a square guess is right for a square image
    (the passing neighbour) and wrong for a 2 x 8 one."""
    model = _conv_on_flat_model()
    n = int(np.prod(chw))
    lo = (np.arange(n) / 8.0 - 1.0).reshape(chw)
    hi = lo + 0.25
    input_set = flat_set(kind, lo, hi)
    try:
        bounds = compute_intermediate_bounds(model, input_set, method="ibp",
                                             input_shape=chw)
    except NotImplementedError:
        return                                  # a refusal is a right answer
    points = samples(lo, hi, 32)
    refs = {1: [_exact_layer(model[0], _qarr(x), _qarr(x)) for x in points]}
    _check_recorded(bounds, refs, f"ibp pre-pass, Conv2d on a flat "
                                  f"{kind} of a {chw} image")


@pytest.mark.parametrize("kind", ["Box", "Star"])
@pytest.mark.parametrize("chw", [pytest.param((1, 4, 4), id="square-4x4"),
                                 pytest.param((1, 2, 8), id="2x8")])
def test_zonotope_pass_on_a_flat_input_in_front_of_a_conv_refuses_or_holds(
        chw, kind):
    """The zonotope pre-pass on a Conv2d network given a flat input set of
    an image and ``input_shape=(C, H, W)`` refuses (``ValueError``, a set
    that does not fit the layer, or ``NotImplementedError``) or records
    bounds that hold the real pre-activation of every member point (exact
    rationals). Today it refuses with ``ValueError`` ('Conv2D requires
    ImageZono input').

    Catches a zonotope pass that reads the flat set as an image of another
    shape (the interval pass's square guess, registry SHAPE-10) or answers
    with bounds of the wrong pixels."""
    model = _conv_on_flat_model()
    n = int(np.prod(chw))
    lo = (np.arange(n) / 8.0 - 1.0).reshape(chw)
    hi = lo + 0.25
    input_set = flat_set(kind, lo, hi)
    try:
        bounds = compute_intermediate_bounds(model, input_set, method="zono",
                                             input_shape=chw)
    except (ValueError, NotImplementedError):
        return                                  # a refusal is a right answer
    points = samples(lo, hi, 32)
    refs = {1: [_exact_layer(model[0], _qarr(x), _qarr(x)) for x in points]}
    _check_recorded(bounds, refs, f"zono pre-pass, Conv2d on a flat {kind} "
                                  f"of a {chw} image")


@pytest.mark.parametrize("method", ["ibp", "zono"])
def test_an_unknown_bounds_method_is_refused(method):
    """``compute_intermediate_bounds`` with a method other than 'ibp' or
    'zono', and reach with such a ``precompute_bounds``, raise ValueError.

    Catches a misspelt method silently running one of the passes (or none)
    instead of refusing."""
    model = FLAT_MODELS["relu"]()
    star = flat_set("Star", FLAT_LO, FLAT_HI)
    with pytest.raises(ValueError):
        compute_intermediate_bounds(model, star, method=method.upper())
    with pytest.raises(ValueError):
        reach_model(model, star, "approx", "unknown pre-pass",
                    precompute_bounds=method + "x")


_REACH7 = known_bug(
    16, "REACH-7: precompute_bounds='ibp' or 'zono' on a Hexatope or an "
        "Octatope input, through reach and through a direct "
        "compute_intermediate_bounds call, crashes with TypeError "
        "('Unsupported input_set type') from _extract_bounds or "
        "_convert_to_zono instead of refusing the option",
    raises=TypeError)


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("kind", [
    pytest.param("Hexatope", marks=_REACH7),
    pytest.param("Octatope", marks=_REACH7),
    "Star"])
def test_the_pre_pass_refuses_hexatope_and_octatope(kind, method):
    """The pre-pass has no rule for a Hexatope or Octatope input: reach with
    ``precompute_bounds`` and a direct call refuse with
    ``NotImplementedError`` or ``ValueError`` (which of the two is the
    owner's open question; DESIGN section 7 names a refusal of the option)
    rather than answer with bounds of some other set or crash. The reach is
    checked first. The Star neighbour goes through the same two calls and
    must answer, with recorded bounds that hold every member point.

    Catches an input set of an unsupported type read as if it were another
    type, a refusal swallowed on the way out of reach, and the option
    ignored by reach (both make the reach answer)."""
    model = FLAT_MODELS["relu"]()
    s = flat_set(kind, FLAT_LO, FLAT_HI)
    if kind == "Star":
        sets = reach_model(model, s, "approx", "Star with the pre-pass",
                           precompute_bounds=method)
        points = samples(FLAT_LO, FLAT_HI, 32)
        assert_covered(sets, replay_torch(model, points, (1, 4)),
                       f"Star, approx, precompute_bounds={method}")
        bounds = compute_intermediate_bounds(model, s, method=method)
        if not bounds:
            pytest.fail("the Star neighbour recorded nothing")
        _check_recorded(bounds, _sequential_refs(model, points, "index"),
                        f"{method} pre-pass on the Star neighbour")
        return
    with pytest.raises((NotImplementedError, ValueError)):
        reach_model(model, s, "approx", f"{kind} with the pre-pass",
                    precompute_bounds=method)
    with pytest.raises((NotImplementedError, ValueError)):
        compute_intermediate_bounds(model, s, method=method)


def _softmax_model():
    """Linear, ReLU, Linear, Softmax (no interval rule), Linear, ReLU."""
    l1 = _set(nn.Linear(4, 3).double(), _dyadic((3, 4), 61), [0.25, 0, -0.25])
    l2 = _set(nn.Linear(3, 3).double(), _dyadic((3, 3), 62), [0, 0.5, 0])
    l3 = _set(nn.Linear(3, 2).double(), [[2, -1, 0], [0, 1, -2]], [-0.5, 0.5])
    return nn.Sequential(l1, nn.ReLU(), l2, nn.Softmax(dim=1), l3,
                         nn.ReLU()).double().eval()


def test_a_layer_without_an_interval_rule_is_refused_or_skipped():
    """A plain Sequential with a Softmax (which the interval pass has no rule
    for) is refused by the direct call (NotImplementedError); the traced
    graph walk records the ReLU in front of it, and the ReLU behind it gets
    either no bounds or bounds that hold its real input (torch in float64,
    one-sided, a pad of 1e-12: Softmax values are not dyadic).

    Catches the per-layer dispatch passing an unknown layer through as the
    identity (the ReLU behind it would get the Softmax input's bounds), and
    the graph walk trusting the output of a layer it could not propagate."""
    model = _softmax_model()
    star = flat_set("Star", FLAT_LO, FLAT_HI)
    with pytest.raises(NotImplementedError):
        compute_intermediate_bounds(model, star, method="ibp")
    traced = fx.symbolic_trace(model)
    bounds = compute_intermediate_bounds(traced, star, method="ibp")
    names = {n.target: n.name for n in traced.graph.nodes
             if n.op == "call_module"}
    if names["1"] not in bounds:
        pytest.fail(f"the ReLU in front of the Softmax was not recorded "
                    f"(recorded {sorted(bounds)})")
    if names["5"] in bounds:
        lo, hi = (np.asarray(v, np.float64).reshape(-1)
                  for v in bounds[names["5"]])
        for x in samples(FLAT_LO, FLAT_HI, 64):
            v = replay_torch(model[:5], [x], (1, 4))[0]
            if np.any(v < lo - 1e-12) or np.any(v > hi + 1e-12):
                raise AssertionError(
                    f"the ReLU behind the Softmax has recorded bounds "
                    f"[{lo}, {hi}] that miss its real input {v} at {x}")


class _ReluBeforeLinear(nn.Module):
    """Neither a Sequential nor a GraphModule: the children are registered as
    ``act`` (a ReLU) then ``fc``, while the forward is ``act(fc(x))``. The
    ReLU's real input is fc(x) = (x0 + x1 + 3, 2 x1 - x0 - 10), far from the
    input box."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
        self.fc = _set(nn.Linear(2, 2).double(), [[1, 1], [-1, 2]], [3, -10])

    def forward(self, x):
        return self.act(self.fc(x))


def _walker_model(name):
    """``(model, refs builder)`` for the sequential-walker cells. 'plain' is
    Linear, ReLU, Linear, ReLU, Linear, Sigmoid with dyadic weights and two
    distinct ReLU objects; 'custom' is :class:`_ReluBeforeLinear`. (A
    Sequential that holds one module object at two places is the case of
    ``tests/soundness/test_soundness_prepass_handover.py``, which marks it.)"""
    if name == "custom":
        model = _ReluBeforeLinear().double().eval()

        def refs(points):
            return {0: [_exact_layer(model.fc, _qarr(x), _qarr(x))
                        for x in points]}
        return model, refs
    layers = [_set(nn.Linear(2, 2).double(), [[1, 0], [0, 1]], [0, 0]),
              nn.ReLU(),
              _set(nn.Linear(2, 2).double(), [[1, 0], [0, 1]], [-1.25, 0.5]),
              nn.ReLU(),
              _set(nn.Linear(2, 2).double(), [[1, 1], [-1, 1]], [0.25, 0]),
              nn.Sigmoid()]
    model = nn.Sequential(*layers).double().eval()
    return model, lambda points: _sequential_refs(model, points, "index")


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("name", [
    "plain",
    pytest.param("custom", marks=known_bug(
        16, "registry REACH-16: "
            "the sequential walkers of the bound pre-pass walk "
            "model.children() of any module that is not a GraphModule as if "
            "it were a chain: a module that is not a Sequential (the "
            "docstring admits a Sequential or a GraphModule) is walked in the "
            "order its children were registered, not refused, so the ReLU "
            "called on fc(x) is recorded with the bounds of x")),
])
def test_the_sequential_walker_follows_the_forward_order(name, method):
    """A direct ``compute_intermediate_bounds`` call on a module that is not
    a ``GraphModule`` walks it as a chain: every interval it records under
    a layer's index must hold the real input of that layer (exact rationals
    on dyadic data), or the call must refuse with ``NotImplementedError``.
    'plain' is the passing neighbour (a Sequential of distinct modules);
    'custom' is a module whose children are registered in another order
    than its forward calls them, which the docstring does not admit (a
    Sequential or a GraphModule), so a refusal is due. Reach traces every
    model and is not affected. Both walkers (the interval and the zonotope
    pass) are checked.

    Catches a walker that follows the order of registration instead of the
    forward order, a walker that records a layer's bounds after propagating
    through it, and bounds recorded under the index of another layer."""
    model, refs = _walker_model(name)
    lo, hi = np.array([-1.0, 0.5]), np.array([0.75, 1.5])
    star = flat_set("Star", lo, hi)
    what = f"{method} pre-pass, sequential walker, {name} model"
    try:
        bounds = compute_intermediate_bounds(model, star, method=method)
    except NotImplementedError:
        if name == "plain":
            raise
        return                                  # a refusal is a right answer
    if name == "plain" and set(bounds) != {1, 3, 5}:
        pytest.fail(f"{what}: recorded {sorted(bounds)}, the nonlinear "
                    f"layers are 1, 3, 5")
    _check_recorded(bounds, refs(samples(lo, hi, 32)), what)


@pytest.mark.parametrize("name", [
    "nested-sequential",
    pytest.param("batch-statistics-batchnorm", marks=known_bug(
        16, "registry REACH-17: "
            "the sequential walker of the interval pre-pass crashes "
            "with AttributeError ('NoneType' object has no attribute "
            "'detach') on a BatchNorm without running statistics "
            "(track_running_stats=False), where it must refuse; reach "
            "refuses that layer with NotImplementedError",
        raises=AttributeError)),
])
def test_the_sequential_walker_refuses_what_it_cannot_walk(name):
    """The interval pass's sequential walker (a direct call on a module that
    is not a ``GraphModule``) refuses with ``NotImplementedError`` a child it
    cannot walk as one layer: a Sequential nested in a Sequential, and a
    BatchNorm that normalises by batch statistics (no running statistics;
    the reach refuses it too).

    Catches a nested container or a batch-statistics BatchNorm passed
    through as the identity (later bounds of the wrong layer) and a crash
    where the refusal is due."""
    lin = _set(nn.Linear(2, 2).double(), [[1, 0.5], [-0.5, 1]], [0.25, 0])
    if name == "nested-sequential":
        model = nn.Sequential(lin, nn.Sequential(nn.ReLU(), nn.Linear(2, 2)),
                              nn.ReLU()).double().eval()
    else:
        model = nn.Sequential(lin, nn.BatchNorm1d(2, track_running_stats=False),
                              nn.ReLU()).double().eval()
    star = flat_set("Star", np.array([-1.0, 0.5]), np.array([0.75, 1.5]))
    with pytest.raises(NotImplementedError):
        compute_intermediate_bounds(model, star, method="ibp")


class _Indexed(nn.Module):
    """``relu(fc(x)[:, index])``: fc has three outputs with ranges that
    differ (centres 10, -10 and 0), traced by torch.fx as a ``getitem`` of a
    tensor (not of a multi-output node)."""

    def __init__(self, index):
        super().__init__()
        self.fc = _set(nn.Linear(2, 3).double(), [[1, 0], [0, 1], [1, -1]],
                       [10, -10, 0])
        self.index = index

    def forward(self, x):
        return torch.relu(self.fc(x)[:, self.index])


@pytest.mark.parametrize("index", [
    pytest.param([0, 1, 2], id="identity"),
    pytest.param([2, 0, 1], id="permuted", marks=known_bug(
        16, "registry REACH-18: "
            "the interval pre-pass's graph walk copies the bounds and the "
            "trust of a getitem's source whatever the index, so behind "
            "x[:, [2, 0, 1]] of a tensor the ReLU is recorded with the "
            "unpermuted bounds (and behind a slice with too many entries); "
            "reach refuses such a graph, the direct call answers")),
])
def test_a_getitem_on_a_tensor_records_no_wrong_bounds(index):
    """A direct interval pass on a traced module that indexes a tensor
    (``fc(x)[:, index]``, a ``getitem`` node) records for the ReLU behind it
    bounds that hold its real input (exact rationals), or nothing.
    'identity' is the passing neighbour (the same entries in the same
    order).

    Catches the graph walk handing a ``getitem`` of a tensor the bounds of
    its whole source, as it rightly does for one output of a multi-output
    node whose bounds it does not trust."""
    model = _Indexed(index).eval()
    traced = fx.symbolic_trace(model)
    lo, hi = np.array([-1.0, 0.5]), np.array([0.75, 1.5])
    points = samples(lo, hi, 32)
    refs = {"relu": []}
    for x in points:
        a, z = _exact_layer(model.fc, _qarr(x), _qarr(x))
        refs["relu"].append((a[index], z[index]))
    bounds = compute_intermediate_bounds(traced, flat_set("Star", lo, hi),
                                         method="ibp")
    if index == [0, 1, 2] and "relu" not in bounds:
        pytest.fail(f"the identity index recorded nothing ({sorted(bounds)})")
    _check_recorded(bounds, refs, f"ibp pre-pass, getitem {index}")


def _graph_broadcast_const(shape):
    """``Relu(X + C)`` with X of shape (1, 3) and the constant C of
    ``shape``: (1, 3) adds entry by entry, (3, 1) broadcasts the sum to
    (3, 3), whose nine entries are x_j + c_i."""
    nodes = [_node("Add", ["X", "C"], ["p"]),
             _node("Relu", ["p"], ["Y"], name="act")]
    const = np.array([0.5, -1.0, 2.0]).reshape(shape)
    return nodes, {"C": const}, [1, 3], {"act": "p"}, TensorProto.DOUBLE


@pytest.mark.parametrize("shape", [
    pytest.param((1, 3), id="same-shape"),
    pytest.param((3, 1), id="broadcast-larger", marks=known_bug(
        16, "registry REACH-19: "
            "the interval pre-pass applies an ONNX Add/Sub/Mul/Div "
            "constant with as many entries as the operand entry by entry, "
            "also when the broadcast makes the result larger ((1, 3) + "
            "(3, 1) is (3, 3)), and records 3 intervals for a Relu of 9 "
            "neurons; the reach handler has the same mechanism (GRAPH-9)")),
])
def test_onnx_constant_that_broadcasts_to_a_larger_tensor_records_no_wrong_bounds(
        shape, tmp_path):
    """The interval pre-pass behind an ONNX Add of a constant records, for
    the Relu, intervals that hold every entry of its real input
    (onnxruntime in float64, exact on dyadic data), or nothing. With a
    constant of shape (3, 1) on a (1, 3) input the Relu input has nine
    entries. 'same-shape' is the passing neighbour.

    Catches a constant rule that ignores a broadcast which adds entries."""
    model, ref_path, in_shape, pre = _load_graph(
        tmp_path, lambda: _graph_broadcast_const(shape))
    lo, hi = np.array([-1.0, 0.5, -2.0]), np.array([0.5, 1.0, -1.0])
    bounds = compute_intermediate_bounds(model, flat_set("Star", lo, hi),
                                         method="ibp", input_shape=(3,))
    if shape == (1, 3) and "act" not in bounds:
        pytest.fail(f"the same-shape Add recorded nothing ({sorted(bounds)})")
    _check_recorded(bounds, _onnx_refs_any(ref_path, in_shape, pre,
                                           samples(lo, hi, 16)),
                    f"ibp pre-pass, Add of a {shape} constant")


def test_onnx_div_by_a_constant_with_a_zero_entry_records_no_wrong_bounds(
        tmp_path):
    """``Relu(X / [2, 0, 4])`` with the middle input in [0.5, 1]: its
    quotient is +inf. The interval pre-pass records nothing for the Relu, or
    intervals that hold the real quotient on the finite entries (exact
    rationals) and an infinite upper bound on the middle one.

    Catches a Div rule that divides by a zero entry as if it were a number
    (a finite or NaN bound for an infinite input)."""
    from n2v.utils.model_loader import load_onnx
    nodes = [_node("Div", ["X", "C"], ["p"]),
             _node("Relu", ["p"], ["Y"], name="act")]
    path = _save_graph(tmp_path / "div.onnx", nodes, [1, 3], ["Y"],
                       {"C": np.array([2.0, 0.0, 4.0])})
    lo, hi = np.array([-1.0, 0.5, -2.0]), np.array([0.5, 1.0, 3.0])
    bounds = compute_intermediate_bounds(load_onnx(path),
                                         flat_set("Star", lo, hi),
                                         method="ibp", input_shape=(3,))
    if "act" not in bounds:
        return                                  # skipping the Relu is right
    rec_lo, rec_hi = (np.ravel(v) for v in bounds["act"])
    if not (rec_hi[1] == np.inf and not np.isnan(rec_lo[1])):
        raise AssertionError(f"the quotient by 0 is +inf, recorded "
                             f"[{rec_lo[1]}, {rec_hi[1]}]")
    for j, c in ((0, 2), (2, 4)):
        for x in (lo[j], hi[j]):
            if not _q_bound(rec_lo[j], "div") <= F(x) / c <= _q_bound(
                    rec_hi[j], "div"):
                raise AssertionError(f"entry {j}: {x} / {c} outside the "
                                     f"recorded [{rec_lo[j]}, {rec_hi[j]}]")


# --------------------------------------------------------------------------- #
# 6. End to end: reach with the pre-pass on is still sound                     #
# --------------------------------------------------------------------------- #
# The ReLU and LeakyReLU kernels of a Star or ImageStar read the recorded
# bounds (``precomputed_bounds``): the approximate kernels use them as given
# for the phase and the triangle, the exact ones to skip neurons they call
# stable. A recorded interval moved inward, or handed to another node,
# zeroes an active neuron or leaves out a piece, and true outputs escape.
# The other set types do not read the bounds, but the pass runs for them
# too and must not break the reach. Reference: torch in float64 or
# onnxruntime on dyadic data (exact), membership by the LP oracle against
# the returned sets' rows; samples include the corners, where a neuron that
# crosses 0 only near a corner is positive.
def _e2e_flat_input(kind, n):
    if kind == "Star-split":
        return (split_cover("Star", FLAT_LO, FLAT_HI, 3),
                samples(FLAT_LO, FLAT_HI, n))
    return _flat_input(kind, n)


def _flat_bn_unfused():
    """Linear, ReLU, BatchNorm1d, ReLU, Linear: a BatchNorm after a ReLU is
    not fused into the Linear by reach, so the reach takes the interval
    BatchNorm rule and the second ReLU reads its bounds. eps 1/16 and
    variances 3/16, 15/16, 63/16, 3/16 (var + eps is a square), one
    negative gamma; all data dyadic."""
    l1 = _set(nn.Linear(4, 4).double(),
              [[1, 0.5, 0, 0], [0, 0, -1, 0.25], [0.5, 0, 0, 1], [0, 1, 0.5, 0]],
              [0.125, 0.5, 0, -0.25])
    bn = _set_bn(nn.BatchNorm1d(4, eps=1.0 / 16).double(),
                 [0.25, 0.5, 0, 1], [3 / 16, 15 / 16, 63 / 16, 3 / 16],
                 [1, -0.5, 2, 0.25], [-0.5, 0.25, -1, 0.125])
    l2 = _set(nn.Linear(4, 2).double(), [[1, -1, 0.5, 0], [0, 0.5, 1, -1]],
              [0, 0.25])
    return nn.Sequential(l1, nn.ReLU(), bn, nn.ReLU(), l2).double().eval()


E2E_FLAT_MODELS = dict(FLAT_MODELS, **{"bn-unfused": _flat_bn_unfused})


def _e2e_flat_params():
    out = []
    for name in ("relu", "leaky", "misc", "smooth"):
        for kind in ("Star", "Star+row", "Star-split", "Zono", "Box"):
            for method in ("approx", "exact"):
                if method == "exact" and (name not in ("relu", "leaky")
                                          or kind not in ("Star",
                                                          "Star+row")):
                    continue
                for pre in ("ibp", "zono"):
                    out.append(pytest.param(
                        name, kind, method, pre,
                        id=f"{name}-{kind}-{method}-{pre}"))
    out.append(pytest.param("relu", "Star", "approx", True,
                            id="relu-Star-approx-True"))
    for pre in ("ibp", "zono"):
        out.append(pytest.param("bn-unfused", "Star", "approx", pre,
                                id=f"bn-unfused-Star-approx-{pre}"))
    return out


@pytest.mark.parametrize("name, kind, method, pre", _e2e_flat_params())
def test_reach_with_the_pre_pass_holds_every_flat_model_output(name, kind,
                                                                method, pre):
    """Reach of a flat network (ReLU layers, LeakyReLU of slopes 1/4 and
    -1/2, Linear plus BatchNorm, Sigmoid and Tanh) with
    ``precompute_bounds`` 'ibp', 'zono' or True holds every true output, for
    one Star, a Star with a binding row, three Stars that cover the box, a
    Zono and a Box, approx and exact.

    Catches the bounds of one layer handed to the kernel of another, and
    any recorded interval moved inward far enough to turn a crossing neuron
    stable or to lower a triangle's top. The 'bn-unfused' network keeps a
    BatchNorm after a ReLU, so the reach consumes the interval BatchNorm
    rule."""
    model = E2E_FLAT_MODELS[name]()
    input_sets, points = _e2e_flat_input(kind, 48 if method == "approx"
                                         else 12)
    what = f"{name} net, {kind}, {method}, precompute_bounds={pre}"
    sets = reach_model(model, input_sets, method, what,
                       precompute_bounds=pre)
    assert_covered(sets, replay_torch(model, points, (1, 4)), what)


def _sparse_image_box():
    """A (2, 3, 5) box with three free pixels in different channels and
    rows (every other pixel fixed), small enough for the exact method."""
    lo, hi = _image_box()
    hi = lo.copy()
    for c, y, x in ((0, 0, 1), (1, 2, 3), (0, 1, 4)):
        hi[c, y, x] = lo[c, y, x] + 0.75
    return lo, hi


def _img_relu_first():
    """A ReLU directly on the (2, 3, 5) image of ``_image_box``, then
    Flatten, Linear and ReLU. The pixels' intervals overlap across channels
    and rows (crossing, stably inactive and stably active ones side by
    side), so bounds read in (channel, height, width) order where the kernel
    reads (height, width, channel) hand a crossing pixel the overlapping
    interval of a stable one (the other image networks
    set their channels apart, so the kernel dropped misplaced bounds)."""
    l = _set(nn.Linear(30, 3).double(), _dyadic((3, 30), 20, denom=8),
             [0.25, -0.25, 0.5])
    return nn.Sequential(nn.ReLU(), nn.Flatten(), l, nn.ReLU()).double().eval()


E2E_IMAGE_MODELS = dict(IMAGE_MODELS, **{"relu-first": _img_relu_first})


def _e2e_image_params():
    out = []
    for name in ("conv", "pool", "smooth", "padmodes"):
        for kind in ("ImageStar", "ImageStar+row", "ImageZono"):
            for pre in ("ibp", "zono"):
                out.append(pytest.param(name, kind, "approx", pre,
                                        id=f"{name}-{kind}-approx-{pre}"))
    for pre in ("ibp", "zono"):
        out.append(pytest.param("relu-first", "ImageStar", "approx", pre,
                                id=f"relu-first-ImageStar-approx-{pre}"))
        out.append(pytest.param("conv", "ImageStar-split", "approx", pre,
                                id=f"conv-ImageStar-split-approx-{pre}"))
        for name in ("conv", "pool"):
            out.append(pytest.param(name, "ImageStar-sparse", "exact", pre,
                                    id=f"{name}-ImageStar-sparse-exact-"
                                       f"{pre}"))
    return out


@pytest.mark.parametrize("name, kind, method, pre", _e2e_image_params())
def test_reach_with_the_pre_pass_holds_every_image_model_output(
        name, kind, method, pre):
    """Reach of a convolutional network on a (2, 3, 5) image with the
    pre-pass on holds every true output: an ImageStar, one with a binding
    row, three that cover the box, an ImageZono, and the exact method on a
    box with three free pixels.

    Catches recorded image bounds read by the ReLU kernel in another row
    order than the ImageStar's (height, width, channel) (the 'relu-first'
    network, whose pixel intervals overlap), bounds of the Flatten's ReLU
    laid out as an image, and inward bounds that zero an active neuron of a
    convolution."""
    model = E2E_IMAGE_MODELS[name]()
    if kind == "ImageStar-sparse":
        lo, hi = _sparse_image_box()
        input_sets, points = image_set("ImageStar", lo, hi), samples(lo, hi,
                                                                       16)
    elif kind == "ImageStar-split":
        lo, hi = _image_box()
        input_sets = image_split_cover("ImageStar", lo, hi, 3)
        points = samples(lo, hi, 64)
    else:
        input_sets, points = _image_input(kind, n=64)
    what = f"{name} image net, {kind}, {method}, precompute_bounds={pre}"
    sets = reach_model(model, input_sets, method, what,
                       precompute_bounds=pre)
    assert_covered(sets, replay_torch(model, points, (1,) + IMG_CHW), what)


@pytest.mark.parametrize("pre", ["ibp", "zono"])
def test_reach_with_the_pre_pass_holds_conv1d_model_outputs(pre):
    """Reach of the Conv1d network (two convolutions, a ReLU after each and
    after the Flatten) on a flat Star of 2 channels by 6 with
    ``input_shape=(2, 6)`` and the pre-pass on holds every true output
    (torch in float64, LP membership): the reach takes the interval Conv1d
    branch, whose (channels, length) shape is read from the flat vector.

    Catches a Conv1d bound built from the wrong weight signs, stride,
    padding or layout that the ReLU kernel then uses."""
    model = _conv1d_model()
    lo = (np.arange(12) % 5 - 2).reshape(2, 6) / 4.0
    hi = lo + (np.arange(12) % 3 + 1).reshape(2, 6) / 4.0
    points = samples(lo.reshape(-1), hi.reshape(-1), 64)
    what = f"Conv1d net, Star, approx, precompute_bounds={pre}"
    sets = reach_model(model, flat_set("Star", lo, hi), "approx", what,
                       precompute_bounds=pre, input_shape=(2, 6))
    assert_covered(sets, replay_torch(model, points, (1, 2, 6)), what)


@pytest.mark.parametrize("pre", ["ibp", "zono"])
@pytest.mark.parametrize("graph, kind, method", [
    ("flat", "Star", "approx"), ("flat", "Star+row", "approx"),
    ("flat", "Star", "exact"), ("residual", "Star", "approx"),
    ("residual", "Star+row", "exact"), ("image", "ImageStar", "approx"),
    ("image", "ImageStar+row", "approx")])
def test_reach_with_the_pre_pass_holds_every_onnx_model_output(
        graph, kind, method, pre, tmp_path):
    """Reach of an ONNX graph loaded with ``load_onnx`` (MatMul, Add, Sub,
    Mul and Div by constants, Neg, a residual Add and Sub of two branches,
    Transposes of an image and a Flatten) with the pre-pass on holds every
    onnxruntime output.

    Catches the graph walk's bounds attached to the wrong node name,
    a residual or Transpose bound that the ReLU kernel then trusts, and the
    image bounds read in the wrong row order by the kernel."""
    build = {"flat": _graph_flat, "residual": _graph_residual,
             "image": _graph_image}[graph]
    model, _, in_shape, _ = _load_graph(tmp_path, build)
    model_path = str(tmp_path / "model.onnx")
    if graph == "image":
        input_set, points = _image_input(kind, n=64)
    else:
        input_set, points = _flat_input(kind, n=64 if method == "approx"
                                        else 16)
    what = f"ONNX {graph}, {kind}, {method}, precompute_bounds={pre}"
    sets = reach_model(model, input_set, method, what, precompute_bounds=pre,
                       input_shape=tuple(in_shape[1:]))
    assert_covered(sets, replay_onnx(model_path, points, in_shape), what)


def _softmax_crossing_model():
    """Linear(1, 2) giving (x, 0), Softmax, Linear(2, 1) giving s0 - 5/8,
    ReLU, for x in [3/8, 5/8]: the real ReLU input sigmoid(x) - 5/8 crosses
    0 (it reaches about +0.026), while the Softmax input passed through as
    if it were the output gives x - 5/8 in [-1/4, 0], which overlaps the
    star's own enclosure of the ReLU input and calls the neuron stably
    inactive."""
    l1 = _set(nn.Linear(1, 2).double(), [[1], [0]], [0, 0])
    l3 = _set(nn.Linear(2, 1).double(), [[1, 0]], [-0.625])
    return nn.Sequential(l1, nn.Softmax(dim=1), l3, nn.ReLU()).double().eval()


@pytest.mark.parametrize("pre", ["ibp", "zono"])
def test_reach_behind_a_layer_without_an_interval_rule_holds_outputs(pre):
    """Reach with the pre-pass on, of Linear, ReLU, Linear, Softmax, Linear,
    ReLU on a Star of four inputs, and of the one-input network of
    ``_softmax_crossing_model``, holds every true output (torch in float64,
    LP membership): the ReLU behind the Softmax must not be decided by
    bounds the pass could not compute.

    Catches a graph walk that keeps trusting past a layer it has no rule
    for and hands the ReLU behind it the Softmax input's bounds (on the
    second network they make a crossing neuron look stably inactive)."""
    lo1, hi1 = np.array([0.375]), np.array([0.625])
    cases = [("softmax net", _softmax_model(), *_flat_input("Star", 64),
              (1, 4)),
             ("softmax crossing net", _softmax_crossing_model(),
              flat_set("Star", lo1, hi1), samples(lo1, hi1, 64), (1, 1))]
    for name, model, star, points, in_shape in cases:
        what = f"{name}, Star, approx, precompute_bounds={pre}"
        sets = reach_model(model, star, "approx", what, precompute_bounds=pre)
        assert_covered(sets, replay_torch(model, points, in_shape), what)


def _relaxed_params():
    out = []
    for method, factor, kinds in (("range", 0.5, ("Star", "Star+row")),
                                  ("area", 0.5, ("Star", "Star+row")),
                                  ("bound", 0.5, ("Star", "Star+row")),
                                  ("area", 1.0, ("Star",)),
                                  ("standard", 0.0, ("Star",))):
        for kind in kinds:
            for pre in ("ibp", "zono"):
                out.append(pytest.param(
                    kind, {"relax_method": method, "relax_factor": factor},
                    pre, id=f"{kind}-{method}-{factor}-{pre}"))
    return out


@pytest.mark.parametrize("kind, options, pre", _relaxed_params())
def test_relaxed_relu_with_the_pre_pass_holds_every_output(kind, options,
                                                           pre):
    """Approx reach of the three-ReLU network with the relaxed ReLU kernels
    (``relax_method`` range, area and bound, ``relax_factor`` 0.5 and 1, and
    0, which is the exact kernel) and the pre-pass on holds every true
    output (torch in float64, LP membership).

    Catches a relaxed kernel that reads the recorded bounds of another
    neuron, and recorded bounds moved inward far enough to change a
    relaxation."""
    model = FLAT_MODELS["relu"]()
    star, points = _flat_input(kind, 32 if options["relax_factor"] else 12)
    what = f"relu net, {kind}, approx {options}, precompute_bounds={pre}"
    sets = reach_model(model, star, "approx", what, precompute_bounds=pre,
                       **options)
    assert_covered(sets, replay_torch(model, points, (1, 4)), what)


def _split_then_mixed_model():
    """Linear (identity), ReLU, Linear, ReLU on two inputs: the first ReLU
    splits a star of [-1, 1]^2 into several, so the second ReLU gets
    several stars; its neuron A = r0 + r1 - 1/2 crosses 0 and its neuron
    B = -r0 - 1/4 is stably inactive, with overlapping ranges, so the
    bounds of B handed to A make A look inactive."""
    l1 = _set(nn.Linear(2, 2).double(), [[1, 0], [0, 1]], [0, 0])
    l2 = _set(nn.Linear(2, 2).double(), [[1, 1], [-1, 0]], [-0.5, -0.25])
    return nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU()).double().eval()


@pytest.mark.parametrize("pre", ["ibp", "zono"])
def test_parallel_exact_relu_with_the_pre_pass_holds_every_output(pre):
    """Exact reach with ``parallel=True`` (two workers) and the pre-pass on
    holds every true output of the three-ReLU network and of the network of
    ``_split_then_mixed_model``, whose second ReLU layer gets several stars
    and has a stable neuron next to a crossing one.

    Catches the parallel exact ReLU path misrouting the recorded bounds it
    passes to its workers (for example in reversed neuron order)."""
    lo2, hi2 = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    cases = [("relu net", FLAT_MODELS["relu"](), *_flat_input("Star", 16),
              (1, 4)),
             ("split-then-mixed net", _split_then_mixed_model(),
              flat_set("Star", lo2, hi2), samples(lo2, hi2, 32), (1, 2))]
    for name, model, star, points, in_shape in cases:
        what = f"{name}, Star, exact parallel, precompute_bounds={pre}"
        sets = reach_model(model, star, "exact", what, precompute_bounds=pre,
                           parallel=True, n_workers=2)
        assert_covered(sets, replay_torch(model, points, in_shape), what)


@pytest.mark.parametrize("dtype", [
    pytest.param(np.float64, id="float64"),
    pytest.param(np.float32, id="float32", marks=known_bug(
        7, "ROUND-11 (another trigger of its mechanism): given a "
           "float32 Zono, the zonotope pre-pass bounds none of its own "
           "rounding (the error step returns None for a set that is not "
           "float64) and records the unwidened row bounds [0, 0] of a ReLU "
           "input whose real value is -2^-53 + 2^-60")),
])
def test_zonotope_pass_on_a_float32_zono_bounds_its_rounding(dtype):
    """The zonotope pre-pass given a Zono stored in float32 (a legal
    constructor input; its points are the float32 numbers as reals) records
    a ReLU interval that holds the real pre-activation, here
    (1 + 2^-30)(x0 + 2^-30) - (1 + 2^-23 + 2^-29 + 2^-52) = -2^-53 + 2^-60
    at x0 = 1 + 2^-23, which one rounding of the product turns into 0.

    Catches the pass dropping its rounding bound for an input set whose
    arrays are not float64 (the float64 Zono with the same values is the
    passing neighbour)."""
    x0 = 1.0 + 2.0 ** -23
    l1 = _set(nn.Linear(2, 2).double(), [[1, 0], [0, 1]], [E30, 0])
    l2 = _set(nn.Linear(2, 2).double(), [[C1, 0], [0, 1]],
              [-(1.0 + 2.0 ** -23 + E29 + 2.0 ** -52), 0])
    model = nn.Sequential(l1, l2, nn.ReLU()).double().eval()
    zono = Zono(np.array([[x0], [0.0]], dtype=dtype),
                np.array([[0.0], [1.0]], dtype=dtype))
    points = [np.array([x0, a]) for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    refs = _sequential_refs(model, points, "index")
    bounds = compute_intermediate_bounds(model, zono, method="zono")
    if 2 not in bounds and dtype == np.float64:
        pytest.fail(f"the ReLU was not recorded (recorded {sorted(bounds)})")
    _check_recorded(bounds, refs, f"zono pre-pass, {dtype.__name__} Zono")


# --------------------------------------------------------------------------- #
# 7. The kernels given bounds through the dispatcher                           #
# --------------------------------------------------------------------------- #
# The ReLU and LeakyReLU kernels of a Star take ``precomputed_bounds`` from
# the dispatcher (``reach_layer``): the pre-pass's, or a caller's such as
# branch and bound. Given an interval per neuron that holds the neuron's
# real range, the result must hold the exact image; given one that misses
# the star's own enclosure (bounds of other neurons), the kernel must not
# use it. Reference: the exact image in rationals (``rational.relu``,
# ``rational.leaky_relu`` on the input polytope), every vertex of every
# piece in one returned set, and for the exact method every vertex of a
# returned set in the image.
_KERNEL_C = [0.0, 0.25, 0.25]
_KERNEL_G = [[1.0, 1.0], [1.0, -1.0], [0.5, 0.0]]
# The predicate box [-1, 1]^2 cut by a0 + a1 <= 1/2: the real ranges of the
# three neurons are [-2, 1/2], [-1.75, 2.25] and [-0.25, 0.75], the first
# much tighter than the box estimate [-2, 2].
_KERNEL_RANGE = ([-2.0, -1.75, -0.25], [0.5, 2.25, 0.75])


def _kernel_parts(parts):
    """``[(star, (A, b))]``: the input Star (one, or the two halves a0 <= 0
    and a0 >= 0) with the rows of its predicate polytope."""
    box_rows = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]]
    box_rhs = [1, 1, 1, 1, 0.5]
    cuts = [([], [])] if parts == 1 else [([[1, 0]], [0]), ([[-1, 0]], [0])]
    out = []
    V = np.hstack([np.array(_KERNEL_C)[:, None], np.array(_KERNEL_G)])
    for rows, rhs in cuts:
        C = np.array([[1.0, 1.0]] + [list(map(float, r)) for r in rows])
        d = np.array([0.5] + list(map(float, rhs))).reshape(-1, 1)
        star = Star(V.copy(), C, d, -np.ones((2, 1)), np.ones((2, 1)))
        out.append((star, (box_rows + rows, box_rhs + rhs)))
    return out


def _kernel_bounds(which):
    lo, hi = (np.array(v) for v in _KERNEL_RANGE)
    if which == "real-range":
        return lo.reshape(-1, 1), hi.reshape(-1, 1)
    if which == "wider":
        return (lo - 0.5).reshape(-1, 1), (hi + 0.25).reshape(-1, 1)
    # bounds of other neurons: the first interval, [3, 5.5], misses the
    # star's own enclosure [-2, 2], so the whole layer's bounds must be
    # dropped (an interval that overlaps the enclosure is used as given:
    # holding the real range is the caller's promise)
    return (lo + np.array([5.0, 0, 0])).reshape(-1, 1), \
        (hi + np.array([5.0, 0, 0])).reshape(-1, 1)


@pytest.mark.parametrize("bounds, parts", [
    ("real-range", 1), ("real-range", 2), ("wider", 1), ("other-neurons", 1)])
@pytest.mark.parametrize("layer, method", [
    (nn.ReLU(), "approx"), (nn.ReLU(), "exact"),
    (nn.LeakyReLU(0.25), "approx"), (nn.LeakyReLU(-0.5), "approx"),
    (nn.LeakyReLU(-0.5), "exact")],
    ids=["relu-approx", "relu-exact", "leaky0.25-approx",
         "leaky-0.5-approx", "leaky-0.5-exact"])
def test_kernels_given_bounds_hold_the_exact_image(layer, method, parts,
                                                   bounds):
    """``reach_layer`` of a ReLU or LeakyReLU on a Star with a binding row,
    or on its two halves passed together, with ``precomputed_bounds`` that
    hold the real range (exactly, or wider) or that belong to other neurons,
    holds the exact image in rationals; the exact method returns nothing
    outside it.

    Catches a kernel that reads the given interval of another neuron,
    keeps bounds that miss its own enclosure, or (exact method) splits on
    the given bounds into pieces that are not in the image."""
    from tests.cell_support import reach_one_layer

    def image(rows, rhs):
        pieces = rational.affine(rational.polytope(rows, rhs), _KERNEL_G,
                                 _KERNEL_C)
        return (rational.relu(pieces) if isinstance(layer, nn.ReLU)
                else rational.leaky_relu(pieces, layer.negative_slope))

    sets_in = _kernel_parts(parts)
    what = (f"{type(layer).__name__}({getattr(layer, 'negative_slope', '')})"
            f" {method}, {parts} part(s), {bounds} bounds")
    out = reach_one_layer(layer, [st for st, _ in sets_in], method, what,
                          precomputed_bounds=_kernel_bounds(bounds))
    computed = [rational.star(st) for st in out]
    for _, (rows, rhs) in sets_in:
        pieces = image(rows, rhs)
        esc = rational.escapes(pieces, *computed)
        if esc:
            raise AssertionError(f"{what}: {len(esc)} vertices of the exact "
                                 f"image are in no returned set, e.g. "
                                 f"{[float(v) for v in esc[0][1]]}")
        if rational.straddlers(pieces, *computed):
            pytest.fail(f"{what}: a piece is spread over several returned "
                        f"sets; the vertex test proves nothing for it")
    if method == "exact":
        whole = image(*_kernel_parts(1)[0][1])
        for st in computed:
            far = rational.unreached(whole, rational.set_vertices(st))
            if far:
                raise AssertionError(
                    f"{what}: a returned set has the vertex "
                    f"{[float(v) for v in far[0][1]]}, which no input point "
                    f"reaches")


@pytest.mark.parametrize("layer, method, options", [
    (nn.ReLU(), "approx", {}), (nn.ReLU(), "exact", {}),
    (nn.ReLU(), "approx", {"relax_method": "area", "relax_factor": 0.5}),
    (nn.LeakyReLU(0.25), "approx", {}), (nn.LeakyReLU(-0.5), "exact", {})],
    ids=["relu-approx", "relu-exact", "relu-relaxed-area", "leaky-approx",
         "leaky-exact"])
def test_kernels_refuse_given_bounds_of_another_size(layer, method, options):
    """``reach_layer`` of a ReLU or LeakyReLU on a 3-neuron Star given
    ``precomputed_bounds`` with one entry raises ``ValueError``: the bounds
    belong to another node, and numpy would broadcast a one-entry bound to
    every neuron (with it, a ReLU given the bounds of one neuron lost the
    true output of the others). Bounds with more entries than neurons are
    not a case of their own: numpy refuses to broadcast them with a
    ``ValueError`` whatever the kernel checks.

    Catches the size check of the given bounds dropped or weakened in the
    approx, exact and relaxed ReLU kernels and the LeakyReLU kernels (each
    kernel calls the check from its own place)."""
    from tests.cell_support import reach_one_layer
    star = _kernel_parts(1)[0][0]
    lo, hi = _kernel_bounds("real-range")
    with pytest.raises(ValueError):
        reach_one_layer(layer, [star], method, f"{layer} {method}",
                        precomputed_bounds=(lo[:1], hi[:1]), **options)


class _FunctionalFlat(nn.Module):
    """The layers of the three-ReLU network with functional activations:
    ``F.relu``, ``F.leaky_relu`` (slope 1/4, a keyword), ``torch.relu``."""

    def __init__(self):
        super().__init__()
        net = _flat_relu()
        self.l1, self.l2, self.l3, self.l4 = net[0], net[2], net[4], net[6]

    def forward(self, x):
        h = torch.nn.functional.relu(self.l1(x))
        h = torch.nn.functional.leaky_relu(self.l2(h), negative_slope=0.25)
        return self.l4(torch.relu(self.l3(h)))


@pytest.mark.parametrize("pre", ["ibp", "zono"])
@pytest.mark.parametrize("kind, method", [
    ("Star", "approx"), ("Star+row", "approx"), ("Star", "exact")])
def test_reach_with_the_pre_pass_holds_functional_model_outputs(kind,
                                                                method, pre):
    """Reach of a module with functional activations (traced: call_function
    nodes) with the pre-pass on holds every true output.

    Catches the bounds of a functional activation looked up under another
    node's name, and a functional activation converted wrongly (another
    slope or another function)."""
    model = _FunctionalFlat().double().eval()
    star, points = _flat_input(kind, 64 if method == "approx" else 16)
    what = f"functional net, {kind}, {method}, precompute_bounds={pre}"
    sets = reach_model(model, star, method, what, precompute_bounds=pre)
    assert_covered(sets, replay_torch(model, points, (1, 4)), what)
