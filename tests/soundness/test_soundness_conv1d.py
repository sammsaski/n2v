"""Soundness cells for Conv1d reachability (``n2v/nn/layer_ops/conv1d_reach.py``).

Rows: (method, set type) for the seven set types and the two methods. The
three kernels (Star, Zono, Box) read no option and not the method: each
builds the convolution matrix from the layer's own ``_conv_forward`` on
float64 identity columns and calls the set's ``affine_map``. A flat set
stands for the (C, L) tensor in its own row-major order.

- Answered: Star (exact and approx), Zono and Box (approx). An image set
  reaches a Conv1d in a legal model only after ``Flatten(start_dim=2)`` or a
  view or Reshape to (1, C, H*W), which hand the kernel a flat set in the
  tensor's order (cells below); a Conv1d handed directly to an image set is
  registry AFFINE-5, marked in ``tests/contracts/test_layer_refusals.py``.
- Refused: ``method='exact'`` on a Zono, ImageZono, Box, Hexatope or
  Octatope at the model entry (``ValueError``), and a Conv1d on a Hexatope or
  Octatope under approx (``NotImplementedError``: their lanes have no Conv1d
  branch).

Oracles: the exact image in rationals from the convolution's definition
(``_exact_conv1d``; for a Box, the exact interval hull of the image), and
torch in float64 or onnxruntime on 128 member points (corners included) with
membership decided by one LP on the returned set's own rows. Every input has
two channels, a coordinate of zero width and coordinates on both sides of
zero; every kernel has distinct entries. The exact-image helpers live in
``test_soundness_conv2d.py`` (one copy for all the convolution files).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Box, Star, Zono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (
    assert_covered, constrained_image_star, constrained_star, flat_box,
    flat_set, image_set, pixel_box, reach_model, reach_one_layer,
    replay_onnx, replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.soundness.test_soundness_conv2d import (
    _assert_exact_image, _distinct, _many_bits, _piece, _set_params, _source,
    _star_predicate)

# id: (C, L) of the input, keyword arguments of nn.Conv1d besides
# in_channels.
_CONV1D = {
    "stride-pad-dilation": ((2, 6), dict(
        out_channels=3, kernel_size=3, stride=2, padding=2, dilation=2)),
    "groups-no-bias": ((2, 5), dict(
        out_channels=4, kernel_size=2, padding=1, groups=2, bias=False)),
    "reflect": ((2, 5), dict(
        out_channels=2, kernel_size=3, stride=2, padding=2,
        padding_mode="reflect")),
    "replicate": ((2, 6), dict(
        out_channels=3, kernel_size=2, padding=1, dilation=3,
        padding_mode="replicate")),
    "circular": ((2, 5), dict(
        out_channels=2, kernel_size=3, padding=2, groups=2,
        padding_mode="circular")),
    "same": ((2, 5), dict(out_channels=3, kernel_size=4, padding="same")),
}
_SAME_WARNING = pytest.mark.filterwarnings(
    "ignore:Using padding='same' with even kernel lengths:UserWarning")


def _make_conv1d(cfg, seed=0, bits=None):
    cl, kwargs = _CONV1D[cfg]
    layer = nn.Conv1d(cl[0], **kwargs)
    return _set_params(layer, np.random.default_rng(seed), bits), cl


def _out_dim(module, cl):
    with torch.no_grad():
        return int(module(torch.zeros((1,) + tuple(cl))).numel())


def _conv1d_pads(layer):
    """(left, right) of an nn.Conv1d's padding."""
    if layer.padding == "valid":
        return (0, 0)
    if layer.padding == "same":            # the extra entry goes at the end
        total = layer.dilation[0] * (layer.kernel_size[0] - 1)
        return (total // 2, total - total // 2)
    return (layer.padding[0], layer.padding[0])


def _exact_conv1d(layer, cl):
    """``f(x) -> (y, (O, Lo))``: nn.Conv1d by its definition on a flat
    (C, L) list of Fractions, output flat in (O, Lo) order."""
    C, L = cl
    w = [[[R.frac(v) for v in k] for k in o]
         for o in layer.weight.detach().double().numpy()]
    b = (None if layer.bias is None
         else [R.frac(v) for v in layer.bias.detach().double().numpy()])
    O, cg, kl = len(w), len(w[0]), len(w[0][0])
    og = O // layer.groups
    s, d = layer.stride[0], layer.dilation[0]
    left, right = _conv1d_pads(layer)
    Lo = (L + left + right - d * (kl - 1) - 1) // s + 1

    def f(x):
        y = []
        for o in range(O):
            g = o // og
            for i in range(Lo):
                acc = R.Fraction(0) if b is None else b[o]
                for cc in range(cg):
                    c = g * cg + cc
                    for p in range(kl):
                        t = _source(i * s - left + p * d, L, layer.padding_mode)
                        if t is not None:
                            acc += w[o][cc][p] * x[c * L + t]
                y.append(acc)
        return y, (O, Lo)
    return f


# =========================================================================== #
# Through reach_layer                                                           #
# =========================================================================== #
def _cells(configs, row_configs):
    out = []
    for cfg in configs:
        variants = [("Star", "exact", "split"), ("Zono", "approx", "split"),
                    ("Box", "approx", "split")]
        if cfg in row_configs:
            variants.append(("Star", "approx", "row"))
        for v in variants:
            out.append(pytest.param(
                cfg, *v, id=f"{cfg}-{v[0]}-{v[1]}-{v[2]}",
                marks=_SAME_WARNING if cfg == "same" else ()))
    return out


def _inputs(kind, variant, cl, seed=0):
    """``(sets, points)``: three sets of ``kind`` covering a box with a
    zero-width coordinate, cut along its widest coordinate (``split``), or one
    Star with a binding row that is not a box row (``row``)."""
    lo, hi = flat_box(cl[0] * cl[1], seed=seed, zero_width=True)
    if variant == "split":
        return split_cover(kind, lo, hi, 3), samples(lo, hi)
    star, points = constrained_star(lo, hi, seed=seed)
    return [star], points


# The point replay runs two layer configurations: zero padding with stride and
# dilation, and groups with circular padding (the branch of ``_conv_forward``
# that pads by ``F.pad``). Every configuration of ``_CONV1D`` (no bias,
# reflect, replicate, padding='same' with an even kernel) runs through the
# exact image in rationals below, the stronger oracle.
_SAMPLED = ("stride-pad-dilation", "circular")


@pytest.mark.parametrize("cfg,kind,method,variant", _cells(_SAMPLED, _SAMPLED))
def test_conv1d_layer_holds_every_torch_output(cfg, kind, method, variant):
    """Conv1d through ``reach_layer`` on three sets at once or on a Star with
    a binding row, against torch in float64 on corners and grid points,
    membership by LP on the rows (the point replay of each answering row).
    Catches a result that keeps only one of several input sets, a missing
    bias or a bias tiled across channels, a convolution matrix built with
    channels and positions exchanged, a padding mode reached as zeros, and a
    result with fewer or more entries than the output."""
    layer, cl = _make_conv1d(cfg)
    sets, points = _inputs(kind, variant, cl)
    what = f"Conv1d {cfg} on {kind} ({variant}, {method})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + cl), what)


def _few_var_flat(n, seed, count=2, bits=20, scale=1):
    """``count`` triples ``(star, zono, c, G, star_rows, zono_rows)`` on three
    predicate variables, centre and generators dense with ``_many_bits``
    values times ``scale`` (a power of two), the Star on the predicate
    polytope of ``_star_predicate`` (the second set on an asymmetric box), cut
    by a binding row that is not a box row."""
    rng = np.random.default_rng(seed)
    k = 3
    out = []
    for index in range(count):
        c = scale * _many_bits(rng, n, bits, span=2)
        G = scale * _many_bits(rng, (n, k), bits)
        lb, ub, A, b = _star_predicate(rng, index, k)
        box_A = np.vstack([np.eye(k), -np.eye(k)])
        star = Star(np.hstack([c[:, None], G]), A, b.reshape(-1, 1),
                    lb.reshape(-1, 1), ub.reshape(-1, 1))
        zono = Zono(c.reshape(-1, 1), G.copy())
        out.append((star, zono, c, G, (A, b), (box_A, np.ones(2 * k))))
    return out


def _many_bit_boxes(n, seed, count=2, bits=20, scale=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        lo = scale * _many_bits(rng, n, bits, span=2)
        out.append((lo, lo + scale * np.abs(_many_bits(rng, n, bits))))
    return out


def _assert_box_holds_the_hull(f, n, boxes, result, what):
    """Each input box's exact image has its interval hull (each output's
    minimum and maximum over the box, in rationals) inside one returned Box."""
    zero = [R.Fraction(0)] * n
    y0, _ = f(zero)
    cols = [[u - v for u, v in zip(f([R.Fraction(int(i == j))
                                       for i in range(n)])[0], y0)]
            for j in range(n)]
    got = [([R.frac(v) for v in np.asarray(s.lb).reshape(-1)],
            [R.frac(v) for v in np.asarray(s.ub).reshape(-1)])
           for s in result]
    for k, (lo, hi) in enumerate(boxes):
        lo, hi = [R.frac(v) for v in lo], [R.frac(v) for v in hi]
        hull_lo = [y0[m] + sum(min(col[m] * a, col[m] * z) for col, a, z
                               in zip(cols, lo, hi)) for m in range(len(y0))]
        hull_hi = [y0[m] + sum(max(col[m] * a, col[m] * z) for col, a, z
                               in zip(cols, lo, hi)) for m in range(len(y0))]
        assert any(len(glo) == len(y0)
                   and all(g <= h for g, h in zip(glo, hull_lo))
                   and all(h <= g for g, h in zip(ghi, hull_hi))
                   for glo, ghi in got), (
            f"{what}: the exact interval hull of input box {k} is in no "
            f"returned Box")


def _assert_flat_exact_image(pieces, results, what, exact, max_vars=3):
    """``_assert_exact_image`` for flat results, after cheap checks that fail
    with a message instead of letting the vertex enumeration run for hours:
    one returned set per input set (the kernels' promise), at most
    ``max_vars`` predicate variables in each, and as many entries in each as
    the exact output has (the shared helper pairs a point with a set by
    ``zip``, which truncates silently)."""
    size = len(pieces[0].c)
    assert len(results) == len(pieces), (
        f"{what}: {len(results)} sets returned for {len(pieces)} input sets")
    for k, s in enumerate(results):
        assert s.dim == size, (f"{what}: returned set {k} has {s.dim} "
                               f"entries, the output has {size}")
        n_var = len(R.star(s)[1][0])
        assert n_var <= max_vars, (f"{what}: returned set {k} has {n_var} "
                                   f"predicate variables, more than the "
                                   f"{max_vars} of the input")
    _assert_exact_image(pieces, results, what, exact)


@pytest.mark.parametrize("kind,method", [("Star", "exact"), ("Star", "approx"),
                                         ("Zono", "approx"), ("Box", "approx")])
@pytest.mark.parametrize("cfg", [
    pytest.param(c, marks=_SAME_WARNING) if c == "same" else c
    for c in _CONV1D])
def test_conv1d_layer_is_the_exact_image_in_rationals(cfg, kind, method):
    """Two input sets at once, each on three predicate variables (the Star
    with a binding row, the second on an asymmetric predicate box; for a
    Box, two boxes), with data of about 20 significant bits so float32
    arithmetic is not exact. The exact image is the convolution's definition
    in rationals. Catches every wrong entry of the convolution matrix even
    when sampled points are inside, a lost input set, an output with an entry
    too few or too many, a computation in float32, and, for the exact method,
    a result larger than the image."""
    layer, cl = _make_conv1d(cfg, seed=1, bits=20)
    f = _exact_conv1d(layer, cl)
    n = cl[0] * cl[1]
    what = f"Conv1d {cfg} on {kind} ({method}), exact image"
    if kind == "Box":
        boxes = _many_bit_boxes(n, seed=2)
        result = reach_one_layer(layer, [Box(lo, hi) for lo, hi in boxes],
                                 method, what)
        _assert_box_holds_the_hull(f, n, boxes, result, what)
        return
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _few_var_flat(n, seed=2):
        sets.append(star if kind == "Star" else zono)
        pieces.append(_piece(f, c, G, star_rows if kind == "Star"
                             else zono_rows, stored_hwc=False)[0])
    result = reach_one_layer(layer, sets, method, what)
    _assert_flat_exact_image(pieces, result, what, exact=(method == "exact"))


@pytest.mark.parametrize("kind,method", [("Star", "exact"), ("Zono", "approx"),
                                         ("Box", "approx")])
def test_float64_conv1d_layer_is_the_exact_image_in_rationals(kind, method):
    """A float64 Conv1d in the state a user hands it over (parameters frozen
    with ``requires_grad_(False)``), whose weights need 28 significant bits
    and whose bias is 1024 plus a 20-bit fraction, more than float32 holds,
    on two input sets of three predicate variables with 12-bit centre and
    generators times 8 (ends up to 16 and intervals up to 48 wide; the Star
    with a binding row, the second on an asymmetric predicate box; for a Box,
    two boxes). Every product and partial sum is exact in float64, so the
    exact image is the definition in rationals with no slack. Catches a
    weight or a bias read through float32 on its way to the float64 matrix
    (the bias then moves by up to 2**-14), which reaches a float64 model as
    another network."""
    cl, kwargs = _CONV1D["stride-pad-dilation"]
    layer = nn.Conv1d(cl[0], **kwargs).double()
    rng = np.random.default_rng(7)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(
            _many_bits(rng, tuple(layer.weight.shape), 28)))
        layer.bias.copy_(torch.as_tensor(
            1024 + _many_bits(rng, tuple(layer.bias.shape), 20)))
    layer.requires_grad_(False)
    f = _exact_conv1d(layer, cl)
    n = cl[0] * cl[1]
    what = f"float64 Conv1d with 28-bit weights on {kind} ({method})"
    if kind == "Box":
        boxes = _many_bit_boxes(n, seed=8, bits=12, scale=8)
        result = reach_one_layer(layer, [Box(lo, hi) for lo, hi in boxes],
                                 method, what)
        _assert_box_holds_the_hull(f, n, boxes, result, what)
        return
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _few_var_flat(
            n, seed=8, bits=12, scale=8):
        sets.append(star if kind == "Star" else zono)
        pieces.append(_piece(f, c, G, star_rows if kind == "Star"
                             else zono_rows, stored_hwc=False)[0])
    result = reach_one_layer(layer, sets, method, what)
    _assert_flat_exact_image(pieces, result, what, exact=(method == "exact"))


def _non_dyadic_boxes(n, seed):
    """Two boxes of ``n`` entries whose ends are float64 numbers with full
    mantissas (eighths of thirds and of sevenths of random numbers, up to
    8/3), cut from one box along its first entry, as a split leaves them."""
    rng = np.random.default_rng(seed)
    lo = 8 * rng.uniform(-1.0, 1.0, n) / 3.0
    hi = lo + 8 * rng.uniform(0.1, 1.0, n) / 7.0
    cut = lo[0] + (hi[0] - lo[0]) / 3.0
    first_hi, second_lo = hi.copy(), lo.copy()
    first_hi[0], second_lo[0] = cut, cut
    return [(lo, first_hi), (second_lo, hi)]


def test_conv1d_on_boxes_holds_the_exact_hull_of_non_dyadic_data():
    """``Box.affine_map`` promises bounds rounded outward, so Conv1d on a Box
    must hold the exact interval hull of the image of each box under the
    stored weights and bias even when every product and sum rounds. Weights
    and bias are float32 numbers with full mantissas, on replicate padding
    with dilation (an edge entry enters some outputs twice), bounds are
    float64 thirds and sevenths; the reference is the hull in rationals of
    exactly those numbers, with no slack, and 128 points replayed by torch in
    float64. Catches a Box kernel that rounds to nearest (a centre and radius
    form, or bounds summed without the outward move), which leaves bounds one
    unit inside the hull."""
    cl, kwargs = _CONV1D["replicate"]
    layer = nn.Conv1d(cl[0], **kwargs)
    rng = np.random.default_rng(14)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(
            rng.uniform(-1, 1, tuple(layer.weight.shape))))
        layer.bias.copy_(torch.from_numpy(rng.uniform(-1, 1, cl[0] + 1)))
    n = cl[0] * cl[1]
    boxes = _non_dyadic_boxes(n, seed=14)
    what = "Conv1d replicate on two Boxes with non-dyadic data"
    result = reach_one_layer(layer, [Box(lo, hi) for lo, hi in boxes],
                             "approx", what)
    _assert_box_holds_the_hull(_exact_conv1d(layer, cl), n, boxes, result,
                               what)
    points = [x for lo, hi in boxes for x in samples(lo, hi, n=64)]
    assert_covered(result, replay_torch(layer, points, (1,) + cl), what)


_EXACT_BOX_IS_ITS_BOUNDING_BOX = known_bug(
    18, "reach_layer with method='exact' on a Box does not read the method "
        "and returns the outward bounding box of the image, which is larger "
        "than the image of a Conv1d whose outputs share input entries; the "
        "model entry refuses 'exact' on a Box before any layer (REACH-8, as "
        "for Linear and Softmax)")

# Boxes of a (2, 2) tensor whose ends are -1, 0 and 1: with dyadic weights
# every product is exact, and so is every bound of the Box rule on an output
# that reads one input entry (a single rounding, corrected by TwoSum).
_UNIT_BOXES = ((np.array([0.0, -1.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0, 1.0])),
               (np.array([-1.0, 0.0, 0.0, -1.0]), np.array([0.0, 1.0, 1.0, 1.0])))


@pytest.mark.parametrize("weights", [
    "per-entry",
    pytest.param("mixing", marks=_EXACT_BOX_IS_ITS_BOUNDING_BOX)])
def test_exact_conv1d_on_a_box_refuses_or_is_the_image(weights):
    """DESIGN section 2, rule 4: 'exact' is exact up to float64 rounding or
    raises. ``reach_layer(nn.Conv1d, boxes, 'exact')``: a refusal
    (``NotImplementedError`` or ``ValueError``, as the model entry gives)
    keeps the promise; an answer must be the image, in both directions. A Box
    is the image only of a map whose outputs read one input entry each: a
    size-1 kernel with one input channel per group (each output a scaled
    entry plus the bias) is answered exactly today; a kernel of length 2
    mixes neighbouring entries and gets the bounding box of a zonotope.
    Catches the exact method returning an over-approximation under its
    name."""
    cl = (2, 2)
    if weights == "per-entry":
        layer = nn.Conv1d(2, 2, 1, groups=2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[[0.75]], [[-1.25]]]))
            layer.bias.copy_(torch.tensor([0.125, -0.5]))
    else:
        layer = _set_params(nn.Conv1d(2, 2, 2), np.random.default_rng(81))
    f = _exact_conv1d(layer, cl)
    n = cl[0] * cl[1]
    box_rows = (np.vstack([np.eye(n), -np.eye(n)]), np.ones(2 * n))
    pieces = [_piece(f, (lo + hi) / 2, np.diag((hi - lo) / 2), box_rows,
                     stored_hwc=False)[0] for lo, hi in _UNIT_BOXES]
    what = f"exact Conv1d ({weights}) on two Boxes"
    try:
        result = reach_one_layer(
            layer, [Box(lo, hi) for lo, hi in _UNIT_BOXES], "exact", what)
    except (NotImplementedError, ValueError):
        return
    _assert_flat_exact_image(pieces, result, what, exact=True, max_vars=n)


class _DoubledConv1d(nn.Conv1d):
    """A Conv1d subclass whose forward is not the stock one: twice the
    convolution."""

    def forward(self, x):
        return 2.0 * super().forward(x)


class _ShiftedWeightConv1d(nn.Conv1d):
    """A Conv1d subclass that changes its weight inside ``_conv_forward``, as
    weight standardization does (here 2 w - 1/2, dyadic, so the float64
    replay is exact)."""

    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, 2.0 * weight - 0.5, bias)


_OWN_FORWARD = known_bug(
    7, "reach_layer matches a Conv1d subclass by isinstance and convolves "
       "with the stock rule, so a subclass whose forward is not the stock "
       "one (here twice the convolution) is answered for another function "
       "and every true output lies outside, on Star, Zono and Box alike; the "
       "model entry traces into such a subclass and refuses its functional "
       "conv1d (REACH-9)")


@pytest.mark.parametrize("subclass", [
    pytest.param(_DoubledConv1d, marks=_OWN_FORWARD, id="own-forward"),
    pytest.param(_ShiftedWeightConv1d, id="own-conv-forward")])
def test_conv1d_subclass_refuses_or_holds_every_torch_output(subclass):
    """A Conv1d subclass (reflect padding, stride 2) handed to
    ``reach_layer`` on a flat Star: the reach must refuse it
    (``NotImplementedError``) or hold every output of the subclass's own
    forward (torch in float64 on 128 corners and grid points, membership by
    LP on the rows). The kernels call the layer's ``_conv_forward``, so a
    subclass that overrides only that is followed: it is the passing
    neighbour of the marked case, and it catches a kernel that convolves
    with ``F.conv1d`` and the layer's parameters instead of the layer's own
    convolution."""
    cl, kwargs = _CONV1D["reflect"]
    layer = _set_params(subclass(cl[0], **kwargs), np.random.default_rng(0))
    lo, hi = flat_box(cl[0] * cl[1], seed=26, zero_width=True)
    what = f"{subclass.__name__} on a Star"
    try:
        result = reach_one_layer(layer, [flat_set("Star", lo, hi)], "exact",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(layer, samples(lo, hi), (1,) + cl),
                   what)


# =========================================================================== #
# Through the model entry point                                                 #
# =========================================================================== #
def _chain(op, cl, seed=3, head=None):
    """``head`` (default: a size-1 Conv1d that mixes the channels) -> ``op``
    -> Flatten -> Linear to 3 outputs."""
    rng = np.random.default_rng(seed)
    if head is None:
        head = _set_params(nn.Conv1d(cl[0], cl[0], 1), rng)
    return nn.Sequential(head, op, nn.Flatten(),
                         _set_params(nn.Linear(_out_dim(op, cl), 3), rng))


@pytest.mark.parametrize("cfg,kind,method,variant", [
    ("stride-pad-dilation", "Star", "exact", "split"),
    ("groups-no-bias", "Star", "approx", "row"),
    ("reflect", "Zono", "approx", "split"),
    ("circular", "Box", "approx", "split"),
])
def test_conv1d_inside_a_model_holds_every_torch_output(cfg, kind, method,
                                                         variant):
    """Conv1d between a channel-mixing Conv1d and Flatten > Linear, through
    ``reach_pytorch_model`` on a flat set that stands for a (1, C, L)
    tensor (``input_shape`` given, as the rule for a flat set of a rank-3
    input asks). Catches convolution rows that only the downstream affine
    map exposes, and a node loop that hands a node the wrong sets."""
    op, cl = _make_conv1d(cfg)
    model = _chain(op, cl)
    sets, points = _inputs(kind, variant, cl)
    what = f"conv1d > Conv1d {cfg} > flatten > linear on {kind} ({method})"
    result = reach_model(model, sets, method, what, input_shape=cl)
    assert_covered(result, replay_torch(model, points, (1,) + cl), what)


def test_conv1d_after_an_exact_relu_split_holds_every_torch_output():
    """ReLU > Conv1d > Flatten > Linear with the exact method on a box where
    two coordinates straddle zero: the convolution receives the four sets of
    the exact split together. Catches a Conv1d that returns the image of only
    the first or only the last of several input sets."""
    op, cl = _make_conv1d("replicate")
    model = _chain(op, cl, head=nn.ReLU())
    lo, hi = flat_box(cl[0] * cl[1], seed=4, zero_width=True)
    shift = 0.25 - lo.min()
    lo, hi = lo + shift, hi + shift
    lo[1], hi[1] = -0.75, 0.5
    lo[-2], hi[-2] = -0.25, 1.0
    what = "relu > Conv1d replicate > flatten > linear, exact"
    result = reach_model(model, flat_set("Star", lo, hi), "exact", what,
                         input_shape=cl)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + cl),
                   what)


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_conv1d_after_a_relu_relaxation_holds_every_torch_output(kind):
    """ReLU > Conv1d (groups 2, no bias) > Flatten > Linear with the approx
    method through ``reach_pytorch_model``, on a box times 8 (ends well
    beyond plus and minus 1, intervals wider than 2) whose straddling entries
    make the ReLU relaxation add predicate variables (Star) or generators
    (Zono): the Conv1d receives a set with more variables than entries, a
    shape ``from_bounds`` never gives. Catches a kernel that maps only as many
    generators as the set has entries, and a no-bias branch that only the
    model entry reaches with such a set."""
    op, cl = _make_conv1d("groups-no-bias")
    model = _chain(op, cl, head=nn.ReLU())
    n = cl[0] * cl[1]
    lo, hi = flat_box(n, seed=8, zero_width=True)
    lo, hi = 8 * lo, 8 * hi
    what = f"relu (approx) > Conv1d groups-no-bias > flatten > linear on {kind}"
    (relaxed,) = reach_one_layer(nn.ReLU(), [flat_set(kind, lo, hi)],
                                 "approx", what)
    n_var = relaxed.nVar if kind == "Star" else relaxed.V.shape[1]
    assert n_var > n, (f"{what}: setup: the relaxation leaves {n_var} "
                       f"variables for {n} entries")
    result = reach_model(model, flat_set(kind, lo, hi), "approx", what,
                         input_shape=cl)
    assert len(result) == 1, f"{what}: {len(result)} sets for one input set"
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + cl),
                   what)


# An image input reaches a Conv1d only after its pixels are flattened: the
# image (C, H, W) = (2, 3, 4) becomes a (1, 2, 12) signal.
_IMAGE = (2, 3, 4)


class _PixelsAsSignal(nn.Module):
    """``x.view(1, C, -1)`` then the Conv1d: the torch spelling of a Conv1d
    over the flattened pixels of each channel."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x.view(1, _IMAGE[0], -1))


def _signal_conv1d(seed):
    return _set_params(nn.Conv1d(_IMAGE[0], **_CONV1D["stride-pad-dilation"][1]),
                       np.random.default_rng(seed))


@pytest.mark.parametrize("route,kind,method", [
    ("flatten", "ImageStar", "exact"), ("flatten", "ImageStar", "approx"),
    ("flatten", "ImageZono", "approx"), ("view", "ImageStar", "exact"),
    ("view", "ImageZono", "approx"), ("onnx-reshape", "ImageStar", "exact")])
def test_conv1d_on_an_image_input_after_its_pixels_are_flattened_holds_every_output(
        tmp_path, route, kind, method):
    """The image rows in ordinary use: an image set of a (2, 3, 4) input,
    a different interval per pixel times 4, then ``Flatten(start_dim=2)``, a
    view to (1, 2, -1), or an ONNX Reshape to (1, 2, 12), then a Conv1d over
    the 12 pixels, through the model entry; the ImageStar has a binding row
    that is not a box row. The convolution must receive the pixels in the
    tensor's (C, H, W) order, not in the stored (H, W, C) order. Reference:
    torch in float64 or onnxruntime on 128 member points, membership by LP.
    Catches a layout confusion between the image set and the flat set the
    Conv1d works on, and a Conv1d kernel handed the image set itself."""
    conv = _signal_conv1d(seed=9)
    lo, hi = pixel_box(*_IMAGE, seed=9)
    lo, hi = 4 * lo, 4 * hi
    if kind == "ImageStar":
        input_set, points = constrained_image_star(lo, hi, seed=9)
    else:
        input_set, points = image_set(kind, lo, hi), samples(lo, hi)
    what = f"{route} > Conv1d on {kind} ({method})"
    if route == "onnx-reshape":
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        inits = {"shape": np.array([1, _IMAGE[0], -1], dtype=np.int64),
                 "w": f32(conv.weight.detach().numpy()),
                 "b": f32(conv.bias.detach().numpy())}
        nodes = [helper.make_node("Reshape", ["X", "shape"], ["T"]),
                 helper.make_node("Conv", ["T", "w", "b"], ["Y"], pads=[2, 2],
                                  strides=[2], dilations=[2], kernel_shape=[3])]
        with torch.no_grad():
            out_shape = list(conv(torch.zeros(1, _IMAGE[0], 12)).shape)
        path = save_onnx(tmp_path, nodes, [1, *_IMAGE], out_shape, inits,
                         name="reshape_conv1d")
        result = reach_model(load_onnx(path), input_set, method, what)
        outputs = replay_onnx(path, points, [1, *_IMAGE])
    else:
        model = (nn.Sequential(nn.Flatten(start_dim=2), conv)
                 if route == "flatten" else _PixelsAsSignal(conv))
        result = reach_model(model, input_set, method, what)
        outputs = replay_torch(model, points, (1, *_IMAGE))
    assert_covered(result, outputs, what)


_RANK_FOUR_ANSWERED = known_bug(
    12, "a flat set that stands for a (1, 2, 3, 4) tensor (input_shape given) "
        "through a model whose first layer is a Conv1d, which torch cannot "
        "apply to a rank-4 tensor, is answered as if its 24 entries were a "
        "(2, 12) signal; no Conv1d path reads the tracked tensor shape "
        "(registry AFFINE-6)", raises=DID_NOT_RAISE)


@pytest.mark.parametrize("route", [
    pytest.param("conv1d", marks=_RANK_FOUR_ANSWERED), "flatten-conv1d"])
def test_conv1d_on_a_flat_set_of_a_rank_four_tensor_refuses_or_holds_every_output(
        route):
    """A flat Star that stands for a (1, 2, 3, 4) tensor (``input_shape``
    given) through ``Conv1d`` alone, which torch refuses (a Conv1d takes a
    rank-2 or rank-3 input), must be refused (``NotImplementedError`` or
    ``ValueError``); through the legal spelling ``Flatten(start_dim=2) >
    Conv1d`` it must hold every torch float64 output. Catches an answer for a
    network torch cannot run, and on the passing neighbour a flat set read in
    another order than the tensor's."""
    conv = _signal_conv1d(seed=10)
    model = (nn.Sequential(conv) if route == "conv1d"
             else nn.Sequential(nn.Flatten(start_dim=2), conv))
    lo, hi = flat_box(int(np.prod(_IMAGE)), seed=10, zero_width=True)
    points = samples(lo, hi)
    what = f"{route} on a flat Star of a (1, 2, 3, 4) tensor"
    if route == "conv1d":
        try:
            replay_torch(model, points[:1], (1, *_IMAGE))
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{what}: setup: torch answers")
        with pytest.raises((NotImplementedError, ValueError)):
            reach_model(model, flat_set("Star", lo, hi), "exact", what,
                        input_shape=_IMAGE)
        return
    result = reach_model(model, flat_set("Star", lo, hi), "exact", what,
                         input_shape=_IMAGE)
    assert_covered(result, replay_torch(model, points, (1, *_IMAGE)), what)


class _ConvOnAReshapedBatch(nn.Module):
    """``x.view(*shape)``, the Conv1d, then ``view(1, -1)``: with ``shape =
    (2, -1, 3)`` the Conv1d works on a batch of two signals, (1, 4, 3) to
    (2, 2, 3); with ``(1, -1, 6)`` on one signal, (1, 2, 6)."""

    def __init__(self, conv, shape):
        super().__init__()
        self.conv, self.shape = conv, shape

    def forward(self, x):
        return self.conv(x.view(*self.shape)).view(1, -1)


_BATCH_READ_AS_ONE_SIGNAL = known_bug(
    7, "a view or Reshape to (2, -1, L) in front of a Conv1d makes a batch of "
       "two; the reshape rule takes the 2 as the batch axis of one sample and "
       "the -1 absorbs it, so the Conv1d kernel reads both batch items as one "
       "signal of length dim / in_channels and convolves across them: every "
       "torch output lies outside, on Star, Zono and Box and through an ONNX "
       "Reshape alike (GRAPH-23; a target without -1 raises "
       "ValueError instead)")


@pytest.mark.parametrize("shape", [
    pytest.param((2, -1, 3), marks=_BATCH_READ_AS_ONE_SIGNAL,
                 id="batch-of-two"),
    pytest.param((1, -1, 6), id="batch-of-one")])
def test_conv1d_on_a_batch_made_by_a_view_refuses_or_holds_every_torch_output(
        shape):
    """A view in front of a Conv1d that moves entries into the batch axis
    (a (1, 4, 3) input viewed as (2, 2, 3)), then a view back to (1, 12),
    through the model entry on a flat Star: torch convolves each batch item
    on its own; the reach must refuse (``NotImplementedError`` or
    ``ValueError``) or hold every torch float64 output. The batch-of-one view
    (1, 2, 6) is the passing neighbour. Catches a Conv1d that reads a batch
    of signals as one longer signal."""
    conv = _set_params(nn.Conv1d(2, 2, 3, padding=1), np.random.default_rng(11))
    model = _ConvOnAReshapedBatch(conv, shape).eval()
    lo, hi = flat_box(12, seed=11, zero_width=True)
    what = f"view{shape} > Conv1d > view(1, -1) on a Star"
    try:
        result = reach_model(model, flat_set("Star", lo, hi), "exact", what,
                             input_shape=(4, 3))
    except (NotImplementedError, ValueError):
        return
    assert_covered(result, replay_torch(model, samples(lo, hi), (1, 4, 3)),
                   what)


def _onnx_conv1d_chain(tmp_path, pads, name):
    """X -> Conv(size 1) -> Conv(kernel 3, strides 2, dilations 2, ``pads``)
    -> Conv(size 1) -> Y on a (1, 2, 6) tensor, dyadic float32 weights.
    Returns ``(path, (C, L) of the input)``."""
    rng = np.random.default_rng(5)
    C, L = 2, 6
    lo_out = (L + sum(pads) - 2 * 2 - 1) // 2 + 1
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (C, C, 1))), "a_b": f32(_distinct(rng, (C,))),
             "m_w": f32(_distinct(rng, (3, C, 3))), "m_b": f32(_distinct(rng, (3,))),
             "z_w": f32(_distinct(rng, (2, 3, 1))), "z_b": f32(_distinct(rng, (2,)))}
    nodes = [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]),
             helper.make_node("Conv", ["T0", "m_w", "m_b"], ["T1"], pads=pads,
                              strides=[2], dilations=[2], kernel_shape=[3]),
             helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    out = [1, 2, lo_out]
    return save_onnx(tmp_path, nodes, [1, C, L], out, inits, name=name), (C, L)


@pytest.mark.parametrize("kind,method", [("Star", "exact"), ("Zono", "approx"),
                                         ("Box", "approx")])
def test_onnx_conv1d_holds_every_onnxruntime_output(tmp_path, kind, method):
    """An ONNX Conv on a (1, 2, 6) tensor (pads 2 and 2, strides 2,
    dilations 2) between two size-1 Convs, loaded with ``load_onnx`` and
    replayed on onnxruntime (``input_shape`` given). Catches a Conv1d whose
    matrix the converted module disagrees with."""
    path, cl = _onnx_conv1d_chain(tmp_path, [2, 2], "conv1d")
    lo, hi = flat_box(cl[0] * cl[1], seed=6, zero_width=True)
    what = f"ONNX Conv1d on {kind} ({method})"
    result = reach_model(load_onnx(path), flat_set(kind, lo, hi), method, what,
                         input_shape=cl)
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, *cl]), what)


def test_onnx_conv1d_without_optional_attributes_holds_every_onnxruntime_output(
        tmp_path):
    """An ONNX Conv on a (1, 2, 6) tensor with a kernel of length 3 that
    leaves out kernel_shape, strides, pads, dilations, group and auto_pad, so
    each takes the operator's default (the kernel length of the weight, 1,
    0, 1, 1, NOTSET). A default filled in wrong (a stride or a dilation of
    2, a pad of 1) changes the output; the loaded model must hold every
    onnxruntime output. Catches a loader that reads a missing attribute as
    another value."""
    rng = np.random.default_rng(12)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"w": f32(_distinct(rng, (3, 2, 3))), "b": f32(_distinct(rng, (3,)))}
    path = save_onnx(tmp_path, [helper.make_node("Conv", ["X", "w", "b"], ["Y"])],
                     [1, 2, 6], [1, 3, 4], inits, name="conv1d_bare")
    lo, hi = flat_box(12, seed=12, zero_width=True)
    what = "ONNX Conv1d without optional attributes on a Star (exact)"
    result = reach_model(load_onnx(path), flat_set("Star", lo, hi), "exact",
                         what, input_shape=(2, 6))
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, 2, 6]), what)


_PAD_ON_FLAT_TYPE_ERROR = known_bug(
    23, "an ONNX Conv1d with unequal begin and end pads is loaded as a static "
        "Pad in front of a Conv1d, and the Pad kernel raises TypeError on a "
        "flat Star or Zono instead of the refusal of an unsupported "
        "combination (NotImplementedError) (CONV-2)", raises=TypeError)


@pytest.mark.parametrize("kind,method", [
    pytest.param("Star", "exact", marks=_PAD_ON_FLAT_TYPE_ERROR),
    pytest.param("Zono", "approx", marks=_PAD_ON_FLAT_TYPE_ERROR),
    ("Box", "approx")])
def test_onnx_conv1d_with_unequal_pads_refuses_or_is_sound(tmp_path, kind,
                                                            method):
    """The same chain with pads 1 (begin) and 2 (end): the loader puts a
    static Pad of the last axis in front of the Conv1d. No rule pads a flat
    set, so the reach must refuse (NotImplementedError), or answer with sets
    that hold every onnxruntime output. ``input_shape`` is given, so the cell
    stays in the regime it means once a flat set of a rank-3 input without
    it is refused (registry SHAPE-2)."""
    path, cl = _onnx_conv1d_chain(tmp_path, [1, 2], "conv1d_unequal")
    lo, hi = flat_box(cl[0] * cl[1], seed=6, zero_width=True)
    what = f"ONNX Conv1d with pads [1, 2] on {kind} ({method})"
    try:
        result = reach_model(load_onnx(path), flat_set(kind, lo, hi), method,
                             what, input_shape=cl)
    except NotImplementedError:
        return
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, *cl]), what)


# =========================================================================== #
# Refusals                                                                      #
# =========================================================================== #
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_conv1d_on_a_hexatope_or_octatope_is_refused(kind):
    """No Conv1d rule exists for these set types: ``reach_layer`` must refuse,
    never answer."""
    lo, hi = flat_box(10)
    layer, _ = _make_conv1d("groups-no-bias")
    with pytest.raises(NotImplementedError):
        reach_layer(layer, [flat_set(kind, lo, hi)], "approx")


@pytest.mark.parametrize("kind,method", [
    ("Zono", "exact"), ("ImageZono", "exact"), ("Box", "exact"),
    ("Hexatope", "exact"), ("Hexatope", "approx"), ("Octatope", "exact"),
    ("Octatope", "approx")])
def test_conv1d_model_on_a_set_type_without_a_rule_is_refused(kind, method):
    """The rows the library does not support, through the model entry:
    ``method='exact'`` on a Zono, ImageZono, Box, Hexatope or Octatope is
    refused before the walk (today ``ValueError``, '... only supports
    approx'; the planned ``UnsupportedError`` subclasses
    ``NotImplementedError``), and a Conv1d on a Hexatope or Octatope under
    approx with ``NotImplementedError`` (their lanes of the dispatcher have
    no Conv1d branch). Catches the model entry passing 'exact' to a lane that
    does not read it (for a Box the bounding box under the name exact), and
    an answer on a set type that has no rule."""
    cl, kwargs = _CONV1D["groups-no-bias"]
    conv = _set_params(nn.Conv1d(cl[0], **kwargs), np.random.default_rng(13))
    if kind == "ImageZono":
        lo, hi = pixel_box(cl[0], 1, cl[1], seed=13)
        model = nn.Sequential(nn.Flatten(start_dim=2), conv)
        input_set, options = image_set(kind, lo, hi), {}
    else:
        lo, hi = flat_box(cl[0] * cl[1], seed=13)
        model = nn.Sequential(conv)
        input_set, options = flat_set(kind, lo, hi), {"input_shape": cl}
    refusals = ((NotImplementedError, ValueError) if method == "exact"
                else NotImplementedError)
    with pytest.raises(refusals):
        reach_model(model, input_set, method, f"Conv1d model on {kind} "
                    f"({method})", **options)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_conv1d_on_a_set_whose_size_is_not_a_multiple_of_the_channels_is_refused(kind):
    """A 7-entry set cannot be a (2, L) tensor, and torch refuses such an
    input: the reach must refuse, never answer. The refusal is a
    ``NotImplementedError``, or a ``ValueError`` that names the channel
    count, as the kernels' own check gives today; a ``ValueError`` from
    deeper in the arithmetic (numpy's shape error in the Box rule when the
    kernel's check is gone) is a crash, not the refusal."""
    lo, hi = flat_box(7)
    layer, _ = _make_conv1d("groups-no-bias")
    with pytest.raises((ValueError, NotImplementedError)) as info:
        reach_layer(layer, [flat_set(kind, lo, hi)],
                    "exact" if kind == "Star" else "approx")
    assert (isinstance(info.value, NotImplementedError)
            or "channel" in str(info.value)), (
        f"Conv1d on a 7-entry {kind}: the error is not the refusal of the "
        f"channel count: {info.value!r}")
