"""Soundness cells for ConvTranspose2d (``conv_transpose2d_reach.py``).

Rows: (method, set type) for the seven set types and the two methods. The
kernels read no option and run the same code for both methods. ImageStar and
ImageZono answer; a flat Star or Zono is refused with ``ValueError`` (marked:
the settled refusal is ``NotImplementedError``) unless a Reshape in front of
the layer turns it into an image first; Box, Hexatope and Octatope have no
rule in the dispatcher and are refused with ``NotImplementedError``.

What is covered: the layer through ``reach_layer`` on ImageStar (exact and
approx) and ImageZono, on several input sets at once and on an ImageStar
with a binding row that is not a box row, also with the basis columns
mapped in several memory batches and with a float64 layer whose parameters
are not float32 numbers; inside a torch model (``reach_pytorch_model``),
behind an approximate ReLU (more predicate variables or generators than
entries) and after an exact ReLU split; called with ``output_size`` in a
torch forward; as an ONNX ConvTranspose loaded with ``load_onnx`` (every
attribute given, every attribute left out, ``output_shape``, ``auto_pad``
SAME_UPPER, a latent vector reshaped into the layer); a subclass with its
own forward; the refusals (a flat set, a Box, Hexatope and Octatope, the
wrong channel count, padding that leaves no output, asymmetric ONNX pads, a
list of two set families). Every layer has stride, padding, output padding
and dilation that differ between height and width, and a kernel with
distinct entries.

Oracles: torch in float64 on member points (corners and grid points) or
onnxruntime for the ONNX graphs, with membership decided by one LP on the
returned set's own rows (``tests/_oracles/membership.py``); and the exact
image in rationals (``tests/_oracles/rational.py``) from the transposed
convolution's definition below, which shares no code with torch or n2v, on
sets of three or four predicate variables with data of about 20 significant
bits (float32 arithmetic is not exact on them).

Layout: a tensor is (C, H, W); an image set stores (H, W, C); every
comparison converts explicitly (``out_chw`` of ``assert_covered``, and
``hwc_index`` in the exact check). Inputs are (2, 3, 5) and (2, 4, 3)
images, and every output has height different from width.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, constrained_image_star,
                                flat_box, flat_set, hwc_index, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx)
from tests.known_bugs import known_bug
from tests.soundness.test_soundness_conv2d import (_assert_exact_image,
                                                   _distinct, _few_var_inputs,
                                                   _piece, _set_params,
                                                   _sets_on_four_variables)

# id: ((C, H, W) of the input, keyword arguments of nn.ConvTranspose2d
# besides in_channels). Output shapes: 5x9, 6x10, 6x9, 9x5.
_CONVT = {
    "stride-pad-outpad-dilation": ((2, 3, 5), dict(
        out_channels=3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0),
        output_padding=(1, 0), dilation=(1, 2))),
    "groups-no-bias": ((2, 4, 3), dict(
        out_channels=4, kernel_size=(3, 2), stride=(1, 3), padding=(1, 0),
        output_padding=(0, 2), dilation=(2, 1), groups=2, bias=False)),
    "large-stride": ((2, 3, 5), dict(
        out_channels=2, kernel_size=(2, 2), stride=(3, 2), padding=(2, 1),
        output_padding=(2, 1))),
    "equal-stride": ((2, 4, 3), dict(
        out_channels=3, kernel_size=(2, 3), stride=(2, 2), padding=(0, 1),
        output_padding=(1, 0))),
}


def _make_convt(cfg, seed=0, bits=None):
    chw, kwargs = _CONVT[cfg]
    layer = nn.ConvTranspose2d(chw[0], **kwargs)
    return _set_params(layer, np.random.default_rng(seed), bits), chw


def _out_chw(module, chw):
    with torch.no_grad():
        return tuple(module(torch.zeros((1,) + tuple(chw))).shape[1:])


def _exact_convt(layer, chw):
    """``f(x) -> (y, (O, Ho, Wo))``: nn.ConvTranspose2d by its definition on
    a flat (C, H, W) list of Fractions. Input entry (c, h, w) adds
    ``weight[c, o', p, q] * x`` to output (g * O/groups + o', h * sh - ph +
    p * dh, w * sw - pw + q * dw), g the group of c; output height
    (H - 1) sh - 2 ph + dh (kh - 1) + oph + 1, and the same for the width."""
    C, H, W = chw
    w = [[[[R.frac(v) for v in row] for row in o] for o in c]
         for c in layer.weight.detach().double().numpy()]
    b = (None if layer.bias is None
         else [R.frac(v) for v in layer.bias.detach().double().numpy()])
    og, kh, kw = len(w[0]), len(w[0][0]), len(w[0][0][0])
    cg = C // layer.groups
    (sh, sw), (ph, pw) = layer.stride, layer.padding
    (dh, dw), (oph, opw) = layer.dilation, layer.output_padding
    Ho = (H - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    Wo = (W - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    O = og * layer.groups

    def f(x):
        y = [R.Fraction(0) if b is None else b[o]
             for o in range(O) for _ in range(Ho * Wo)]
        for c in range(C):
            g = c // cg
            for h in range(H):
                for v in range(W):
                    xv = x[(c * H + h) * W + v]
                    if xv == 0:
                        continue
                    for o in range(og):
                        oc = g * og + o
                        for p in range(kh):
                            i = h * sh - ph + p * dh
                            if not 0 <= i < Ho:
                                continue
                            for q in range(kw):
                                j = v * sw - pw + q * dw
                                if 0 <= j < Wo:
                                    y[(oc * Ho + i) * Wo + j] += w[c][o][p][q] * xv
        return y, (O, Ho, Wo)
    return f


def _inputs(kind, variant, chw, seed=0):
    """``(sets, points)``: three sets of ``kind`` covering a pixel box cut
    along its widest pixel (``split``; the pixel of the first channel at the
    bottom-left corner has zero width, so an ImageStar has fewer predicate
    variables than pixels and an ImageZono a zero generator), one ImageStar
    with a binding row that is not a box row (``row``), or one set of
    ``kind`` over a box whose every pixel straddles zero (``relaxed``, for a
    ReLU in front); points in (C, H, W)."""
    lo, hi = pixel_box(*chw, seed=seed)
    if variant == "split":
        hi[0, chw[1] - 1, 0] = lo[0, chw[1] - 1, 0]
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    if variant == "relaxed":
        middle = (lo + hi) / 2 - 1.0 / 16
        lo, hi = lo - middle, hi - middle
        return [image_set(kind, lo, hi)], samples(lo, hi)
    star, points = constrained_image_star(lo, hi, seed=seed)
    return [star], points


def _cells(configs, row_configs):
    out = []
    for cfg in configs:
        variants = [("ImageStar", "exact", "split"),
                    ("ImageZono", "approx", "split")]
        if cfg in row_configs:
            variants.append(("ImageStar", "approx", "row"))
        out += [pytest.param(cfg, *v, id=f"{cfg}-{v[0]}-{v[1]}-{v[2]}")
                for v in variants]
    return out


def _recorder(layer_type):
    """``(seen, callback)`` for the model entry's ``_on_layer_input``: the
    callback appends the list of sets that enter each layer of
    ``layer_type``, so a cell can check that its input shape reached the
    layer (a test-setup check, not the verdict)."""
    seen = []

    def record(name, module, sets):
        if isinstance(module, layer_type):
            seen.append(list(sets))
    return seen, record


# --------------------------------------------------------------------------- #
# The kernel through reach_layer                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg,kind,method,variant", _cells(
    ("stride-pad-outpad-dilation",), ("stride-pad-outpad-dilation",)))
def test_conv_transpose2d_layer_holds_every_torch_output(cfg, kind, method,
                                                          variant):
    """The point replay of each answering row (ImageStar exact, ImageZono
    approx, ImageStar approx) through ``reach_layer``, on the layer whose
    kernel, stride, padding, output padding and dilation differ between the
    axes: three sets at once covering a pixel box with one zero-width pixel,
    or one ImageStar with a binding row; torch in float64 on 128 corners and
    grid points, membership by LP on the rows. The other layer
    configurations (groups, no bias, a stride larger than the kernel, equal
    strides) are held by the exact-image cells below. Catches a result that keeps only one of several input
    sets, a bias missing or written over the centre, a flipped kernel, and
    stride, padding, output padding or dilation read for the wrong axis (the
    result is then laid out for another output shape)."""
    layer, chw = _make_convt(cfg)
    sets, points = _inputs(kind, variant, chw)
    what = f"ConvTranspose2d {cfg} on {kind} ({variant}, {method})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_out_chw(layer, chw))


@pytest.mark.parametrize("cfg,kind,method", [
    pytest.param(cfg, kind, method, id=f"{cfg}-{kind}-{method}")
    for cfg in _CONVT
    for kind, method in [("ImageStar", "exact"), ("ImageZono", "approx")]
    + ([("ImageStar", "approx")] if cfg == "large-stride" else [])])
def test_conv_transpose2d_layer_is_the_exact_image_in_rationals(cfg, kind,
                                                                 method):
    """Two input sets at once on three predicate variables each (the
    ImageStar with a binding row, the second on an asymmetric predicate
    box), weights, bias, centre and generators of about 20 significant bits.
    The exact image is the definition of the transposed convolution in
    rationals. Catches every wrong generator even when sampled points are
    inside, a transposed convolution run in float32, a lost input set,
    predicate bounds not passed on as they are, and, for the exact method, a
    result larger than the image."""
    layer, chw = _make_convt(cfg, seed=1, bits=20)
    f = _exact_convt(layer, chw)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _few_var_inputs(chw, seed=2):
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"ConvTranspose2d {cfg} on {kind} ({method}), exact image"
    result = reach_one_layer(layer, sets, method, what)
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_float64_conv_transpose2d_layer_is_the_exact_image_in_rationals(
        kind, method):
    """A float64 ConvTranspose2d (kernel, stride, padding, output padding
    and dilation different per axis), frozen with ``requires_grad_(False)``
    as a user hands it over, whose weights and bias need 28 significant
    bits, more than float32 holds, on two input sets of three predicate
    variables with 12-bit centre and generators (the ImageStar with a
    binding row, the second on an asymmetric predicate box): every product
    and partial sum needs at most 45 bits, so a float64 transposed
    convolution is exact and the exact image is the definition in rationals
    with no slack. Catches a weight, bias or basis column rounded to float32
    on its way to the float64 map, which reaches a float64 model as another
    network, and a parameter read only when it requires a gradient."""
    chw, kwargs = _CONVT["stride-pad-outpad-dilation"]
    layer = _set_params(nn.ConvTranspose2d(chw[0], **kwargs).double(),
                        np.random.default_rng(1), bits=28)
    layer.requires_grad_(False)
    f = _exact_convt(layer, chw)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _few_var_inputs(
            chw, seed=2, bits=12):
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"float64 ConvTranspose2d with 28-bit weights on {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


def _spy_on_column_batches(monkeypatch):
    """The batch sizes of every ``map_basis_columns`` call the
    ConvTranspose2d kernels make, one list per call (the probe on an empty
    batch left out). The kernel module holds its own reference to the
    function, so the spy replaces that one; it calls the original, so
    whatever the original does still runs."""
    from n2v.nn.layer_ops import conv_transpose2d_reach
    calls = []
    original = conv_transpose2d_reach.map_basis_columns

    def spy(op, V, scratch_factor=1):
        sizes = []

        def counted(x):
            if x.shape[0]:
                sizes.append(int(x.shape[0]))
            return op(x)
        result = original(counted, V, scratch_factor)
        calls.append(sizes)
        return result
    monkeypatch.setattr(conv_transpose2d_reach, "map_basis_columns", spy)
    return calls


@pytest.mark.parametrize("batches", ["one-column", "short-last"])
@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_conv_transpose2d_in_several_column_batches_is_the_exact_image_in_rationals(
        monkeypatch, kind, method, batches):
    """``map_basis_columns`` maps the basis columns in batches whose size
    comes from a memory budget (``_COLUMN_BATCH_BYTES``); on test-sized
    images every column fits in one batch, so the budget is lowered here, to
    one column per batch and to three columns per batch: a batch of three
    and a shorter last batch of two for the five columns of each ImageStar,
    of three and one for the four generators of each ImageZono. The budget is
    the column's size in the kernel's own reckoning (input, output and
    kernel-area scratch, 8 bytes each), and a spy fails the test if other
    batches ran. Two sets at once on four predicate variables with dense
    generators, a binding row and an asymmetric predicate box, so no
    generator column can be exchanged unseen; the exact image in rationals,
    both directions for the exact method. Catches a batch read from or
    written to the wrong columns, a shorter last batch left out or written
    in another order, and a lost set."""
    from n2v.nn.layer_ops import conv2d_reach
    layer, chw = _make_convt("stride-pad-outpad-dilation", seed=1, bits=20)
    out_chw = _out_chw(layer, chw)
    n_in, n_out = int(np.prod(chw)), int(np.prod(out_chw))
    area = int(np.prod(layer.kernel_size))
    column = 8 * (n_in + n_out + area * max(n_in, n_out))
    per_batch = 1 if batches == "one-column" else 3
    monkeypatch.setattr(conv2d_reach, "_COLUMN_BATCH_BYTES",
                        per_batch * column)
    calls = _spy_on_column_batches(monkeypatch)
    f = _exact_convt(layer, chw)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _sets_on_four_variables(
            chw, seed=3):
        sets.append(star if kind == "ImageStar" else zono)
        piece, _ = _piece(f, c, G, star_rows if kind == "ImageStar"
                          else zono_rows)
        pieces.append(piece)
    what = (f"ConvTranspose2d on {kind} ({method}) in batches of "
            f"{per_batch} column(s)")
    result = reach_one_layer(layer, sets, method, what)
    generators = [s for s in calls if sum(s) >= 4]
    if batches == "one-column":
        ran = bool(generators) and all(max(s) == 1 for s in generators)
    else:
        ran = bool(generators) and all(len(s) > 1 and s[-1] < s[0]
                                       for s in generators)
    if not ran:
        pytest.fail(f"test setup: the batches that ran were {calls}, not "
                    f"{batches}")
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Through the model entry point                                                 #
# --------------------------------------------------------------------------- #
def _chain(op, chw, seed=3, head=None):
    """``head`` (default: a 1x1 convolution that mixes the channels) ->
    ``op`` -> Flatten -> Linear to 3 outputs."""
    rng = np.random.default_rng(seed)
    if head is None:
        head = _set_params(nn.Conv2d(chw[0], chw[0], 1), rng)
    n_out = int(np.prod(_out_chw(op, chw)))
    return nn.Sequential(head, op, nn.Flatten(),
                         _set_params(nn.Linear(n_out, 3), rng))


def _more_columns_than_entries(s):
    if isinstance(s, ImageStar):
        return s.V.shape[-1] - 1 > s.height * s.width * s.num_channels
    return s.V.shape[1] > s.V.shape[0]


@pytest.mark.parametrize("cfg,kind,method,variant", [
    ("stride-pad-outpad-dilation", "ImageStar", "exact", "split"),
    ("groups-no-bias", "ImageZono", "approx", "split"),
    ("large-stride", "ImageStar", "approx", "row"),
    ("equal-stride", "ImageStar", "approx", "relaxed"),
    ("stride-pad-outpad-dilation", "ImageZono", "approx", "relaxed"),
])
def test_conv_transpose2d_inside_a_model_holds_every_torch_output(
        cfg, kind, method, variant):
    """ConvTranspose2d between a channel-mixing convolution and Flatten >
    Linear, through ``reach_pytorch_model``; in the ``relaxed`` cells an
    approximate ReLU over a box whose every pixel straddles zero is in front
    instead, so the layer receives an ImageStar with more predicate
    variables than entries, or an ImageZono with more generators than
    entries (the shape a ReLU relaxation leaves; checked as test setup).
    Catches rows or a layout that only the downstream affine map and the
    Flatten conversion expose, and a kernel that reads the number of
    generator columns from the number of entries."""
    op, chw = _make_convt(cfg)
    relaxed = variant == "relaxed"
    model = _chain(op, chw, head=nn.ReLU() if relaxed else None)
    sets, points = _inputs(kind, variant, chw)
    what = f"conv > ConvTranspose2d {cfg} > flatten > linear on {kind}"
    seen, record = _recorder(nn.ConvTranspose2d)
    result = reach_model(model, sets, method, what, _on_layer_input=record)
    if relaxed and not (seen and all(_more_columns_than_entries(s)
                                     for sets_in in seen for s in sets_in)):
        pytest.fail(f"test setup: the sets entering the layer do not have "
                    f"more generator columns than entries: "
                    f"{[[s.V.shape for s in x] for x in seen]}")
    assert_covered(result, replay_torch(model, points, (1,) + chw), what)


def _coupled_straddlers(chw, seed=4):
    """``(image_star, points)``: a pixel box with every pixel positive except
    two, in different channels, rows and columns, whose intervals contain 0,
    as an ImageStar with one predicate variable a in [-1, 1] per pixel and
    the binding row ``a_p + a_q <= 0`` over the two. An exact ReLU splits it
    into four sets, and the row couples them: the output with pixel p at 0
    and pixel q at its maximum comes only from the piece where p is inactive
    and q active, the output with p at its maximum and q at 0 only from the
    piece where p is active and q inactive, so no piece holds the outputs of
    the others. ``points`` are member points in (C, H, W), among them those
    two."""
    C, H, W = chw
    lo, hi = pixel_box(*chw, seed=seed)
    shift = 0.25 - lo.min()
    lo, hi = lo + shift, hi + shift
    p, q = (0, 1, 0), (C - 1, H - 1, W - 1)
    lo[p], hi[p] = -0.75, 0.5
    lo[q], hi[q] = -0.25, 1.0
    order = hwc_index(C, H, W)
    centre = ((lo + hi) / 2).reshape(-1)[order]
    radius = ((hi - lo) / 2).reshape(-1)[order]
    n = centre.size
    ip = int(np.flatnonzero(order == np.ravel_multi_index(p, chw))[0])
    iq = int(np.flatnonzero(order == np.ravel_multi_index(q, chw))[0])
    row = np.zeros(n)
    row[ip] = row[iq] = 1.0
    rows = np.vstack([np.eye(n), -np.eye(n), row])
    rhs = np.concatenate([np.ones(2 * n), [0.0]]).reshape(-1, 1)
    V = np.hstack([centre[:, None], np.diag(radius)]).reshape(H, W, C, n + 1)
    star = ImageStar(V, rows, rhs, -np.ones((n, 1)), np.ones((n, 1)),
                     height=H, width=W, num_channels=C)
    rng = np.random.default_rng(seed)
    pairs = [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (0.25, -0.25),
             (-1.0, 0.5), (0.5, -1.0), (0.0, 0.0), (-0.5, 0.5)]
    points = []
    for k in range(128):
        a = (rng.choice([-1.0, 1.0], size=n) if k % 2 == 0
             else rng.integers(-2, 3, size=n) / 2.0)
        a[ip], a[iq] = pairs[k % len(pairs)]
        x = np.empty(n)
        x[order] = centre + radius * a
        points.append(x.reshape(chw))
    return star, points


def test_conv_transpose2d_after_an_exact_relu_split_holds_every_torch_output():
    """ReLU > ConvTranspose2d through ``reach_pytorch_model`` with the exact
    method on an ImageStar with two pixels that straddle zero and a binding
    row that couples them: the layer receives the four sets of the exact
    split together (checked as test setup), each with the split's own rows,
    and no one of them holds the outputs of the others. The model ends at
    the layer, whose output is compared as the (C, H, W) image. Catches a
    kernel that returns the image of only the first or only the last of
    several input sets, and one that loses the rows an exact split adds."""
    op, chw = _make_convt("equal-stride")
    model = nn.Sequential(nn.ReLU(), op)
    star, points = _coupled_straddlers(chw)
    what = "relu > ConvTranspose2d, exact, after a split of a coupled box"
    seen, record = _recorder(nn.ConvTranspose2d)
    result = reach_model(model, star, "exact", what, _on_layer_input=record)
    if not (seen and len(seen[0]) >= 3):
        pytest.fail(f"test setup: the layer received "
                    f"{[len(x) for x in seen]} sets, not the pieces of a split")
    assert_covered(result, replay_torch(model, points, (1,) + chw), what,
                   out_chw=_out_chw(op, chw))


class _UpsampleTo(nn.Module):
    """A 1x1 convolution, then a ConvTranspose2d (kernel (2, 3), stride 2)
    called with ``output_size`` in the forward, as a decoder does to match
    the size of a skip connection."""

    def __init__(self, size):
        super().__init__()
        rng = np.random.default_rng(31)
        self.head = _set_params(nn.Conv2d(2, 2, 1), rng)
        self.up = _set_params(nn.ConvTranspose2d(2, 3, (2, 3), stride=2), rng)
        self.size = list(size)

    def forward(self, x):
        return self.up(self.head(x), output_size=self.size)


_OUTPUT_SIZE_DROPPED = known_bug(
    16, "the graph walk hands a ConvTranspose2d only the first argument of "
        "its call, so an output_size given in a torch forward is dropped and "
        "the reach is computed for the default output size: the result "
        "stores one row and one column fewer than the layer outputs, and "
        "behind a global average pool every true output lies outside (GRAPH-25)")


@pytest.mark.parametrize("size", [
    pytest.param((6, 11), id="default-size"),
    pytest.param((7, 12), marks=_OUTPUT_SIZE_DROPPED,
                 id="one-more-row-and-column")])
def test_conv_transpose2d_called_with_an_output_size_holds_every_torch_output(
        size):
    """A torch model whose forward calls its ConvTranspose2d with
    ``output_size`` (the default 6 x 11 of a (2, 3, 5) input, and 7 x 12,
    which torch reaches by one more row and column of output padding),
    through ``reach_pytorch_model`` on an ImageStar (exact); torch in
    float64 on 128 member points, the output compared as the (C, H, W)
    image torch returns. Catches an output size dropped on the way to the
    kernel, which reaches another layer. The default size is the passing
    neighbour of the marked case."""
    chw = (2, 3, 5)
    model = _UpsampleTo(size)
    lo, hi = pixel_box(*chw, seed=32)
    what = f"ConvTranspose2d called with output_size {size}"
    result = reach_model(model, image_set("ImageStar", lo, hi), "exact", what)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what, out_chw=_out_chw(model, chw))


# --------------------------------------------------------------------------- #
# ONNX ConvTranspose through load_onnx                                          #
# --------------------------------------------------------------------------- #
def _onnx_convt_chain(tmp_path, pads, name):
    """X -> Conv 1x1 -> ConvTranspose(kernel (2, 3), strides (2, 1),
    ``pads``, output_padding (1, 0), dilations (1, 2)) -> Conv 1x1 -> Y on a
    (1, 2, 3, 5) tensor with dyadic float32 weights."""
    rng = np.random.default_rng(5)
    C, H, W = 2, 3, 5
    mid = nn.ConvTranspose2d(C, 3, (2, 3), stride=(2, 1), padding=(1, 0),
                             output_padding=(1, 0), dilation=(1, 2))
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (C, C, 1, 1))),
             "a_b": f32(_distinct(rng, (C,))),
             "t_w": f32(_distinct(rng, (C, 3, 2, 3))),
             "t_b": f32(_distinct(rng, (3,))),
             "z_w": f32(_distinct(rng, (2, 3, 1, 1))),
             "z_b": f32(_distinct(rng, (2,)))}
    out_chw = (2,) + _out_chw(mid, (C, H, W))[1:]
    nodes = [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]),
             helper.make_node("ConvTranspose", ["T0", "t_w", "t_b"], ["T1"],
                              kernel_shape=[2, 3], strides=[2, 1], pads=pads,
                              output_padding=[1, 0], dilations=[1, 2]),
             helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, *out_chw], inits,
                     name=name)
    return path, (C, H, W), out_chw


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_onnx_conv_transpose_holds_every_onnxruntime_output(tmp_path, kind,
                                                            method):
    """An ONNX ConvTranspose with pads (1, 0) on both sides, strides (2, 1),
    output_padding (1, 0) and dilations (1, 2) between two 1x1 Convs, loaded
    with ``load_onnx`` and replayed on onnxruntime. Catches a transposed
    convolution that the converted module and the ONNX file disagree on."""
    path, chw, out_chw = _onnx_convt_chain(tmp_path, [1, 0, 1, 0], "convt")
    lo, hi = pixel_box(*chw, seed=6)
    what = f"ONNX ConvTranspose on {kind} ({method})"
    result = reach_model(load_onnx(path), image_set(kind, lo, hi), method,
                         what)
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, *chw]),
                   what, out_chw=out_chw)


def _onnx_convt_with(tmp_path, name, out_hw, **attrs):
    """X -> Conv 1x1 -> ConvTranspose(weight (2, 3, 2, 3), bias, ``attrs``)
    -> Conv 1x1 -> Y on a (1, 2, 3, 5) tensor with dyadic float32 weights;
    ``out_hw`` is the output's height and width as onnxruntime computes
    them. Returns the path and the (C, H, W) of the output."""
    rng = np.random.default_rng(7)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (2, 2, 1, 1))),
             "a_b": f32(_distinct(rng, (2,))),
             "t_w": f32(_distinct(rng, (2, 3, 2, 3))),
             "t_b": f32(_distinct(rng, (3,))),
             "z_w": f32(_distinct(rng, (2, 3, 1, 1))),
             "z_b": f32(_distinct(rng, (2,)))}
    nodes = [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]),
             helper.make_node("ConvTranspose", ["T0", "t_w", "t_b"], ["T1"],
                              **attrs),
             helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    out_chw = (2, *out_hw)
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, *out_chw], inits,
                     name=name)
    return path, out_chw


_OUTPUT_SHAPE_IGNORED = known_bug(
    23, "the vendored onnx2torch converter never reads the output_shape "
        "attribute of an ONNX ConvTranspose, so load_onnx builds the layer "
        "of the default output size and the reach answers for it: the "
        "result stores 6 x 11 where onnxruntime outputs 7 x 12 (LOAD-10)")


@pytest.mark.parametrize("attrs,out_hw", [
    pytest.param({}, (4, 7), id="every-attribute-left-out"),
    pytest.param(dict(strides=[2, 2], output_shape=[6, 11]), (6, 11),
                 id="output-shape-of-the-default-size"),
    pytest.param(dict(strides=[2, 2], output_shape=[7, 12]), (7, 12),
                 marks=_OUTPUT_SHAPE_IGNORED,
                 id="output-shape-one-larger")])
def test_onnx_conv_transpose_attributes_hold_every_onnxruntime_output(
        tmp_path, attrs, out_hw):
    """An ONNX ConvTranspose with a (2, 3) kernel between two 1x1 Convs on a
    (2, 3, 5) input, loaded with ``load_onnx``, reached on an ImageStar
    (exact) and replayed on onnxruntime. With every attribute left out the
    converter's defaults (kernel shape from the weight, strides and
    dilations 1, no pads, no output padding, one group) must be the
    operator's own, on a shape where a wrong default changes the output;
    with ``output_shape`` the output has the size the file asks for (one
    more row and column than the default, which the ONNX padding rule puts
    at the end). Catches a default other than the operator's and an
    attribute the converter does not read. The default-size case is the
    passing neighbour of the marked one."""
    path, out_chw = _onnx_convt_with(tmp_path, "convt_attrs", out_hw, **attrs)
    lo, hi = pixel_box(2, 3, 5, seed=8)
    what = f"ONNX ConvTranspose with {attrs or 'no attributes'}"
    result = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                         "exact", what)
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, 2, 3, 5]),
                   what, out_chw=out_chw)


def test_onnx_conv_transpose_with_auto_pad_same_upper_refuses_or_holds_every_onnxruntime_output(
        tmp_path):
    """An ONNX ConvTranspose with ``auto_pad = SAME_UPPER``, strides (2, 1)
    and a (2, 3) kernel on a (2, 3, 5) image, between two 1x1 Convs: the
    output is the input size times the stride, (6, 5), and the odd entry
    the cropping leaves goes at the end. The vendored onnx2torch refuses
    SAME_UPPER and SAME_LOWER at load (``NotImplementedError``); a loader or
    converter that learns to read them must hold every onnxruntime output.
    Catches SAME_UPPER read as SAME_LOWER or as no padding at all."""
    path, out_chw = _onnx_convt_with(tmp_path, "convt_same_upper", (6, 5),
                                     strides=[2, 1], auto_pad="SAME_UPPER")
    lo, hi = pixel_box(2, 3, 5, seed=9)
    what = "ONNX ConvTranspose with auto_pad SAME_UPPER on ImageStar (approx)"
    try:
        result = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                             "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, 2, 3, 5]),
                   what, out_chw=out_chw)


def _onnx_latent_chain(tmp_path):
    """The generator pattern of the cgan benchmarks: X of shape (1, 4) ->
    Gemm to 30 -> Reshape to (1, 2, 3, 5) -> ConvTranspose(kernel (2, 3),
    strides (2, 1), pads (1, 0), output_padding (1, 0), dilations (1, 2))
    -> Y of shape (1, 3, 5, 9), with dyadic float32 weights. Every value
    onnxruntime computes on the cell's points needs at most 23 significant
    bits, so its float32 outputs are exact (a Gemm after the layer would
    need more, and onnxruntime's rounding would land outside the result).
    Returns the path and the (C, H, W) of the output."""
    rng = np.random.default_rng(11)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    mid = nn.ConvTranspose2d(2, 3, (2, 3), stride=(2, 1), padding=(1, 0),
                             output_padding=(1, 0), dilation=(1, 2))
    out_chw = _out_chw(mid, (2, 3, 5))
    inits = {"g_w": f32(_distinct(rng, (30, 4))),
             "g_b": f32(_distinct(rng, (30,))),
             "shape": np.array([1, 2, 3, 5], dtype=np.int64),
             "t_w": f32(_distinct(rng, (2, 3, 2, 3))),
             "t_b": f32(_distinct(rng, (3,)))}
    nodes = [helper.make_node("Gemm", ["X", "g_w", "g_b"], ["T0"], transB=1),
             helper.make_node("Reshape", ["T0", "shape"], ["T1"]),
             helper.make_node("ConvTranspose", ["T1", "t_w", "t_b"], ["Y"],
                              kernel_shape=[2, 3], strides=[2, 1],
                              pads=[1, 0, 1, 0], output_padding=[1, 0],
                              dilations=[1, 2])]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, *out_chw], inits,
                     name="latent")
    return path, out_chw


@pytest.mark.parametrize("kind,method", [("Star", "exact"),
                                         ("Zono", "approx")])
def test_onnx_latent_vector_reshaped_into_conv_transpose_holds_every_onnxruntime_output(
        tmp_path, kind, method):
    """A flat Star or Zono over a 4-dimensional latent box (sides beyond
    [-1, 1]) through Gemm, Reshape into an image and ConvTranspose, loaded
    with ``load_onnx`` and replayed on onnxruntime, the output compared as
    the (C, H, W) image: the Reshape
    turns the flat set into an ImageStar or ImageZono with four generator
    columns over 30 entries, which is the only way a flat set reaches this
    layer. Catches a kernel that mishandles a set whose generators are
    dense rows built by a Reshape, and a lost route from a flat set to the
    layer (a refusal here fails the test)."""
    path, out_chw = _onnx_latent_chain(tmp_path)
    lo = np.array([-2.0, -0.5, 0.0, 0.25])
    hi = np.array([1.0, 1.5, 0.5, 2.5])
    what = f"ONNX latent {kind} > Gemm > Reshape > ConvTranspose"
    result = reach_model(load_onnx(path), flat_set(kind, lo, hi), method,
                         what)
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, 4]), what,
                   out_chw=out_chw)


def test_onnx_conv_transpose_with_asymmetric_pads_is_refused(tmp_path):
    """An ONNX ConvTranspose whose begin and end pads differ: the loader has
    no module for it and must refuse, never load a different layer."""
    path, _, _ = _onnx_convt_chain(tmp_path, [1, 0, 0, 1], "convt_asym")
    with pytest.raises(NotImplementedError):
        load_onnx(path)


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
_FLAT_REFUSED_BY_VALUE_ERROR = known_bug(
    23, "ConvTranspose2d on a flat Star or Zono raises ValueError instead of "
        "the refusal of an unsupported combination (NotImplementedError, "
        "which UnsupportedError will subclass), so a caller that tries the "
        "next method on a refusal stops instead", raises=ValueError)


@pytest.mark.parametrize("kind", [
    pytest.param("Star", marks=_FLAT_REFUSED_BY_VALUE_ERROR),
    pytest.param("Zono", marks=_FLAT_REFUSED_BY_VALUE_ERROR),
    "Box", "Hexatope", "Octatope"])
def test_conv_transpose2d_on_a_flat_set_is_refused(kind):
    """A flat set carries no height, width or channel count and no rule
    reaches ConvTranspose2d on it: ``reach_layer`` must refuse."""
    lo, hi = flat_box(30)
    layer, _ = _make_convt("stride-pad-outpad-dilation")
    with pytest.raises(NotImplementedError):
        reach_layer(layer, [flat_set(kind, lo, hi)],
                    "exact" if kind == "Star" else "approx")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_conv_transpose2d_with_the_wrong_channel_count_is_refused(kind):
    """A 3-channel ConvTranspose2d on a 2-channel image: torch itself raises,
    so the reach must raise too (ValueError, or a refusal). The passing
    neighbour of the marked case below."""
    lo, hi = pixel_box(2, 3, 5)
    layer = _set_params(nn.ConvTranspose2d(3, 2, (2, 3), stride=(2, 1)),
                        np.random.default_rng(0))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_layer(layer, [image_set(kind, lo, hi)], "approx")


@known_bug(23, "a ConvTranspose2d whose padding leaves no output lets "
               "torch's RuntimeError escape from the probe in "
               "map_basis_columns instead of a refusal (CONV-3, also for "
               "Conv2d, AvgPool2d and MaxPool2d kernels larger than the "
               "image; the same probe)", raises=RuntimeError)
def test_conv_transpose2d_whose_padding_leaves_no_output_is_refused():
    """A ConvTranspose2d with a (2, 2) kernel and padding 3 on the height of
    a (2, 3, 5) image: the output height would be -2, and torch raises. The
    reach must refuse (``ValueError`` for an input that cannot be read, or
    ``NotImplementedError``). Catches a crash in place of the refusal."""
    layer = _set_params(nn.ConvTranspose2d(2, 3, 2, padding=(3, 0)),
                        np.random.default_rng(0))
    try:
        with torch.no_grad():
            layer(torch.zeros(1, 2, 3, 5))
    except RuntimeError:
        pass
    else:
        pytest.fail("test setup: torch answers for this layer")
    lo, hi = pixel_box(2, 3, 5)
    with pytest.raises((ValueError, NotImplementedError)):
        reach_layer(layer, [image_set("ImageStar", lo, hi)], "approx")


@pytest.mark.parametrize("first,second", [("ImageStar", "ImageZono"),
                                          ("ImageZono", "ImageStar")])
def test_conv_transpose2d_on_a_list_of_mixed_set_types_refuses_or_is_sound(
        first, second):
    """An ImageStar and an ImageZono in one ``reach_layer`` call (only a
    direct call can build such a list). The dispatcher routes by the first
    set; the kernel must refuse the other (``ValueError`` for an input it
    cannot read, ``NotImplementedError`` for an unsupported combination),
    or answer with sets that hold the outputs of both. Both orders are
    refused with ``ValueError`` today. Catches a kernel that skips a set of
    the wrong type, returns after the first set, or crashes on the second
    (any other exception fails the test)."""
    layer, chw = _make_convt("stride-pad-outpad-dilation")
    lo, hi = pixel_box(*chw)
    try:
        result = reach_layer(layer, [image_set(first, lo, hi),
                                     image_set(second, lo + 4, hi + 4)],
                             "approx")
    except (ValueError, NotImplementedError):
        return
    points = samples(lo, hi, 32) + samples(lo + 4, hi + 4, 32)
    assert_covered(result, replay_torch(layer, points, (1,) + chw),
                   "ConvTranspose2d on a mixed list",
                   out_chw=_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# A ConvTranspose2d subclass                                                    #
# --------------------------------------------------------------------------- #
class _DoubledConvTranspose2d(nn.ConvTranspose2d):
    """A ConvTranspose2d subclass whose forward is not the stock one: twice
    the transposed convolution."""

    def forward(self, x, output_size=None):
        return 2.0 * super().forward(x, output_size)


class _NamedConvTranspose2d(nn.ConvTranspose2d):
    """A ConvTranspose2d subclass that keeps the stock forward (it only adds
    a class attribute)."""

    role = "decoder"


_OWN_FORWARD = known_bug(
    7, "reach_layer matches a ConvTranspose2d subclass by isinstance and "
       "maps the set with the stock transposed convolution, so a subclass "
       "whose forward is not the stock one (here twice the map) is answered "
       "for another function and every true output lies outside (REACH-9; "
       "the model entry traces into the "
       "subclass and refuses its functional call)")


@pytest.mark.parametrize("kind,subclass", [
    pytest.param("ImageStar", _DoubledConvTranspose2d, marks=_OWN_FORWARD,
                 id="ImageStar-own-forward"),
    pytest.param("ImageZono", _DoubledConvTranspose2d, marks=_OWN_FORWARD,
                 id="ImageZono-own-forward"),
    pytest.param("ImageStar", _NamedConvTranspose2d,
                 id="ImageStar-stock-forward")])
def test_conv_transpose2d_subclass_refuses_or_holds_every_torch_output(
        kind, subclass):
    """A ConvTranspose2d subclass handed to ``reach_layer`` on a (2, 3, 5)
    pixel box: the reach must refuse it (``NotImplementedError``) or hold
    every output of the subclass's own forward (torch in float64 on 128
    corners and grid points, membership by LP on the rows). The Star lane
    and the Zono lane match the layer type in separate branches, so each
    has a marked case; a subclass that keeps the stock forward is answered
    and is the passing neighbour."""
    chw, kwargs = _CONVT["stride-pad-outpad-dilation"]
    layer = _set_params(subclass(chw[0], **kwargs), np.random.default_rng(0))
    lo, hi = pixel_box(*chw, seed=26)
    what = f"{subclass.__name__} on {kind}"
    try:
        result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(layer, samples(lo, hi), (1,) + chw),
                   what, out_chw=_out_chw(layer, chw))
