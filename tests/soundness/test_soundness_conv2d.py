"""Soundness cells for Conv2d reachability (``n2v/nn/layer_ops/conv2d_reach.py``).

Rows: (method, set type) for the seven set types and the two methods. The
kernels read no option and run the same code for both methods. ImageStar and
ImageZono answer; a flat Star or Zono is refused with ``ValueError`` (marked:
the settled refusal is ``NotImplementedError``); Box, Hexatope and Octatope
have no rule in the dispatcher and are refused with ``NotImplementedError``;
the placeholder kernels ``conv2d_hexatope`` and ``conv2d_octatope``, which no
public entry reaches, are checked by direct calls (registry HEXOCT-5).

Several other soundness files import the exact-image helpers below
(``_assert_exact_image``, ``_few_var_inputs``, ``_piece``, ``_chain``,
``_inputs`` and others); keep their names and behaviour.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono


# =========================================================================== #
# Soundness cells                                                               #
#                                                                               #
# The cells below follow tests/README.md "A soundness cell". Oracles: torch in  #
# float64 on member points (corners included) with membership decided by one   #
# LP on the returned set's own rows (tests/_oracles/membership.py), and the     #
# exact image in rationals (tests/_oracles/rational.py) built from a direct     #
# implementation of the convolution's definition below, which shares no code    #
# with torch or n2v. Layout: a tensor is (C, H, W); an image set stores        #
# (H, W, C); every comparison converts explicitly (``out_chw``, ``hwc_index``). #
# Every image has two or more channels and height different from width, every  #
# kernel has distinct entries, and strides, pads and dilations differ between   #
# the two axes.                                                                 #
# =========================================================================== #
import pytest  # noqa: E402
from onnx import helper  # noqa: E402

from n2v.nn.layer_ops.conv2d_reach import conv2d_box  # noqa: E402
from n2v.nn.layer_ops.dispatcher import reach_layer  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from tests._oracles import rational as R  # noqa: E402
from tests.cell_support import (  # noqa: E402
    assert_covered, constrained_image_star, flat_box, flat_set, hwc_index,
    image_set, image_split_cover, pixel_box, reach_model, reach_one_layer,
    replay_onnx, replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import known_bug  # noqa: E402

# id: ((C, H, W) of the input, keyword arguments of nn.Conv2d besides
# in_channels). Every output is non-square, so a result stored with height
# and width exchanged fails the shape check of ``assert_covered``.
_CONV2D = {
    "stride-pad-dilation": ((2, 3, 5), dict(
        out_channels=3, kernel_size=(2, 3), stride=(1, 2), padding=(1, 2),
        dilation=(2, 1))),
    "groups-no-bias": ((2, 4, 3), dict(
        out_channels=4, kernel_size=(3, 2), stride=(2, 1), padding=(1, 2),
        dilation=(1, 2), groups=2, bias=False)),
    "reflect": ((2, 3, 5), dict(
        out_channels=2, kernel_size=(2, 3), stride=(2, 1), padding=(1, 2),
        padding_mode="reflect")),
    "replicate": ((2, 4, 3), dict(
        out_channels=3, kernel_size=(3, 2), stride=(1, 2), padding=(2, 1),
        dilation=(1, 2), padding_mode="replicate")),
    "circular": ((2, 3, 5), dict(
        out_channels=2, kernel_size=(2, 2), stride=(2, 3), padding=(1, 2),
        dilation=(2, 1), groups=2, padding_mode="circular")),
    "same": ((2, 3, 5), dict(
        out_channels=3, kernel_size=(2, 4), padding="same", dilation=(1, 2))),
}
# padding='same' with an even kernel pads one more entry at the end; torch
# warns that it copies the input to do so (the warning is not about n2v).
_SAME_WARNING = pytest.mark.filterwarnings(
    "ignore:Using padding='same' with even kernel lengths:UserWarning")


def _distinct(rng, shape, denom=16, span=2):
    """Distinct nonzero multiples of 1/denom in [-span, span] (the span grows
    when the tensor has more entries than that), so that a kernel read with
    two axes exchanged is a different kernel."""
    size = int(np.prod(shape))
    span = max(span, -(-size // (2 * denom)))
    pool = np.array([v for v in range(-span * denom, span * denom + 1) if v])
    return rng.choice(pool, size=size, replace=False).reshape(shape) / denom


def _many_bits(rng, shape, bits=20, span=1):
    """Multiples of 2**-bits in [-span, span): a product of two of them needs
    about 2 * bits significant bits, exact in float64, not in float32."""
    top = span * 2 ** bits
    return rng.integers(-top, top, size=shape) / 2.0 ** bits


def _set_params(layer, rng, bits=None):
    """Fill the layer's weight and bias: distinct dyadic values, or
    ``_many_bits`` values when ``bits`` is given."""
    with torch.no_grad():
        for p in (layer.weight, layer.bias):
            if p is None:
                continue
            v = (_distinct(rng, tuple(p.shape)) if bits is None
                 else _many_bits(rng, tuple(p.shape), bits))
            p.copy_(torch.as_tensor(v, dtype=p.dtype))
    return layer


def _make_conv2d(cfg_id, seed=0, bits=None):
    chw, kwargs = _CONV2D[cfg_id]
    layer = nn.Conv2d(chw[0], **kwargs)
    return _set_params(layer, np.random.default_rng(seed), bits), chw


def _out_chw(module, chw):
    with torch.no_grad():
        return tuple(module(torch.zeros((1,) + tuple(chw))).shape[1:])


# --------------------------------------------------------------------------- #
# The convolution by its definition, in rationals                               #
# --------------------------------------------------------------------------- #
def _source(t, n, mode):
    """Index of the input entry that padded position ``t`` reads, or None
    for a zero (torch's padding modes; ``t`` in input coordinates)."""
    if 0 <= t < n:
        return t
    if mode == "zeros":
        return None
    if mode == "reflect":
        return -t if t < 0 else 2 * (n - 1) - t
    if mode == "replicate":
        return min(max(t, 0), n - 1)
    if mode == "circular":
        return t % n
    raise ValueError(mode)


def _conv2d_pads(layer):
    """(top, bottom, left, right) of an nn.Conv2d's padding."""
    if layer.padding == "valid":
        return (0, 0, 0, 0)
    if layer.padding == "same":            # the extra entry goes at the end
        out = []
        for d, k in zip(layer.dilation, layer.kernel_size):
            total = d * (k - 1)
            out += [total // 2, total - total // 2]
        return tuple(out)
    ph, pw = layer.padding
    return (ph, ph, pw, pw)


def _rational_params(layer):
    w = [[[[R.frac(v) for v in row] for row in ch] for ch in o]
         for o in layer.weight.detach().double().numpy()]
    b = (None if layer.bias is None
         else [R.frac(v) for v in layer.bias.detach().double().numpy()])
    return w, b


def _exact_conv2d(layer, chw):
    """``f(x) -> (y, (O, Ho, Wo))``: nn.Conv2d by its definition on a flat
    (C, H, W) list of Fractions, output flat in (O, Ho, Wo) order."""
    C, H, W = chw
    w, b = _rational_params(layer)
    O, cg, kh, kw = (len(w), len(w[0]), len(w[0][0]), len(w[0][0][0]))
    og = O // layer.groups
    (sh, sw), (dh, dw) = layer.stride, layer.dilation
    top, bottom, left, right = _conv2d_pads(layer)
    Ho = (H + top + bottom - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + left + right - dw * (kw - 1) - 1) // sw + 1
    mode = layer.padding_mode

    def f(x):
        y = []
        for o in range(O):
            g = o // og
            for i in range(Ho):
                for j in range(Wo):
                    acc = R.Fraction(0) if b is None else b[o]
                    for cc in range(cg):
                        c = g * cg + cc
                        for p in range(kh):
                            r = _source(i * sh - top + p * dh, H, mode)
                            if r is None:
                                continue
                            for q in range(kw):
                                s = _source(j * sw - left + q * dw, W, mode)
                                if s is not None:
                                    acc += w[o][cc][p][q] * x[(c * H + r) * W + s]
                    y.append(acc)
        return y, (O, Ho, Wo)
    return f


# --------------------------------------------------------------------------- #
# Input sets on three predicate variables, and the exact image check            #
# --------------------------------------------------------------------------- #
# Predicate boxes of the stars of ``_few_var_inputs``: the first set has the
# usual [-1, 1]^3, the second an asymmetric box (one side at 0), so a kernel
# that rebuilds the predicate bounds from one side, or exchanges or negates
# them, is seen; a symmetric box hides such a bug.
_PREDICATE_BOXES = (
    (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
    (np.array([-0.5, 0.0, -2.0]), np.array([1.5, 1.0, 0.25])),
)


def _star_predicate(rng, index, k=3):
    """``(lb, ub, A, b)`` of the ``index``-th star's predicate polytope: the
    box of ``_PREDICATE_BOXES`` cut by one binding row ``r a <= rhs`` that is
    not a box row (coefficients 1, 1/2 and 1 with random signs; ``rhs`` is a
    quarter below the row's maximum over the box, so the row removes a corner
    and the polytope stays full-dimensional). All dyadic."""
    lb, ub = _PREDICATE_BOXES[index % len(_PREDICATE_BOXES)]
    row = rng.permutation([1.0, 0.5, 1.0]) * rng.choice([-1.0, 1.0], k)
    rhs = float(np.sum(np.maximum(row * lb, row * ub))) - 0.25
    A = np.vstack([np.eye(k), -np.eye(k), row])
    b = np.concatenate([ub, -lb, [rhs]])
    return lb, ub, A, b


def _few_var_inputs(chw, seed, count=2, bits=20):
    """``count`` pairs of image sets on three predicate variables each, with
    centre and generators dense over every pixel (``_many_bits`` values):
    an ImageStar on the predicate polytope of ``_star_predicate`` (the second
    set on an asymmetric box), whose binding row is not a box row, and the
    ImageZono on the same centre and generators. Returns a list of
    ``(image_star, image_zono, c, G, star_rows, zono_rows)``: ``c`` and ``G``
    in tensor (C, H, W) order, each ``rows`` an ``(A, b)`` pair of the
    predicate polytope in rationals."""
    C, H, W = chw
    n, k = C * H * W, 3
    rng = np.random.default_rng(seed)
    order = hwc_index(C, H, W)
    out = []
    for index in range(count):
        c = _many_bits(rng, n, bits, span=2)
        G = _many_bits(rng, (n, k), bits)
        lb, ub, A, b = _star_predicate(rng, index, k)
        box_A = np.vstack([np.eye(k), -np.eye(k)])
        V = np.hstack([c[order][:, None], G[order]])
        star = ImageStar(V, A, b.reshape(-1, 1), lb.reshape(-1, 1),
                         ub.reshape(-1, 1), height=H, width=W, num_channels=C)
        zono = ImageZono(c[order].reshape(-1, 1), G[order].copy(), H, W, C)
        out.append((star, zono, c, G, (A, b), (box_A, np.ones(2 * k))))
    return out


def _piece(f, c, G, rows, stored_hwc=True):
    """The exact image of ``{c + G a : A a <= b}`` under the affine ``f`` as
    one rational piece, in the order the result stores: ``(piece,
    out_shape)``. ``f`` is evaluated at the centre and at the centre plus each
    generator; the image of a polytope under an affine map is the hull of the
    images of its vertices."""
    cf = [R.frac(v) for v in c]
    y0, shape = f(cf)
    cols = []
    for j in range(G.shape[1]):
        yj, _ = f([ci + R.frac(g) for ci, g in zip(cf, G[:, j])])
        cols.append([u - v for u, v in zip(yj, y0)])
    order = hwc_index(*shape) if stored_hwc else range(len(y0))
    A, b = rows
    return (R.Piece(R.matrix(A.tolist()), R.vector(b.tolist()),
                    [[col[p] for col in cols] for p in order],
                    [y0[p] for p in order]), shape)


class _Solver:
    """Exact solutions of ``M x = r`` for one matrix ``M`` and many ``r``.

    When ``M`` has full column rank (always so for the dense random
    generators of these cells) the solution is unique: it is found from a
    square subsystem of independent rows and checked against every row, all
    in rationals. Otherwise ``unique`` is False and the caller falls back to
    the simplex of ``tests/_oracles/rational.py``."""

    def __init__(self, M):
        self.M = M
        n = len(M[0]) if M else 0
        self.rows = []
        for i, row in enumerate(M):
            if len(self.rows) == n:
                break
            if R.rank([M[j] for j in self.rows] + [row]) > len(self.rows):
                self.rows.append(i)
        self.unique = n > 0 and len(self.rows) == n

    def __call__(self, r):
        """The solution, or None when there is none."""
        x = R.solve([self.M[i] for i in self.rows], [r[i] for i in self.rows])
        if x is None or any(R.dot(row, x) != v for row, v in zip(self.M, r)):
            return None
        return x


def _member(computed, solver, y):
    """Is ``y`` in the computed set ``(c, G, C, d, lb, ub)``? Exact."""
    if not solver.unique:
        return R.member(computed, y)
    c, G, C, d, lb, ub = computed
    a = solver([yi - ci for yi, ci in zip(y, c)])
    return a is not None and all(
        R.dot(row, a) <= v for row, v in zip(C, d)) and all(
        (lo is None or lo <= ai) and (hi is None or ai <= hi)
        for ai, lo, hi in zip(a, lb, ub))


def _reached(piece, solver, y):
    """Is ``y`` the image of a point of the piece's region? Exact."""
    if not solver.unique:
        return R.image_contains([piece], y)
    x = solver([yi - ci for yi, ci in zip(y, piece.c)])
    return x is not None and piece.holds(x)


def _assert_exact_image(pieces, results, what, exact, out_chw=None):
    """Every vertex of every exact piece is in one returned set, the same set
    for all vertices of a piece (``rational.escapes`` and ``straddlers``), in
    rationals with no slack; with ``exact`` also every vertex of every
    returned set has a preimage (``rational.unreached``)."""
    if out_chw is not None:
        C, H, W = out_chw
        for s in results:
            assert (s.height, s.width, s.num_channels) == (H, W, C), (
                f"{what}: a result stores {s.height}x{s.width}x"
                f"{s.num_channels}, the output is (C, H, W) = {out_chw}")
    computed = [R.star(s) for s in results]
    solvers = [_Solver(s[1]) for s in computed]
    for k, piece in enumerate(pieces):
        image = piece.image()
        inside = [[_member(s, m, y) for y in image]
                  for s, m in zip(computed, solvers)]
        out = [y for i, y in enumerate(image) if not any(r[i] for r in inside)]
        assert not out, (f"{what}: {len(out)} of {len(image)} vertices of "
                         f"exact piece {k} are in no returned set, first "
                         f"{[float(v) for v in out[0]]}")
        assert any(all(r) for r in inside), (
            f"{what}: exact piece {k} is spread over several returned sets")
    if exact:
        image_solvers = [_Solver(p.W) for p in pieces]
        for k, s in enumerate(computed):
            far = [y for y in R.set_vertices(s) if not any(
                _reached(p, m, y) for p, m in zip(pieces, image_solvers))]
            assert not far, (f"{what}: returned set {k} has {len(far)} "
                             f"vertices that no input point reaches")


# --------------------------------------------------------------------------- #
# Cells                                                                         #
# --------------------------------------------------------------------------- #
def _cells(configs, row_configs, same_mark=None):
    """The (config, set kind, method, input variant) cells of a sampled
    test: every config on several ImageStars (exact) and several ImageZonos,
    and the configs in ``row_configs`` on an ImageStar with a binding row
    (approx). ``same_mark`` marks the configs whose torch call warns."""
    out = []
    for cfg in configs:
        variants = [("ImageStar", "exact", "split"),
                    ("ImageZono", "approx", "split")]
        if cfg in row_configs:
            variants.append(("ImageStar", "approx", "row"))
        for v in variants:
            marks = same_mark(cfg) if same_mark else ()
            out.append(pytest.param(cfg, *v, marks=marks,
                                    id=f"{cfg}-{v[0]}-{v[1]}-{v[2]}"))
    return out


def _exact_cells(configs, approx_configs, same_mark=None):
    """(config, set kind, method) of a rational cell: every config on
    ImageStars (exact) and ImageZonos, ``approx_configs`` on ImageStars
    (approx) too."""
    out = []
    for cfg in configs:
        kinds = [("ImageStar", "exact"), ("ImageZono", "approx")]
        if cfg in approx_configs:
            kinds.append(("ImageStar", "approx"))
        for kind, method in kinds:
            marks = same_mark(cfg) if same_mark else ()
            out.append(pytest.param(cfg, kind, method, marks=marks,
                                    id=f"{cfg}-{kind}-{method}"))
    return out


def _same_mark(cfg):
    return _SAME_WARNING if cfg == "same" else ()


def _inputs(kind, variant, chw, seed=0):
    """``(sets, points)``: three sets of ``kind`` covering a pixel box cut
    along its widest pixel (``split``; the pixel of the first channel at the
    bottom-left corner has zero width, so an ImageStar has fewer predicate
    variables than pixels and an ImageZono a zero generator), or one
    ImageStar with a binding row that is not a box row (``row``); points are
    member points in (C, H, W)."""
    lo, hi = pixel_box(*chw, seed=seed)
    if variant == "split":
        hi[0, chw[1] - 1, 0] = lo[0, chw[1] - 1, 0]
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    star, points = constrained_image_star(lo, hi, seed=seed)
    return [star], points


@pytest.mark.parametrize("cfg,kind,method,variant", _cells(
    ("stride-pad-dilation",), ("stride-pad-dilation",)))
def test_conv2d_layer_holds_every_torch_output(cfg, kind, method, variant):
    """The point replay of each answering row (ImageStar exact, ImageZono
    approx, ImageStar approx) through ``reach_layer``, on the layer whose
    kernel, stride, padding and dilation differ between the axes: three sets
    at once covering a pixel box with one zero-width pixel (so the ImageStar
    has fewer predicate variables than pixels and the ImageZono a zero
    generator), or one ImageStar with a binding row over all its variables;
    torch in float64 on 128 corners and grid points, membership by LP on the
    rows. The other layer configurations (groups, no bias, every padding
    mode, padding='same') are held by
    ``test_conv2d_layer_is_the_exact_image_in_rationals``. Catches a result
    that keeps only one of several input sets, a bias missing or added to a
    generator, an axis parameter read for the wrong axis, and an output
    stored with height, width and channels confused."""
    layer, chw = _make_conv2d(cfg)
    sets, points = _inputs(kind, variant, chw)
    what = f"Conv2d {cfg} on {kind} ({variant}, {method})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_out_chw(layer, chw))


@pytest.mark.parametrize("cfg,kind,method", _exact_cells(
    _CONV2D, ("groups-no-bias", "reflect"), _same_mark))
def test_conv2d_layer_is_the_exact_image_in_rationals(cfg, kind, method):
    """Two input sets at once, each on three predicate variables (the
    ImageStar with a binding row, the second on an asymmetric predicate
    box), with centre, generators, weights and bias of about 20 significant
    bits, so float32 arithmetic is not exact. The exact image is the
    definition of the convolution in rationals. Catches every wrong
    generator even when sampled points happen to be inside, a convolution
    run in float32, a lost input set, predicate bounds not passed on as they
    are, and, for the exact method, a result larger than the image."""
    layer, chw = _make_conv2d(cfg, seed=1, bits=20)
    f = _exact_conv2d(layer, chw)
    inputs = _few_var_inputs(chw, seed=2)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in inputs:
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"Conv2d {cfg} on {kind} ({method}), exact image"
    result = reach_one_layer(layer, sets, method, what)
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_float64_conv2d_layer_is_the_exact_image_in_rationals(kind, method):
    """A float64 Conv2d (kernel, stride, padding and dilation different per
    axis) whose weights and bias need 28 significant bits, more than float32
    holds, on two input sets of three predicate variables with 12-bit centre
    and generators (the ImageStar with a binding row, the second on an
    asymmetric predicate box): every product and partial sum needs at most
    45 bits, so a float64 convolution is exact and the exact image is the
    definition in rationals with no slack. Catches a weight, bias or basis
    column rounded to float32 on its way to the float64 convolution, which
    reaches a float64 model as another network."""
    chw, kwargs = _CONV2D["stride-pad-dilation"]
    layer = _set_params(nn.Conv2d(chw[0], **kwargs).double(),
                        np.random.default_rng(1), bits=28)
    f = _exact_conv2d(layer, chw)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _few_var_inputs(
            chw, seed=2, bits=12):
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"float64 Conv2d with 28-bit weights on {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


# Predicate box of the second star of ``_sets_on_four_variables``: one side
# at 0 and sides beyond [-1, 1], so predicate bounds that are reset,
# exchanged or rebuilt from one side are seen.
_FOUR_VARIABLE_BOX = (np.array([-0.5, 0.0, -2.0, 0.25]),
                      np.array([1.5, 1.0, 0.25, 1.0]))


def _sets_on_four_variables(chw, seed, bits=20):
    """``_few_var_inputs`` on four predicate variables instead of three: two
    pairs ``(image_star, image_zono, c, G, star_rows, zono_rows)`` with dense
    centre and generators (``_many_bits`` values, ``c`` and ``G`` in tensor
    order). Each ImageStar's predicate polytope is a box (the first
    [-1, 1]^4, the second ``_FOUR_VARIABLE_BOX``) cut by one binding row over
    all four variables with distinct coefficients, so no two of its
    generator columns can be exchanged without changing the set; each
    ImageZono has the same centre and generators on the cube. Five basis
    columns per star are enough for a batch of three and a shorter one of
    two."""
    C, H, W = chw
    n, k = C * H * W, 4
    rng = np.random.default_rng(seed)
    order = hwc_index(C, H, W)
    cube = (np.vstack([np.eye(k), -np.eye(k)]), np.ones(2 * k))
    out = []
    for lb, ub in ((-np.ones(k), np.ones(k)), _FOUR_VARIABLE_BOX):
        c = _many_bits(rng, n, bits, span=2)
        G = _many_bits(rng, (n, k), bits)
        row = (rng.permutation([1.0, 0.5, 0.25, 0.75])
               * rng.choice([-1.0, 1.0], k))
        rhs = float(np.sum(np.maximum(row * lb, row * ub))) - 0.25
        A = np.vstack([np.eye(k), -np.eye(k), row])
        b = np.concatenate([ub, -lb, [rhs]])
        V = np.hstack([c[order][:, None], G[order]])
        star = ImageStar(V, A, b.reshape(-1, 1), lb.reshape(-1, 1),
                         ub.reshape(-1, 1), height=H, width=W, num_channels=C)
        zono = ImageZono(c[order].reshape(-1, 1), G[order].copy(), H, W, C)
        out.append((star, zono, c, G, (A, b), cube))
    return out


def _spy_on_column_batches(monkeypatch):
    """The batch sizes of every ``map_basis_columns`` call the Conv2d kernels
    make, one list per call (the probe on an empty batch left out). The spy
    calls the kernel module's own function, so whatever it does still runs."""
    from n2v.nn.layer_ops import conv2d_reach
    calls = []
    original = conv2d_reach.map_basis_columns

    def spy(op, V, scratch_factor=1):
        sizes = []

        def counted(x):
            if x.shape[0]:
                sizes.append(int(x.shape[0]))
            return op(x)
        result = original(counted, V, scratch_factor)
        calls.append(sizes)
        return result
    monkeypatch.setattr(conv2d_reach, "map_basis_columns", spy)
    return calls


@pytest.mark.parametrize("batches", ["one-column", "short-last"])
@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_conv2d_in_several_column_batches_is_the_exact_image_in_rationals(
        monkeypatch, kind, method, batches):
    """``map_basis_columns`` convolves the basis columns in batches whose
    size comes from a memory budget (``_COLUMN_BATCH_BYTES``); on test-sized
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
    layer, chw = _make_conv2d("stride-pad-dilation", seed=1, bits=20)
    out_chw = _out_chw(layer, chw)
    n_in, n_out = int(np.prod(chw)), int(np.prod(out_chw))
    area = int(np.prod(layer.kernel_size))
    column = 8 * (n_in + n_out + area * max(n_in, n_out))
    per_batch = 1 if batches == "one-column" else 3
    monkeypatch.setattr(conv2d_reach, "_COLUMN_BATCH_BYTES",
                        per_batch * column)
    calls = _spy_on_column_batches(monkeypatch)
    f = _exact_conv2d(layer, chw)
    sets, pieces = [], []
    for star, zono, c, G, star_rows, zono_rows in _sets_on_four_variables(
            chw, seed=3):
        sets.append(star if kind == "ImageStar" else zono)
        piece, _ = _piece(f, c, G, star_rows if kind == "ImageStar"
                          else zono_rows)
        pieces.append(piece)
    what = f"Conv2d on {kind} ({method}) in batches of {per_batch} column(s)"
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
    """``head`` (default: a 1x1 convolution that mixes the channels, with
    bias) -> ``op`` -> Flatten -> Linear to 3 outputs: the rows the operation
    emits are consumed by an affine map and by the layout conversion at
    Flatten."""
    rng = np.random.default_rng(seed)
    if head is None:
        head = _set_params(nn.Conv2d(chw[0], chw[0], 1), rng)
    n_out = int(np.prod(_out_chw(op, chw)))
    return nn.Sequential(head, op, nn.Flatten(),
                         _set_params(nn.Linear(n_out, 3), rng))


def _two_straddlers(chw, seed=0):
    """A pixel box with every pixel positive except two, in different
    channels, rows and columns, whose intervals contain 0: an exact ReLU
    splits it into four sets, which then reach the next layer together."""
    lo, hi = pixel_box(*chw, seed=seed)
    shift = 0.25 - lo.min()
    lo, hi = lo + shift, hi + shift
    C, H, W = chw
    lo[0, 0, 1], hi[0, 0, 1] = -0.75, 0.5
    lo[C - 1, H - 1, W - 2], hi[C - 1, H - 1, W - 2] = -0.25, 1.0
    return lo, hi


@pytest.mark.parametrize("cfg,kind,method,variant", [
    ("stride-pad-dilation", "ImageStar", "exact", "split"),
    ("reflect", "ImageZono", "approx", "split"),
    ("circular", "ImageStar", "approx", "row"),
])
def test_conv2d_inside_a_model_holds_every_torch_output(cfg, kind, method,
                                                         variant):
    """Conv2d between a channel-mixing convolution and Flatten > Linear,
    through ``reach_pytorch_model``. Catches a convolution whose rows or
    layout are wrong in a way only the downstream affine map and the Flatten
    conversion expose, and a node loop that hands a node the wrong sets."""
    op, chw = _make_conv2d(cfg)
    model = _chain(op, chw)
    sets, points = _inputs(kind, variant, chw)
    what = f"conv > Conv2d {cfg} > flatten > linear on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    assert_covered(result, replay_torch(model, points, (1,) + chw), what)


@pytest.mark.parametrize("cfg", ["replicate"])
def test_conv2d_after_an_exact_relu_split_holds_every_torch_output(cfg):
    """ReLU > Conv2d > Flatten > Linear with the exact method on a box with
    two pixels that straddle zero: the convolution receives the four sets of
    the exact split together. Catches a convolution that returns the image
    of only the first or only the last of several input sets."""
    op, chw = _make_conv2d(cfg)
    model = _chain(op, chw, head=nn.ReLU())
    lo, hi = _two_straddlers(chw)
    what = f"relu > Conv2d {cfg} > flatten > linear, exact"
    result = reach_model(model, image_set("ImageStar", lo, hi), "exact", what)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


def _onnx_conv(src, name, dst, **attrs):
    """A Conv node ``src -> dst`` with initializers ``<name>_w``, ``<name>_b``."""
    return helper.make_node("Conv", [src, f"{name}_w", f"{name}_b"], [dst],
                            name=name, **attrs)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
def test_onnx_conv_with_asymmetric_pads_holds_every_onnxruntime_output(
        tmp_path, kind, method):
    """An ONNX Conv whose four pads all differ (top 1, left 0, bottom 0,
    right 2; the loader puts a static Pad in front of a padding-free
    nn.Conv2d), with strides (2, 1), dilations (1, 2) and a (2, 3) kernel,
    between two 1x1 Convs, loaded with ``load_onnx`` and replayed on
    onnxruntime. Catches the pad amounts read in the wrong order and the
    padded image placed at the wrong offset."""
    C, H, W = 2, 3, 5
    rng = np.random.default_rng(4)
    mid = nn.Sequential(nn.ZeroPad2d((0, 2, 1, 0)),
                        nn.Conv2d(C, 3, (2, 3), stride=(2, 1),
                                  dilation=(1, 2)))
    _set_params(mid[1], rng)
    out_chw = _out_chw(mid, (C, H, W))
    w0, b0 = _distinct(rng, (C, C, 1, 1)), _distinct(rng, (C,))
    w2, b2 = _distinct(rng, (2, 3, 1, 1)), _distinct(rng, (2,))
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"Conv0_w": f32(w0), "Conv0_b": f32(b0),
             "Conv1_w": f32(mid[1].weight.detach()),
             "Conv1_b": f32(mid[1].bias.detach()),
             "Conv2_w": f32(w2), "Conv2_b": f32(b2)}
    nodes = [_onnx_conv("X", "Conv0", "T0"),
             _onnx_conv("T0", "Conv1", "T1", pads=[1, 0, 0, 2], strides=[2, 1],
                        dilations=[1, 2], kernel_shape=[2, 3]),
             _onnx_conv("T1", "Conv2", "Y")]
    final = (2,) + out_chw[1:]
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, *final], inits,
                     name="conv_asym")
    lo, hi = pixel_box(C, H, W, seed=5)
    if kind == "ImageStar" and method == "approx":
        start, points = constrained_image_star(lo, hi, seed=5)
    else:
        start, points = image_set(kind, lo, hi), samples(lo, hi)
    what = f"ONNX Conv with pads [1, 0, 0, 2] on {kind} ({method})"
    result = reach_model(load_onnx(path), start, method, what)
    assert_covered(result, replay_onnx(path, points, [1, C, H, W]), what,
                   out_chw=final)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_onnx_conv_with_groups_and_per_axis_pads_holds_every_onnxruntime_output(
        tmp_path, kind, method):
    """An ONNX Conv with group 2, pads (top 1, left 2, bottom 1, right 2),
    strides (1, 2), dilations (2, 1) and a (2, 3) kernel, loaded with
    ``load_onnx`` as an nn.Conv2d with padding (1, 2) and replayed on
    onnxruntime. Catches the ONNX pads, strides, dilations or group read
    for the wrong axis between the file and the reach."""
    C, H, W = 2, 4, 3
    rng = np.random.default_rng(14)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"g_w": f32(_distinct(rng, (4, 1, 2, 3))),
             "g_b": f32(_distinct(rng, (4,))),
             "z_w": f32(_distinct(rng, (2, 4, 1, 1))),
             "z_b": f32(_distinct(rng, (2,)))}
    ref = nn.Conv2d(C, 4, (2, 3), stride=(1, 2), padding=(1, 2),
                    dilation=(2, 1), groups=2)
    out_chw = (2,) + _out_chw(ref, (C, H, W))[1:]
    nodes = [helper.make_node("Conv", ["X", "g_w", "g_b"], ["T"], group=2,
                              pads=[1, 2, 1, 2], strides=[1, 2],
                              dilations=[2, 1], kernel_shape=[2, 3]),
             _onnx_conv("T", "z", "Y")]
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, *out_chw], inits,
                     name="conv_groups")
    lo, hi = pixel_box(C, H, W, seed=15)
    what = f"ONNX Conv group 2 on {kind} ({method})"
    result = reach_model(load_onnx(path), image_set(kind, lo, hi), method,
                         what)
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, C, H, W]),
                   what, out_chw=out_chw)


def test_onnx_conv_with_auto_pad_same_upper_refuses_or_holds_every_onnxruntime_output(
        tmp_path):
    """An ONNX Conv with ``auto_pad = SAME_UPPER`` and the even kernel (2, 4)
    on a (2, 3, 5) image, between two 1x1 Convs: the padding keeps the
    image size, and the odd extra entry of each axis goes at the end. The
    vendored onnx2torch refuses SAME_UPPER and SAME_LOWER at load
    (``NotImplementedError``); a loader or converter that learns to read
    them must hold every onnxruntime output. Catches SAME_UPPER read as
    SAME_LOWER (the extra entry at the start) or as explicit pads in the
    wrong order."""
    C, H, W = 2, 3, 5
    rng = np.random.default_rng(24)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"Conv0_w": f32(_distinct(rng, (C, C, 1, 1))),
             "Conv0_b": f32(_distinct(rng, (C,))),
             "Conv1_w": f32(_distinct(rng, (3, C, 2, 4))),
             "Conv1_b": f32(_distinct(rng, (3,))),
             "Conv2_w": f32(_distinct(rng, (2, 3, 1, 1))),
             "Conv2_b": f32(_distinct(rng, (2,)))}
    nodes = [_onnx_conv("X", "Conv0", "T0"),
             _onnx_conv("T0", "Conv1", "T1", auto_pad="SAME_UPPER",
                        kernel_shape=[2, 4]),
             _onnx_conv("T1", "Conv2", "Y")]
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, 2, H, W], inits,
                     name="conv_same_upper")
    lo, hi = pixel_box(C, H, W, seed=25)
    what = "ONNX Conv with auto_pad SAME_UPPER on ImageStar (approx)"
    try:
        result = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                             "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, C, H, W]),
                   what, out_chw=(2, H, W))


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
_FLAT_REFUSED_BY_VALUE_ERROR = known_bug(
    23, "Conv2d on a flat Star or Zono raises ValueError instead of the "
        "refusal of an unsupported combination (NotImplementedError, which "
        "UnsupportedError will subclass), so a caller that tries the next "
        "method on a refusal stops instead", raises=ValueError)


@pytest.mark.parametrize("kind", [
    pytest.param("Star", marks=_FLAT_REFUSED_BY_VALUE_ERROR),
    pytest.param("Zono", marks=_FLAT_REFUSED_BY_VALUE_ERROR),
    "Box", "Hexatope", "Octatope"])
def test_conv2d_on_a_flat_set_is_refused(kind):
    """A flat set carries no height, width or channel count, and no rule
    reaches Conv2d on it: ``reach_layer`` must refuse, never answer. The set
    has 32 entries, which a square 4 x 4 image of the layer's 2 channels
    could hold, so a kernel that guesses a square image (as
    ``conv2d_hexatope`` does) answers here instead of failing on the size.
    Catches such a guess in the Star and Zono kernels, which reads the
    tensor-order entries as height, width, channel, and a dispatcher that
    routes Box, Hexatope or Octatope to a convolution rule."""
    lo, hi = flat_box(32)
    method = "exact" if kind == "Star" else "approx"
    layer, _ = _make_conv2d("stride-pad-dilation")
    with pytest.raises(NotImplementedError):
        reach_layer(layer, [flat_set(kind, lo, hi)], method)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_conv2d_model_on_a_hexatope_or_octatope_is_refused(kind):
    """``reach_pytorch_model`` on a model whose first layer is a Conv2d, with
    a flat Hexatope or Octatope of 32 entries (a square 2 x 4 x 4 image
    could hold them): the model entry accepts the set type for the approx
    method, and the convolution has no rule for it, so the reach must
    refuse. Catches the placeholder kernels ``conv2d_hexatope`` and
    ``conv2d_octatope``, whose result is the input's bounds whatever the
    layer (registry HEXOCT-5), routed into the model entry."""
    lo, hi = flat_box(32)
    layer, _ = _make_conv2d("stride-pad-dilation")
    with pytest.raises(NotImplementedError):
        reach_model(nn.Sequential(layer), flat_set(kind, lo, hi), "approx",
                    f"Conv2d model on a {kind}")


def test_conv2d_box_kernel_is_refused():
    """``conv2d_box`` (not routed by the dispatcher) has no image shape to
    convolve: a direct call must refuse."""
    from n2v.sets import Box
    lo, hi = flat_box(30)
    layer, _ = _make_conv2d("stride-pad-dilation")
    with pytest.raises(NotImplementedError):
        conv2d_box(layer, [Box(lo, hi)])


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_conv2d_with_the_wrong_channel_count_is_refused(kind):
    """A 3-channel Conv2d on a 2-channel image: torch itself raises, so the
    reach must raise too (a malformed input: ValueError, or a refusal)."""
    lo, hi = pixel_box(2, 3, 5)
    layer = _set_params(nn.Conv2d(3, 2, (2, 3)), np.random.default_rng(0))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_layer(layer, [image_set(kind, lo, hi)], "approx")


_MIXED_LIST_TYPE_ERROR = known_bug(
    23, "conv2d_star handed an ImageZono after an ImageStar raises TypeError "
        "('conv2d_star expects Star or ImageStar') instead of the refusal of "
        "an unsupported combination (NotImplementedError) or of an input "
        "that cannot be read (ValueError)", raises=TypeError)


@pytest.mark.parametrize("first,second", [
    pytest.param("ImageStar", "ImageZono", marks=_MIXED_LIST_TYPE_ERROR),
    ("ImageZono", "ImageStar")])
def test_conv2d_on_a_list_of_mixed_set_types_refuses_or_is_sound(first,
                                                                  second):
    """An ImageStar and an ImageZono in one ``reach_layer`` call (only a
    direct call can build such a list). The dispatcher routes by the first
    set; the kernel must refuse the other (``ValueError`` for an input it
    cannot read, ``NotImplementedError`` for an unsupported combination),
    or answer with sets that hold the outputs of both. Catches a kernel
    that skips a set of the wrong type, or appends the previous result in
    its place. The ImageZono-led list is refused with ``ValueError`` today
    and is the passing neighbour of the marked case."""
    layer, chw = _make_conv2d("stride-pad-dilation")
    lo, hi = pixel_box(*chw)
    try:
        result = reach_layer(layer, [image_set(first, lo, hi),
                                     image_set(second, lo + 4, hi + 4)],
                             "approx")
    except (ValueError, NotImplementedError):
        return
    points = samples(lo, hi, 32) + samples(lo + 4, hi + 4, 32)
    assert_covered(result, replay_torch(layer, points, (1,) + chw),
                   "Conv2d on a mixed list", out_chw=_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# A Conv2d subclass                                                             #
# --------------------------------------------------------------------------- #
class _DoubledConv2d(nn.Conv2d):
    """A Conv2d subclass whose forward is not the stock one: twice the
    convolution."""

    def forward(self, x):
        return 2.0 * super().forward(x)


class _ShiftedWeightConv2d(nn.Conv2d):
    """A Conv2d subclass that changes its weight inside ``_conv_forward``,
    as weight standardization does (here 2 w - 1/2, dyadic, so the float64
    replay is exact)."""

    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, 2.0 * weight - 0.5, bias)


_OWN_FORWARD = known_bug(
    7, "reach_layer matches a Conv2d subclass by isinstance and convolves "
       "with the stock rule, so a subclass whose forward is not the stock "
       "one (here twice the convolution) is answered for another function "
       "and every true output lies outside (REACH-9; the model entry traces "
       "into such a subclass and refuses its functional conv2d)")


@pytest.mark.parametrize("subclass", [
    pytest.param(_DoubledConv2d, marks=_OWN_FORWARD, id="own-forward"),
    pytest.param(_ShiftedWeightConv2d, id="own-conv-forward")])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_conv2d_subclass_refuses_or_holds_every_torch_output(kind, subclass):
    """A Conv2d subclass handed to ``reach_layer`` on a (2, 3, 5) pixel box:
    the reach must refuse it (``NotImplementedError``) or hold every output
    of the subclass's own forward (torch in float64 on 128 corners and grid
    points, membership by LP on the rows). The kernels call the layer's
    ``_conv_forward``, so a subclass that overrides only that is followed:
    it is the passing neighbour of the marked case, and it catches a kernel
    that convolves with ``F.conv2d`` and the layer's parameters instead of
    the layer's own convolution."""
    chw, kwargs = _CONV2D["stride-pad-dilation"]
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


# --------------------------------------------------------------------------- #
# Hexatope and Octatope (direct kernel calls; the dispatcher refuses them)      #
# --------------------------------------------------------------------------- #
_HEXOCT_PLACEHOLDER = known_bug(
    9, "conv2d_hexatope and conv2d_octatope return the input's bounds "
       "whatever the layer (a placeholder, registry HEXOCT-5), so the true "
       "outputs of any convolution other than the identity lie outside the "
       "result")

# id: ((C, H, W), the layer). The identity is the one layer the placeholder
# kernels answer soundly; (2, 2, 2) takes conv2d_hexatope's square-image
# branch, (2, 3, 5) its fallback for an image that is not square
# (conv2d_octatope has no branch). "-30bit" uses bounds with 31 significant
# bits (kept exactly by ``from_bounds``, not representable in float32), so a
# bound rounded to float32 is seen.
_HEXOCT_LAYERS = {
    "identity-2x2x2": (2, 2, 2),
    "identity-2x3x5-30bit": (2, 3, 5),
    "conv-2x2x2": (2, 2, 2),
    "conv-2x3x5": (2, 3, 5),
}


def _hexoct_box(layer_id):
    chw = _HEXOCT_LAYERS[layer_id]
    if not layer_id.endswith("30bit"):
        return pixel_box(*chw, seed=7)
    rng = np.random.default_rng(7)
    lo = _many_bits(rng, chw, bits=30, span=2)
    return lo, lo + np.abs(_many_bits(rng, chw, bits=30)) + 2.0 ** -30


def _hexoct_layer(layer_id):
    chw = _HEXOCT_LAYERS[layer_id]
    if layer_id.startswith("identity"):
        layer = nn.Conv2d(chw[0], chw[0], 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(chw[0]).reshape(chw[0], chw[0], 1, 1))
        return layer, chw
    # same output shape as the input, so the failure is an escape
    layer = nn.Conv2d(chw[0], chw[0], (3, 3), padding=(1, 1))
    return _set_params(layer, np.random.default_rng(6)), chw


@pytest.mark.parametrize("layer_id,kind", [
    ("identity-2x2x2", "Hexatope"), ("identity-2x3x5-30bit", "Hexatope"),
    pytest.param("conv-2x2x2", "Hexatope", marks=_HEXOCT_PLACEHOLDER),
    pytest.param("conv-2x3x5", "Hexatope", marks=_HEXOCT_PLACEHOLDER),
    ("identity-2x2x2", "Octatope"), ("identity-2x3x5-30bit", "Octatope"),
    pytest.param("conv-2x2x2", "Octatope", marks=_HEXOCT_PLACEHOLDER)])
def test_conv2d_hexatope_and_octatope_kernels_are_sound_or_refuse(kind,
                                                                  layer_id):
    """``conv2d_hexatope`` and ``conv2d_octatope`` called directly on three
    flat sets (tensor order) that cover a box: every torch output must be in
    the result, or the kernel must refuse (NotImplementedError). Catches a
    kernel that keeps one of several sets, a bound read from the wrong side,
    and a bound rounded to float32."""
    from n2v.nn.layer_ops.conv2d_reach import conv2d_hexatope, conv2d_octatope
    kernel = conv2d_hexatope if kind == "Hexatope" else conv2d_octatope
    layer, chw = _hexoct_layer(layer_id)
    lo, hi = _hexoct_box(layer_id)
    try:
        result = kernel(layer, split_cover(kind, lo, hi, 3))
    except NotImplementedError:
        return
    what = f"{kernel.__name__} with {layer_id}"
    assert_covered(result, replay_torch(layer, samples(lo, hi, 64), (1,) + chw),
                   what)
    # The outputs of the two extreme corners again, in rationals with no
    # slack: a bound moved inward by less than the LP oracle's tolerance
    # (a float32 rounding of a 31-bit bound moves it by up to 6e-8) is seen.
    computed = [R.star(s) for s in result]
    for name, y in zip(("lower", "upper"),
                       replay_torch(layer, [lo, hi], (1,) + chw)):
        assert any(R.member(s, y) for s in computed), (
            f"{what}: the output of the {name} corner of the input box is "
            f"in no returned set (exact check)")
