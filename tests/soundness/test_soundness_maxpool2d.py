"""
Soundness tests for MaxPool2D layer reachability.

Tests verify mathematical correctness of MaxPool2D operations.
MaxPool is a non-linear operation that may require splitting for exact analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar
from n2v.nn.layer_ops.maxpool2d_reach import maxpool2d_hexatope, maxpool2d_octatope


class TestMaxPool2DEdgeCases:
    """Edge case tests for MaxPool2D."""

    def test_negative_values(self):
        """A point input (every pixel of zero width, so the ImageStar has no
        predicate variable) whose values are all below zero, through a pool
        with an asymmetric window and stride, padded along the height: by
        both methods the result holds the torch output (LP on the rows), and
        every set of the exact result is that point or empty (rationals).
        Catches a padding position taken as a max candidate of value 0 (the
        September fix: MaxPool pads with -inf), the maximum of negative
        values read from another pixel, and a set without predicate
        variables mishandled. Under a padding-candidate bug the exact split
        stays at most 2**8 pieces (one padding row, eight windows)."""
        chw = (2, 3, 4)
        lo, _ = pixel_box(*chw, seed=4)
        x = lo - 3.0
        layer = nn.MaxPool2d((2, 1), stride=(2, 1), padding=(1, 0))
        out_chw = _mp_out_chw(layer, chw)
        y = replay_torch(layer, [x], (1,) + chw)
        y_stored = tuple(Fraction(float(v)) for v in
                         y[0][hwc_index(*out_chw)])
        for method in ("approx", "exact"):
            what = f"{method} MaxPool2d on a point below zero"
            out = reach_one_layer(layer, [image_set("ImageStar", x, x)],
                                  method, what)
            assert_covered(out, y, what, out_chw=out_chw)
            if method == "exact":
                for s in out:
                    vertices = rational.set_vertices(rational.star(s))
                    assert vertices in ([], [y_stored]), (
                        f"{what}: a returned set holds "
                        f"{[[float(v) for v in p] for p in vertices]}, the "
                        f"output of the point is {[float(v) for v in y_stored]}")


# =========================================================================== #
# Soundness cells: MaxPool2d on every set type, method and option              #
# =========================================================================== #
# A cell is (set type, method, option). Two oracles, neither from n2v:
#
# - the exact image in rationals (tests/_oracles/rational.py, ``maxpool``) of
#   an ImageStar of two predicate variables whose pixels are dyadic; it
#   decides containment with zero slack and, for method="exact", the other
#   direction (no vertex of a returned piece outside the image) and the
#   number of pieces;
# - torch in float64 on sample points (corners, grid points, points on a
#   binding row), each a member of some returned set by one LP on the set's
#   own rows (tests/cell_support.assert_covered).
#
# Layout: the windows and true outputs are computed in the network's
# (C, H, W) order and converted to the stored (H, W, C) order explicitly
# (``out_chw=``, ``hwc_index``); every image here has two channels and
# height different from width.
from fractions import Fraction

import pytest
from onnx import helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Zono
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (N_SAMPLES, assert_covered,
                                constrained_image_star, constrained_star,
                                flat_set, hwc_index, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug


def _mp_pair(v):
    return (int(v), int(v)) if isinstance(v, int) else tuple(int(t) for t in v)


def _mp_out_chw(layer, chw):
    """The (C, H, W) shape of the layer's output on a (C, H, W) input, by
    torch (its first output when it returns a tuple, as ``replay_torch``
    reads it)."""
    with torch.no_grad():
        y = layer(torch.zeros((1,) + tuple(chw), dtype=torch.float64))
    if isinstance(y, (tuple, list)):
        y = y[0]
    return tuple(int(t) for t in y.shape[1:])


def _mp_groups(layer, chw):
    """For each output entry, in (C, H_out, W_out) row-major order, the flat
    (C, H, W) indices of the input entries in its window: torch's rule, the
    window of output (i, j) starts at (i * stride - padding) and steps by the
    dilation; padding positions are left out (MaxPool pads with -inf). Checked
    against torch on a random input before use."""
    C, H, W = chw
    kh, kw = _mp_pair(layer.kernel_size)
    sh, sw = _mp_pair(layer.stride)
    ph, pw = _mp_pair(layer.padding)
    dh, dw = _mp_pair(layer.dilation)
    _, h_out, w_out = _mp_out_chw(layer, chw)
    groups = []
    for c in range(C):
        for i in range(h_out):
            for j in range(w_out):
                groups.append([c * H * W + h * W + w
                               for u in range(kh) for v in range(kw)
                               for h in [i * sh - ph + u * dh]
                               for w in [j * sw - pw + v * dw]
                               if 0 <= h < H and 0 <= w < W])
    x = np.random.default_rng(7).normal(size=chw)
    with torch.no_grad():
        y = layer(torch.from_numpy(x[None])).numpy().reshape(-1)
    mine = np.array([x.reshape(-1)[g].max() for g in groups])
    if not np.array_equal(mine, y):
        pytest.fail(f"test setup: the window groups of {layer} disagree with "
                    f"torch on a {chw} input")
    return groups


def _mp_layer(kwargs):
    return nn.MaxPool2d(**kwargs)


# --------------------------------------------------------------------------- #
# The approximate method on box, constrained and split inputs                  #
# --------------------------------------------------------------------------- #
# name: ((C, H, W), MaxPool2d keywords). Kernels and strides differ between
# the two axes; padding on one axis only; overlapping windows; ceil_mode where
# it does not change the output shape (the star kernels accept it there).
_MP_CONFIGS = {
    "k1x2-s1x2": ((2, 3, 5), dict(kernel_size=(1, 2), stride=(1, 2))),
    "k2x1-s2x1-p1x0": ((2, 4, 3), dict(kernel_size=(2, 1), stride=(2, 1),
                                       padding=(1, 0))),
    "k3-s3-p1x0": ((2, 4, 3), dict(kernel_size=3, stride=3, padding=(1, 0))),
    "k2-s1": ((2, 3, 5), dict(kernel_size=2, stride=1)),
    "k2x3-s1x2-p1": ((2, 3, 5), dict(kernel_size=(2, 3), stride=(1, 2),
                                     padding=1)),
    "k3x2-s2x1-p1x0-ceil": ((2, 3, 4), dict(kernel_size=(3, 2), stride=(2, 1),
                                           padding=(1, 0), ceil_mode=True)),
}

# Accepted by the ImageZono kernel (it pools the bounds with torch itself),
# refused by the star kernels: dilation, and ceil_mode where it adds windows.
# The dilated shapes are chosen so that the undilated pool has the same
# output shape: a dilation that is dropped gives a set of the right size.
_MP_ZONO_ONLY = {
    "k2-s2-d2": ((2, 3, 5), dict(kernel_size=2, stride=2, dilation=2)),
    "k2-s2-d1x2": ((2, 3, 5), dict(kernel_size=2, stride=2, dilation=(1, 2))),
    "k2-s2-ceil": ((2, 3, 5), dict(kernel_size=2, stride=2, ceil_mode=True)),
}


def _mp_const_box(chw, groups, seed):
    """A pixel box in which windows are led by a constant pixel: for each
    window in turn, the pixel with the largest lower bound is made constant
    and another pixel of the window straddles that value by 1/4 on each side.
    The regime where the window's largest lower bound belongs to a pixel of
    zero width, so that pixel must stay a max candidate."""
    lo, hi = pixel_box(*chw, seed=seed)
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    for group in groups:
        if len(group) < 2:
            continue
        top = max(group, key=lambda p: flat_lo[p])
        other = next(p for p in group if p != top)
        flat_hi[top] = flat_lo[top]
        flat_lo[other] = flat_lo[top] - 0.25
        flat_hi[other] = flat_lo[top] + 0.25
    return flat_lo.reshape(chw), flat_hi.reshape(chw)


def _mp_inputs(kind, variant, chw, layer, seed):
    """``(input sets, sample points)`` of one input variant, points in the
    network's (C, H, W) order."""
    if variant == "box":
        lo, hi = pixel_box(*chw, seed=seed)
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if variant == "const":
        lo, hi = _mp_const_box(chw, _mp_groups(layer, chw), seed)
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if variant == "split":
        lo, hi = pixel_box(*chw, seed=seed)
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    if variant == "rows":
        lo, hi = pixel_box(*chw, seed=seed)
        star, points = constrained_image_star(lo, hi, seed=seed)
        return [star], points
    raise ValueError(variant)


# The ImageStar loop over several input sets is the same code for every
# layer configuration; two configurations keep the "split" variant.
_MP_STAR_SPLIT = ("k1x2-s1x2", "k2x3-s1x2-p1")

# The ImageZono rule pools the two bound images with torch whatever the
# input: a constant pixel (a zero generator) and a list of several sets are
# the same code for every layer configuration, so two configurations (one
# unpadded, one padded on both axes) keep the "const" and "split" variants
# and every configuration keeps "box".
_MP_ZONO_VARIANTS = ("k1x2-s1x2", "k2x3-s1x2-p1")


def _mp_cells():
    cells = []
    for name, (chw, kw) in _MP_CONFIGS.items():
        star = ("box", "const", "split", "rows") if name in _MP_STAR_SPLIT \
            else ("box", "const", "rows")
        zono = ("box", "const", "split") if name in _MP_ZONO_VARIANTS \
            else ("box",)
        for kind, variants in (("ImageStar", star), ("ImageZono", zono)):
            for variant in variants:
                cells.append(pytest.param(kind, variant, chw, kw,
                                          id=f"{kind}-{name}-{variant}"))
    for name, (chw, kw) in _MP_ZONO_ONLY.items():
        cells.append(pytest.param("ImageZono", "box", chw, kw,
                                  id=f"ImageZono-{name}-box"))
    return cells


@pytest.mark.parametrize("kind,variant,chw,kw", _mp_cells())
def test_maxpool_approx_holds_every_torch_output(kind, variant, chw, kw):
    """MaxPool2d by the approximate method (ImageStar) and by the bound rule
    (ImageZono) through ``reach_layer``: every torch output of a sample point
    is in a returned set, by LP on the set's rows. Catches a window read with
    the kernel height as its width or the vertical stride as its horizontal
    one, padding read from the wrong axis, bounds of one entry used for
    another (height, width and channel confused), a constant pixel with the
    window's largest lower bound dropped from the candidates, a relaxation row
    with the wrong sign, dilation or ceil_mode not passed to the ImageZono
    pooling, only one of several input sets answered, and the approximate
    method routed to the splitting kernel (one set per input set is checked
    before the membership LPs, whose cost grows with the number of sets, so
    that this fails at once and not after hours)."""
    layer = _mp_layer(kw)
    what = f"MaxPool2d({kw}) {kind} {variant}"
    sets, points = _mp_inputs(kind, variant, chw, layer, seed=sum(chw))
    out = reach_one_layer(layer, sets, "approx", what)
    assert len(out) == len(sets), (
        f"{what}: {len(out)} sets for {len(sets)} input set(s); the "
        f"approximate method does not split")
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))


# (configuration, input shape, input variant): pixel boxes on which the exact
# split stays below about forty pieces.
_MP_EXACT_BOX = [
    ("k1x2-s1x2", (2, 2, 3), "box"), ("k1x2-s1x2", (2, 2, 3), "const"),
    ("k3-s3-p1x0", (2, 4, 3), "box"),
    ("k3-s3-p1x0", (2, 4, 3), "const"), ("k3-s3-p1x0", (2, 4, 3), "split"),
    ("k2-s1", (2, 3, 2), "box"), ("k2-s1", (2, 3, 2), "const"),
    ("k2x3-s1x2-p1", (2, 2, 3), "box"), ("k2x3-s1x2-p1", (2, 2, 3), "split"),
]


@pytest.mark.parametrize("name,chw,variant", _MP_EXACT_BOX,
                         ids=[f"{n}-{c[1]}x{c[2]}-{v}" for n, c, v in
                              _MP_EXACT_BOX])
def test_maxpool_exact_on_a_box_holds_every_torch_output(name, chw, variant):
    """MaxPool2d by the exact method on a pixel box (one predicate variable
    per pixel, several uncertain windows): every torch output is in some
    returned piece. Catches a dropped split piece, a split row with the wrong
    right-hand side, and a constant pixel dropped from the candidates of the
    exact split."""
    layer = _mp_layer(_MP_CONFIGS[name][1])
    what = f"exact MaxPool2d {name} on {chw} {variant}"
    sets, points = _mp_inputs("ImageStar", variant, chw, layer, seed=11)
    out = reach_one_layer(layer, sets, "exact", what)
    assert_covered(out, replay_torch(layer, points[:64], (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# The exact image in rationals, on an ImageStar of two predicate variables     #
# --------------------------------------------------------------------------- #
# The predicate polygon: |a1| <= 1, |a2| <= 1 and the binding row
# a1 + a2 / 2 <= 1, which cuts off the corner (1, 1).
_MP_POLY_A = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, Fraction(1, 2)]]
_MP_POLY_B = [1, 1, 1, 1, 1]

# The candidates of a contested window: (centre offset from the window's top
# level, generator). Each pixel listed is the window's strict largest on a
# region of positive area of the polygon; the generators have |g|_1 <= 1/4.
_MP_PATTERNS = {
    # two pixels whose centres differ by 1/8
    "two": [(0.0, (0.25, 0.0)), (0.125, (0.0, -0.25))],
    # three pixels with three different centres
    "three": [(0.0, (0.25, 0.0)), (0.125, (0.0, -0.25)),
              (-0.0625, (-0.125, 0.125))],
    # a constant pixel holds the largest lower bound, another straddles it
    "const": [(0.0, (0.0, 0.0)), (0.0, (0.25, 0.0))],
    # another pixel's upper bound equals the constant pixel's value
    "touch": [(0.0, (0.0, 0.0)), (-0.25, (0.25, 0.0))],
}

# name: ((C, H, W), MaxPool2d keywords, {window index: pattern}). The window
# index is the output entry in (C, H_out, W_out) order; the contested windows
# of one configuration share no pixel.
_MP_EXACT = {
    "k2-s1x2": ((2, 3, 5), dict(kernel_size=2, stride=(1, 2)),
                {1: "three", 4: "two", 7: "const"}),
    "k3-s3-p1x0": ((2, 4, 3), dict(kernel_size=3, stride=3, padding=(1, 0)),
                   {0: "two", 3: "const"}),
    "k1x2-s1x2": ((2, 3, 5), dict(kernel_size=(1, 2), stride=(1, 2)),
                  {0: "two", 3: "touch", 8: "const", 11: "two"}),
    "k3x2-s2x1-p1x0-ceil": ((2, 3, 4), dict(kernel_size=(3, 2),
                                            stride=(2, 1), padding=(1, 0),
                                            ceil_mode=True),
                            {2: "three", 9: "two"}),
}


def _mp_level_pixels(chw, groups, contest, seed):
    """``(centre, gen)`` of every pixel in (C, H, W) order: pixel p is
    ``centre[p] + gen[p] . a`` over the polygon. Centres are distinct integers
    and ``|gen|_1 <= 1/8``, so pixels of different integer centres never
    overlap and a window has one certain largest pixel, except the windows of
    ``contest``, whose top pixels get the pattern's offsets and generators."""
    n = int(np.prod(chw))
    rng = np.random.default_rng(seed)
    centre = rng.permutation(n).astype(np.float64) - n // 2
    gen = rng.choice([-0.0625, 0.0, 0.0625], size=(n, 2))
    used = set()
    for w, pattern in contest.items():
        ranked = sorted(groups[w], key=lambda p: -centre[p])
        top = centre[ranked[0]]
        chosen = ranked[:len(_MP_PATTERNS[pattern])]
        if used & set(chosen) or len(chosen) < len(_MP_PATTERNS[pattern]):
            pytest.fail(f"test setup: contested window {w} shares a pixel "
                        f"with another or is too small")
        used |= set(chosen)
        for p, (offset, g) in zip(chosen, _MP_PATTERNS[pattern]):
            centre[p] = top + offset
            gen[p] = g
    return centre, gen


def _mp_poly_points(extra_rows=(), extra_b=()):
    """Member points of the polygon (and of the extra rows): its vertices and
    the points of the half grid inside it, as rational pairs."""
    A = rational.matrix(_MP_POLY_A + list(extra_rows))
    b = rational.vector(_MP_POLY_B + list(extra_b))
    pts = set(rational.vertices(A, b))
    half = [Fraction(k, 2) for k in range(-2, 3)]
    for a in ((x, y) for x in half for y in half):
        if all(rational.dot(r, a) <= rhs for r, rhs in zip(A, b)):
            pts.add(a)
    return sorted(pts)


def _mp_low_star(chw, centre, gen, extra_rows=(), extra_b=(), kind="ImageStar",
                 bounds="finite"):
    """The image ``centre + gen . a`` over the polygon (cut by
    ``extra_rows``) as an ImageStar in stored (H, W, C) order, or as an
    ImageZono over the square [-1, 1]^2 (a zonotope has no rows).
    ``bounds``: "finite" (the box [-1, 1]^2), "none" (rows only, the
    constructor's default) or "infinite" (-inf and +inf)."""
    C, H, W = chw
    order = hwc_index(C, H, W)
    c, G = centre[order], gen[order]
    if kind == "ImageZono":
        return ImageZono(c.reshape(-1, 1), G.copy(), H, W, C)
    A = np.array([[float(v) for v in r] for r in _MP_POLY_A + list(extra_rows)])
    b = np.array([float(v) for v in _MP_POLY_B + list(extra_b)]).reshape(-1, 1)
    lb, ub = {"finite": (-np.ones((2, 1)), np.ones((2, 1))),
              "none": (None, None),
              "infinite": (np.full((2, 1), -np.inf),
                           np.full((2, 1), np.inf))}[bounds]
    V = np.hstack([c[:, None], G])
    return ImageStar(V, A, b, lb, ub, H, W, C)


def _mp_live_groups(A, b, centre, gen, groups):
    """``groups`` without the pixels that are never strictly larger than
    another pixel of their window over the polytope ``A a <= b``: pixel p is
    left out when its largest value over the polytope is at most the smallest
    value of some other pixel q of the window, decided exactly at the
    polytope's vertices. Then p <= q everywhere, so the window's maximum is
    the same function without p; only the oracle's work shrinks (pieces where
    p ties for the maximum have no interior and the same value)."""
    corners = rational.vertices(rational.matrix(A), rational.vector(b))
    gen_q = [rational.vector(row) for row in gen.tolist()]
    centre_q = rational.vector(centre.tolist())
    values = [[rational.dot(g, a) + c for a in corners]
              for g, c in zip(gen_q, centre_q)]
    live = []
    for group in groups:
        keep = [p for p in group
                if not any(q != p and max(values[p]) <= min(values[q])
                           for q in group)]
        live.append(keep or list(group))
    return live


def _mp_pool_pieces(A, b, centre, gen, groups):
    """The exact pieces of ``a -> pool(centre + gen a)`` over ``A a <= b``,
    outputs in the network's (C, H_out, W_out) order."""
    pieces = rational.polytope(A, b)
    pieces = rational.affine(pieces, gen.tolist(), centre.tolist())
    return rational.maxpool(pieces, _mp_live_groups(A, b, centre, gen,
                                                    groups))


def _mp_exact_pieces(chw, centre, gen, groups, out_chw, square=False):
    """The exact image of the polygon (the square for a zonotope) under the
    pool, as rational pieces whose outputs are in stored (H, W, C) order."""
    A = [[1, 0], [-1, 0], [0, 1], [0, -1]] if square else _MP_POLY_A
    b = [1, 1, 1, 1] if square else _MP_POLY_B
    pieces = _mp_pool_pieces(A, b, centre, gen, groups)
    order = hwc_index(*out_chw)
    m = len(order)
    perm = [[int(order[k] == j) for j in range(m)] for k in range(m)]
    return rational.affine(pieces, perm)


def _mp_pred_rows(computed):
    """The rows ``A a <= b`` of a computed set's predicate polytope: its
    constraint rows and its finite predicate bounds, in rationals."""
    _, _, C, d, lb, ub = computed
    n = len(lb)
    A, b = [list(r) for r in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return A, b


def _mp_full_dim(computed):
    """Does a computed set's predicate polytope have an interior?"""
    A, b = _mp_pred_rows(computed)
    return rational.Piece(A, b, [], []).is_full_dimensional()


def _mp_unreached(pieces, computed):
    """Vertices of a computed set that are no output of the pool: each vertex
    ``c + G a`` of the set (``a`` a vertex of its predicate polytope) is
    first certified cheaply, when ``a`` is a point of the input polytope and
    the exact pieces evaluated at ``a`` give that vertex (the set then holds
    the pool's own output of that input point); a vertex not certified so is
    decided by ``rational.unreached`` against every piece."""
    c, G = computed[0], computed[1]
    A, b = _mp_pred_rows(computed)
    if not computed[4]:
        corners = [()] if all(r >= 0 for r in b) else []
    else:
        corners = rational.vertices(A, b)
    n_in = len(pieces[0].A[0])
    rest = []
    for a in corners:
        y = tuple(rational.dot(row, a) + ci for row, ci in zip(G, c))
        if len(a) == n_in:
            try:
                if tuple(rational.evaluate(pieces, a)) == y:
                    continue
            except ValueError:                  # a is outside every piece
                pass
        rest.append(y)
    return rational.unreached(pieces, rest)


def _mp_contains_pieces(pieces, computed, what):
    """Every exact piece has all its vertices in one computed set (so the
    union of the computed sets holds the exact image), with zero slack."""
    tests = [rational.Membership(s) for s in computed]
    for i, piece in enumerate(pieces):
        image = piece.image()
        if not any(all(t(y) for y in image) for t in tests):
            missing = [y for y in image if not any(t(y) for t in tests)]
            raise AssertionError(
                f"{what}: exact piece {i} of {len(pieces)} is not inside one "
                f"returned set; {len(missing)} of its {len(image)} vertices "
                f"are in none of the {len(computed)} sets (first "
                f"{[float(v) for v in missing[0]] if missing else None})")


def _mp_check_n_var(out, n_var, what):
    """The exact method adds no predicate variable. A returned set with more
    than the input's ``n_var`` fails here, before any vertex enumeration,
    whose cost grows fast with the number of variables (a bug that answers
    with a relaxation would otherwise make the case run for hours)."""
    for k, s in enumerate(out):
        if s.nVar > n_var:
            raise AssertionError(
                f"{what}: returned piece {k} has {s.nVar} predicate "
                f"variables, the input {n_var}; the exact method adds none")


def _mp_check_exact(out, pieces, what, n_var=2):
    """Both directions of an exact result, and its piece count."""
    _mp_check_n_var(out, n_var, what)
    computed = [rational.star(s) for s in out]
    _mp_contains_pieces(pieces, computed, what)
    for k, s in enumerate(computed):
        bad = _mp_unreached(pieces, s)
        assert not bad, (f"{what}: returned piece {k} has a vertex that is no "
                         f"output of the pool: {[float(v) for v in bad[0][1]]}")
    full_out = sum(_mp_full_dim(s) for s in computed)
    full_exact = sum(p.is_full_dimensional() for p in pieces)
    assert full_out == full_exact, (
        f"{what}: {full_out} returned pieces have an interior, the argmax "
        f"regions of the windows are {full_exact}")


def _mp_low_case(name, seed=3):
    chw, kw, contest = _MP_EXACT[name]
    layer = _mp_layer(kw)
    groups = _mp_groups(layer, chw)
    centre, gen = _mp_level_pixels(chw, groups, contest, seed)
    return chw, layer, groups, centre, gen


def _mp_replay_points(chw, centre, gen, points):
    return [(centre + gen @ np.array([float(v) for v in a])).reshape(chw)
            for a in points]


@pytest.mark.parametrize("split", [False, True], ids=["one-star", "two-halves"])
@pytest.mark.parametrize("name", sorted(_MP_EXACT))
def test_maxpool_exact_is_the_exact_image(name, split):
    """Exact MaxPool2d on an ImageStar of two predicate variables with a
    binding row, alone or as the two halves a1 <= 0 and a1 >= 0 of an earlier
    split: every vertex of the exact image (rationals) is in one returned
    piece, every vertex of every returned piece is an output of the pool, and
    there are as many pieces with an interior as argmax regions. Catches a
    dropped or duplicated piece, a split row with the wrong right-hand side,
    a certain window copied from the wrong pixel (row and column swapped,
    stride or padding of the other axis), a constant pixel dropped from the
    candidates, and a second input star not answered."""
    chw, layer, groups, centre, gen = _mp_low_case(name)
    out_chw = _mp_out_chw(layer, chw)
    pieces = _mp_exact_pieces(chw, centre, gen, groups, out_chw)
    if split:
        halves = [([[1, 0]], [0]), ([[-1, 0]], [0])]
        sets = [_mp_low_star(chw, centre, gen, r, b) for r, b in halves]
        pieces = rational.split(pieces, [1, 0], 0)
    else:
        sets = [_mp_low_star(chw, centre, gen)]
    what = f"exact MaxPool2d {name} ({'two halves' if split else 'one star'})"
    out = reach_one_layer(layer, sets, "exact", what)
    _mp_check_exact(out, pieces, what)
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=out_chw)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("name", sorted(_MP_EXACT))
def test_maxpool_approx_contains_the_exact_image(name, kind):
    """Approximate MaxPool2d (ImageStar) and the bound rule (ImageZono) on a
    two-variable set: every vertex of every exact piece (rationals) is a
    member of the returned set, with zero slack. The ImageStar result keeps
    the coupling of the new variable to the window's pixels, so a relaxation
    row with a wrong coefficient or sign shows here even when every entry's
    range is right. Catches the constant-pixel and touching-bound regimes of
    the candidate choice."""
    chw, layer, groups, centre, gen = _mp_low_case(name)
    out_chw = _mp_out_chw(layer, chw)
    square = kind == "ImageZono"
    pieces = _mp_exact_pieces(chw, centre, gen, groups, out_chw, square=square)
    what = f"approx MaxPool2d {name} on a two-variable {kind}"
    out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen, kind=kind)],
                          "approx", what)
    _mp_contains_pieces(pieces, [rational.star(s) for s in out], what)


_MP_ROUNDING = known_bug(
    7, "approx MaxPool2d on an ImageStar computes a contested window's new "
       "variable range and coupling rows by round-to-nearest (centre "
       "(lb + ub) / 2, half-width (ub - lb) / 2, right-hand sides centre "
       "minus pixel centre) with no outward term: on non-dyadic data a true "
       "output needs the variable 4.4e-16 below its lower bound and violates "
       "a coupling row by 5.2e-18 (ROUND-21; "
       "a rounding of the kernel's relaxation, not of "
       "the affine map that PLAN step 7b narrows)")


def _mp_non_dyadic(name, seed=3):
    """The two-variable case of ``name`` with every centre shifted by a
    non-dyadic hundredth-multiple and every generator scaled by a
    non-dyadic factor, so float64 rounds in the kernel's arithmetic."""
    chw, kw, contest = _MP_EXACT[name]
    layer = _mp_layer(kw)
    groups = _mp_groups(layer, chw)
    centre, gen = _mp_level_pixels(chw, groups, contest, 3 + seed)
    rng = np.random.default_rng(seed)
    centre = centre + rng.integers(1, 10, size=centre.shape) / 1000.0
    gen = gen * (1 + rng.integers(1, 10, size=gen.shape) / 30.0)
    return chw, layer, groups, centre, gen


@pytest.mark.parametrize("name,kind", [
    pytest.param("k1x2-s1x2", "ImageStar", marks=_MP_ROUNDING),
    ("k1x2-s1x2", "ImageZono"), ("k2-s1x2", "ImageZono")])
def test_maxpool_approx_contains_the_exact_image_of_non_dyadic_data(name,
                                                                    kind):
    """The cell of ``test_maxpool_approx_contains_the_exact_image`` on data
    that float64 cannot hold exactly after the kernel's arithmetic: every
    vertex of the exact image (rationals of the stored float64 inputs) is a
    member of the result with zero slack. The ImageZono kernel pools its
    bounds in float64 and widens them outward (passes); catches the bounds
    pooled in float32 or rounded inward, and the output's centre and radius
    taken by round-to-nearest (two configurations: the rounding does not
    depend on the windows). The ImageStar kernel rounds to nearest
    (marked)."""
    chw, layer, groups, centre, gen = _mp_non_dyadic(name)
    out_chw = _mp_out_chw(layer, chw)
    pieces = _mp_exact_pieces(chw, centre, gen, groups, out_chw,
                              square=kind == "ImageZono")
    what = f"approx MaxPool2d {name} on a non-dyadic two-variable {kind}"
    out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen, kind=kind)],
                          "approx", what)
    _mp_contains_pieces(pieces, [rational.star(s) for s in out], what)


def _mp_union_contains(pieces, out, what, depth=24):
    """The union of the returned sets holds the exact image, with zero slack,
    also where an exact piece is spread over several returned sets (the
    one-set check ``_mp_contains_pieces`` proves nothing there). A piece
    whose vertices are not all in one returned set is cut, exactly, along a
    row of a returned set's predicate polytope that separates its corners
    (rows in the input's predicate variables, which the exact method keeps),
    then bisected through its centroid along its widest axis, until every
    part has all its vertices in one returned set. A vertex, or the centroid
    of a part that no row cuts, in no returned set is the true output of a
    member point outside the result."""
    computed = [rational.star(s) for s in out]
    tests = [rational.Membership(s) for s in computed]
    n_in = len(pieces[0].A[0])
    cuts = [(row, rhs) for s in computed
            for row, rhs in zip(*_mp_pred_rows(s)) if len(row) == n_in]
    for i, piece in enumerate(pieces):
        stack = [(piece, 0)]
        while stack:
            part, level = stack.pop()
            image = part.image()
            if any(all(t(y) for y in image) for t in tests):
                continue
            missing = [y for y in image if not any(t(y) for t in tests)]
            assert not missing, (
                f"{what}: a vertex of exact piece {i} of {len(pieces)} is in "
                f"none of the {len(out)} returned sets: "
                f"{[float(v) for v in missing[0]]}")
            corners = part.corners()
            cut = next(((row, rhs) for row, rhs in cuts
                        if min(rational.dot(row, v) for v in corners) < rhs
                        < max(rational.dot(row, v) for v in corners)), None)
            middle = [sum(v[j] for v in corners) / len(corners)
                      for j in range(n_in)]
            if cut is None:
                y = part.value(middle)
                assert any(t(y) for t in tests), (
                    f"{what}: the true output at the member point "
                    f"{[float(v) for v in middle]} (exact piece {i}) is in "
                    f"none of the {len(out)} returned sets")
                assert level < depth, (
                    f"{what}: exact piece {i} is not shown to lie in the "
                    f"union of the returned sets after {depth} bisections")
                axis = max(range(n_in), key=lambda j: max(v[j] for v in corners)
                           - min(v[j] for v in corners))
                unit = [int(j == axis) for j in range(n_in)]
                cut = (unit, middle[axis])
                level += 1
            stack.extend((child, level)
                         for child in rational.split([part], *cut))


_MP_EXACT_ROUNDING = known_bug(
    7, "exact MaxPool2d on an ImageStar writes the split rows of a contested "
       "window ('candidate pixel >= other candidate') from the difference of "
       "the two pixels' generators and centres computed by round-to-nearest, "
       "with no outward term: on non-dyadic data the returned pieces leave "
       "out a sliver of member points whose true outputs are in no returned "
       "piece (a row violated by 2.6e-18 at the same predicate point), and "
       "returned vertices lie off the exact image (ROUND-15; the exact-method "
       "counterpart of ROUND-21)")


@pytest.mark.parametrize("name,seed", [
    ("k1x2-s1x2", 4), pytest.param("k1x2-s1x2", 3, marks=_MP_EXACT_ROUNDING)])
def test_maxpool_exact_holds_the_exact_image_of_non_dyadic_data(name, seed):
    """Exact MaxPool2d on the two-variable ImageStar of
    ``test_maxpool_approx_contains_the_exact_image_of_non_dyadic_data``
    (every centre and generator non-dyadic): the union of the returned
    pieces holds the exact image in rationals of the stored float64 inputs,
    with zero slack, decided where a piece straddles returned sets by
    cutting it along their rows, and every vertex of every returned piece is
    an output of the pool. Catches split rows rounded inward or written with
    a dropped term, which leave a sliver between pieces that the dyadic
    cells cannot see. Seed 4 passes today; seed 3 is marked."""
    chw, layer, groups, centre, gen = _mp_non_dyadic(name, seed)
    out_chw = _mp_out_chw(layer, chw)
    pieces = _mp_exact_pieces(chw, centre, gen, groups, out_chw)
    what = f"exact MaxPool2d {name} on non-dyadic data (seed {seed})"
    out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen)], "exact",
                          what)
    _mp_check_n_var(out, 2, what)
    _mp_union_contains(pieces, out, what)
    for k, s in enumerate(out):
        bad = _mp_unreached(pieces, rational.star(s))
        assert not bad, (f"{what}: returned piece {k} has a vertex that is no "
                         f"output of the pool: {[float(v) for v in bad[0][1]]}")


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_maxpool_with_verbose_logging_is_sound(method, caplog):
    """The verbose option only logs; the result is still sound and the split
    (exact) or the new variables (approx) are reported at debug level."""
    chw, layer, groups, centre, gen = _mp_low_case("k3-s3-p1x0")
    what = f"{method} MaxPool2d with verbose=True"
    with caplog.at_level("DEBUG", logger="n2v.nn.layer_ops.maxpool2d_reach"):
        out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen)], method,
                              what, verbose=True)
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))
    assert caplog.records, f"{what}: nothing was logged"


# --------------------------------------------------------------------------- #
# Composition: affine, MaxPool2d, affine through the model entry points        #
# --------------------------------------------------------------------------- #
_MP_COMP_CHW = (2, 3, 4)
_MP_COMP_POOL = dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0))
_MP_COMP_SCALE = np.array([2.0, 0.5])         # per-channel 1x1 convolution
_MP_COMP_BIAS = np.array([0.25, -0.5])


_MP_COMP_MIX = np.array([[[[1.0, -0.5]], [[0.25, 0.5]]],      # 1x2 kernels
                         [[[-0.5, 0.25]], [[1.0, 0.75]]]])     # mixing channels


def _mp_comp_head(n_in, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.integers(-4, 5, size=(3, n_in)) / 4.0,
            rng.integers(-4, 5, size=3) / 8.0)


def _mp_comp_onnx(tmp_path, mixing):
    """Conv (1x1 per channel, or a mixing 1x2 kernel) > MaxPool > Flatten >
    Gemm, dyadic weights, saved as ONNX. Returns (path, Gemm weight, bias)."""
    C, H, W = _MP_COMP_CHW
    if mixing:
        w = _MP_COMP_MIX.astype(np.float32)
        conv_out = (2, H, W - 1)
    else:
        w = np.zeros((2, 2, 1, 1), dtype=np.float32)
        w[0, 0, 0, 0], w[1, 1, 0, 0] = _MP_COMP_SCALE
        conv_out = (2, H, W)
    pool = _mp_layer(_MP_COMP_POOL)
    pooled = _mp_out_chw(pool, conv_out)
    G, h = _mp_comp_head(int(np.prod(pooled)))
    kh, kw = _MP_COMP_POOL["kernel_size"]
    nodes = [helper.make_node("Conv", ["X", "w", "b"], ["c"],
                              kernel_shape=list(w.shape[2:])),
             helper.make_node("MaxPool", ["c"], ["p"], kernel_shape=[kh, kw],
                              strides=list(_MP_COMP_POOL["stride"]),
                              pads=[1, 0, 1, 0]),
             helper.make_node("Flatten", ["p"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "g", "h"], ["Y"], transB=1)]
    inits = {"w": w, "b": _MP_COMP_BIAS.astype(np.float32),
             "g": G.astype(np.float32), "h": h.astype(np.float32)}
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, 3], inits,
                     name="conv_maxpool_gemm")
    return path, G, h


@pytest.mark.parametrize("kind,entry", [
    pytest.param("ImageStar", "onnx", id="ImageStar"),
    pytest.param("ImageZono", "onnx", id="ImageZono"),
    pytest.param("ImageStar", "torch", id="ImageStar-torch"),
    pytest.param("ImageZono", "torch", id="ImageZono-torch")])
def test_maxpool_in_an_onnx_chain_approx(tmp_path, kind, entry):
    """Conv (mixing channels) > MaxPool (kernel 3x2, stride 2x1, pads on the
    height only) > Flatten > Gemm, loaded from ONNX (list-valued kernel and
    strides), or the same network as a torch module (tuple-valued): every
    onnxruntime or torch output of a sample point is in the result. Catches
    rows the pool emits that the next layers read wrongly, and the layout
    change at Flatten."""
    path, G, h = _mp_comp_onnx(tmp_path, mixing=True)
    lo, hi = pixel_box(*_MP_COMP_CHW, seed=2)
    what = f"Conv > MaxPool > Flatten > Gemm, approx, {kind}, {entry}"
    points = samples(lo, hi)
    if entry == "onnx":
        out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                          what)
        outputs = replay_onnx(path, points, [1] + list(_MP_COMP_CHW))
    else:
        conv = nn.Conv2d(2, 2, (1, 2)).double()
        head = nn.Linear(G.shape[1], 3).double()
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(_MP_COMP_MIX))
            conv.bias.copy_(torch.from_numpy(_MP_COMP_BIAS))
            head.weight.copy_(torch.from_numpy(G))
            head.bias.copy_(torch.from_numpy(h))
        model = nn.Sequential(conv, _mp_layer(_MP_COMP_POOL), nn.Flatten(),
                              head).eval()
        out = reach_model(model, image_set(kind, lo, hi), "approx", what)
        outputs = replay_torch(model, points, (1,) + _MP_COMP_CHW)
    assert_covered(out, outputs, what)


@pytest.mark.parametrize("entry", ["onnx", "torch"])
def test_maxpool_in_a_chain_exact_is_the_exact_image(tmp_path, entry):
    """Conv (a positive scale per channel) > MaxPool > Flatten > Gemm on a
    two-variable ImageStar, by the exact method through the model entry (an
    ONNX file, whose MaxPool carries list-valued kernel and strides, and the
    torch module): the exact image in rationals is covered piece by piece,
    no returned vertex is outside it, and onnxruntime or torch outputs are
    members. Catches a wrong split row consumed by the Gemm, and the layout
    change at Flatten after an exact split."""
    chw = _MP_COMP_CHW
    pool = _mp_layer(_MP_COMP_POOL)
    groups = _mp_groups(pool, chw)
    centre, gen = _mp_level_pixels(chw, groups, {1: "three", 9: "two"}, 4)
    path, G, h = _mp_comp_onnx(tmp_path, mixing=False)
    if entry == "onnx":
        model = load_onnx(path)
    else:
        conv = nn.Conv2d(2, 2, 1).double()
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0, 0, 0], conv.weight[1, 1, 0, 0] = _MP_COMP_SCALE
            conv.bias.copy_(torch.from_numpy(_MP_COMP_BIAS))
        head = nn.Linear(G.shape[1], 3).double()
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(G))
            head.bias.copy_(torch.from_numpy(h))
        model = nn.Sequential(conv, pool, nn.Flatten(), head).eval()
    n = int(np.prod(chw))
    scale = np.repeat(_MP_COMP_SCALE, n // 2)
    shift = np.repeat(_MP_COMP_BIAS, n // 2)
    pieces = _mp_pool_pieces(_MP_POLY_A, _MP_POLY_B, scale * centre + shift,
                             scale[:, None] * gen, groups)
    pieces = rational.affine(pieces, G.tolist(), h.tolist())
    what = f"Conv > MaxPool > Flatten > Gemm, exact, {entry}"
    out = reach_model(model, _mp_low_star(chw, centre, gen), "exact", what)
    _mp_check_exact(out, pieces, what)
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    outputs = (replay_onnx(path, points, [1] + list(chw)) if entry == "onnx"
               else replay_torch(model, points, (1,) + chw))
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# Wide values, more predicate variables than entries, rows that tie pixels     #
# --------------------------------------------------------------------------- #
_MP_WIDE_CHW = (2, 4, 6)


def _mp_wide_star(chw=_MP_WIDE_CHW, n_extra=4, n=N_SAMPLES, seed=21):
    """``(image_star, points)`` for ``nn.MaxPool2d(2)`` (2x2 windows, stride
    2): pixel p is ``centre[p] + 1.5 a_p + 0.5 b_q`` with its own variable
    a_p and one of ``n_extra`` shared variables b_q (signs random), so the
    set has more predicate variables than entries and every pixel interval
    is 4 wide. Centres are distinct multiples of 8 (up to about 4 times the
    entry count), so a window has one certain largest pixel, except the
    first window of each channel, whose two top pixels get the same centre
    (contested, two candidates). Two rows tie pixels together: a_i + a_j <=
    0 for the two candidates i, j of channel 0's contested window (the
    candidate kept and the one dropped), and a_k - a_i <= 1/2 for channel
    1's candidate k. ``points`` are ``n`` member points as (C, H, W) arrays:
    predicate corners, half-grid points and points on the two rows' faces."""
    C, H, W = chw
    m = C * H * W
    k = m + n_extra
    rng = np.random.default_rng(seed)
    centre = 8.0 * rng.permutation(m) - 4.0 * m
    gen = np.zeros((m, k))
    gen[np.arange(m), np.arange(m)] = 1.5 * rng.choice([-1.0, 1.0], size=m)
    gen[np.arange(m), m + rng.integers(0, n_extra, size=m)] = (
        0.5 * rng.choice([-1.0, 1.0], size=m))
    pairs = []
    for c in range(C):
        window = [c * H * W + h * W + w for h in (0, 1) for w in (0, 1)]
        top, second = sorted(window, key=lambda p: -centre[p])[:2]
        centre[second] = centre[top]
        pairs.append((top, second))
    (i, j), (kk, _) = pairs
    rows = np.zeros((2, k))
    rows[0, i], rows[0, j] = 1.0, 1.0
    rows[1, kk], rows[1, i] = 1.0, -1.0
    A = np.vstack([np.eye(k), -np.eye(k), rows])
    b = np.concatenate([np.ones(2 * k), [0.0, 0.5]])
    points = []
    for attempt in range(200 * n):
        if len(points) >= n:
            break
        a = (rng.choice([-1.0, 1.0], size=k) if attempt % 4 == 0
             else rng.integers(-2, 3, size=k) / 2.0)
        if attempt % 4 == 1:
            a[j] = -a[i]                                   # face of row 1
        elif attempt % 4 == 2:
            a[kk] = min(1.0, a[i] + 0.5)                   # face of row 2
        if np.all(A @ a <= b):
            points.append((centre + gen @ a).reshape(chw))
    if len(points) < n:
        pytest.fail("test setup: could not draw enough member points")
    order = hwc_index(C, H, W)
    star = ImageStar(np.hstack([centre[order][:, None], gen[order]]), A,
                     b.reshape(-1, 1), -np.ones((k, 1)), np.ones((k, 1)),
                     H, W, C)
    return star, points


# The exact case uses a smaller image (four windows), so a bug that contests
# every window splits into at most 4**4 pieces instead of 4**12.
@pytest.mark.parametrize("method,chw", [("approx", _MP_WIDE_CHW),
                                        ("exact", (2, 2, 4))],
                         ids=["approx", "exact"])
def test_maxpool_default_spelling_on_a_wide_coupled_image_star(method, chw):
    """``nn.MaxPool2d(2)`` (kernel and stride given as one int, the most
    common spelling) through the model entry, on the ImageStar of
    ``_mp_wide_star``: more predicate variables than entries (52 for 48;
    20 for 16 in the exact case), pixel intervals 4 wide and values up to
    about 190, a contested window in each channel tied to other pixels by
    rows. Every torch output of a member
    point is in a returned set, by LP on the rows. Catches a stride given as
    one int read for one axis only, a new variable whose bounds are the unit
    interval while the window's range is wider than 2, a coupling or split
    row built from the pixel of another channel, and a relaxation or input
    variable read in place of another, which sets whose values all lie in
    [-1, 1] cannot show (the B1 lesson on scale)."""
    star, points = _mp_wide_star(chw)
    layer = nn.MaxPool2d(2)
    what = f"{method} MaxPool2d(2) on a wide coupled ImageStar {chw}"
    out = reach_model(nn.Sequential(layer), star, method, what)
    if method == "exact":
        _mp_check_n_var(out, star.nVar, what)
    assert_covered(out, replay_torch(layer, points, (1,) + tuple(chw)), what,
                   out_chw=_mp_out_chw(layer, chw))


def _mp_wide_zono(chw, n_extra=4, n=N_SAMPLES, seed=22):
    """``(centre, gen, points)`` of a zonotope over [-1, 1]^k in the network's
    (C, H, W) order: pixel p is ``centre[p] + 1.5 a_p + 0.5 b_q`` with its own
    generator a_p and one of ``n_extra`` shared generators b_q (signs random),
    so there are more generators than entries (k = entries + ``n_extra``, the
    inner shape a ReLU relaxation leaves, not the one ``from_bounds`` gives)
    and every pixel interval is 4 wide. Centres are distinct multiples of 8
    around zero, up to about 4 times the entry count, half of them negative.
    ``points``: ``n`` member points as (C, H, W) arrays, alternately a random
    corner of [-1, 1]^k and a point of its half grid."""
    C, H, W = chw
    m = C * H * W
    k = m + n_extra
    rng = np.random.default_rng(seed)
    centre = 8.0 * rng.permutation(m) - 4.0 * m
    gen = np.zeros((m, k))
    gen[np.arange(m), np.arange(m)] = 1.5 * rng.choice([-1.0, 1.0], size=m)
    gen[np.arange(m), m + rng.integers(0, n_extra, size=m)] = (
        0.5 * rng.choice([-1.0, 1.0], size=m))
    points = []
    for t in range(n):
        a = (rng.choice([-1.0, 1.0], size=k) if t % 2 == 0
             else rng.integers(-2, 3, size=k) / 2.0)
        points.append((centre + gen @ a).reshape(chw))
    return centre, gen, points


# The ImageZono route: kernel 3, stride 2, padding 1 on a (2, 4, 6) image
# (overlapping windows, padding on both axes, the int spelling); with the seed
# of _mp_wide_zono two windows that touch the padding hold only pixels below 0.
_MP_WIDE_ZONO_POOL = dict(kernel_size=3, stride=2, padding=1)


@pytest.mark.parametrize("route", ["image-zono", "flat-zono-view"])
def test_maxpool_on_a_wide_zonotope_with_more_generators_than_entries(route):
    """MaxPool2d by the zonotope bound rule through the model entry on the
    zonotope of ``_mp_wide_zono``: more generators than entries (52 for 48,
    28 for 24), pixel intervals 4 wide, values up to about 190. ``image-zono``:
    an ImageZono into ``MaxPool2d(3, stride=2, padding=1)``, where two padded
    windows hold only pixels below 0; ``flat-zono-view``: a flat Zono in the
    tensor's (C, H, W) order that the graph reshapes (``view``) into the image
    feeding the pool, so the walker turns it into an ImageZono (the one route
    on which a flat Zono reaches MaxPool2d and is answered), then flatten and
    a Linear head. One set per input set, then every torch output of a member
    point is in the result, by LP on the generators. Catches each entry's
    radius taken from as many generators as entries (the shape
    ``from_bounds`` gives), an output zonotope with unit generators in place
    of the pooled half-widths, which data whose intervals are narrower than 2
    cannot show (the B1 lesson on scale), padding taken as zero where
    MaxPool pads with -inf, and the flat (C, H, W) entries read as stored
    (H, W, C) ones at the conversion."""
    if route == "image-zono":
        chw = (2, 4, 6)
        centre, gen, points = _mp_wide_zono(chw)
        layer = _mp_layer(_MP_WIDE_ZONO_POOL)
        groups = _mp_groups(layer, chw)
        upper = centre + np.abs(gen).sum(axis=1)
        padded = [g for g in groups
                  if len(g) < 9 and all(upper[p] < 0 for p in g)]
        if not padded:
            pytest.fail("test setup: no padded window holds only pixels "
                        "below 0")
        C, H, W = chw
        order = hwc_index(C, H, W)
        model = nn.Sequential(layer)
        input_set = ImageZono(centre[order].reshape(-1, 1), gen[order], H, W, C)
        out_chw = _mp_out_chw(layer, chw)
    else:
        chw = (2, 3, 4)
        centre, gen, points = _mp_wide_zono(chw)
        head = nn.Linear(2 * 2 * 2, 3).double()
        rng = np.random.default_rng(9)
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, 8))
                                               / 4.0))
            head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
        model = _MpViewPool(chw, head).eval()
        input_set = Zono(centre.reshape(-1, 1), gen)
        out_chw = None
    what = f"approx MaxPool2d on a wide zonotope, {route}"
    out = reach_model(model, input_set, "approx", what)
    assert len(out) == 1, (f"{what}: {len(out)} sets for one input set; the "
                           f"bound rule does not split")
    in_shape = (1,) + chw if out_chw else (1, int(np.prod(chw)))
    assert_covered(out, replay_torch(model, points, in_shape), what,
                   out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Refusals                                                                       #
# --------------------------------------------------------------------------- #
# The star kernels refuse what _MP_ZONO_ONLY holds, and a ceil_mode pool that
# adds a window along the width only (the height keeps its floor-mode count).
_MP_STAR_REFUSED = dict(_MP_ZONO_ONLY, **{
    "k2-s2-ceil-width": ((2, 4, 5), dict(kernel_size=2, stride=2,
                                         ceil_mode=True))})


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("name", sorted(_MP_STAR_REFUSED))
def test_maxpool_star_refuses_dilation_and_extra_ceil_windows(name, method):
    """The star kernels build undilated floor-mode windows only, so a dilated
    pool (on both axes or one) and a ceil_mode pool that adds windows (on
    both axes, or along the width only) must raise. The dilated shapes are
    such that the undilated pool has the layer's output shape, so answering
    would give a set of the right size over the wrong pixels. Catches a
    dilation check that needs both axes dilated, and a ceil_mode check that
    compares one axis of the output shape only."""
    chw, kw = _MP_STAR_REFUSED[name]
    lo, hi = pixel_box(*chw)
    with pytest.raises(NotImplementedError):
        reach_layer(_mp_layer(kw), [image_set("ImageStar", lo, hi)], method)


def test_maxpool_with_asymmetric_onnx_pads_is_refused(tmp_path):
    """An ONNX MaxPool padded on one side only loads as a pad with -inf in
    front of the pool; reach must refuse it, not pad with zeros."""
    C, H, W = 2, 3, 4
    path = save_onnx(tmp_path, [helper.make_node(
        "MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[1, 2],
        pads=[1, 0, 0, 0])], [1, C, H, W], [1, C, 3, 2], name="maxpool_asym")
    lo, hi = pixel_box(C, H, W)
    for kind in ("ImageStar", "ImageZono"):
        with pytest.raises(NotImplementedError):
            reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                        f"asymmetric MaxPool on {kind}")


@pytest.mark.parametrize("kind", ["Box", "Hexatope", "Octatope"])
def test_maxpool_on_a_flat_set_type_without_a_rule_is_refused(kind):
    """The dispatcher has no MaxPool2d rule for Box, Hexatope or Octatope; it
    must raise, never return the input or a placeholder."""
    lo, hi = pixel_box(2, 2, 3)
    for method in ("approx", "exact"):
        with pytest.raises(NotImplementedError):
            reach_layer(nn.MaxPool2d(2), [flat_set(kind, lo, hi)], method)


# One rule per lane: Box, Hexatope and Octatope have no MaxPool2d rule
# (NotImplementedError from the dispatcher), and the model entry has no exact
# method for a set outside the Star family (ValueError before the walk).
_MP_MODEL_REFUSED = [("Box", "approx"), ("Box", "exact"),
                     ("Hexatope", "approx"), ("Hexatope", "exact"),
                     ("Octatope", "approx"), ("Octatope", "exact"),
                     ("Zono", "exact"), ("ImageZono", "exact")]


@pytest.mark.parametrize("kind,method", _MP_MODEL_REFUSED,
                         ids=[f"{k}-{m}" for k, m in _MP_MODEL_REFUSED])
def test_maxpool_through_the_model_entry_is_refused_where_there_is_no_rule(
        kind, method):
    """MaxPool2d inside a model (pool > Flatten, a flat set given its
    (2, 2, 3) shape by ``input_shape``) through ``reach_pytorch_model``: a
    Box, Hexatope or Octatope reaches a lane with no MaxPool2d rule and must
    be refused with ``NotImplementedError``; ``method="exact"`` on any set
    outside the Star family must be refused too (today the model entry's
    deliberate ``ValueError`` before the walk; which of the two types is the
    settled one is decided in ``tests/contracts/test_reach_coverage.py``).
    Catches a lane that answers a pool it has no rule for (the input passed
    through, a placeholder) and an exact request on a zonotope, box or
    Hexatope answered with an over-approximation. The ``reach_layer``
    neighbours are ``test_maxpool_on_a_flat_set_type_without_a_rule_is_refused``
    and ``test_maxpool_exact_on_an_image_zono_through_reach_layer_is_refused``."""
    chw = (2, 2, 3)
    lo, hi = pixel_box(*chw)
    model = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())
    what = f"{method} MaxPool2d on a {kind} through the model entry"
    if kind == "ImageZono":
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set, options = flat_set(kind, lo, hi), {"input_shape": (1,) + chw}
    expected = (NotImplementedError if method == "approx"
                else (NotImplementedError, ValueError))
    with pytest.raises(expected):
        reach_model(model, input_set, method, what, **options)


@known_bug(
    18, "reach_layer with method='exact' on an ImageZono (or a Zono) runs the "
        "MaxPool2d bound rule and returns the pooled bounding box with no "
        "warning, which is not the image of the zonotope under the pool: the "
        "Zono lane of the dispatcher does not read the method (the MaxPool2d "
        "branch of REACH-8); the model entry "
        "refuses exact on a Zono before the walk",
    raises=DID_NOT_RAISE)
def test_maxpool_exact_on_an_image_zono_through_reach_layer_is_refused():
    """``method="exact"`` on a two-variable ImageZono through ``reach_layer``:
    the zonotope rule pools the bounds, so its result holds points that no
    input reaches (the upper corner of the returned box, decided in
    rationals), and the call must raise
    ``NotImplementedError``. Neighbour that passes today with the same
    assertion: ``test_maxpool_on_a_flat_set_type_without_a_rule_is_refused``
    (exact on a Box through ``reach_layer``)."""
    chw, layer, groups, centre, gen = _mp_low_case("k1x2-s1x2")
    with pytest.raises(NotImplementedError):
        reach_layer(layer, [_mp_low_star(chw, centre, gen, kind="ImageZono")],
                    "exact")


@known_bug(
    23, "MaxPool2d on a reach_layer list led by an ImageZono that also holds "
        "a set of another family crashes in maxpool2d_zono instead of "
        "refusing: it calls get_bounds on every set, so an ImageStar or a Box "
        "raises AttributeError and a Hexatope TypeError (REACH-6: "
        "the dispatcher picks the kernel by the first set)",
    raises=AttributeError)
def test_maxpool_on_a_list_led_by_an_image_zono_mixing_set_types_raises():
    """A list holding an ImageZono and then an ImageStar of the same image,
    through ``reach_layer`` (the only route that builds such a list): the
    call must refuse (``NotImplementedError``, or ``ValueError`` for an input
    it cannot read), never answer for the first set only, and never crash
    with another exception type. Today ``maxpool2d_zono`` reads the
    ImageStar's missing ``get_bounds`` (marked). Passing neighbour on the
    same lane with a list of ImageZonos:
    ``test_maxpool_approx_holds_every_torch_output[ImageZono-k1x2-s1x2-split]``."""
    lo, hi = pixel_box(2, 2, 3)
    sets = [image_set("ImageZono", lo, hi), image_set("ImageStar", lo, hi)]
    with pytest.raises((NotImplementedError, ValueError)):
        reach_layer(nn.MaxPool2d(2), sets, "approx")


_MP_FLAT_REFUSAL = known_bug(
    23, "MaxPool2d on a flat Star raises TypeError, not the declared refusal "
        "(NotImplementedError, later UnsupportedError): the kernels' type "
        "checks at maxpool2d_reach.py lines 199 (exact) and 367 (approx) "
        "(POOL-6)",
    raises=TypeError)
_MP_FLAT_ZONO_CRASH = known_bug(
    23, "MaxPool2d on a flat Zono crashes with AttributeError ('Zono' object "
        "has no attribute 'height', maxpool2d_reach.py line 389) instead of "
        "refusing, through the model entry with an input shape and through "
        "reach_layer by either method, since the Zono lane does not read the "
        "method (POOL-6)",
    raises=AttributeError)


# (the model entry refuses method="exact" for a Zono before any layer runs,
# with a ValueError, so that combination does not reach the pool; a flat Star
# handed to reach_layer reaches the same raise as through the model entry,
# and test_maxpool_on_a_list_mixing_set_types_raises marks that route; a flat
# Zono handed to reach_layer with approx reaches the raise of
# [Zono-approx-model], and [Zono-exact-layer] keeps the exact route, which
# the Zono lane answers instead of refusing because it does not read the
# method, so a fix of that alone shows there)
@pytest.mark.parametrize("kind,method,entry", [
    pytest.param("Star", "approx", "model", marks=_MP_FLAT_REFUSAL),
    pytest.param("Star", "exact", "model", marks=_MP_FLAT_REFUSAL),
    pytest.param("Zono", "exact", "layer", marks=_MP_FLAT_ZONO_CRASH),
    pytest.param("Zono", "approx", "model", marks=_MP_FLAT_ZONO_CRASH),
])
def test_maxpool_on_a_flat_star_or_zono_is_refused(kind, method, entry):
    """A flat Star or Zono carries no image shape, so MaxPool2d cannot place
    its windows: the call must refuse (NotImplementedError), through
    ``reach_layer`` and through the model entry with an input shape. The Box,
    Hexatope and Octatope neighbours of
    ``test_maxpool_on_a_flat_set_type_without_a_rule_is_refused`` pass with
    the same inputs."""
    chw = (2, 2, 3)
    lo, hi = pixel_box(*chw)
    s = flat_set(kind, lo, hi)
    with pytest.raises(NotImplementedError):
        if entry == "layer":
            reach_layer(nn.MaxPool2d(2), [s], method)
        else:
            reach_model(nn.Sequential(nn.MaxPool2d(2), nn.Flatten()), s,
                        method, f"MaxPool2d on a flat {kind}",
                        input_shape=(1,) + chw)


_MP_MIXED_LIST = known_bug(
    23, "MaxPool2d on a list holding an ImageStar and a flat Star raises the "
        "kernel's TypeError ('MaxPool2D expects ImageStar input') at the flat "
        "Star instead of the declared refusal (POOL-6)",
    raises=TypeError)


class _MpViewPool(nn.Module):
    """x.view(1, C, H, W) > MaxPool2d((2, 1), stride (1, 2)) > flatten >
    Linear: a flat input reshaped into an image right before the pool."""

    def __init__(self, chw, head):
        super().__init__()
        self.chw = tuple(chw)
        self.pool = nn.MaxPool2d((2, 1), stride=(1, 2))
        self.head = head

    def forward(self, x):
        return self.head(torch.flatten(self.pool(x.view(1, *self.chw)), 1))


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_maxpool_on_a_flat_star_reshaped_into_an_image_holds_every_output(
        method):
    """The one route on which a flat Star reaches MaxPool2d and is answered:
    the graph reshapes it (``view``) into a (2, 3, 4) image that feeds the
    pool, the walker turns it into an ImageStar, then the ImageStar kernel
    and a Linear head. Input: a Star with a binding row that is not a box
    row, values up to 16 and intervals up to 10 wide. Every torch output of
    a member point is in a returned set, by LP on the rows. Catches the
    flat (C, H, W) entries read as stored (H, W, C) ones at the conversion,
    and the pooled rows read wrongly by the head."""
    chw = (2, 3, 4)
    lo, hi = pixel_box(*chw, seed=3)
    star, points = constrained_star(8 * lo.reshape(-1), 8 * hi.reshape(-1))
    head = nn.Linear(2 * 2 * 2, 3).double()
    rng = np.random.default_rng(9)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, 8)) / 4.0))
        head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    model = _MpViewPool(chw, head).eval()
    what = f"view > MaxPool2d > Linear on a flat Star, {method}"
    out = reach_model(model, star, method, what)
    assert_covered(out, replay_torch(model, points, (1, int(np.prod(chw)))),
                   what)


class _MpTwice(nn.MaxPool2d):
    """A subclass of nn.MaxPool2d whose forward is twice the max pooling."""

    def forward(self, x):
        return 2 * super().forward(x)


_MP_SUBCLASS = known_bug(
    7, "reach_layer matches a subclass of nn.MaxPool2d by isinstance and "
       "pools with the stock rule, so a subclass whose forward is not the "
       "stock one (twice the max) is answered for another function: 128 of "
       "128 torch outputs outside, largest distance 2.25 (the MaxPool2d "
       "branch of REACH-9)")


@pytest.mark.parametrize("entry", [pytest.param("reach_layer",
                                                marks=_MP_SUBCLASS),
                                   "reach_pytorch_model"])
def test_maxpool_subclass_with_another_forward_refuses_or_holds_every_output(
        entry):
    """A subclass of nn.MaxPool2d with its own forward: the reach must
    refuse (NotImplementedError) or hold every torch output of the subclass.
    The model entry traces into the forward and refuses its functional
    max_pool2d (passes); a direct ``reach_layer`` call answers with the
    stock rule (marked)."""
    chw = (2, 3, 4)
    lo, hi = pixel_box(*chw, seed=3)
    layer = _MpTwice((2, 1), stride=(1, 2))
    what = f"MaxPool2d subclass through {entry}"
    try:
        if entry == "reach_layer":
            out = reach_one_layer(layer, [image_set("ImageStar", lo, hi)],
                                  "approx", what)
            out_chw = _mp_out_chw(layer, chw)
        else:
            out = reach_model(nn.Sequential(layer, nn.Flatten()),
                              image_set("ImageStar", lo, hi), "approx", what)
            out_chw = None
    except NotImplementedError:
        return
    model = layer if out_chw else nn.Sequential(layer, nn.Flatten())
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw), what,
                   out_chw=out_chw)


@known_bug(
    7, "reach_layer matches a subclass of nn.MaxPool2d by isinstance on the "
       "Zono lane too and pools an ImageZono's bounds with the stock rule, so "
       "a subclass whose forward is twice the max is answered for another "
       "function: 128 of 128 torch outputs outside, largest distance 2.25 "
       "(the Zono-lane MaxPool2d branch of REACH-9)")
def test_maxpool_subclass_on_an_image_zono_refuses_or_holds_every_output():
    """The subclass of ``test_maxpool_subclass_with_another_forward_refuses_or_holds_every_output``
    (forward twice the max) handed to ``reach_layer`` with an ImageZono: the
    reach must refuse (``NotImplementedError``) or hold every torch output of
    the subclass, by LP on the generators. The Zono lane has its own
    ``isinstance`` branch (dispatcher line 388), so it keeps its own marked
    case; the model entry refuses the subclass before any set type matters
    (``...[reach_pytorch_model]`` of that test passes)."""
    chw = (2, 3, 4)
    lo, hi = pixel_box(*chw, seed=3)
    layer = _MpTwice((2, 1), stride=(1, 2))
    what = "MaxPool2d subclass on an ImageZono through reach_layer"
    try:
        out = reach_one_layer(layer, [image_set("ImageZono", lo, hi)],
                              "approx", what)
    except NotImplementedError:
        return
    assert_covered(out, replay_torch(layer, samples(lo, hi), (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))


class _MpFunctional(nn.Module):
    def forward(self, x):
        return torch.flatten(torch.nn.functional.max_pool2d(x, (2, 1),
                                                            (1, 2)), 1)


class _MpUnpacked(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 1), stride=(1, 2), return_indices=True)

    def forward(self, x):
        values, _ = self.pool(x)
        return torch.flatten(values, 1)


# form: (model or layer, reached through reach_layer?)
_MP_OTHER_FORMS = {
    "AdaptiveMaxPool2d": (lambda: nn.Sequential(nn.AdaptiveMaxPool2d((1, 2)),
                                                nn.Flatten()), False),
    "functional": (_MpFunctional, False),
    "return_indices-unpacked": (_MpUnpacked, False),
    "return_indices-layer": (lambda: nn.MaxPool2d((2, 1), stride=(1, 2),
                                                  return_indices=True), True),
}


@pytest.mark.parametrize("form", list(_MP_OTHER_FORMS))
def test_other_max_pooling_forms_refuse_or_hold_every_output(form):
    """Max pooling that is not a plain nn.MaxPool2d call on an ImageStar
    (approx; the routes refuse or answer before the method matters):
    nn.AdaptiveMaxPool2d (no Star rule), F.max_pool2d in a traced model (a
    function node), nn.MaxPool2d with ``return_indices=True`` whose tuple
    the model unpacks (a getitem node), and the same layer handed to
    ``reach_layer`` (its first output, the values). Each must refuse
    (NotImplementedError) or hold every torch output (the values, for a
    layer that returns a tuple). Catches a form answered with another
    function's set: an adaptive pool read as a fixed-window pool, the
    indices tensor taken for the values, a function node skipped."""
    chw = (2, 3, 4)
    lo, hi = pixel_box(*chw, seed=3)
    make, is_layer = _MP_OTHER_FORMS[form]
    module = make()
    what = f"{form} on an ImageStar"
    try:
        if is_layer:
            out = reach_one_layer(module, [image_set("ImageStar", lo, hi)],
                                  "approx", what)
        else:
            out = reach_model(module, image_set("ImageStar", lo, hi),
                              "approx", what)
    except NotImplementedError:
        return
    assert_covered(out, replay_torch(module, samples(lo, hi), (1,) + chw),
                   what, out_chw=_mp_out_chw(module, chw) if is_layer else None)


@pytest.mark.parametrize("method", [pytest.param("approx", marks=_MP_MIXED_LIST),
                                    pytest.param("exact", marks=_MP_MIXED_LIST)])
def test_maxpool_on_a_list_mixing_set_types_raises(method):
    """A list holding an ImageStar and a flat Star: the call must refuse
    (NotImplementedError, or ValueError for an input it cannot read), never
    answer for the first set only, and never crash with another exception
    type. Today the kernels' own type checks raise TypeError (marked)."""
    lo, hi = pixel_box(2, 2, 3)
    sets = [image_set("ImageStar", lo, hi), flat_set("Star", lo, hi)]
    with pytest.raises((NotImplementedError, ValueError)):
        reach_layer(nn.MaxPool2d(2), sets, method)


# --------------------------------------------------------------------------- #
# An ImageStar without a finite predicate box (POOL-4)                          #
# --------------------------------------------------------------------------- #
_MP_NO_BOX = known_bug(
    7, "POOL-4: MaxPool2d on an ImageStar with constraint rows only (no "
       "predicate bounds) crashes with ValueError: _window_bounds_4d reshapes "
       "the state bounds that estimate_ranges() does not store on that "
       "branch, and behind it the approx kernel stacks the missing predicate "
       "bounds (None) with the new variables' bounds, so the ImageStar "
       "constructor raises 'pred_lb size != nVar' (registry POOL-8; the "
       "same marker absorbs both)", raises=ValueError)


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("bounds", [
    pytest.param("none", marks=_MP_NO_BOX), "finite"])
def test_maxpool_on_an_image_star_given_by_rows(bounds, method):
    """The same two-variable ImageStar, its polygon given by constraint rows,
    with no predicate bounds (the constructor's default) or with the finite
    box: every torch output must be in the result."""
    chw, layer, groups, centre, gen = _mp_low_case("k2-s1x2")
    what = f"{method} MaxPool2d on an ImageStar, predicate bounds {bounds}"
    out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen,
                                               bounds=bounds)], method, what)
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))


@known_bug(7, "POOL-4: approx MaxPool2d on an ImageStar whose predicate "
              "bounds are -inf and +inf (the polygon given by rows) takes the "
              "midpoint of infinite window bounds, inf - inf, and returns a "
              "set with NaN in its centre and constraint right-hand sides; "
              "numpy's RuntimeWarning on that sum is an error in the suite",
           raises=RuntimeWarning)
def test_maxpool_approx_on_an_image_star_with_infinite_predicate_bounds():
    """Predicate bounds of -inf and +inf, the polygon given by rows: the
    result must be a set of real numbers holding every torch output. The
    neighbour with finite bounds is
    ``test_maxpool_on_an_image_star_given_by_rows[finite-approx]``. Catches
    window bounds read from the predicate box instead of the rows."""
    chw, layer, groups, centre, gen = _mp_low_case("k2-s1x2")
    what = "approx MaxPool2d on an ImageStar with infinite predicate bounds"
    out = reach_one_layer(layer, [_mp_low_star(chw, centre, gen,
                                               bounds="infinite")],
                          "approx", what)
    for s in out:
        assert np.all(np.isfinite(s.V)) and not np.any(np.isnan(s.d)), (
            f"{what}: the result holds NaN or inf in its basis or rows")
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_mp_out_chw(layer, chw))


_MP_INF_PADDING = known_bug(
    7, "exact MaxPool2d on an ImageStar whose predicate bounds are -inf and "
       "+inf, with padding: every lower bound in a window is -inf, so "
       "_get_local_max_index_4d adds the window's first position, a padding "
       "position, to the max candidates, and _step_split_4d emits pieces "
       "whose output for that window is the padding's zero placeholder; a "
       "true output with a pixel below 0 in a padded window is in no "
       "returned piece (POOL-7; the exact-method path of POOL-4's input)")

# (C, H, W) = (2, 1, 4), kernel (1, 2), stride (1, 2); padded: one column of
# padding on each side, so the edge windows hold one pixel and one padding
# position.
_MP_INF_POOLS = {"unpadded": dict(kernel_size=(1, 2), stride=(1, 2)),
                 "padded": dict(kernel_size=(1, 2), stride=(1, 2),
                                padding=(0, 1))}


@pytest.mark.parametrize("bounds,pool", [
    ("finite", "unpadded"), ("infinite", "unpadded"), ("finite", "padded"),
    pytest.param("infinite", "padded", marks=_MP_INF_PADDING)])
def test_maxpool_exact_with_infinite_predicate_bounds_is_the_exact_image(
        bounds, pool):
    """Exact MaxPool2d through the model entry on a two-variable ImageStar
    whose polygon is given by rows, with predicate bounds -inf and +inf
    (every window's interval bounds are then infinite and it splits on all
    its positions) or the finite box: both directions of the exact image in
    rationals, the number of pieces with an interior, and torch outputs by
    LP. Catches window candidates taken from an infinite predicate box: a
    padding position taken as a candidate (marked today), a pixel left out,
    an infeasible split piece answered with an interior."""
    chw = (2, 1, 4)
    layer = _mp_layer(_MP_INF_POOLS[pool])
    groups = _mp_groups(layer, chw)
    centre, gen = _mp_level_pixels(chw, groups, {1: "two"}, seed=3)
    if pool == "padded" and not any(
            len(g) == 1 and centre[g[0]] < 0 for g in groups):
        pytest.fail("test setup: no padded window holds a pixel below 0")
    out_chw = _mp_out_chw(layer, chw)
    pieces = _mp_exact_pieces(chw, centre, gen, groups, out_chw)
    what = f"exact MaxPool2d {pool}, predicate bounds {bounds}"
    out = reach_model(nn.Sequential(layer),
                      _mp_low_star(chw, centre, gen, bounds=bounds), "exact",
                      what)
    _mp_check_exact(out, pieces, what)
    points = _mp_replay_points(chw, centre, gen, _mp_poly_points())
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Hexatope and Octatope kernels (reachable by a direct call only)              #
# --------------------------------------------------------------------------- #
_MP_HEXOCT_INPUT_BOX = known_bug(
    9, "HEXOCT-6: maxpool2d_hexatope and maxpool2d_octatope return the "
       "input's bounding box whatever the layer: outputs of a size-preserving "
       "pool escape, and a shrinking pool gets a set of the input's dimension "
       "(the same line for both; latent, no public entry calls the kernels)")

_MP_HEXOCT = {"Hexatope": maxpool2d_hexatope, "Octatope": maxpool2d_octatope}


@pytest.mark.parametrize("kind", sorted(_MP_HEXOCT))
@pytest.mark.parametrize("pool,parts", [
    ("k1", 1), ("k1", 3),
    pytest.param("k3-s1-p1", 1, marks=_MP_HEXOCT_INPUT_BOX),
])
def test_maxpool_hexatope_octatope_kernel_is_sound_or_refuses(kind, pool,
                                                              parts):
    """The Hexatope and Octatope MaxPool kernels, called directly on the
    flattened (C, H, W) image, one set or three sets covering the box: the
    result holds every torch output, or the call refuses
    (NotImplementedError). A 1x1 window is the identity, so the result must
    hold every member of every input set. Catches a bound taken from the
    wrong end or the midpoint, and only one of several sets answered."""
    chw = (2, 3, 4)
    layer = {"k1": nn.MaxPool2d(1),
             "k3-s1-p1": nn.MaxPool2d(3, stride=1, padding=1)}[pool]
    lo, hi = pixel_box(*chw, seed=1)
    sets = split_cover(kind, lo, hi, parts) if parts > 1 else [
        flat_set(kind, lo, hi)]
    try:
        out = _MP_HEXOCT[kind](layer, sets)
    except NotImplementedError:
        return
    what = f"{kind} MaxPool2d {pool}, {parts} set(s)"
    assert out, f"{what}: no sets returned"
    assert_covered(out, replay_torch(layer, samples(lo, hi), (1,) + chw), what)


# --------------------------------------------------------------------------- #
# After an exact split inside the model: several sets reach the layer          #
# --------------------------------------------------------------------------- #
def _mp_after_split_box(chw=(2, 2, 4), seed=12):
    """A pixel box with every pixel positive except three that straddle 0,
    so an exact ReLU splits it into at most eight stars. The (1, 2) window
    partner of each straddling pixel (flat indices 0, 7, 13 beside 1, 6, 12)
    is a small positive interval below the straddler's upper end, so after
    the ReLU either pixel can be its window's maximum: the eight stars have
    different pooled images, and a result of one star reused for another
    leaves outputs out."""
    lo, hi = pixel_box(*chw, seed=seed)
    lo, hi = lo + 3.0, hi + 3.0
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    for p, (a, b) in zip((1, 6, 12), ((-0.5, 0.75), (-0.25, 0.5),
                                      (-1.0, 0.25))):
        flat_lo[p], flat_hi[p] = a, b
    for p, (a, b) in zip((0, 7, 13), ((0.25, 0.5), (0.125, 0.375),
                                      (0.0625, 0.1875))):
        flat_lo[p], flat_hi[p] = a, b
    return flat_lo.reshape(chw), flat_hi.reshape(chw)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
def test_maxpool_after_an_exact_relu_split_in_a_model(kind, method):
    """ReLU > MaxPool2d((1, 2), stride (1, 2)) > Flatten > Linear through ``reach_pytorch_model``
    on a (2, 2, 4) box with three pixels straddling zero: by the exact method
    the ReLU hands up to eight stars to the layer in one call. Every torch
    output of a sample point is in a returned set, by LP on the rows.
    Catches only the first or the last of several sets answered inside the
    model, and a result of one set reused for another."""
    chw = (2, 2, 4)
    layer = nn.MaxPool2d((1, 2), stride=(1, 2))
    with torch.no_grad():
        n_mid = int(np.prod(layer(torch.zeros((1,) + chw)).shape[1:]))
    head = nn.Linear(n_mid, 3).double()
    rng = np.random.default_rng(13)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, n_mid))
                                           / 4.0))
        head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    model = nn.Sequential(nn.ReLU(), layer.double(), nn.Flatten(),
                          head).eval()
    lo, hi = _mp_after_split_box(chw)
    what = f"ReLU > MaxPool2d((1, 2), stride (1, 2)) > Flatten > Linear, {method}, {kind}"
    out = reach_model(model, image_set(kind, lo, hi), method, what)
    if method == "exact" and len(out) < 2:
        pytest.fail(f"test setup: the exact ReLU gave {len(out)} set(s)")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)
