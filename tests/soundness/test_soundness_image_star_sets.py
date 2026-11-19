"""Soundness cells for the methods of ``ImageStar`` that reach and
verification rely on.

Covered: the constructor (a 2-D basis in height, width, channel order, a 4-D
basis, a 1-D right-hand side, and the inputs it refuses) and ``from_bounds``
(with an infinite end too); the conversions ``to_star`` (height, width,
channel order) and ``flatten_to_star`` (channel, height, width order, with
the state bounds it passes on); the range queries ``estimate_range``,
``estimate_ranges`` (with a predicate box, without one, and with lower
predicate bounds only, and the state bounds they store), ``get_range``,
``get_range_flat`` and ``get_ranges`` on the batch and the parallel path,
with their fallbacks when an LP gives no optimum, the indices they refuse and
the ranges ``get_ranges`` caches for a later MaxPool2d; ``is_empty_set``; the
set operations ``affine_map`` (diagonal and full, a bias flat or as an image,
and a bias in channel-first layout, which must be refused), ``minkowski_sum``
(also of operands with and without a predicate box), ``convex_hull`` and
``intersect_half_space``, and that none of them hands its input's cached
bounds on; ``evaluate``, ``contains`` (rows that the predicate box alone
does not enforce, no predicate box, a channel-first array) and ``sample``
(no rows, no predicate box, a sliver, N = 0); and the compositions in which
MaxPool2d consumes ``estimate_ranges`` or cached bounds and Flatten then
Linear consume ``flatten_to_star``. Every image has two channels and a
height different from its width: (C, H, W) = (2, 3, 5) or (2, 5, 3), and
(2, 2, 3) or (2, 3, 2) for the exact MaxPool2d cells and the inexact box.

Oracle. The image stars with constraint rows have a few predicate variables
and a basis written down here, so the true set is the exact set
``{c + G a : C a <= d, lb <= a <= ub}`` of those dyadic arrays in the stored
height, width, channel order (``Exact`` of the Star cells,
``tests/soundness/test_soundness_star_sets.py``, built on
``tests/_oracles/rational.py``), and its vertices give the exact range of every
pixel. The image stars built from a box have the box as their exact range. A
computed set is read from its stored arrays and judged by exact membership of
the true vertices, by exact membership of its own vertices in the true set
(for the methods that claim the exact set; ``assert_no_extra`` and
``assert_inside_cut`` of the Star cells), and by the LP oracle on float
points.

Layout. An image set stores height, width, channel; a network tensor is
channel, height, width. A flat index of an ImageStar method is the stored
order. ``flatten_to_star`` is compared with the true set permuted to channel,
height, width order (``cell_support.hwc_index``). Model outputs of the
composition cells (MaxPool2d; Flatten then Linear, through ``reach_layer``
and ``reach_pytorch_model``) are compared with ``assert_covered``, with
``out_chw`` for an image output.

Slack: as in the Star cells: none for set-returning methods and
``estimate_range(s)``; ``1e-12`` relative outward and ``1e-9`` relative
tightness for the LP range queries.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.utils.lpsolver import LPFailure
from tests._oracles import rational as Q
from tests._oracles.membership import LP_TOL, distance
from tests.cell_support import (assert_covered, hwc_index, image_set,
                                pixel_box, reach_model, reach_one_layer,
                                replay_torch, samples)
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.soundness.test_soundness_star_sets import (
    LP_REL, TIGHT_REL, Exact, _cut_slack, assert_encloses, assert_holds_exact,
    assert_inside_cut, assert_no_extra, assert_replay, exact_sum, membership,
    support, sum_vertices)

INF = float("inf")


# --------------------------------------------------------------------------- #
# The input image stars                                                         #
# --------------------------------------------------------------------------- #
# Predicate systems on three variables (rows C a <= d and a box).
_SYSTEMS = {
    "rows": ([[1, 1, 0.5], [-1, 0.5, 0]], [1, 0.5],
             [-1, -0.5, -1], [1, 1, 0.5]),
    "offset-box": ([[1, 0, 1]], [2], [0.5, -1, 0.25], [1.5, -0.25, 2]),
    "no-box": (np.vstack([np.eye(3), -np.eye(3), [[1, 1, 1]]]),
               [1, 1, 1.5, 1, 0.5, 0.5, 1], None, None),
}

# Lower predicate bounds only, bounded above by the rows: the regime between a
# box and no box. Used by the estimate cells only (not added to _SYSTEMS, so
# the other grids keep their cases).
_ONE_SIDED = {
    "lower-only": ([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                   [1, 1, 0.5, 1.5], [-1, -0.5, -1], None),
}

SHAPES = {"wide": (2, 3, 5), "tall": (2, 5, 3)}       # (C, H, W)


def _basis(shape, seed):
    """A dyadic basis ``[c | G]`` of an image of ``shape`` (C, H, W) with
    three predicate variables, rows in the stored (H, W, C) order; every
    pixel has its own centre and generators."""
    C, H, W = shape
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=H * W * C) / 8.0
    G = rng.integers(-4, 5, size=(H * W * C, 3)) / 4.0
    G[np.all(G == 0, axis=1), 0] = 0.25
    return c, G


def low_rank(system, shape_name, seed=0):
    """``(exact, image)``: an ImageStar on three predicate variables whose
    basis and rows are written here, and its exact set in stored order."""
    return low_rank_of_shape(system, SHAPES[shape_name], seed)


def low_rank_of_shape(system, shape, seed=0):
    """:func:`low_rank` for any (C, H, W) ``shape``."""
    C, H, W = shape
    c, G = _basis(shape, seed)
    rows, d, lb, ub = {**_SYSTEMS, **_ONE_SIDED}[system]
    exact = Exact.of(c, G, rows, d, lb, ub)
    image = ImageStar(np.hstack([c[:, None], G]), np.asarray(rows, float),
                      np.asarray(d, float).reshape(-1, 1),
                      None if lb is None else np.asarray(lb, float),
                      None if ub is None else np.asarray(ub, float),
                      H, W, C)
    return exact, image


def box_image(shape_name, zero_pixel=True, seed=1):
    """``(lo, hi, image)``: an ImageStar from (C, H, W) box bounds, one pixel of
    zero width."""
    C, H, W = SHAPES[shape_name]
    lo, hi = pixel_box(C, H, W, seed=seed)
    if zero_pixel:
        hi[1, 0, 1] = lo[1, 0, 1]
    return lo, hi, image_set("ImageStar", lo, hi)


def stored(chw, shape):
    C, H, W = shape
    return np.asarray(chw, dtype=np.float64).reshape(C, H, W).transpose(
        1, 2, 0).reshape(-1)


def chw_rows(shape):
    """The permutation ``q`` with ``y_chw = y_hwc[q]``."""
    return np.argsort(hwc_index(*shape))


RANGE_CASES = ["rows-wide", "rows-tall", "offset-box-wide", "no-box-wide",
               "no-box-tall", "box-wide", "box-tall", "point-wide",
               "inexact-wide", "inexact-tall"]
# The estimate cells also take an image star with lower predicate bounds only.
ESTIMATE_CASES = RANGE_CASES + ["lower-only-tall"]


def inexact_image(shape_name, seed=23):
    """``(exact, image)``: an ImageStar on three predicate variables with a
    non-dyadic centre, generators and predicate box and no rows, so the exact
    range of each pixel is its interval estimate and a sum rounded to nearest
    lands inside it."""
    C, H, W = SHAPES[shape_name]
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(H * W * C)
    G = rng.standard_normal((H * W * C, 3))
    lb, ub = [-1.0, 0.1, -0.3], [1.0, 0.7, 1 / 3]
    exact = Exact.of(c, G, None, None, lb, ub)
    image = ImageStar(np.hstack([c[:, None], G]), None, None, np.array(lb),
                      np.array(ub), H, W, C)
    return exact, image


def range_case(name):
    """``(exact_lo, exact_hi, image)`` of a named case in stored order."""
    kind, shape_name = name.rsplit("-", 1)
    if kind in _SYSTEMS or kind in _ONE_SIDED or kind == "inexact":
        exact, image = (inexact_image(shape_name) if kind == "inexact"
                        else low_rank(kind, shape_name))
        lo, hi = exact.ranges()
        return lo, hi, image
    lo, hi, image = box_image(shape_name)
    if kind == "point":
        image = image_set("ImageStar", lo, lo)
        hi = lo
    shape = SHAPES[shape_name]
    return (Q.vector(stored(lo, shape)), Q.vector(stored(hi, shape)), image)


# --------------------------------------------------------------------------- #
# Construction and conversion                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis", ["2-D", "4-D", "2-D, 1-D d"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_constructed_image_star_is_the_given_set(shape_name, basis):
    """``ImageStar.__init__`` stores the set it is given: a 2-D basis is read
    in height, width, channel order (pixel (h, w, c) is row (h W + w) C + c),
    a 4-D basis is used as it is, and a 1-D right-hand side is a column.
    Catches a 2-D basis read in channel-first order, predicate bounds stored
    for one another, and a dropped row. The stored layout is compared with
    the raw basis written here, not with another ImageStar."""
    exact, _ = low_rank("rows", shape_name)
    C, H, W = SHAPES[shape_name]
    c, G = _basis(SHAPES[shape_name], 0)
    rows, d, lb, ub = _SYSTEMS["rows"]
    d = np.asarray(d, float) if basis == "2-D, 1-D d" else \
        np.asarray(d, float).reshape(-1, 1)
    V_raw = np.hstack([c[:, None], G])
    V = V_raw.reshape(H, W, C, 4) if basis == "4-D" else V_raw
    image = ImageStar(V, np.asarray(rows, float), d, np.asarray(lb, float),
                      np.asarray(ub, float), H, W, C)
    assert (image.height, image.width, image.num_channels) == (H, W, C)
    for h, w, ch in np.ndindex(H, W, C):
        assert np.array_equal(image.V[h, w, ch], V_raw[(h * W + w) * C + ch])
    assert_holds_exact(image, exact.vertices(), "ImageStar constructor")
    assert_no_extra(image, exact, "ImageStar constructor")


def test_constructor_refuses_a_basis_that_contradicts_its_shape():
    """``ImageStar.__init__`` refuses with ValueError: a 4-D basis whose
    height, width or channel count
    contradicts a given one; a 2-D basis without the three sizes or with a
    row count that is not H W C; a basis of rank 1 or 3; a C whose column
    count is not the number of predicate variables or whose row count is not
    d's; predicate bounds of the wrong size. A missing check builds a set
    whose stated shape or predicate box disagrees with its basis."""
    C, H, W = SHAPES["tall"]
    c, G = _basis(SHAPES["tall"], 0)
    V = np.hstack([c[:, None], G])
    V4 = V.reshape(H, W, C, 4)
    box = (-np.ones(3), np.ones(3))
    bad = {
        "4-D, height": lambda: ImageStar(V4, None, None, *box, H + 1, W, C),
        "4-D, width": lambda: ImageStar(V4, None, None, *box, H, W + 1, C),
        "4-D, channels": lambda: ImageStar(V4, None, None, *box, H, W, C + 1),
        "2-D, no sizes": lambda: ImageStar(V, None, None, *box),
        "2-D, row count": lambda: ImageStar(V[:-1], None, None, *box, H, W, C),
        "rank 3": lambda: ImageStar(V.reshape(H, W * C, 4), None, None, *box),
        "rank 1": lambda: ImageStar(c, None, None),
        "C columns": lambda: ImageStar(V, np.ones((1, 2)), np.ones(1), *box,
                                       H, W, C),
        "C and d rows": lambda: ImageStar(V, np.ones((2, 3)), np.ones(1),
                                          *box, H, W, C),
        "lower bounds": lambda: ImageStar(V, None, None, -np.ones(2),
                                          np.ones(3), H, W, C),
        "upper bounds": lambda: ImageStar(V, None, None, -np.ones(3),
                                          np.ones(4), H, W, C),
    }
    answered = []
    for name, build in bad.items():
        try:
            build()
        except ValueError:
            continue
        answered.append(name)
    assert not answered, f"answered instead of refused: {answered}"


@known_bug(7, "ImageStar.__init__ checks the row count of d against C only "
              "when C has entries: with C omitted (or empty) and d given, the "
              "right-hand sides are kept with no row to bind and are ignored "
              "by every consumer, so a d that empties the set is dropped "
              "silently where an input that cannot be read is to be refused",
           raises=DID_NOT_RAISE)
def test_constructor_refuses_right_hand_sides_without_rows():
    """An image star with three predicate variables, C omitted and d = [-1]
    cannot be read: there is no row for the right-hand side, so it must be
    refused with ValueError, as a mismatch of C's and d's row counts is when
    C has rows (the passing neighbour in
    :func:`test_constructor_refuses_a_basis_that_contradicts_its_shape`).
    Today the d is stored and ignored, and is_empty_set answers False."""
    C, H, W = SHAPES["tall"]
    c, G = _basis(SHAPES["tall"], 0)
    V = np.hstack([c[:, None], G])
    with pytest.raises(ValueError):
        ImageStar(V, None, np.array([-1.0]), -np.ones(3), np.ones(3), H, W, C)


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_from_bounds_is_the_box_in_stored_order(shape_name):
    """``ImageStar.from_bounds`` of a (H, W, C) box with one zero-width pixel
    is the box: corners of the box are members, the support in every axis
    direction is the bound of that pixel. Catches a centre chosen from the
    wrong bound and the radius of one perturbed pixel given to another."""
    lo, hi, image = box_image(shape_name)
    shape = SHAPES[shape_name]
    slo, shi = stored(lo, shape), stored(hi, shape)
    assert_replay(image, [slo, shi] + [stored(p, shape)
                                       for p in samples(lo, hi, n=16)],
                  "from_bounds")
    for k in range(slo.size):
        e = np.eye(slo.size)[k]
        assert support(image, e) == shi[k] and -support(image, -e) == slo[k]


def test_from_bounds_holds_every_corner_of_an_inexact_box():
    """``ImageStar.from_bounds`` of a (2, 3, 2) box whose bounds, midpoints and
    half-widths are not short binary fractions (one pixel of zero width):
    the two extreme corners and 30 random corners of the box are members in
    exact arithmetic (Fractions, no slack), at their stored positions. Every
    pixel's lower and upper end is tested by the two extreme corners. Catches
    a centre or radius rounded to nearest instead of outward."""
    C, H, W = 2, 3, 2
    rng = np.random.default_rng(29)
    lo = rng.standard_normal((C, H, W))
    hi = lo + rng.random((C, H, W)) + 0.01
    hi[0, 2, 1] = lo[0, 2, 1]
    image = image_set("ImageStar", lo, hi)
    shape = (C, H, W)
    member = membership(Q.star(image))
    picks = [np.zeros(lo.shape, bool), np.ones(lo.shape, bool)]
    picks += [rng.random(lo.shape) < 0.5 for _ in range(30)]
    missed = [k for k, pick in enumerate(picks)
              if not member(Q.vector(stored(np.where(pick, hi, lo), shape)))]
    assert not missed, f"corners {missed} of the box are not in the image star"


@known_bug(8, "ImageStar.from_bounds of a box with an infinite end puts an "
              "infinite generator into the basis (centre 0, half-width "
              "infinite), so c + G a is NaN at a = 0 and no finite point of "
              "the box is a member of the returned set by its rows; a later "
              "affine map with a zero weight on that pixel puts NaN into the "
              "basis")
def test_from_bounds_of_a_box_with_an_infinite_end_holds_its_points():
    """``from_bounds`` of a (2, 3, 2) box whose one pixel has an infinite
    upper end either refuses with ValueError or returns a set with a finite
    basis that holds points of the box at their stored positions (the lower
    corner, a point with the open pixel at 1e3, their midpoint), by the LP
    oracle on its rows. :func:`test_from_bounds_is_the_box_in_stored_order`
    (a finite box, the same replay check) is the passing neighbour."""
    C, H, W = 2, 3, 2
    lo, hi = pixel_box(C, H, W, seed=4)
    hi[1, 2, 0] = INF
    try:
        image = ImageStar.from_bounds(lo.transpose(1, 2, 0),
                                      hi.transpose(1, 2, 0), H, W, C)
    except ValueError:
        return
    assert np.all(np.isfinite(image.V)), \
        "the basis holds an infinite entry, so c + G a is NaN at a = 0"
    far = np.where(np.isinf(hi), lo + 1e3, hi)
    points = [stored(p, (C, H, W)) for p in (lo, far, (lo + far) / 2)]
    assert_replay(image, points, "from_bounds with an infinite end")


@pytest.mark.parametrize("system", list(_SYSTEMS))
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_to_star_and_flatten_to_star_are_the_set_in_their_orders(
        shape_name, system):
    """``to_star`` is the same set in height, width, channel order and
    ``flatten_to_star`` the same set in channel, height, width order (what
    ``nn.Flatten`` of the tensor gives), both exactly. Catches a basis not
    transposed before flattening, one transposed to (C, W, H), and swapped
    predicate bounds."""
    exact, image = low_rank(system, shape_name)
    shape = SHAPES[shape_name]
    V = exact.vertices()
    star = image.to_star()
    assert isinstance(star, Star) and star.dim == image.dim
    assert_holds_exact(star, V, f"{system}: to_star")
    assert_no_extra(star, exact, f"{system}: to_star")
    q = chw_rows(shape)
    V_chw = [tuple(v[i] for i in q) for v in V]
    flat = image.flatten_to_star()
    assert_holds_exact(flat, V_chw, f"{system}: flatten_to_star")
    assert_no_extra(flat, exact.permuted(q), f"{system}: flatten_to_star")


@pytest.mark.parametrize("method", [
    "to_star",
    pytest.param("flatten_to_star", marks=known_bug(
        8, "ImageStar.flatten_to_star reorders the basis to channel, height, "
           "width order but passes state_lb and state_ub on in height, "
           "width, channel order, so the flat star's stored state bounds "
           "belong to other entries and leave out their true ranges "
           "(registry SET-9)")),
])
def test_conversion_passes_state_bounds_in_its_own_order(method):
    """After ``estimate_ranges`` an ImageStar of (2, 3, 5) with its own
    interval per pixel stores state bounds; a cached bound that the flat star
    of ``to_star`` (height, width, channel) or of ``flatten_to_star``
    (channel, height, width) carries must contain the true range of its own
    entry. Handing no cached bound on keeps that promise (both fixes that
    registry SET-9 allows, reordering and dropping, turn the marked case
    green). Catches cached bounds handed on in another order or swapped."""
    lo, hi, image = box_image("wide", zero_pixel=False)
    image.estimate_ranges()
    if image.state_lb is None or image.state_ub is None:
        pytest.fail("test setup: estimate_ranges stored no state bounds")
    star = getattr(image, method)()
    if method == "to_star":
        e_lo, e_hi = stored(lo, (2, 3, 5)), stored(hi, (2, 3, 5))
    else:
        e_lo, e_hi = lo.reshape(-1), hi.reshape(-1)
    s_lo = np.full(star.dim, -INF) if star.state_lb is None else star.state_lb
    s_hi = np.full(star.dim, INF) if star.state_ub is None else star.state_ub
    assert_encloses(s_lo, s_hi, Q.vector(e_lo), Q.vector(e_hi),
                    f"{method}: stored state bounds")


# --------------------------------------------------------------------------- #
# Range queries                                                                 #
# --------------------------------------------------------------------------- #
def _flat_to_hwc(index, image):
    h = index // (image.width * image.num_channels)
    rest = index % (image.width * image.num_channels)
    return h, rest // image.num_channels, rest % image.num_channels


def _lp_query(image, query):
    n = image.dim
    if query == "get_ranges(parallel=False)":
        return image.get_ranges(parallel=False)
    if query == "get_ranges(parallel=True)":
        return image.get_ranges(parallel=True, n_workers=2)
    if query == "get_ranges()":
        return image.get_ranges()
    if query == "get_range":
        pairs = [image.get_range(*_flat_to_hwc(i, image)) for i in range(n)]
    elif query == "get_range_flat":
        pairs = [image.get_range_flat(i) for i in range(n)]
    else:
        raise ValueError(query)
    return [p[0] for p in pairs], [p[1] for p in pairs]


# The explicit batch path, the explicit parallel path and get_range_flat
# (which calls get_range) run on every range case. The default call
# get_ranges(), which takes the parallel path on these 30-pixel images by
# config.parallel_threshold, and get_range with the index split by the test
# run on a few regimes only: on the others they repeated the parallel=True
# and get_range_flat cells line for line.
_LP_DEFAULT_CASES = ("rows-tall",)
_LP_GET_RANGE_CASES = ("rows-tall", "no-box-wide", "point-wide")


@pytest.mark.parametrize("name, query", [
    (name, query) for name in RANGE_CASES
    for query in ("get_ranges(parallel=False)", "get_ranges(parallel=True)",
                  "get_ranges()", "get_range", "get_range_flat")
    if (query != "get_ranges()" or name in _LP_DEFAULT_CASES)
    and (query != "get_range" or name in _LP_GET_RANGE_CASES)])
def test_lp_range_query_is_the_exact_range_of_each_pixel(name, query):
    """The LP range queries of an ImageStar return the exact range of every
    pixel, at the stored flat index (pixel (h, w, c) at (h W + w) C + c):
    outward with a 1e-12 relative pad and within 1e-9 relative of it, on
    images with height above and below width and two channels, rows that
    bind, no predicate box, and a point image. Catches centres and
    generators flattened in different orders, the point shortcut flattened
    in Fortran order, the flat index split with the height instead of the
    width, a maximum solved as a minimum, and the first pixel of the parallel
    path left at [0, 0]."""
    e_lo, e_hi, image = range_case(name)
    lo, hi = _lp_query(image, query)
    assert_encloses(lo, hi, e_lo, e_hi, f"{name}: {query}", rel=LP_REL,
                    tight=TIGHT_REL)


@pytest.mark.parametrize("name, query", [
    (name, query) for name in ESTIMATE_CASES
    for query in ("estimate_ranges", "estimate_range", "stored state bounds")
    # without a full predicate box the enclosure is returned, not stored
    # (POOL-4)
    if not (query == "stored state bounds"
            and name.startswith(("no-box", "lower-only")))])
def test_estimate_query_encloses_the_exact_range_of_each_pixel(name, query):
    """``estimate_range(index)`` and ``estimate_ranges()`` contain the exact
    range of the pixel at each stored flat index with no slack, with and
    without a predicate box (the LP enclosure), and with lower predicate
    bounds only (bounded above by the rows, which also takes the LP
    enclosure), on two-channel images with height above and below width; so
    do the state bounds ``estimate_ranges`` stores. The non-dyadic image star
    without rows has the interval estimate as its exact range, so a sum
    rounded to nearest lands inside it. Catches the LP enclosure computed on
    the channel-first flattening, the flat index split with the height
    instead of the width, an upper bound built from lower bounds, bounds
    flattened channel-first, interval sums without outward rounding, and a
    one-sided predicate box taken for a box."""
    e_lo, e_hi, image = range_case(name)
    if query == "estimate_ranges":
        lo, hi = image.estimate_ranges()
    elif query == "estimate_range":
        pairs = [image.estimate_range(i) for i in range(image.dim)]
        lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
    else:
        image.estimate_ranges()
        if image.state_lb is None:
            pytest.fail(f"{name}: estimate_ranges stored no state bounds")
        lo, hi = image.state_lb, image.state_ub
    assert_encloses(lo, hi, e_lo, e_hi, f"{name}: {query}")
    finite = np.all(np.isfinite(np.asarray(lo, dtype=np.float64))) and \
        np.all(np.isfinite(np.asarray(hi, dtype=np.float64)))
    assert finite, f"{name}: {query} returned an infinite bound for a " \
                   f"bounded image star"


def _poisoning(real, poison, mode):
    def stub(objectives, *args, **kwargs):
        hit = [any(np.array_equal(np.asarray(f, dtype=np.float64).ravel(), r)
                   for r in poison) for f in objectives]
        if mode == "raise" and any(hit) and len(objectives) == 2:
            raise RuntimeError("injected LP failure")
        values = real(objectives, *args, **kwargs)
        return [None if h else v for v, h in zip(values, hit)]
    return stub


@pytest.mark.parametrize("path, mode", [
    ("parallel=False", "none"),
    ("parallel=True", "raise"),
])
@pytest.mark.parametrize("name", ["rows-wide", "offset-box-tall"])
def test_get_ranges_falls_back_to_an_enclosure_when_an_lp_fails(
        name, path, mode, monkeypatch):
    """When the LP of some pixels returns no optimum on the batch path
    (image_star.py, the fallback of ``_get_ranges_batch``), or raises in a
    worker of the parallel path (the ``except`` of
    ``_get_ranges_parallel``), ``get_ranges`` must still contain the exact
    range of those pixels (the fallback to ``estimate_range``), and keep the
    others exact. The failure is injected for flat indices 4 and 17 only.
    Catches a fallback that returns the lower bound twice, writes to the
    wrong pixel, or leaves the pixel at [0, 0]. An LP without an optimum on
    the parallel path is the defect marked in
    :func:`test_get_ranges_of_an_unbounded_image_star_is_no_finite_bound`."""
    import n2v.sets.image_star as image_module
    kind, shape_name = name.rsplit("-", 1)
    exact, image = low_rank(kind, shape_name)
    V2 = image.V.reshape(image.dim, -1)
    poison = [np.asarray(V2[i, 1:], dtype=np.float64) for i in (4, 17)]
    monkeypatch.setattr(image_module, "solve_lp_batch", _poisoning(
        image_module.solve_lp_batch, poison, mode))
    if path == "parallel=False":
        lo, hi = image.get_ranges(parallel=False)
    else:
        lo, hi = image.get_ranges(parallel=True, n_workers=2)
    e_lo, e_hi = exact.ranges()
    what = f"{name}: get_ranges({path}) with failed LPs ({mode})"
    assert_encloses(lo, hi, e_lo, e_hi, what, rel=LP_REL)
    others = [i for i in range(image.dim) if i not in (4, 17)]
    assert_encloses(lo, hi, e_lo, e_hi, what, rel=LP_REL, tight=TIGHT_REL,
                    rows=others)


def _unbounded_image():
    """An ImageStar of (2, 2, 3) without a predicate box whose rows leave
    ``a0`` open below: every pixel is unbounded below."""
    H, W, C = 2, 3, 2
    V = np.zeros((H * W * C, 3))
    V[:, 0] = np.arange(H * W * C) / 4.0
    V[:, 1] = 1.0
    V[:, 2] = np.arange(H * W * C) / 8.0
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return ImageStar(V, rows, np.ones((3, 1)), None, None, H, W, C)


@pytest.mark.parametrize("parallel", [
    False,
    pytest.param(True, marks=known_bug(
        8, "ImageStar.get_ranges on the parallel path (the default from 10 "
           "pixels) writes NaN bounds for every pixel whose LP returns no "
           "optimum (here an image star unbounded below), where the batch "
           "path and Star.get_ranges fall back to an enclosure; NaN is no "
           "bound and is cached in state_lb and state_ub")),
])
def test_get_ranges_of_an_unbounded_image_star_is_no_finite_bound(parallel):
    """Every pixel of this image star is unbounded below and bounded above
    (rows ``a0 <= 1``, ``-1 <= a1 <= 1``, no predicate box): ``get_ranges``
    must give -inf below (never NaN, never a finite number) and an upper
    bound at or above the true maximum ``c + 1 + |g2|``. The batch path is
    the passing neighbour on the same image."""
    image = _unbounded_image()
    lo, hi = image.get_ranges(parallel=parallel, n_workers=2)
    lo, hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    assert np.all(lo == -INF), f"lower bounds {lo}"
    true_max = image.V[:, :, :, 0].reshape(-1) + 1.0 + \
        np.abs(image.V[:, :, :, 2].reshape(-1))
    assert np.all(hi >= true_max), f"upper bounds {hi}"


def test_get_range_of_an_empty_image_star_is_no_bound():
    """An ImageStar whose rows contradict each other has no range:
    ``get_range`` returns (None, None) (image_star.py, the None return)."""
    image = ImageStar(np.hstack([np.ones((6, 1)), np.ones((6, 2))]),
                      np.array([[1.0, 1.0]]), np.array([[-3.0]]),
                      -np.ones(2), np.ones(2), 1, 3, 2)
    assert image.get_range(0, 2, 1) == (None, None)


@pytest.mark.parametrize("query", [
    "get_range", "get_range_flat",
    pytest.param("estimate_range", marks=known_bug(
        23, "ImageStar.estimate_range with a flat index past the last pixel "
            "lets numpy's IndexError escape where get_range and "
            "get_range_flat refuse the same index with ValueError",
        raises=IndexError)),
])
def test_range_queries_refuse_an_index_that_names_no_pixel(query):
    """An index that names no pixel of a (5, 3, 2) image star is refused
    with ValueError: for ``get_range`` a height, width or channel index equal
    to its size or negative; for the flat queries the index ``dim`` (and -1
    for ``get_range_flat``, whose documented range is 0 to dim - 1). A
    missing check answers the range of another pixel through numpy's
    negative indexing, or crashes."""
    _, image = low_rank("rows", "tall")
    H, W, C = image.height, image.width, image.num_channels
    if query == "get_range":
        bad = [(H, 0, 0), (-1, 0, 0), (0, W, 0), (0, -1, 0), (0, 0, C),
               (0, 0, -1)]
    elif query == "get_range_flat":
        bad = [(image.dim,), (-1,)]
    else:
        bad = [(image.dim,)]
    for index in bad:
        with pytest.raises(ValueError):
            getattr(image, query)(*index)


@pytest.mark.parametrize("parallel", [False, True])
def test_get_ranges_caches_the_range_of_each_pixel_for_a_later_maxpool(
        parallel):
    """``get_ranges`` (batch path and parallel path) caches its ranges in
    ``state_lb`` and ``state_ub``, which approx MaxPool2d reads in place of
    ``estimate_ranges``. A cached bound must contain the exact range of the
    pixel it is stored at (the cache is the LP ranges: 1e-12 relative
    outward pad), and MaxPool2d after the call must hold every float64
    output of the true layer at points of the set. Catches a cache that
    stores the lower bounds as the upper ones or the other way round, or in
    another order. Not caching at all keeps the promise."""
    exact, image = low_rank("rows", "tall")
    C, H, W = SHAPES["tall"]
    image.get_ranges(parallel=parallel, n_workers=2)
    e_lo, e_hi = exact.ranges()
    lo = np.full(image.dim, -INF) if image.state_lb is None else image.state_lb
    hi = np.full(image.dim, INF) if image.state_ub is None else image.state_ub
    assert_encloses(lo, hi, e_lo, e_hi,
                    f"get_ranges(parallel={parallel}): cached bounds",
                    rel=LP_REL)
    pool = nn.MaxPool2d((2, 1), stride=(1, 2))
    out = reach_one_layer(pool, [image], "approx", "maxpool after get_ranges")
    q = chw_rows((C, H, W))
    ys = replay_torch(pool, [p[q] for p in exact.points(extra=12)],
                      (1, C, H, W))
    assert_covered(out, ys, f"maxpool after get_ranges(parallel={parallel})",
                   out_chw=(C, H - 1, (W - 1) // 2 + 1))


# --------------------------------------------------------------------------- #
# is_empty_set                                                                  #
# --------------------------------------------------------------------------- #
def _image_of(rows, d, lb, ub, scale=1.0):
    V = np.zeros((6, 3))
    V[:, 0] = np.arange(6) / 4.0
    V[:, 1:] = scale * np.array([[1, 0], [0, 1], [1, 1], [1, -1], [0.5, 0],
                                 [0, 0.5]])
    return ImageStar(V, None if rows is None else np.asarray(rows, float),
                     None if d is None else np.asarray(d, float),
                     None if lb is None else np.asarray(lb, float),
                     None if ub is None else np.asarray(ub, float), 1, 3, 2)


IMAGE_EMPTINESS = {
    "rows-not-empty": ("low-rank", False),
    "empty-by-rows": (([[1, 1]], [-2.25], [-1, -1], [1, 1]), True),
    "empty-by-rows-no-box": (([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                              [1, 1, 1, 1, -2.25], None, None), True),
    "empty-by-predicate-box": ((None, None, [1, 0], [0, 1]), True),
    "sliver-not-empty": (([[1, 1]], [-2 + 2.0 ** -19], [-1, -1], [1, 1]),
                         False),
    "row-excludes-lower-corner": (([[-1, -1]], [1], [-1, -1], [1, 1]),
                                  False),
    "box-rows-not-empty": (([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1],
                            None, None), False),
    "point-image": ("point", False),
}


@pytest.mark.parametrize("case", [
    c if c != "point-image" else pytest.param(c, marks=known_bug(
        8, "ImageStar.is_empty_set on an image star with no predicate "
           "variable (a point image, from_bounds with lo == hi) hands a model "
           "with no variable to the solver and raises LPFailure instead of "
           "answering 'not empty' as Star.is_empty_set does",
        raises=LPFailure))
    for c in IMAGE_EMPTINESS])
def test_image_star_is_empty_set_answers_as_the_exact_feasibility(case):
    """``ImageStar.is_empty_set`` is True exactly when the predicate polytope
    is empty: a sliver of width about 2e-6, a row that only removes the
    lower corner and a box written as four rows (no point makes the four
    tight together) are not empty, contradictory rows and crossed predicate
    bounds are. A wrong True is a false UNSAT."""
    spec, expected = IMAGE_EMPTINESS[case]
    if spec == "low-rank":
        image = low_rank("rows", "wide")[1]
    elif spec == "point":
        lo, _, _ = box_image("wide")
        image = image_set("ImageStar", lo, lo)
    else:
        image = _image_of(*spec, scale=2.0 ** 20)
    assert image.is_empty_set() is expected, case


# --------------------------------------------------------------------------- #
# Set operations                                                                #
# --------------------------------------------------------------------------- #
def _assert_same_shape(result, image, what):
    assert isinstance(result, ImageStar), f"{what}: not an ImageStar"
    assert (result.height, result.width, result.num_channels) == \
        (image.height, image.width, image.num_channels), what


_AFFINE_FORMS = ["diagonal, flat bias", "diagonal, image bias", "full, bias",
                 "full, no bias"]


@pytest.mark.parametrize("shape_name, system, form", [
    (shape_name, system, form) for shape_name in SHAPES for system in _SYSTEMS
    for form in _AFFINE_FORMS] + [("tall", "rows", "full, image bias")])
def test_image_affine_map_is_the_exact_image(shape_name, system, form):
    """``ImageStar.affine_map`` with a diagonal weight (a vector in stored
    order) or a full square weight on the stored flat order, and a bias given
    flat or as an (H, W, C) image (both paths flatten an image bias, each in
    its own code), is the exact image and keeps the image's shape. Catches a
    diagonal reshaped in another order, a bias added to the wrong pixels
    (an image bias flattened in another order on the full path), and the
    shape of the result swapped."""
    exact, image = low_rank(system, shape_name)
    n = image.dim
    rng = np.random.default_rng(len(form))
    b = rng.integers(-8, 9, size=n) / 8.0
    if form.startswith("diagonal"):
        w = rng.integers(-4, 5, size=n) / 4.0
        W = np.diag(w)
        bias = b if form.endswith("flat bias") else b.reshape(
            image.height, image.width, image.num_channels)
        result = image.affine_map(w, bias)
    else:
        W = rng.integers(-2, 3, size=(n, n)) / 4.0 * \
            (rng.random((n, n)) < 0.2)
        bias = {"full, bias": b, "full, no bias": None,
                "full, image bias": b.reshape(image.height, image.width,
                                              image.num_channels)}[form]
        result = image.affine_map(W, bias)
    target = exact.affine(W, b if bias is not None else None)
    _assert_same_shape(result, image, form)
    V = target.vertices()
    assert_holds_exact(result, V, f"{system}: affine_map {form}")
    assert_no_extra(result, target, f"{system}: affine_map {form}")


def test_image_affine_map_refuses_a_weight_that_changes_the_size():
    """A full weight that is not square, or a diagonal of the wrong length,
    is refused, and so is a bias of the wrong size on either path."""
    _, image = low_rank("rows", "wide")
    with pytest.raises(ValueError):
        image.affine_map(np.ones((image.dim - 1, image.dim)))
    with pytest.raises(ValueError):
        image.affine_map(np.ones(image.dim - 1))
    with pytest.raises(ValueError):
        image.affine_map(np.ones(image.dim), np.ones(image.dim - 1))
    with pytest.raises(ValueError):
        image.affine_map(np.eye(image.dim), np.ones(image.dim - 1))


@pytest.mark.parametrize("weight", [
    pytest.param(w, marks=known_bug(
        7, "ImageStar.affine_map checks only the number of entries of its "
           "bias: a (C, H, W) array is read as (H, W, C), so each pixel gets "
           "another pixel's bias, where a shape that does not fit is to be "
           "refused", raises=DID_NOT_RAISE))
    for w in ("diagonal", "full")])
def test_image_affine_map_refuses_a_channels_first_bias(weight):
    """The bias of ``affine_map`` is documented as flat or (H, W, C); a bias
    of the right size in (C, H, W) layout (an image with two channels and
    height different from width, so the layouts differ) does not fit and
    must be refused with ValueError, on the diagonal and the full path.
    :func:`test_image_affine_map_refuses_a_weight_that_changes_the_size` is
    the passing neighbour."""
    _, image = low_rank("rows", "tall")
    n = image.dim
    b_chw = (np.arange(n) / 8.0).reshape(image.num_channels, image.height,
                                         image.width)
    weight = np.ones(n) if weight == "diagonal" else np.eye(n)
    with pytest.raises(ValueError):
        image.affine_map(weight, b_chw)


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_image_minkowski_sum_is_the_exact_sum(shape_name):
    """``ImageStar.minkowski_sum`` of two image stars of the same shape with
    different centres, rows and predicate boxes is the exact sum, pixel by
    pixel in stored order, with the shape kept. Shapes that differ and a
    non-image operand are refused."""
    ea, a = low_rank("rows", shape_name, seed=0)
    eb, b = low_rank("offset-box", shape_name, seed=1)
    result = a.minkowski_sum(b)
    _assert_same_shape(result, a, "minkowski_sum")
    V = sum_vertices(ea, eb)
    assert_holds_exact(result, V, "image minkowski_sum")
    assert_no_extra(result, exact_sum(ea, eb), "image minkowski_sum")
    other = low_rank("rows", "tall" if shape_name == "wide" else "wide")[1]
    with pytest.raises(ValueError):
        a.minkowski_sum(other)
    with pytest.raises(TypeError):
        a.minkowski_sum(a.to_star())


@known_bug(8, "ImageStar.minkowski_sum delegates to Star.minkowski_sum, which "
              "keeps the operands' predicate bounds only when both have "
              "them: the sum of an image star with a predicate box and one "
              "given by its rows alone has no predicate box and is unbounded "
              "where the sum is bounded (registry SET-10)")
def test_image_minkowski_sum_keeps_each_operands_predicate_box():
    """The sum of an image star whose predicate box excludes 0 and one given
    by its rows alone (no predicate box) is the exact sum: every true vertex
    is a member and every vertex of the result lies in the sum (an unbounded
    result fails here). Registry SET-10; the entry route through ImageStar.
    :func:`test_image_minkowski_sum_is_the_exact_sum` (both operands with a
    box) is the passing neighbour with the same reference code."""
    ea, a = low_rank("offset-box", "tall", seed=1)
    eb, b = low_rank("no-box", "tall", seed=2)
    result = a.minkowski_sum(b)
    _assert_same_shape(result, a, "minkowski_sum")
    assert_holds_exact(result, sum_vertices(ea, eb), "image sum, mixed boxes")
    assert_no_extra(result, exact_sum(ea, eb), "image sum, mixed boxes")


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_image_convex_hull_holds_both_operands(shape_name):
    """``ImageStar.convex_hull`` of an image star whose predicate box excludes
    0 and one without a predicate box holds both, with the shape kept. An
    operand of another shape with the same number of entries ((2, 3, 5)
    against (2, 5, 3)) and a non-image operand are refused."""
    ea, a = low_rank("offset-box", shape_name, seed=2)
    eb, b = low_rank("no-box", shape_name, seed=3)
    result = a.convex_hull(b)
    _assert_same_shape(result, a, "convex_hull")
    assert_holds_exact(result, ea.vertices(), "image hull, first operand")
    assert_holds_exact(result, eb.vertices(), "image hull, second operand")
    other = low_rank("rows", "tall" if shape_name == "wide" else "wide")[1]
    with pytest.raises(ValueError):
        a.convex_hull(other)
    with pytest.raises(TypeError):
        a.convex_hull(b.to_star())


@pytest.mark.parametrize("system", list(_SYSTEMS))
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_image_intersect_half_space_is_the_exact_cut(shape_name, system):
    """``ImageStar.intersect_half_space(H, g)`` with rows over the stored flat
    order that cut the set is the intersection (true vertices are members,
    the result does not cross the rows beyond the rounding pad), with the
    shape kept."""
    exact, image = low_rank(system, shape_name)
    n = image.dim
    rng = np.random.default_rng(7)
    H = rng.integers(-2, 3, size=(2, n)) / 4.0
    H[:, 0] = [1.0, -1.0]
    g = []
    for h in H:
        values = [Q.dot(Q.vector(h), v) for v in exact.vertices()]
        g.append(float((min(values) + max(values)) / 2))
    g = np.array(g)
    result = image.intersect_half_space(H, g)
    _assert_same_shape(result, image, "intersect_half_space")
    cut = exact.cut(H, g)
    assert_holds_exact(result, cut.vertices(), f"{system}: image cut")
    assert_inside_cut(result, exact, H, g, _cut_slack(H, g, image.to_star()),
                      f"{system}: image cut")


def test_a_set_operation_after_estimate_ranges_holds_every_maxpool_output():
    """State left by an earlier call: the operands carry the bounds that
    ``estimate_ranges`` cached, and approx MaxPool2d reads the cached bounds
    of the set it is given. For each set operation (diagonal and full
    ``affine_map``, ``minkowski_sum``, ``convex_hull``,
    ``intersect_half_space``) on the (2, 5, 3) image stars, MaxPool2d of the
    result must hold every float64 output of the true layer at the exact
    vertices of the true result (in channel-first order). Catches a result
    that carries its input's cached bounds, which are not bounds of the
    result."""
    C, H, W = SHAPES["tall"]
    ea, a = low_rank("rows", "tall", seed=0)
    eb, b = low_rank("offset-box", "tall", seed=1)
    n = a.dim
    rng = np.random.default_rng(11)
    w = rng.integers(-4, 5, size=n) / 4.0
    Wf = rng.integers(-2, 3, size=(n, n)) / 4.0 * (rng.random((n, n)) < 0.2)
    bias = rng.integers(-8, 9, size=n) / 8.0
    Hc = rng.integers(-2, 3, size=(1, n)) / 4.0
    Hc[0, 0] = 1.0
    values = [Q.dot(Q.vector(Hc[0]), v) for v in ea.vertices()]
    g = np.array([float((min(values) + max(values)) / 2)])
    operations = {
        "affine_map diagonal": (lambda: a.affine_map(w, bias),
                                lambda: ea.affine(np.diag(w), bias).vertices()),
        "affine_map full": (lambda: a.affine_map(Wf, bias),
                            lambda: ea.affine(Wf, bias).vertices()),
        "minkowski_sum": (lambda: a.minkowski_sum(b),
                          lambda: sum_vertices(ea, eb)),
        "convex_hull": (lambda: a.convex_hull(b),
                        lambda: ea.vertices() + eb.vertices()),
        "intersect_half_space": (lambda: a.intersect_half_space(Hc, g),
                                 lambda: ea.cut(Hc, g).vertices()),
    }
    pool = nn.MaxPool2d((2, 1), stride=(1, 2))
    q = chw_rows((C, H, W))
    for name, (operation, true_vertices) in operations.items():
        a.estimate_ranges()
        b.estimate_ranges()
        if a.state_lb is None or b.state_lb is None:
            pytest.fail("test setup: estimate_ranges cached no bounds")
        result = operation()
        out = reach_one_layer(pool, [result], "approx", f"maxpool after {name}")
        points = [np.array([float(x) for x in v])[q] for v in true_vertices()]
        assert_covered(out, replay_torch(pool, points, (1, C, H, W)),
                       f"maxpool after {name} of image stars with cached bounds",
                       out_chw=(C, H - 1, (W - 1) // 2 + 1))


# --------------------------------------------------------------------------- #
# evaluate, contains, sample                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_evaluate_is_the_basis_at_the_predicate_point(shape_name):
    """``ImageStar.evaluate(a)`` is ``c + G a`` as an (H, W, C) image, exactly
    for dyadic data; a predicate vector of the wrong length is refused."""
    exact, image = low_rank("rows", shape_name)
    c, G = exact.arrays[0], exact.arrays[1]
    for a in ([1.0, -0.5, 0.25], [-1.0, 1.0, 0.5], [0.0, 0.0, 0.0]):
        got = image.evaluate(np.array(a))
        assert got.shape == (image.height, image.width, image.num_channels)
        assert np.array_equal(got.reshape(-1), c + G @ np.array(a))
    with pytest.raises(ValueError):
        image.evaluate(np.zeros(2))


def test_image_contains_answers_by_the_rows():
    """``ImageStar.contains`` accepts vertices of the set (as (H, W, C)
    images and as flat vectors in stored order) and rejects points pushed out
    of the set from a vertex. It also rejects images c + G a whose predicate
    point a lies inside the predicate box but breaks a row: the box image
    holds them, so a contains that drops the
    rows accepts them. On an image star given by its rows alone (no predicate
    box) it accepts the vertices. An array of another size is refused."""
    exact, image = low_rank("rows", "tall")
    V = exact.vertices()
    centre = [sum(v[i] for v in V) / len(V) for i in range(exact.dim)]
    shape = (image.height, image.width, image.num_channels)
    for v in V[:4]:
        y = np.array([float(x) for x in v])
        assert image.contains(y.reshape(shape)) and image.contains(y)
        out = np.array([float(x + (x - m) / 4) for x, m in zip(v, centre)])
        if exact.holds(Q.vector(out)):
            pytest.fail("test data: pushed point is still in the set")
        assert distance(image, out) > 1e-6
        assert not image.contains(out.reshape(shape))
    c, G = exact.arrays[0], exact.arrays[1]
    _, _, lb, ub = _SYSTEMS["rows"]
    for a in ([1.0, 1.0, 0.5], [-1.0, 1.0, 0.0]):
        if not all(lo <= x <= hi for x, lo, hi in zip(a, lb, ub)):
            pytest.fail("test data: the predicate point is outside the box")
        y = c + G @ np.array(a)
        if exact.holds(Q.vector(y)):
            pytest.fail("test data: the image breaks no row")
        assert distance(image, y) > 1e-6
        assert not image.contains(y.reshape(shape))
        assert not image.contains(y)
    rows_only, no_box = low_rank("no-box", "tall")
    for v in rows_only.vertices()[:4]:
        y = np.array([float(x) for x in v])
        assert no_box.contains(y.reshape(shape)) and no_box.contains(y)
    with pytest.raises(ValueError):
        image.contains(np.zeros(image.dim + 1))


@known_bug(15, "ImageStar.contains checks only the number of entries of an "
               "array that is not (H, W, C): a (C, H, W) array is read in "
               "stored order and answered for another point, where a shape "
               "that does not fit is to be refused", raises=DID_NOT_RAISE)
def test_image_contains_refuses_a_channels_first_array():
    """``contains`` takes an (H, W, C) image or a flat vector in stored
    order; a (C, H, W) array of the same size (two channels, height
    different from width, so the layouts differ) does not fit and must be
    refused with ValueError. The refusal of an array of another size in
    :func:`test_image_contains_answers_by_the_rows` is the passing
    neighbour."""
    exact, image = low_rank("rows", "tall")
    y = np.array([float(x) for x in exact.vertices()[0]])
    y_chw = y.reshape(image.height, image.width,
                      image.num_channels).transpose(2, 0, 1)
    with pytest.raises(ValueError):
        image.contains(y_chw)


def _assert_sampled_members(image, images, what):
    """Each sampled image has shape (H, W, C) and is a member of ``image``
    by the LP oracle on its own rows."""
    for y in images:
        assert y.shape == (image.height, image.width, image.num_channels), what
        assert distance(image, y.reshape(-1)) <= LP_TOL * max(
            1.0, float(np.max(np.abs(y)))), what


@pytest.mark.parametrize("system", ["rows", "offset-box", "no-rows"])
def test_image_sample_returns_points_of_the_set(system):
    """``ImageStar.sample(N)`` returns N images of shape (H, W, C), each a
    member of the set by the LP oracle (the predicate sampler must respect
    the rows), also for an image star with a predicate box and no rows (C
    omitted, stored as (0, nVar)); a point image gives N copies of its
    point."""
    image = (inexact_image("wide")[1] if system == "no-rows"
             else low_rank(system, "wide")[1])
    images = image.sample(12)
    assert len(images) == 12
    _assert_sampled_members(image, images, f"sample of {system}")
    lo, _, _ = box_image("wide")
    point = image_set("ImageStar", lo, lo)
    copies = point.sample(3)
    assert len(copies) == 3 and all(
        np.array_equal(p.reshape(-1), stored(lo, (2, 3, 5))) for p in copies)


@pytest.mark.parametrize("case, n", [
    pytest.param("no predicate box", 4, marks=known_bug(
        15, "ImageStar.sample reads the predicate box without checking that "
            "it exists: an image star given by its rows alone raises "
            "AttributeError where N members are due", raises=AttributeError)),
    pytest.param("sliver", 4, marks=known_bug(
        15, "ImageStar.sample draws predicate points from the box and keeps "
            "those that satisfy the rows; when none of its 100 N draws does "
            "(a non-empty sliver, or a predicate box with an infinite bound, "
            "whose draws are not finite) it raises numpy's ValueError "
            "('high <= 0') where N members are due", raises=ValueError)),
    pytest.param("box", 0, marks=known_bug(
        15, "ImageStar.sample(0) on an image star with predicate variables "
            "raises numpy's ValueError ('need at least one array to "
            "concatenate') where an empty list is due", raises=ValueError)),
])
def test_image_sample_answers_for_every_non_empty_image_star(case, n):
    """``sample(N)`` returns N members of every non-empty image star: one
    given by its rows alone (no predicate box), a sliver of the predicate
    box (a0 + a1 + a2 <= -3 + 2^-20, which the corner a = (-1, -1, -1)
    satisfies), and N = 0 (an empty list). Membership by the LP oracle;
    :func:`test_image_sample_returns_points_of_the_set` is the passing
    neighbour with the same check."""
    _, image = low_rank("no-box" if case == "no predicate box" else "rows",
                        "wide")
    if case == "sliver":
        image = ImageStar(image.V, np.ones((1, 3)),
                          np.array([[-3 + 2.0 ** -20]]), -np.ones(3),
                          np.ones(3), image.height, image.width,
                          image.num_channels)
        corner = image.evaluate(-np.ones(3)).reshape(-1)
        if distance(image, corner) > LP_TOL * max(
                1.0, float(np.max(np.abs(corner)))):
            pytest.fail("test data: the sliver is empty")
    images = image.sample(n)
    assert len(images) == n, f"{case}: {len(images)} images for N = {n}"
    _assert_sampled_members(image, images, f"sample of {case}")


# --------------------------------------------------------------------------- #
# Composition: MaxPool2d reads estimate_ranges                                  #
# --------------------------------------------------------------------------- #
# (C, H, W), window, stride and method of the MaxPool2d cells. The approx
# cells use the (2, 3, 5) and (2, 5, 3) images. The exact cells use images
# with four windows, so the split by argmax stays at 16 pieces or fewer: on
# the larger images every one of 12 windows splits and the membership check
# of thousands of pieces took minutes.
POOL_CASES = {
    "wide-approx": ((2, 3, 5), (2, 1), (1, 2), "approx"),
    "tall-approx": ((2, 5, 3), (2, 1), (1, 2), "approx"),
    "wide-exact": ((2, 2, 3), (1, 2), (2, 1), "exact"),
    "tall-exact": ((2, 3, 2), (2, 1), (1, 2), "exact"),
}


@pytest.mark.parametrize("case", list(POOL_CASES))
def test_maxpool_on_a_low_rank_image_star_holds_every_output(case):
    """MaxPool2d with an asymmetric window and stride on a two-channel image
    star with binding rows reads the pixel bounds of ``estimate_ranges`` to
    rule out candidates: every float64 output of the true layer at points of
    the set (vertices and midpoints, in channel-first order) is in a result
    of stored shape (C, H', W'), for an image wider than tall and one taller
    than wide, approx and exact. Catches pixel bounds stored in another
    order, so that the wrong candidates are ruled out, and a dropped argmax
    piece."""
    shape, kernel, stride, method = POOL_CASES[case]
    exact, image = low_rank_of_shape("rows", shape)
    C, H, W = shape
    pool = nn.MaxPool2d(kernel, stride=stride)
    out = reach_one_layer(pool, [image], method, f"maxpool {case}")
    q = chw_rows(shape)
    inputs = [p[q] for p in exact.points(extra=12)]
    ys = replay_torch(pool, inputs, (1, C, H, W))
    Ho = (H - kernel[0]) // stride[0] + 1
    Wo = (W - kernel[1]) // stride[1] + 1
    assert_covered(out, ys, f"maxpool {case}", out_chw=(C, Ho, Wo))


@pytest.mark.parametrize("entry", ["reach_model"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_approx_maxpool_on_a_low_rank_image_star_through_both_entries(
        shape_name, entry):
    """The approx MaxPool2d cell of
    :func:`test_maxpool_on_a_low_rank_image_star_holds_every_output` on the
    two images through ``reach_pytorch_model``, which hands the input image
    star on itself: every float64 output of the true layer at points of the
    set is in the result. Catches pixel bounds read in another order on the
    model path too. (The dispatcher entry of the same cell is
    ``test_maxpool_on_a_low_rank_image_star_holds_every_output[*-approx]``;
    its duplicate here was removed.)"""
    exact, image = low_rank("rows", shape_name)
    C, H, W = SHAPES[shape_name]
    pool = nn.MaxPool2d((2, 1), stride=(1, 2))
    if entry == "reach_layer":
        out = reach_one_layer(pool, [image], "approx", "maxpool")
    else:
        out = reach_model(nn.Sequential(pool), image, "approx", "maxpool")
    q = chw_rows((C, H, W))
    ys = replay_torch(pool, [p[q] for p in exact.points(extra=12)],
                      (1, C, H, W))
    assert_covered(out, ys, f"maxpool via {entry}",
                   out_chw=(C, H - 1, (W - 1) // 2 + 1))


def _flatten_linear(n_in, seed=4):
    rng = np.random.default_rng(seed)
    linear = nn.Linear(n_in, 4)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(
            rng.integers(-4, 5, size=(4, n_in)) / 4.0, dtype=torch.float32))
        linear.bias.copy_(torch.tensor(rng.integers(-8, 9, size=4) / 8.0,
                                       dtype=torch.float32))
    return nn.Sequential(nn.Flatten(), linear)


@pytest.mark.parametrize("entry", ["reach_layer", "reach_model"])
@pytest.mark.parametrize("system", list(_SYSTEMS))
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_flatten_then_linear_on_two_low_rank_image_stars_is_the_exact_image(
        shape_name, system, entry):
    """Flatten (``ImageStar.flatten_to_star``) then a dyadic Linear on two
    two-channel image stars with rows at once (the named system and a second,
    different one), through the dispatcher layer by layer and through
    ``reach_pytorch_model``: the results hold the exact image of every true
    vertex of both (the vertex taken in the network's channel, height, width
    order and mapped by the weight in rationals) with no slack, and float64
    outputs of the true model at points of both sets. Catches Flatten that
    walks the image in height, width, channel order or column by column,
    swapped predicate bounds on the way, and a second input set dropped or
    replaced by the first."""
    C, H, W = SHAPES[shape_name]
    other = "offset-box" if system != "offset-box" else "rows"
    pairs = [low_rank(system, shape_name), low_rank(other, shape_name, seed=5)]
    model = _flatten_linear(C * H * W)
    images = [image for _, image in pairs]
    if entry == "reach_layer":
        mid = reach_one_layer(model[0], images, "approx", "flatten")
        out = reach_one_layer(model[1], mid, "approx", "linear")
    else:
        out = reach_model(model, images, "approx", "flatten linear")
    q = chw_rows((C, H, W))
    Wq = [Q.vector(row) for row in model[1].weight.detach().double().numpy()]
    bq = Q.vector(model[1].bias.detach().double().numpy())
    targets, inputs = [], []
    for exact, _ in pairs:
        targets += [tuple(Q.dot(row, [v[i] for i in q]) + bi
                          for row, bi in zip(Wq, bq))
                    for v in exact.vertices()]
        inputs += [p[q] for p in exact.points(extra=8)]
    assert_holds_exact(out, targets, f"{system}: flatten then linear "
                                     f"via {entry}")
    ys = replay_torch(model, inputs, (1, C, H, W))
    assert_covered(out, ys, f"{system}: flatten then linear via {entry}")


def test_repr_and_image_shape_name_height_width_and_channels():
    """``get_image_shape`` is (height, width, channels) and ``repr`` states
    them in that order, for an image taller than wide; a swap of height and
    width here is the label a layout bug would print."""
    _, image = low_rank("rows", "tall")
    assert image.get_image_shape() == (5, 3, 2)
    assert repr(image) == "ImageStar(height=5, width=3, channels=2, nVar=3)"
