"""Bounds that are not moved outward: range queries, the cache they leave,
Softmax, the Hexatope and Octatope split bounds, a Star without a finite
predicate box, and the slope product of LeakyReLU.

Every defect here is a bound or a row that a computation by round-to-nearest,
or a solver stopped at its own tolerance, leaves on the inner side of the true
value. Most are one unit in the last place, far below any slack a test could
allow, so each cell compares with the exact value and allows nothing: ranges
and memberships are decided in rationals (``tests/_oracles/rational.py``) on
the stored floats, and the values of exp and of the sigmoid are enclosed at
100 bits by the interval arithmetic of ``tests/_oracles/interval_proof.py``.
No reference comes from n2v. Data that passes through an affine map is dyadic,
so that the map itself is exact and the cell sees only the defect it names
(the rounding of the affine map of a Star and of a Zono is a separate, decided
matter).

The sections, each with the registry rows it marks (``DEFECTS.md``):

1. The LP range queries of a Star and an ImageStar add the centre to the LP
   optimum by round-to-nearest, so a bound can lie inside the exact range
   (SET-5). The interval queries ``estimate_ranges`` and ``estimate_range``
   round outward and are the neighbours.
2. ``ImageStar.get_ranges`` stores those unpadded ranges in ``state_lb`` and
   ``state_ub``, and MaxPool2d reads them instead of computing its own
   (POOL-5); a false UNSAT behind a Linear layer with weight 2**60.
3. Softmax followed by an affine layer whose terms cancel: the unrounded
   Softmax bound gives a false UNSAT on every entry point (ROUND-6).
4. Hexatope and Octatope ReLU pieces whose bounds come from OSQP at 1e-7 with
   no pad (HEXOCT-4).
5. A Star without a finite predicate box: HiGHS drops the constraint
   coefficients at or below 1e-12 and nothing certifies the answer (SET-7), a
   range query hands the LP to the caller's solver with no certificate and no
   pad (SET-6), and the approx ReLU and LeakyReLU kernels return rows holding
   NaN or an infinite right-hand side for a predicate box of minus and plus
   infinity (REACH-5).
6. LeakyReLU multiplies the generator rows by the slope by round-to-nearest
   (ROUND-9).
7. The range queries of a Box, a Zono, an ImageZono, a Hexatope and an
   Octatope, on data where a sum computed to nearest lands inside the exact
   range: the Box returns its stored bounds, the Zono family rounds outward
   (the neighbours), the Hexatope and Octatope 'lp' queries take an OSQP
   optimum with no pad (HEXOCT-4) and their 'mcf' queries add the centre
   term to the flow cost to nearest.
8. The set arithmetic of a Zono and a Box other than the affine map: the
   Zono built from a box, the Zono order reduction and the Box sum round
   outward (``Zono.minkowski_sum`` is a site of the step-7b radius).

Expected failures: a known defect keeps one strict expected failure per
distinct failing code path, that is per place where the rounding (or the
missing certificate) happens and per caller in which a fix could be placed
on its own (each query method of each set type, each kernel of each set
type, each solver backend and predicate-box regime of the LP layer), with a
neighbour that passes today and uses the same reference. Families of data,
scales, solver names on one backend and entry routes that reach the same
kernel are not distinct paths.
"""

import itertools
from fractions import Fraction

import cvxpy
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, HalfSpace, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles import interval_proof as proof
from tests._oracles import rational
from tests.known_bugs import known_bug

F = Fraction
TINY = 2.0 ** -60


# --------------------------------------------------------------------------- #
# Shared exact helpers                                                          #
# --------------------------------------------------------------------------- #
def exact_ranges(s):
    """The exact least and greatest value of every entry of the set ``s``
    over its own rows, in rationals: a linear function on a polytope takes
    its extremes at the vertices."""
    points = rational.set_vertices(rational.star(s))
    n = len(points[0])
    return ([min(p[i] for p in points) for i in range(n)],
            [max(p[i] for p in points) for i in range(n)])


def _floats(values):
    """A flat list of the returned values, None kept as None."""
    out = []
    for v in np.asarray(values, dtype=object).reshape(-1):
        out.append(None if v is None else float(v))
    return out


def bounds_inside(exact, lo, hi):
    """The returned bounds that do not hold the exact range: a list of
    ``(entry, side, how far inside)``. A bound that is None or NaN holds
    nothing; an infinite bound on the right side holds everything."""
    elo, ehi = exact
    lo, hi = _floats(lo), _floats(hi)
    if len(lo) != len(elo) or len(hi) != len(ehi):
        pytest.fail(f"{len(lo)} lower and {len(hi)} upper bounds returned for "
                    f"{len(elo)} entries")
    found = []
    for i, (a, b) in enumerate(zip(lo, hi)):
        for side, got, end, sign in (("lower", a, elo[i], 1),
                                     ("upper", b, ehi[i], -1)):
            if got is None or np.isnan(got):
                found.append((i, side, "no value"))
            elif np.isinf(got):
                if (got > 0) != (side == "upper"):
                    found.append((i, side, got))
            elif sign * (F(got) - end) > 0:
                found.append((i, side, float(sign * (F(got) - end))))
    return found


def holders_of(sets):
    return [rational.Membership(rational.star(s)) for s in sets]


def held(holders, point):
    return any(holder(list(point)) for holder in holders)


def box_held(holders, low, high):
    """Is the box ``[low, high]`` (an enclosure of one true output) inside one
    of the sets? A convex set holds a box when it holds its vertices."""
    corners = list(itertools.product(*[sorted({a, b})
                                       for a, b in zip(low, high)]))
    return any(all(holder(list(c)) for c in corners) for holder in holders)


def is_unsat(sets, unsafe):
    return verify_specification(sets, unsafe).verdict == "UNSAT"


# --------------------------------------------------------------------------- #
# 1. The LP range queries of a Star and an ImageStar (SET-5)                    #
# --------------------------------------------------------------------------- #
def _one_ulp_star():
    """x = 1 + 2**-60 a, a in [-1, 1]: the exact range is 1 -+ 2**-60, and
    1.0 + 2**-60 is 1.0 in float64."""
    return [Star(np.array([[1.0, TINY]]), np.zeros((0, 1)), np.zeros((0, 1)),
                 np.array([[-1.0]]), np.array([[1.0]]))]


def _random_stars(row, centre=True, seed=20261111, count=6):
    """Stars of three entries over two predicate variables in [-1, 1], with
    centres from 1e-2 to 5e5 and generators from 1e-9 to 1, so the LP optimum
    is far smaller than the centre it is added to; with ``row`` also the
    binding row a1 + a2 <= 1. ``centre=False`` sets every centre to 0, where
    the sum is exact (the neighbour)."""
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        c = rng.uniform(-5, 5, 3) * 10.0 ** rng.integers(-2, 6, 3)
        G = rng.uniform(-1, 1, (3, 2)) * 10.0 ** rng.integers(-9, 1, (3, 1))
        if not centre:
            c = np.zeros(3)
        C = np.array([[1.0, 1.0]]) if row else np.zeros((0, 2))
        d = np.array([[1.0]]) if row else np.zeros((0, 1))
        stars.append(Star(np.hstack([c.reshape(-1, 1), G]), C, d,
                          -np.ones((2, 1)), np.ones((2, 1))))
    return stars


def _one_ulp_image():
    """A (H, W, C) = (1, 2, 2) ImageStar whose four pixels are 1, -3, 7 and
    0.5 plus 2**-60 times one predicate variable each."""
    V = np.zeros((1, 2, 2, 5))
    for k, (h, w, c, centre) in enumerate([(0, 0, 0, 1.0), (0, 0, 1, -3.0),
                                           (0, 1, 0, 7.0), (0, 1, 1, 0.5)]):
        V[h, w, c, 0] = centre
        V[h, w, c, 1 + k] = TINY
    return [ImageStar(V, np.zeros((0, 4)), np.zeros((0, 1)), -np.ones((4, 1)),
                      np.ones((4, 1)), 1, 2, 2)]


def _random_images(row, centre=True, seed=20261112, count=3):
    """ImageStars of (H, W, C) = (2, 3, 2) over two predicate variables, the
    scales of :func:`_random_stars`."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        c = rng.uniform(-5, 5, (2, 3, 2)) * 10.0 ** rng.integers(-2, 6, (2, 3, 2))
        G = (rng.uniform(-1, 1, (2, 3, 2, 2))
             * 10.0 ** rng.integers(-9, 1, (2, 3, 2, 1)))
        if not centre:
            c = np.zeros_like(c)
        C = np.array([[1.0, 1.0]]) if row else np.zeros((0, 2))
        d = np.array([[1.0]]) if row else np.zeros((0, 1))
        images.append(ImageStar(np.concatenate([c[..., None], G], axis=3), C, d,
                                -np.ones((2, 1)), np.ones((2, 1)), 2, 3, 2))
    return images


def _dyadic_stars(seed=20261115, count=4):
    """Stars of three entries over two predicate variables in [-1, 1] with
    the binding row a1 + a2 <= 1, centres multiples of 1/4 in [-8, 8] and
    generators multiples of 1/8 in [-1, 1]: every vertex value and every
    sum with a centre is a float, so a bound computed by round-to-nearest
    is still exact. The neighbour that passes with a centre that is not 0
    (a query that drops or misplaces the centre fails here)."""
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        c = rng.integers(-32, 33, 3) / 4.0
        G = rng.integers(-8, 9, (3, 2)) / 8.0
        stars.append(Star(np.hstack([c.reshape(-1, 1), G]),
                          np.array([[1.0, 1.0]]), np.array([[1.0]]),
                          -np.ones((2, 1)), np.ones((2, 1))))
    return stars


def _dyadic_images(seed=20261116, count=2):
    """ImageStars of (H, W, C) = (2, 3, 2) with the data of
    :func:`_dyadic_stars`."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        c = rng.integers(-32, 33, (2, 3, 2)) / 4.0
        G = rng.integers(-8, 9, (2, 3, 2, 2)) / 8.0
        images.append(ImageStar(np.concatenate([c[..., None], G], axis=3),
                                np.array([[1.0, 1.0]]), np.array([[1.0]]),
                                -np.ones((2, 1)), np.ones((2, 1)), 2, 3, 2))
    return images


def _point_star():
    """A Star with no predicate variable (the set is its centre), centres
    that differ per entry and are not dyadic: every query must return each
    entry's own centre (``nVar == 0`` branches)."""
    return [Star(np.array([[0.1], [-1 / 3], [7.25]]))]


def _point_image():
    """An ImageStar of (H, W, C) = (1, 2, 2) with no predicate variable."""
    V = np.array([0.1, -1 / 3, 7.25, 1e5 + 0.3]).reshape(1, 2, 2, 1)
    return [ImageStar(V, np.zeros((0, 0)), np.zeros((0, 1)), None, None,
                      1, 2, 2)]


_STAR_FAMILIES = {
    "one_ulp": _one_ulp_star,
    "random": lambda: _random_stars(row=False),
    "random_with_row": lambda: _random_stars(row=True),
    "centre_zero": lambda: _random_stars(row=True, centre=False),
    "dyadic": _dyadic_stars,
    "point": _point_star,
}
_IMAGE_FAMILIES = {
    "image_one_ulp": _one_ulp_image,
    "image_random": lambda: _random_images(row=False),
    "image_random_with_row": lambda: _random_images(row=True),
    "image_centre_zero": lambda: _random_images(row=True, centre=False),
    "image_dyadic": _dyadic_images,
    "image_point": _point_image,
}


def _box_query(s):
    box = s.get_box()
    return box.lb, box.ub


_STAR_QUERIES = {
    "get_ranges": lambda s: s.get_ranges(),
    "get_ranges_batch": lambda s: s.get_ranges(parallel=False),
    "get_ranges_parallel": lambda s: s.get_ranges(parallel=True, n_workers=2),
    "get_range": lambda s: tuple(zip(*[s.get_range(i) for i in range(s.dim)])),
    "get_min_get_max": lambda s: ([s.get_min(i) for i in range(s.dim)],
                                  [s.get_max(i) for i in range(s.dim)]),
    "get_box": _box_query,
    "estimate_ranges": lambda s: s.estimate_ranges(),
    "estimate_range": lambda s: tuple(zip(*[s.estimate_range(i)
                                            for i in range(s.dim)])),
}


def _image_get_range(s):
    """``get_range(h, w, c)`` in the stored (height, width, channel) order."""
    return tuple(zip(*[s.get_range(h, w, c) for h in range(s.height)
                       for w in range(s.width)
                       for c in range(s.num_channels)]))


_IMAGE_QUERIES = {
    "get_ranges": lambda s: s.get_ranges(),
    "get_ranges_batch": lambda s: s.get_ranges(parallel=False),
    "get_ranges_parallel": lambda s: s.get_ranges(parallel=True, n_workers=2),
    "get_range": _image_get_range,
    "get_range_flat": lambda s: tuple(zip(*[s.get_range_flat(i)
                                            for i in range(s.dim)])),
    "estimate_ranges": lambda s: s.estimate_ranges(),
    "estimate_range": lambda s: tuple(zip(*[s.estimate_range(i)
                                            for i in range(s.dim)])),
}

_CENTRE_ADDED_TO_NEAREST = known_bug(
    8, "the LP range queries of a Star and an ImageStar (get_ranges, "
       "get_range, get_min, get_max, get_box, get_range_flat) add the centre "
       "to the LP optimum by round-to-nearest with no outward step, so a "
       "returned bound can lie inside the exact range: centre 1 and "
       "generator 2**-60 give [1.0, 1.0]")


# The marked cells: one per line of library arithmetic that adds the centre
# to nearest, on the family where it shows on every IEEE machine (centre 1,
# generator 2**-60): Star._get_ranges_batch (get_ranges, and through it
# get_box and parallel=False), Star.get_range (and _get_ranges_parallel, which
# calls it), Star.get_min and Star.get_max, ImageStar._get_ranges_batch (a
# set of fewer than ``config.parallel_threshold`` entries takes it by
# default) and ImageStar.get_range (and get_range_flat and
# _get_ranges_parallel, which call it). The random families keep their
# interval queries only; their LP queries ran the same lines.
_MARKED_LP_CELLS = {"one_ulp": ("get_ranges", "get_range", "get_min_get_max"),
                    "image_one_ulp": ("get_ranges", "get_range")}


def _range_cells():
    cells = []
    for families, queries in ((_STAR_FAMILIES, _STAR_QUERIES),
                              (_IMAGE_FAMILIES, _IMAGE_QUERIES)):
        for family in families:
            for query in queries:
                sound = (query.startswith("estimate") or "centre_zero" in family
                         or "dyadic" in family or "point" in family)
                if not sound and query not in _MARKED_LP_CELLS.get(family, ()):
                    continue
                cells.append(pytest.param(
                    family, query,
                    marks=() if sound else _CENTRE_ADDED_TO_NEAREST,
                    id=f"{family}-{query}"))
    return cells


@pytest.mark.parametrize("family,query", _range_cells())
def test_a_range_query_holds_the_exact_range(family, query):
    """Catches a range query whose bound lies inside the exact range of its
    entry (registry SET-5: the centre added by round-to-nearest), and an
    ImageStar query that reads its pixels in the wrong order (the exact
    ranges are in the stored height, width, channel order), or that drops
    or misplaces the centre. Neighbours that pass today: the interval
    queries, which round outward; every query on stars whose centre is 0,
    where the sum is exact and the LP optimum is certified on the finite
    predicate box; and every query on the dyadic family, whose centres are
    not 0 and whose sums are exact; and every query on a set with no
    predicate variable, which must return each entry's own centre. The
    marked cells are the one-ulp family on each line that adds the centre
    (``_MARKED_LP_CELLS``)."""
    families = {**_STAR_FAMILIES, **_IMAGE_FAMILIES}
    queries = _STAR_QUERIES if family in _STAR_FAMILIES else _IMAGE_QUERIES
    found = []
    for k, s in enumerate(families[family]()):
        exact = exact_ranges(s)
        lo, hi = queries[query](s)
        found += [(k,) + miss for miss in bounds_inside(exact, lo, hi)]
    assert not found, (
        f"{query} on the {family} family: {len(found)} bounds inside the "
        f"exact range (set, entry, side, how far inside), the first "
        f"{found[:4]}")


def _without_a_box(s):
    """The same set with its predicate box written as constraint rows and no
    predicate box, so the interval queries take their LP path."""
    k = int(s.nVar)
    C = np.vstack([np.asarray(s.C, dtype=np.float64).reshape(-1, k),
                   np.eye(k), -np.eye(k)])
    d = np.vstack([np.asarray(s.d, dtype=np.float64).reshape(-1, 1),
                   np.asarray(s.predicate_ub).reshape(-1, 1),
                   -np.asarray(s.predicate_lb).reshape(-1, 1)])
    if isinstance(s, ImageStar):
        return ImageStar(s.V, C, d, None, None, s.height, s.width,
                         s.num_channels)
    return Star(s.V, C, d)


def _big_centre_stars(seed=20261117, count=4):
    """Stars of three entries over two predicate variables in [-1, 1] (the
    box as constraint rows) with centres near 2**40 and generators from 0.1
    to 1: one unit in the last place of the centre (2**-12) is far above
    the 1e-7 pad of the LP enclosure, so a centre added to the padded LP
    optimum by round-to-nearest lands inside the exact range about half the
    time."""
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        c = 2.0 ** 40 * rng.choice([-1.0, 1.0], 3) + rng.integers(-64, 64, 3)
        G = rng.uniform(0.1, 1.0, (3, 2)) * rng.choice([-1.0, 1.0], (3, 2))
        stars.append(Star(np.hstack([c.reshape(-1, 1), G]),
                          np.zeros((0, 2)), np.zeros((0, 1)),
                          -np.ones((2, 1)), np.ones((2, 1))))
    return stars


def _big_centre_images():
    """The data of :func:`_big_centre_stars` as (H, W, C) = (1, 3, 2)
    ImageStars, two stars to an image."""
    stars = _big_centre_stars()
    images = []
    for a, b in ((0, 1), (2, 3)):
        V = np.vstack([stars[a].V, stars[b].V]).reshape(1, 3, 2, 3)
        images.append(ImageStar(V, np.zeros((0, 2)), np.zeros((0, 1)),
                                -np.ones((2, 1)), np.ones((2, 1)), 1, 3, 2))
    return images


_NO_BOX_FAMILIES = {"big_centre": _big_centre_stars,
                    "image_big_centre": _big_centre_images}


@pytest.mark.parametrize("family,query", [
    (family, query)
    for family in ("one_ulp", "random_with_row", "dyadic", "big_centre",
                   "image_one_ulp", "image_random_with_row", "image_dyadic",
                   "image_big_centre")
    for query in ("estimate_ranges", "estimate_range")])
def test_an_interval_query_without_a_finite_box_holds_the_exact_range(
        family, query):
    """``estimate_ranges`` and ``estimate_range`` on a set with no predicate
    box take a padded HiGHS LP and then add the centre; the promise is the
    same as with a box: every bound holds the exact range of its entry
    (rationals, over the rows). Catches the centre added to the LP optimum
    by round-to-nearest (centre 1 and generator 2**-60), a pad applied in
    the wrong direction, an upper bound read from the minimisation, and an
    ImageStar that orders its pixels differently from the set it stores.
    The family with centres near 2**40 is the one where a round-to-nearest
    sum shows past the 1e-7 pad. Passes today."""
    families = {**_STAR_FAMILIES, **_IMAGE_FAMILIES, **_NO_BOX_FAMILIES}
    found = []
    for k, s in enumerate(families[family]()):
        s = _without_a_box(s)
        exact = exact_ranges(s)
        lo, hi = _STAR_QUERIES[query](s)
        found += [(k,) + miss for miss in bounds_inside(exact, lo, hi)]
    assert not found, (
        f"{query} on the {family} family without a predicate box: "
        f"{len(found)} bounds inside the exact range (set, entry, side, how "
        f"far inside), the first {found[:4]}")


# An entry whose LP has no optimum. x0 = 1/4 + a1 and x1 = 1/2 + 3/4 a2 over
# -1 <= a1 <= 1 and a2 >= 0, as rows only: the maximum of x1 is unbounded,
# so its LP gives no optimum and the query falls back to an enclosure of
# that entry (``Star._estimate_rows`` on the batch and the parallel path,
# ``ImageStar.estimate_range`` on the image batch path), while x0 keeps its
# LP range. The data are dyadic, so the LP range of x0 is exact. The
# ImageStar parallel path writes NaN here (registry SET-12, marked in
# ``test_soundness_image_star_sets.py``), and an infinite lower bound of x1
# is not exact (registry SET-42) but holds the range.
_NO_OPTIMUM_EXACT = ([F(-3, 4), F(1, 2)], [F(5, 4), float("inf")])


def _no_optimum_star():
    return Star(np.array([[0.25, 1.0, 0.0], [0.5, 0.0, 0.75]]),
                np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
                np.array([[1.0], [1.0], [0.0]]))


_NO_OPTIMUM_QUERIES = {
    "Star-get_ranges_batch": lambda: _no_optimum_star().get_ranges(
        parallel=False),
    "Star-get_ranges_parallel": lambda: _no_optimum_star().get_ranges(
        parallel=True, n_workers=2),
    "Star-get_box": lambda: _box_query(_no_optimum_star()),
    "ImageStar-get_ranges_batch": lambda: ImageStar(
        _no_optimum_star().V.reshape(1, 1, 2, 3), _no_optimum_star().C,
        _no_optimum_star().d, None, None, 1, 1, 2).get_ranges(parallel=False),
}


@pytest.mark.parametrize("query", list(_NO_OPTIMUM_QUERIES))
def test_an_entry_whose_lp_has_no_optimum_gets_an_enclosure(query):
    """Catches the fallback of a range query for an entry whose LP has no
    optimum (an unbounded direction) placed on another entry, given a
    finite bound where the range is unbounded, or written as NaN, and the
    entries that do have an optimum disturbed by it: every bound must hold
    the exact ranges [-3/4, 5/4] and [1/2, +inf) (by hand, rationals, zero
    slack). Passes today."""
    lo, hi = _NO_OPTIMUM_QUERIES[query]()
    found = bounds_inside(_NO_OPTIMUM_EXACT, lo, hi)
    assert not found, (
        f"{query}: bounds inside the exact range (entry, side, how far "
        f"inside): {found}")


# --------------------------------------------------------------------------- #
# 2. The cache ImageStar.get_ranges() leaves behind (POOL-5)                    #
# --------------------------------------------------------------------------- #
# An ImageStar of (H, W, C) = (2, 4, 2) over a1, a2 in [-1, 1]. Channel 0 holds
# 1 + 2**-60 a1 and 1 + 2**-60 a2 in the first 2x2 window, so get_ranges()
# gives both [1.0, 1.0] and the window's true maximum 1 + 2**-60 at
# a = (-1, 1) is above the cached upper bound; 10 + a1 in the second window
# pins a1 in the output. Channel 1 pins a2 (10 + a2) and holds two pixels that
# straddle zero, one of them with an upper end that get_ranges() rounds inward
# (0.5 + 1.5 a1 + 2**-60 a2). Channel 1 also holds two negative pixels whose
# ranges overlap, -2 + a1 / 2 and -1.75 + a2 / 4 at (0, 1) and (0, 2): with
# padding 1 they share a border window with two padded positions, so a pool
# that took a padded position as a candidate (value 0) would leave out the
# window's negative maximum; without padding each is dominated.
_IMAGE_SHAPE = (2, 4, 2)                        # height, width, channels


def _cache_image():
    V = np.zeros(_IMAGE_SHAPE + (3,))
    V[0, 0, 0] = [1.0, TINY, 0.0]
    V[0, 1, 0] = [1.0, 0.0, TINY]
    V[1, 2, 0] = [10.0, 1.0, 0.0]
    V[0, 0, 1] = [0.5, 1.5, TINY]
    V[0, 3, 1] = [10.0, 0.0, 1.0]
    V[1, 1, 1] = [-0.25, 0.5, 0.25]
    V[0, 1, 1] = [-2.0, 0.5, 0.0]
    V[0, 2, 1] = [-1.75, 0.0, 0.25]
    return ImageStar(V, np.zeros((0, 2)), np.zeros((0, 1)), -np.ones((2, 1)),
                     np.ones((2, 1)), *_IMAGE_SHAPE)


def _pixels(a):
    """The exact image at the predicate point ``a`` as a (C, H, W) nested
    list, in the network's order."""
    V = _cache_image().V
    height, width, channels = _IMAGE_SHAPE
    return [[[rational.frac(V[h, w, c, 0]) + rational.frac(V[h, w, c, 1]) * a[0]
              + rational.frac(V[h, w, c, 2]) * a[1]
              for w in range(width)] for h in range(height)]
            for c in range(channels)]


def _max_pool(x, padding):
    """2x2 max pooling with stride 2 of a (C, H, W) nested list; padding with
    -inf, as torch and ONNX pad, so a padded position is never the max."""
    out = []
    for channel in x:
        height, width = len(channel), len(channel[0])

        def at(h, w):
            h, w = h - padding, w - padding
            if 0 <= h < height and 0 <= w < width:
                return channel[h][w]
            return None
        rows = (height + 2 * padding) // 2
        cols = (width + 2 * padding) // 2
        out.append([[max(v for v in (at(2 * i, 2 * j), at(2 * i, 2 * j + 1),
                                     at(2 * i + 1, 2 * j),
                                     at(2 * i + 1, 2 * j + 1)) if v is not None)
                     for j in range(cols)] for i in range(rows)])
    return out


def _stored(x):
    """A (C, H, W) nested list as the flat vector an image set stores: height,
    width, channel order."""
    return [x[c][h][w] for h in range(len(x[0])) for w in range(len(x[0][0]))
            for c in range(len(x))]


_CACHE_LAYERS = {
    # name: (layer, the exact function on a (C, H, W) nested list)
    "MaxPool2d": (lambda: nn.MaxPool2d(2), lambda x: _max_pool(x, 0)),
    "MaxPool2d_padding_1": (lambda: nn.MaxPool2d(2, padding=1),
                            lambda x: _max_pool(x, 1)),
    "ReLU": (nn.ReLU, lambda x: [[[max(v, F(0)) for v in r] for r in ch]
                                 for ch in x]),
    "LeakyReLU": (lambda: nn.LeakyReLU(0.125),
                  lambda x: [[[v if v >= 0 else v / 8 for v in r] for r in ch]
                             for ch in x]),
}
# Predicate points: the quarter grid of [-1, 1]^2, corners included.
_GRID = [(F(i, 2), F(j, 2)) for i in range(-2, 3) for j in range(-2, 3)]

_MAXPOOL_READS_THE_CACHE = known_bug(
    7, "MaxPool2d reads the ranges ImageStar.get_ranges() cached in state_lb "
       "and state_ub, which are not rounded outward (the centre is added by "
       "round-to-nearest), instead of computing outward ranges of its own: "
       "after a get_ranges() call a window maximum one unit in the last place "
       "above the cached bound is left out")


def _cache_cells():
    cells = []
    for name in _CACHE_LAYERS:
        for method in ("approx", "exact"):
            for cached in (False, True):
                marked = cached and name == "MaxPool2d"
                cells.append(pytest.param(
                    name, method, cached,
                    marks=_MAXPOOL_READS_THE_CACHE if marked else (),
                    id=f"{name}-{method}-{'after' if cached else 'without'}"
                       f"-get_ranges"))
    return cells


@pytest.mark.parametrize("name,method,cached", _cache_cells())
def test_a_kernel_after_get_ranges_holds_every_output(name, method, cached):
    """Catches MaxPool2d taking the input's ranges from the cache an earlier
    ``ImageStar.get_ranges()`` call left (registry POOL-5, without padding,
    approx and exact), a result stored in the wrong (height, width, channel)
    order, and a window or a neuron whose true value is left out (padding
    read as a candidate, a ReLU or LeakyReLU relaxation row that cuts the
    graph, a dropped exact piece). True outputs are exact at the 25
    predicate points of the quarter grid. Neighbours that pass today: the
    same calls without the earlier ``get_ranges()``; MaxPool2d with padding,
    which pools a padded copy that has no cache; ReLU and LeakyReLU. For
    these two a kernel that read the cache would not be seen (planted and
    run: nothing is left out), because their relaxation slope is moved
    outward by 2**-50, far more than the cache's error, and a pixel whose
    range touches zero has a generator as large as its centre, so its cached
    end is not rounded inward. By code reading only MaxPool2d
    (``_window_bounds_4d``) and ``Star.sample`` read the cache."""
    make_layer, function = _CACHE_LAYERS[name]
    image = _cache_image()
    if cached:
        image.get_ranges()
        if image.state_ub is None:
            pytest.fail("get_ranges() left no cache; the cell tests nothing")
    sets = reach_layer(make_layer(), [image], method)
    if not sets:
        pytest.fail(f"{name} {method}: no reach sets; an empty result is not "
                    f"the defect the marker names")
    holders = holders_of(sets)
    missed = [a for a in _GRID
              if not held(holders, _stored(function(_pixels(a))))]
    assert not missed, (
        f"{name} {method}, {'after' if cached else 'without'} get_ranges(): "
        f"the true output at the predicate points {[tuple(map(float, a)) for a in missed]} "
        f"is in none of the {len(sets)} returned sets")


def _pool_then_linear():
    """MaxPool2d(2), Flatten, then y = (2**60 m0 - 2**60, m1 - 10) where m0 is
    the first window of channel 0 and m1 the second. The pooled tensor is
    (C, H, W) = (2, 1, 2), flattened in that order to four entries."""
    linear = nn.Linear(4, 2)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = 2.0 ** 60
        linear.weight[1, 1] = 1.0
        linear.bias.copy_(torch.tensor([-(2.0 ** 60), -10.0]))
    return nn.Sequential(nn.MaxPool2d(2), nn.Flatten(), linear)


def _pool_then_gemm_onnx(tmp_path):
    """The network of :func:`_pool_then_linear` as an ONNX graph (MaxPool,
    Flatten, Gemm; every weight is a float32 number), and its weight and
    bias as stored."""
    W = np.zeros((2, 4), dtype=np.float32)
    W[0, 0], W[1, 1] = 2.0 ** 60, 1.0
    B = np.array([-(2.0 ** 60), -10.0], dtype=np.float32)
    nodes = [helper.make_node("MaxPool", ["X"], ["P"], kernel_shape=[2, 2],
                              strides=[2, 2]),
             helper.make_node("Flatten", ["P"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W", "B"], ["Y"], transB=1)]
    graph = helper.make_graph(
        nodes, "pool_gemm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 2, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        initializer=[numpy_helper.from_array(W, "W"),
                     numpy_helper.from_array(B, "B")])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = str(tmp_path / "pool_gemm.onnx")
    onnx.save(model, path)
    return path, W, B


# One marked cell per kernel that reads the cache (approx and exact
# MaxPool2d) on the model entry; NeuralNetwork.reach and an ONNX graph loaded
# with load_onnx (which builds the same nn.MaxPool2d) reach the same kernel,
# so they keep only the neighbour without the earlier get_ranges() call.
@pytest.mark.parametrize("entry,method,cached", [
    pytest.param(entry, method, cached,
                 marks=_MAXPOOL_READS_THE_CACHE if cached else (),
                 id=f"{entry}-{method}-{'after' if cached else 'without'}"
                    f"-get_ranges")
    for entry in ("reach_pytorch_model", "NeuralNetwork", "onnx")
    for method in ("approx", "exact") for cached in (False, True)
    if entry == "reach_pytorch_model" or not cached])
def test_no_unsat_after_get_ranges_on_a_pooled_image(tmp_path, entry, method,
                                                     cached):
    """The verdict behind POOL-5. At a = (-1, 1) the exact output of the
    network with its stored weights is y = (1, -1), in the unsafe region
    -y0 + y1 <= -1.5; the window maximum 1 + 2**-60 that the cached bound
    leaves out is multiplied by 2**60. Without the earlier ``get_ranges()``
    the answer is UNKNOWN today, the neighbour, on ``reach_pytorch_model``,
    ``NeuralNetwork.reach`` and the ONNX graph loaded with ``load_onnx``.
    Catches a false UNSAT from a pooled bound or a Flatten or Gemm that
    drops the pooled maximum on any of the three entries."""
    if entry == "onnx":
        path, W_stored, b_stored = _pool_then_gemm_onnx(tmp_path)
        model = load_onnx(path)
        W, b = rational.matrix(W_stored), rational.vector(b_stored)
    else:
        model = _pool_then_linear()
        W = rational.matrix(model[2].weight.detach().numpy())
        b = rational.vector(model[2].bias.detach().numpy())
    pooled = _max_pool(_pixels((F(-1), F(1))), 0)
    flat = [v for channel in pooled for row in channel for v in row]
    y = [rational.dot(row, flat) + bi for row, bi in zip(W, b)]
    if not -y[0] + y[1] <= F(-3, 2):
        pytest.fail(f"the exact output {y} is not in the unsafe region")
    image = _cache_image()
    if cached:
        image.get_ranges()
    if entry == "NeuralNetwork":
        sets = NeuralNetwork(model).reach(image, method=method)
    else:
        sets = reach_pytorch_model(model, image, method=method)
    unsafe = HalfSpace(np.array([[-1.0, 1.0]]), np.array([[-1.5]]))
    assert not is_unsat(sets, unsafe), (
        f"UNSAT, and the exact output at a = (-1, 1) is "
        f"{[float(v) for v in y]}, inside the unsafe region -y0 + y1 <= -1.5")


# --------------------------------------------------------------------------- #
# 3. Softmax, then an affine layer whose terms cancel (ROUND-6)                 #
# --------------------------------------------------------------------------- #
# y = w p0 - w, with w the float32 the network stores, so y = -w (1 - p0)
# exactly. With the logits x0 in [40, 80] and x1 = 0 the least value of p0 is
# 1 / (1 + e**-40) = 1 - 4.2e-18, which float64 rounds to 1.0: a Softmax bound
# that is not moved outward gives y the range [0, 0], while the exact output
# at the member point (40, 0) is -0.4248, inside the unsafe region y <= -0.25.
# The neighbour has logits x0 in [4, 8], where p0 is far from 1, and w = 1024;
# its member point (5, 0) is inside the logit box and not its centre, with
# the exact output -6.853 in the unsafe region y <= -2, far from any bound, so
# it passes on every machine, and a Softmax that answers the value at the
# centre of the logit box (y = -2.532) instead of a set leaves it out. (At the vertex (4, 0) the lower bound of p0 that the Zono and
# Star routes return is one unit in the last place above 1 / (1 + e**-4) on
# this machine, which w = 1024 carries to 1.1e-14: the same defect, but
# whether it shows hangs on the last bit of this machine's exp, so it is not
# a cell.)
_SOFTMAX_VERDICT_CASES = {
    # name: (lower logits, upper logits, w, member point, unsafe y <= this)
    "saturated": ([40.0, 0.0], [80.0, 0.0], 1e17, (40, 0), -0.25),
    "unsaturated": ([4.0, 0.0], [8.0, 0.0], 1024.0, (5, 0), -2.0),
}


def _softmax_then_linear(w):
    """Softmax over two logits, then y = w p0 - w, stored in float32."""
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[w, 0.0]]))
        linear.bias.copy_(torch.tensor([-w]))
    return nn.Sequential(nn.Softmax(dim=-1), linear)


def _softmax_then_gemm_onnx(tmp_path, w, opset):
    """The same network as an ONNX graph: Softmax (axis 1 at opset 11, -1 at
    opset 13), then Gemm with float32 initializers."""
    nodes = [helper.make_node("Softmax", ["X"], ["S"],
                              axis=1 if opset < 13 else -1),
             helper.make_node("Gemm", ["S", "W", "B"], ["Y"], transB=1)]
    graph = helper.make_graph(
        nodes, "softmax_gemm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(np.array([[w, 0.0]], dtype=np.float32), "W"),
            numpy_helper.from_array(np.array([-w], dtype=np.float32), "B")])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = str(tmp_path / f"softmax_gemm_{opset}.onnx")
    onnx.save(model, path)
    return path


def _exact_softmax_output(w, b, point):
    """An enclosure, in rationals, of w p0 + b at the logits ``point``, with
    p0 = 1 / (1 + exp(x1 - x0)) at 100 bits."""
    def f(xs):
        e = proof.IV.exp(xs[1] - xs[0])
        return [proof.IV.mpf(w.numerator) / w.denominator / (1 + e)
                + proof.IV.mpf(b.numerator) / b.denominator]
    low, high = proof.value(proof.chain(proof.function(f)),
                            [F(point[0]), F(point[1])])
    return low[0], high[0]


def _softmax_verdict_cells():
    """Every route and set type with the unsaturated neighbour; the saturated
    case once per kernel that computes the bound: ``softmax_box`` and
    ``softmax_star`` (``reach_layer``) and ``softmax_attn_box``, ``_zono``
    and ``_star`` (every model entry, here ``reach_pytorch_model`` of the
    ``nn.Sequential``; ``NeuralNetwork.reach`` and the ONNX graphs at opsets
    13 and 11 reach the same three). ``reach_layer`` refuses a Zono."""
    cells = []
    for route in ("reach_layer", "Sequential", "NeuralNetwork", "onnx13",
                  "onnx11"):
        for kind in ("Box", "Zono", "Star"):
            for case in _SOFTMAX_VERDICT_CASES:
                refused = route == "reach_layer" and kind == "Zono"
                marked = case != "unsaturated" and not refused
                if marked and route not in ("reach_layer", "Sequential"):
                    continue
                cells.append(pytest.param(
                    route, kind, case,
                    marks=_SOFTMAX_UNROUNDED if marked else (),
                    id=f"{route}-{kind}-{case}"))
    return cells


_SOFTMAX_UNROUNDED = known_bug(
    7, "the Softmax bounds are exp and division by round-to-nearest with no "
       "outward step (softmax_reach._softmax_bounds for reach_layer, "
       "correlated_row_softmax_bounds for every model reach of a Softmax "
       "node): on saturated logits the least value 1 - 4.2e-18 of p0 becomes "
       "1.0, so y = w p0 - w gets the range [0, 0] and the verdict is a false "
       "UNSAT although the exact output at (40, 0) is -0.4248")


@pytest.mark.parametrize("route,kind,case", _softmax_verdict_cells())
def test_softmax_then_cancelling_linear_is_not_unsat(tmp_path, route, kind,
                                                     case):
    """The verdict behind registry ROUND-6: Softmax
    over two logits, then y = w p0 - w with float32 weights, unsafe region
    y <= t. The exact output at a member point (an enclosure at 100 bits, of
    the network with its stored weights) is checked to be in the unsafe
    region first; then it must be in a returned set (decided in rationals)
    and the verdict must not be UNSAT. Entry points: ``reach_layer`` twice,
    ``reach_pytorch_model`` of the ``nn.Sequential`` and of the ONNX graph at
    opsets 13 and 11, and ``NeuralNetwork.reach``. A refusal
    (``NotImplementedError``, today ``reach_layer`` on a Zono) keeps the
    promise. Catches Softmax bounds rounded inward, a bound placed on the
    wrong entry, and a Softmax that answers a point where the input has
    width."""
    lo, hi, w, point, threshold = _SOFTMAX_VERDICT_CASES[case]
    if route.startswith("onnx"):
        path = _softmax_then_gemm_onnx(tmp_path, w, int(route[4:]))
        stored = onnx.load(path).graph.initializer
        weights = {t.name: numpy_helper.to_array(t) for t in stored}
        w_q, b_q = F(float(weights["W"][0, 0])), F(float(weights["B"][0]))
        model = load_onnx(path)
    else:
        model = _softmax_then_linear(w)
        w_q = F(float(model[1].weight.detach().numpy()[0, 0]))
        b_q = F(float(model[1].bias.detach().numpy()[0]))
    y_low, y_high = _exact_softmax_output(w_q, b_q, point)
    if not y_high <= F(threshold):
        pytest.fail(f"the exact output [{float(y_low)}, {float(y_high)}] at "
                    f"{point} is not in the unsafe region y <= {threshold}")
    lo_a, hi_a = np.array(lo).reshape(-1, 1), np.array(hi).reshape(-1, 1)
    input_set = (Box(lo_a, hi_a) if kind == "Box"
                 else {"Zono": Zono, "Star": Star}[kind].from_bounds(lo_a, hi_a))
    try:
        if route == "reach_layer":
            sets = reach_layer(model[1], reach_layer(model[0], [input_set],
                                                     "approx"), "approx")
        elif route == "NeuralNetwork":
            sets = NeuralNetwork(model).reach(input_set, method="approx")
        else:
            sets = reach_pytorch_model(model, input_set, method="approx")
    except NotImplementedError:
        return
    if not sets:
        pytest.fail(f"Softmax then Linear on {kind} through {route}: no sets")
    assert box_held(holders_of(sets), [y_low], [y_high]), (
        f"{route} on {kind}, {case}: the exact output "
        f"{float(y_high)!r} at {point} is in none of the {len(sets)} sets")
    unsafe = HalfSpace(np.array([[1.0]]), np.array([[threshold]]))
    assert not is_unsat(sets, unsafe), (
        f"{route} on {kind}, {case}: UNSAT, and the exact output "
        f"{float(y_high)!r} at {point} is in the unsafe region "
        f"y <= {threshold}")


# --------------------------------------------------------------------------- #
# 4. Hexatope and Octatope ReLU pieces (HEXOCT-4)                               #
# --------------------------------------------------------------------------- #
# Linear(3, 3) then ReLU on ``from_bounds`` of a box in three dimensions: a
# random layer (seed 1), its weights and bias rounded to
# multiples of 2**-10 so that the exact image in rationals is the network's
# own. The ReLU splits each input into pieces whose new constraints come from
# LPs solved by OSQP at 1e-7 and stored with no pad: a vertex of the exact
# image lies outside every piece, by 2.4e-4 to 5.0e-4 at half-width 1e4 (3e-8
# to 1.8e-7 on the unit box). One marked cell per set type, since each has its
# own split (``Hexatope._dcs_bounding_box`` and ``_optimize_with_constraint``,
# ``Octatope._utvpi_bounding_box``); the unit box, the option ``solver='lp'``
# (``intersect_half_space`` ignores it) and ReLU alone on a Hexatope box whose
# centre is not 0 run the same split and were removed as excess expected
# failures. ReLU alone on ``from_bounds`` of a box is the
# neighbour: on a symmetric box, and on an Octatope box whose centre is not 0;
# the crossing points of the asymmetric boxes (-c / h per entry) are dyadic,
# so the direct branch of the Octatope split divides exactly (a crossing
# point that is not a float64 number is ROUND-26, marked in
# ``test_soundness_hexoct_relu.py``). Membership is decided in rationals on
# the rows that ``membership.rows`` reads from each piece.
_HEXOCT_SCALES = {"unit": (1.0, 0.3), "half_width_1e4": (1e4, 300.0)}
# Boxes whose centre is not 0, for ReLU alone, with crossing points -1/4,
# -1/2, 1/2 (unit) and 1/4, -1/4, 1/2 (half-width 1e4) in kernel units.
_HEXOCT_ASYMMETRIC = {"unit": ([-0.75, -0.5, -1.5], [1.25, 1.5, 0.5]),
                      "half_width_1e4": ([-1e4, -6000.0, -7500.0],
                                         [6000.0, 1e4, 2500.0])}


def _hexoct_net(scale, seed=1):
    """(lb, ub, W, b) with W and b multiples of 2**-10."""
    half, offset = _HEXOCT_SCALES[scale]
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1, 1, (3, 3))
    b = rng.uniform(-1, 1, 3) * offset
    W = np.round(W * 2 ** 10) / 2 ** 10
    b = np.round(b * 2 ** 10) / 2 ** 10
    return -np.full(3, half), np.full(3, half), W, b


_HEXOCT_UNPADDED = known_bug(
    9, "the Hexatope and Octatope ReLU split stores the optima of LPs solved "
       "by CVXPY OSQP at a hard-coded 1e-7 (behind a bare except) as bounds "
       "of the new constraints, with no pad: a vertex of the exact image of "
       "Linear then ReLU is in no returned piece (2.4e-4 to 5.0e-4 out at "
       "half-width 1e4, 3.1e-8 to 1.8e-7 on the unit box)")


def _hexoct_cells():
    cells = []
    for cls in ("Hexatope", "Octatope"):
        cells.append(pytest.param(
            cls, "half_width_1e4", "linear_relu", "default",
            marks=_HEXOCT_UNPADDED,
            id=f"{cls}-half_width_1e4-linear_relu-default"))
        for scale in _HEXOCT_SCALES:
            if cls == "Octatope":
                cells.append(pytest.param(
                    cls, scale, "relu_asymmetric", "default",
                    id=f"{cls}-{scale}-relu_asymmetric-default"))
            cells.append(pytest.param(cls, scale, "relu", "default",
                                      id=f"{cls}-{scale}-relu-default"))
    return cells


def _box_outside_the_union(holders, low, high, depth):
    """A point of the box ``[low, high]`` that is in none of the convex sets
    of ``holders``, or None when the box is inside their union. A box whose
    corners are all in one set is inside it. Otherwise it is cut in two
    across its widest side, at most ``depth`` times along a path; a box that
    still straddles two sets at the end counts as not shown inside (its
    centre is returned with the word 'undecided')."""
    corners = list(itertools.product(*[sorted({a, b})
                                       for a, b in zip(low, high)]))
    inside = [[holder(list(c)) for holder in holders] for c in corners]
    for corner, row in zip(corners, inside):
        if not any(row):
            return corner
    if any(all(row[j] for row in inside) for j in range(len(holders))):
        return None
    if depth == 0:
        return ("undecided",) + tuple((a + b) / 2 for a, b in zip(low, high))
    axis = max(range(len(low)), key=lambda i: high[i] - low[i])
    middle = (low[axis] + high[axis]) / 2
    for part_low, part_high in ((low, high[:axis] + [middle] + high[axis + 1:]),
                                (low[:axis] + [middle] + low[axis + 1:], high)):
        found = _box_outside_the_union(holders, part_low, part_high,
                                       depth - 1)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("cls,scale,case,solver", _hexoct_cells())
def test_hexoct_relu_pieces_hold_the_exact_image(cls, scale, case, solver):
    """Catches a ReLU split piece of a Hexatope or an Octatope whose
    constraint bound is a solver optimum not moved outward (registry
    HEXOCT-4), a piece dropped, and a split on the wrong side or about the
    wrong point. Linear then ReLU: every vertex of the exact image
    (rationals, so the points on each neuron's zero crossing are among them)
    must be a member of some returned piece, with no slack. ReLU alone: the
    image of each exact piece is a box, and each must lie inside the union of
    the returned pieces, decided piece by piece in rationals: a box whose
    corners are all in one returned piece is inside it, and a box that
    straddles two is cut in two until each part is (a split threshold moved
    off the crossing point by any amount leaves a part with a corner in no
    piece); on a Hexatope, whose split leaves a sliver of 1e-26 out today
    (HEXOCT-4, marked by the linear_relu cell), the corners, the middle and
    the points at 2**-10 of each range from the crossing point. Neighbours
    that pass today: ReLU alone on a symmetric box, and on an Octatope box
    whose centre is not 0."""
    lb, ub, W, b = _hexoct_net(scale)
    if case == "relu_asymmetric":
        lb, ub = map(np.array, _HEXOCT_ASYMMETRIC[scale])
    input_set = {"Hexatope": Hexatope, "Octatope": Octatope}[cls].from_bounds(
        lb.reshape(-1, 1), ub.reshape(-1, 1))
    options = {} if solver == "default" else {"solver": solver}
    if case == "linear_relu":
        linear = nn.Linear(3, 3).double()
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(W))
            linear.bias.copy_(torch.tensor(b))
        sets = reach_layer(linear, [input_set], "approx")
        pieces = rational.relu(rational.affine(rational.box(lb, ub), W, b))
    else:
        sets = [input_set]
        pieces = rational.relu(rational.box(lb, ub))
    sets = reach_layer(nn.ReLU(), sets, "approx", **options)
    if not sets:
        pytest.fail(f"{cls} {case}: no pieces returned")
    holders = holders_of(sets)
    if case == "linear_relu":
        outputs = sorted({y for piece in pieces for y in piece.image()})
        missed = next((y for y in outputs if not held(holders, y)), None)
        assert missed is None, (
            f"{cls} {scale} {case} solver={solver}: the true output "
            f"{[float(v) for v in missed]} is in none of the {len(sets)} "
            f"pieces (the search stops at the first of {len(outputs)})")
        return
    for piece in pieces:
        image = piece.image()
        low = [min(y[i] for y in image) for i in range(3)]
        high = [max(y[i] for y in image) for i in range(3)]
        if cls == "Octatope":
            missed = _box_outside_the_union(holders, low, high, depth=12)
        else:
            # The Hexatope split leaves out a sliver next to each crossing
            # point today (the OSQP bound of HEXOCT-4: y2 >= 1.0e-26 in the
            # active piece of the box of half-width 1e4, marked by the
            # linear_relu cell), so the neighbour checks the points at
            # 2**-10 of the range from each crossing point and the middle:
            # a split threshold off by more than that leaves one out.
            axes = [sorted({lo, hi, lo + (hi - lo) / 1024, (lo + hi) / 2})
                    for lo, hi in zip(low, high)]
            missed = next((y for y in itertools.product(*axes)
                           if not held(holders, y)), None)
        assert missed is None, (
            f"{cls} {scale} {case}: the exact image "
            f"{[float(v) for v in low]}..{[float(v) for v in high]} of one "
            f"piece is not inside the {len(sets)} returned pieces: "
            f"{missed[0] if missed[0] == 'undecided' else ''} "
            f"{[float(v) for v in missed if v != 'undecided']}")


# --------------------------------------------------------------------------- #
# 5. A Star without a finite predicate box (SET-7, SET-6, REACH-5)              #
# --------------------------------------------------------------------------- #
# 5a. A coefficient at or below HiGHS's small-coefficient threshold (SET-7).
# x = 2**13 a0 over a0 - 2**-41 a1 <= 0, -a0 <= 0, -2**30 <= a1 <= 2**30 (all
# four as constraint rows): a0 ranges over [0, 2**-11], so x over [0, 4]. The
# entry 2**-41 (4.5e-13) is dropped by HiGHS, which then sees a0 <= 0. The
# dyadic form of the Star first measured (coefficient 5e-13, range
# [0, 5]): every end here is a float, so an LP that keeps the row gets 4.0
# exactly and the cells see only the dropped coefficient. Neighbours that pass
# today: the same rows with a finite predicate box (the certificate takes
# over), and row 0 multiplied by 2**20, the same set with no small entry.
_TINY_ROW_V = np.array([[0.0, 2.0 ** 13, 0.0]])
_TINY_ROW_C = np.array([[1.0, -2.0 ** -41], [-1.0, 0.0], [0.0, 1.0],
                        [0.0, -1.0]])
_TINY_ROW_D = np.array([[0.0], [0.0], [2.0 ** 30], [2.0 ** 30]])
# A member point and its output: a = (2**-11, 2**30) meets row 0 with
# equality, and x = 4.
_TINY_ROW_POINT = (F(1, 2 ** 11), F(2 ** 30))


def _tiny_row_star(variant):
    C, d = _TINY_ROW_C.copy(), _TINY_ROW_D.copy()
    if variant == "lifted":
        C[0] *= 2.0 ** 20
        d[0] *= 2.0 ** 20
    if variant in ("none", "lifted"):
        return Star(_TINY_ROW_V.copy(), C, d)
    if variant == "infinite":
        return Star(_TINY_ROW_V.copy(), C, d, np.full((2, 1), -np.inf),
                    np.full((2, 1), np.inf))
    return Star(_TINY_ROW_V.copy(), C, d, np.array([[0.0], [-2.0 ** 30]]),
                np.array([[2.0 ** -10], [2.0 ** 30]]))


_TINY_COEFFICIENT_DROPPED = known_bug(
    17, "HiGHS drops constraint coefficients at or below 1e-12 and a Star "
        "without a finite predicate box has no certificate that would catch "
        "it: the range LPs, ensure_finite_predicate_box and the "
        "disjointness LP of verify_specification work on a larger cut "
        "(a0 <= 0 for a0 <= 2**-41 a1), so x = 2**13 a0 gets the range "
        "[0, 0] for the exact [0, 4], true outputs are left out and the "
        "verdict is a false UNSAT")

_TINY_ROW_QUERIES = {
    "get_ranges": lambda s: s.get_ranges(),
    "get_range": lambda s: s.get_range(0),
    "get_min_get_max": lambda s: ([s.get_min(0)], [s.get_max(0)]),
    "get_box": _box_query,
    "estimate_ranges": lambda s: s.estimate_ranges(),
}


def _tiny_row_marks(variant, query):
    """SET-7 on a Star whose predicate box is absent or infinite: one marked
    cell per path that solves the bound LPs, the batch of ``get_ranges``
    (which ``get_range`` and ``get_box`` reach as well, so their cells keep
    only the neighbours) and the single LPs of ``get_min`` and ``get_max``,
    for each of the two kinds of missing box; and the interval query of a
    Star with no box, which takes its LP box from
    ``ensure_finite_predicate_box`` (an infinite box gives infinite
    interval bounds, which hold the range: a neighbour)."""
    if variant not in ("none", "infinite"):
        return ()
    if query in ("get_ranges", "get_min_get_max"):
        return _TINY_COEFFICIENT_DROPPED
    if query == "estimate_ranges":
        return _TINY_COEFFICIENT_DROPPED if variant == "none" else ()
    return None


@pytest.mark.parametrize("variant,query", [
    pytest.param(v, q, marks=_tiny_row_marks(v, q), id=f"{v}-{q}")
    for v in ("none", "infinite", "finite", "lifted")
    for q in _TINY_ROW_QUERIES if _tiny_row_marks(v, q) is not None])
def test_a_range_query_keeps_a_tiny_constraint_coefficient(variant, query):
    """Catches a range query that answers for a polytope with a constraint
    coefficient dropped (registry SET-7, direct calls): the exact range of
    x over the stored rows is [0, 4] (rationals), and every returned bound
    must hold it. A refusal (``NotImplementedError`` or ``ValueError``)
    keeps the promise too. ``estimate_ranges`` of the Star with no box
    returns [-8.2e-4, 8.2e-4] today. Neighbours: a finite predicate box, and
    the row rescaled by 2**20."""
    s = _tiny_row_star(variant)
    exact = exact_ranges(s)
    if exact != ([F(0)], [F(4)]):
        pytest.fail(f"the exact range is {exact}, not [0, 4]")
    try:
        lo, hi = _TINY_ROW_QUERIES[query](s)
    except (NotImplementedError, ValueError):
        return
    found = bounds_inside(exact, lo, hi)
    assert not found, (
        f"{query} on the Star with predicate box {variant}: bounds inside the "
        f"exact range [0, 4]: {found}")


def _identity_linear():
    linear = nn.Linear(1, 1)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(0.0)
    return linear


_TINY_ROW_LAYERS = {
    "Linear": _identity_linear,
    "ReLU": nn.ReLU,
    "LeakyReLU": lambda: nn.LeakyReLU(0.125),
}


# The marked cells: one per place that answers on the dropped coefficient.
# Through reach_layer, the approx ReLU and LeakyReLU kernels each take their
# ranges from such an LP, and Linear (approx or exact, the same affine map)
# and the exact ReLU and LeakyReLU hold the true output and leave the false
# UNSAT to the disjointness LP of verify_specification, so Linear stands for
# all three; through the model entry ``ensure_finite_predicate_box`` makes
# the box too narrow before any layer runs, so one layer stands for all.
# NeuralNetwork.reach calls reach_pytorch_model with the same arguments; its
# cells were removed as duplicates of the reach_pytorch_model
# ones.
_TINY_ROW_MARKED = {("reach_layer", "Linear", "approx"),
                    ("reach_layer", "ReLU", "approx"),
                    ("reach_layer", "LeakyReLU", "approx"),
                    ("reach_pytorch_model", "Linear", "approx")}


def _tiny_row_cells():
    cells = []
    for entry in ("reach_layer", "reach_pytorch_model"):
        for layer in _TINY_ROW_LAYERS:
            for method in ("approx", "exact"):
                for variant in ("none", "finite", "lifted"):
                    marked = variant == "none"
                    if marked and (entry, layer, method) not in _TINY_ROW_MARKED:
                        continue
                    cells.append(pytest.param(
                        entry, layer, method, variant,
                        marks=_TINY_COEFFICIENT_DROPPED if marked else (),
                        id=f"{entry}-{layer}-{method}-{variant}"))
    return cells


@pytest.mark.parametrize("entry,layer,method,variant", _tiny_row_cells())
def test_a_star_with_a_tiny_coefficient_is_not_unsat(entry, layer, method,
                                                     variant):
    """The verdict behind registry SET-7. At the member point
    a = (2**-11, 2**30) the
    input is x = 4 and the layer (identity Linear, ReLU, LeakyReLU 0.125)
    gives 4, in the unsafe region y >= 3: it must be in a returned set
    (rationals) and the verdict must not be UNSAT. A refusal keeps the
    promise. Through ``reach_layer`` and the model entry point, where
    ``ensure_finite_predicate_box`` first gives the Star a box. Neighbours:
    a finite predicate box, and the row rescaled by 2**20."""
    s = _tiny_row_star(variant)
    a = _TINY_ROW_POINT
    c, G, C, d, _, _ = rational.star(s)
    if not all(rational.dot(row, a) <= rhs for row, rhs in zip(C, d)):
        pytest.fail("the member point breaks a stored row")
    x = c[0] + rational.dot(G[0], a)
    if x != 4:
        pytest.fail(f"the member point gives x = {x}, not 4")
    module = _TINY_ROW_LAYERS[layer]()
    try:
        if entry == "reach_layer":
            sets = reach_layer(module, [s], method)
        else:
            sets = reach_pytorch_model(nn.Sequential(module), s, method=method)
    except (NotImplementedError, ValueError):
        return
    if not sets:
        pytest.fail(f"{layer} {method} through {entry}: no sets")
    assert held(holders_of(sets), [x]), (
        f"{layer} {method} through {entry}, predicate box {variant}: the true "
        f"output 4 at a = (2**-11, 2**30) is in none of the {len(sets)} sets")
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-3.0]]))
    assert not is_unsat(sets, unsafe), (
        f"{layer} {method} through {entry}, predicate box {variant}: UNSAT, "
        f"and the true output 4 is in the unsafe region y >= 3")


# 5b. A doubly scaled Star (variables and rows both scaled, seed 1): its
# row 7 has the entry -6.48e-13, which HiGHS drops, so the range of neuron 0
# that LeakyReLU, Sigmoid and Tanh read ends at 0.153 where the exact range
# ends at 0.253, and the output at vertex 8 of the predicate polytope is left
# out by 0.0999. Neighbours: the same rows with the small row lifted by 2**20
# (the same set), and with the finite predicate box the rows imply.
def _doubly_scaled_rows(seed=1):
    """(V, C, d, s) of the doubly scaled Star: a cube in three
    variables cut by two rows, with the variables scaled by 1e-6, 1 and 1e6
    and every row by 10**U(-6, 6)."""
    n_out, n_var = 4, 3
    rng = np.random.default_rng(1000 + seed)
    cuts = rng.normal(size=(2, n_var))
    cuts /= np.linalg.norm(cuts, axis=1, keepdims=True)
    C0 = np.vstack([np.eye(n_var), -np.eye(n_var), cuts])
    d0 = np.r_[np.ones(2 * n_var), rng.uniform(0.3, 0.8, 2)]
    c = rng.uniform(-0.3, 0.3, n_out)
    G = rng.uniform(-1, 1, (n_out, n_var))
    s = np.array([1e-6, 1.0, 1e6])[rng.permutation(n_var)]
    r = 10.0 ** rng.uniform(-6, 6, len(d0))
    C = (C0 / s) * r[:, None]
    d = d0 * r
    return np.hstack([c.reshape(-1, 1), G / s]), C, d.reshape(-1, 1), s


def _doubly_scaled_star(variant):
    V, C, d, s = _doubly_scaled_rows()
    if variant == "none":
        return Star(V, C, d)
    if variant == "lifted":
        small = np.any((np.abs(C) <= 1e-12) & (C != 0), axis=1)
        C, d = C.copy(), d.copy()
        C[small] *= 2.0 ** 20
        d[small] *= 2.0 ** 20
        return Star(V, C, d)
    return Star(V, C, d, -s.reshape(-1, 1), s.reshape(-1, 1))


_SMOOTH = {"Sigmoid": (nn.Sigmoid, proof.SIGMOID),
           "Tanh": (nn.Tanh, proof.TANH)}


# One marked cell per kernel that reads the ranges: the LeakyReLU kernel and
# the S-curve kernel that Sigmoid and Tanh share (``_s_curve_single_star_
# approx``); the Tanh cell without a box was removed as an excess
# expected failure of the same path. Tanh keeps its neighbours.
@pytest.mark.parametrize("layer,variant", [
    pytest.param(layer, variant, marks=_TINY_COEFFICIENT_DROPPED
                 if variant == "none" else (), id=f"{layer}-{variant}")
    for layer in ("LeakyReLU", "Sigmoid", "Tanh")
    for variant in ("none", "lifted", "finite")
    if not (layer == "Tanh" and variant == "none")])
def test_a_kernel_on_a_doubly_scaled_star_holds_every_output(layer, variant):
    """Catches a kernel whose range LP on a Star without a finite predicate
    box answers for a polytope with a coefficient dropped (registry SET-7,
    through ``reach_layer``), and a relaxation row that cuts the graph. For
    LeakyReLU every vertex of the exact image (rationals) must be in the
    returned set; for Sigmoid and Tanh the enclosure (100 bits) of the
    output at every vertex of the predicate polytope must be. Neighbours:
    the small row lifted by 2**20, and the finite predicate box."""
    s = _doubly_scaled_star(variant)
    c, G, C, d, _, _ = rational.star(_doubly_scaled_star("none"))
    vertices = rational.vertices(C, d)
    if len(vertices) != 12:
        pytest.fail(f"the predicate polytope has {len(vertices)} vertices, "
                    f"not 12")
    if layer == "LeakyReLU":
        sets = reach_layer(nn.LeakyReLU(0.125), [s], "approx")
        pieces = rational.leaky_relu(
            rational.affine(rational.polytope(C, d), G, c), F(1, 8))
        holders = holders_of(sets)
        outputs = sorted({y for piece in pieces for y in piece.image()})
        missed = next((y for y in outputs if not held(holders, y)), None)
    else:
        module, function = _SMOOTH[layer]
        sets = reach_layer(module(), [s], "approx")
        holders = holders_of(sets)
        enclose = proof.chain(function)
        outputs = [[ci + rational.dot(row, a) for ci, row in zip(c, G)]
                   for a in vertices]
        missed = next((x for x in outputs
                       if not box_held(holders, *proof.value(enclose, x))),
                      None)
    assert missed is None, (
        f"{layer} on the doubly scaled Star, predicate box {variant}: the "
        f"true output at {[float(v) for v in missed]} is in none of the "
        f"{len(sets)} sets (the search stops at the first of {len(outputs)})")


# 5c. Direct range calls with the caller's solver (SET-6). The well-scaled and
# the column-scaled (1e-9, 1, 1e9) families, centre 0 so
# that the sum with the centre is exact and a bound inside the exact range is
# the LP's own. Without a finite predicate box the optimum of every solver is
# returned with no certificate and no pad; with a finite box the HiGHS
# solvers are certified (the neighbours) and the CVXPY solvers are not.
_HIGHS_SOLVERS = ("default", "linprog", "highs-ds", "highs-ipm")
_CVXPY_SOLVERS = ("SCS", "CLARABEL", "OSQP")


def _scaled_rows(kind, seed):
    """(V, C, d, s) of ``n4_nobox.make(kind, seed)`` with centre 0."""
    n_out, n_var = 4, 3
    rng = np.random.default_rng(1000 + seed)
    cuts = rng.normal(size=(2, n_var))
    cuts /= np.linalg.norm(cuts, axis=1, keepdims=True)
    C0 = np.vstack([np.eye(n_var), -np.eye(n_var), cuts])
    d0 = np.r_[np.ones(2 * n_var), rng.uniform(0.3, 0.8, 2)]
    rng.uniform(-0.3, 0.3, n_out)                   # the centre, set to 0
    G = rng.uniform(-1, 1, (n_out, n_var))
    s = np.ones(n_var)
    if kind == "cols9":
        s = np.array([1e-9, 1.0, 1e9])[rng.permutation(n_var)]
    C = C0 / s
    V = np.hstack([np.zeros((n_out, 1)), G / s])
    return V, C, d0.reshape(-1, 1), s


def _scaled_set(kind, seed, set_kind, box):
    V, C, d, s = _scaled_rows(kind, seed)
    if np.any((np.abs(C) <= 1e-12) & (C != 0)):
        pytest.fail("a constraint entry is at or below 1e-12 (that is SET-7)")
    bounds = {"none": (None, None),
              "infinite": (np.full((3, 1), -np.inf), np.full((3, 1), np.inf)),
              "finite": (-s.reshape(-1, 1), s.reshape(-1, 1))}[box]
    if set_kind == "Star":
        return Star(V, C, d, *bounds)
    return ImageStar(V.reshape(1, 2, 2, 4), C, d, *bounds, 1, 2, 2)


def _star_solver_queries(s, solver):
    yield "get_ranges", s.get_ranges(lp_solver=solver)
    yield "get_range", tuple(zip(*[s.get_range(i, solver)
                                   for i in range(s.dim)]))
    yield "get_min_get_max", ([s.get_min(i, solver) for i in range(s.dim)],
                              [s.get_max(i, solver) for i in range(s.dim)])
    box = s.get_box(lp_solver=solver)
    yield "get_box", (box.lb, box.ub)


def _image_solver_queries(s, solver):
    yield "get_ranges", s.get_ranges(lp_solver=solver)
    yield "get_range", tuple(zip(*[s.get_range(h, w, c, solver)
                                   for h in range(s.height)
                                   for w in range(s.width)
                                   for c in range(s.num_channels)]))
    yield "get_range_flat", tuple(zip(*[s.get_range_flat(i, solver)
                                        for i in range(s.dim)]))


_RAW_SOLVER_OPTIMUM = known_bug(
    17, "on a Star or ImageStar without a finite predicate box the range "
        "queries (get_ranges, get_range, get_min, get_max, get_box, "
        "get_range_flat) return the caller's solver's optimum plus the "
        "centre with no certificate and no pad: HiGHS ends one unit in the "
        "last place inside the exact range, SCS, CLARABEL and OSQP by up to "
        "the whole range width")
_CVXPY_OPTIMUM_WITH_A_BOX = known_bug(
    17, "with a finite predicate box only the HiGHS solvers' optima are "
        "certified and pushed outward; with lp_solver set to a CVXPY solver "
        "(SCS, CLARABEL, OSQP) the range queries return its optimum as it "
        "comes, inside the exact range by up to 3.8 on the column-scaled "
        "family")


# The marked cells, one per branch of the LP layer that returns an optimum
# with no certificate (the Star cells run the batch of get_ranges, get_range
# and get_box and the single LPs of get_min and get_max): the highspy branch
# ('default'; 'linprog' takes the same one), the scipy branch ('highs-ds';
# 'highs-ipm' the same) and the CVXPY branch (SCS, CLARABEL and OSQP share
# ``_solve_lp_cvxpy``), each with no predicate box and with an infinite one,
# and the CVXPY branch with a finite box, which is never certified. The
# ImageStar queries forward to the same branches: one marked cell. The other
# marked cells were removed as excess expected failures.
_SOLVER_MARKED = {("Star", "none", "default"), ("Star", "none", "highs-ds"),
                  ("Star", "none", "SCS"), ("Star", "infinite", "default"),
                  ("Star", "infinite", "highs-ds"),
                  ("Star", "infinite", "CLARABEL"), ("Star", "finite", "SCS"),
                  ("ImageStar", "none", "default")}


def _solver_cells():
    cells = []
    for set_kind in ("Star", "ImageStar"):
        for box in ("none", "infinite", "finite"):
            for solver in _HIGHS_SOLVERS + _CVXPY_SOLVERS:
                if box != "finite":
                    marks = _RAW_SOLVER_OPTIMUM
                elif solver in _CVXPY_SOLVERS:
                    marks = _CVXPY_OPTIMUM_WITH_A_BOX
                else:
                    marks = ()
                if marks and (set_kind, box, solver) not in _SOLVER_MARKED:
                    continue
                cells.append(pytest.param(set_kind, box, solver, marks=marks,
                                          id=f"{set_kind}-{box}-{solver}"))
    return cells


@pytest.mark.parametrize("set_kind,box,solver", _solver_cells())
def test_a_range_query_with_each_solver_holds_the_exact_range(set_kind, box,
                                                              solver):
    """Catches a range query that returns a solver optimum with no
    certificate and no pad (registry SET-6: without a finite predicate box,
    and with a finite box and a CVXPY solver), the certificate of the HiGHS
    solvers dropped on a finite box, a solver name read as another, and an
    ImageStar query that reads its pixels in the wrong order. Every bound of
    every query must hold the exact range (vertices of the predicate
    polytope in rationals), on two seeds of the well-scaled and the
    column-scaled families. A refusal (``NotImplementedError`` or
    ``ValueError``) keeps the promise. The neighbours: a finite predicate box
    with each HiGHS solver, on both set types."""
    if solver in _CVXPY_SOLVERS and solver not in cvxpy.installed_solvers():
        pytest.skip(f"cvxpy has no {solver}")
    queries = (_star_solver_queries if set_kind == "Star"
               else _image_solver_queries)
    found = []
    for kind, seed in itertools.product(("well", "cols9"), (1, 2)):
        s = _scaled_set(kind, seed, set_kind, box)
        exact = exact_ranges(s)
        try:
            for name, (lo, hi) in queries(s, solver):
                found += [(kind, seed, name) + miss
                          for miss in bounds_inside(exact, lo, hi)]
        except (NotImplementedError, ValueError):
            continue
        if found:
            break                   # one set with a bound inside is enough
    assert not found, (
        f"{set_kind} with predicate box {box}, lp_solver={solver}: "
        f"{len(found)} bounds inside the exact range (family, seed, query, "
        f"entry, side, how far inside), the first {found[:4]}")


@pytest.mark.parametrize("layer,solver", [
    (layer, solver) for layer in ("Sigmoid", "Tanh")
    for solver in _CVXPY_SOLVERS])
def test_a_kernel_given_a_cvxpy_solver_holds_every_output(layer, solver):
    """The range LPs of the Sigmoid and Tanh Star kernels go to the certified
    HiGHS path whatever ``lp_solver`` the caller names (``_bound_lp_solver``),
    so ``reach_layer`` is sound with SCS, CLARABEL and OSQP although their
    own optima land up to 3.8 inside on these sets (the SET-6 cells above).
    Catches a kernel that hands the caller's solver to its range LPs, and a
    relaxation row that cuts the graph: on the column-scaled Stars of seeds 1
    and 2 with a finite predicate box, the enclosure (100 bits) of the output
    at every vertex of the predicate polytope must be in the returned set.
    Seed 2 is the one where CLARABEL's own optimum lies inside the exact
    range (by 0.994; on seed 1 it does not), so the CLARABEL cells can see a
    kernel that uses it. Passes today."""
    if solver in _CVXPY_SOLVERS and solver not in cvxpy.installed_solvers():
        pytest.skip(f"cvxpy has no {solver}")
    module, function = _SMOOTH[layer]
    enclose = proof.chain(function)
    for seed in (1, 2):
        s = _scaled_set("cols9", seed, "Star", "finite")
        c, G, C, d, _, _ = rational.star(s)
        sets = reach_layer(module(), [s], "approx", lp_solver=solver)
        holders = holders_of(sets)
        for a in rational.vertices(C, d):
            x = [ci + rational.dot(row, a) for ci, row in zip(c, G)]
            assert box_held(holders, *proof.value(enclose, x)), (
                f"{layer} with lp_solver={solver}, seed {seed}: the output "
                f"at the input {[float(v) for v in x]} is in none of the "
                f"{len(sets)} sets")


# 5d. A predicate box of minus and plus infinity through the approx ReLU and
# LeakyReLU kernels (REACH-5): the ReLU result holds NaN in its rows, the
# LeakyReLU result a right-hand side of +inf. The promise: finite rows that
# hold every true output, or a refusal. Neighbours: the same rows with no
# predicate box, and the exact method.
def _infinite_box_star(box):
    V, C, d, _ = _scaled_rows("well", 1)
    V = V.copy()
    V[:, 0] = [0.25, -0.125, 0.0625, -0.25]
    if box == "infinite":
        return Star(V, C, d, np.full((3, 1), -np.inf), np.full((3, 1), np.inf))
    return Star(V, C, d)


_INFINITE_BOX_TEXT = (
    "registry REACH-5: the approx ReLU and LeakyReLU kernels accept a Star "
    "whose predicate box is -inf and +inf and compute with it, and "
    "reach_layer neither makes the box finite nor refuses: ")
_NAN_ROWS_FOR_AN_INFINITE_BOX = known_bug(
    16, _INFINITE_BOX_TEXT + "ReLU computes rows holding NaN, and numpy's "
        "invalid-value RuntimeWarning of that computation is raised first "
        "under the suite's warnings policy", raises=RuntimeWarning)
_INF_ROWS_FOR_AN_INFINITE_BOX = known_bug(
    16, _INFINITE_BOX_TEXT + "LeakyReLU returns rows with a +inf right-hand "
        "side")


def _infinite_box_marks(layer, method, box):
    if box != "infinite" or method != "approx":
        return ()
    return (_NAN_ROWS_FOR_AN_INFINITE_BOX if layer == "ReLU"
            else _INF_ROWS_FOR_AN_INFINITE_BOX)


@pytest.mark.parametrize("layer,method,box", [
    pytest.param(layer, method, box,
                 marks=_infinite_box_marks(layer, method, box),
                 id=f"{layer}-{method}-{box}")
    for layer in ("ReLU", "LeakyReLU") for method in ("approx", "exact")
    for box in ("infinite", "none")])
def test_a_star_with_an_infinite_box_gives_a_set_or_a_refusal(layer, method,
                                                              box):
    """Catches a kernel that returns NaN or an infinite value in the rows of
    a returned Star (registry REACH-5), and one that leaves out a true
    output: every vertex of the exact image (rationals) must be a member.
    A refusal (``NotImplementedError`` or ``ValueError``) keeps the
    promise."""
    s = _infinite_box_star(box)
    c, G, C, d, _, _ = rational.star(_infinite_box_star("none"))
    module = nn.ReLU() if layer == "ReLU" else nn.LeakyReLU(0.125)
    try:
        sets = reach_layer(module, [s], method)
    except (NotImplementedError, ValueError):
        return
    if not sets:
        pytest.fail(f"{layer} {method}: no sets")
    for k, o in enumerate(sets):
        arrays = (o.V, o.C, o.d)
        assert all(np.all(np.isfinite(np.asarray(a, dtype=np.float64)))
                   for a in arrays), (
            f"{layer} {method}, predicate box {box}: returned set {k} has NaN "
            f"or an infinite value in V, C or d")
    slope = F(0) if layer == "ReLU" else F(1, 8)
    pieces = rational.leaky_relu(
        rational.affine(rational.polytope(C, d), G, c), slope)
    holders = holders_of(sets)
    outputs = sorted({y for piece in pieces for y in piece.image()})
    missed = next((y for y in outputs if not held(holders, y)), None)
    assert missed is None, (
        f"{layer} {method}, predicate box {box}: the vertex "
        f"{[float(v) for v in missed]} of the exact image is in no returned "
        f"set (the search stops at the first of {len(outputs)})")


# --------------------------------------------------------------------------- #
# 6. The slope product of LeakyReLU (ROUND-9)                                   #
# --------------------------------------------------------------------------- #
# Stars of three entries over two predicate variables in [-1, 1]: x0 = c + g1
# a1 + g2 a2 with |c| from 0.3 to 3 and g1, g2 from 1e-9 to 1e-1, so x0 has
# one sign over the whole set, and x1 = a1, x2 = a2, which pin the predicate
# point in the output. Where x0 < 0 the output is s x0, and a kernel that
# multiplies c, g1 and g2 by the slope s with round-to-nearest leaves the true
# value out unless every product is exact: a slope of float32(0.1) does that,
# a dyadic slope (0.125, 0.25) is exact and is the neighbour.
def _slope_stars(seed=7, count=10):
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(count):
        c = rng.uniform(0.3, 3.0) * rng.choice([-1, 1])
        g1, g2 = 10.0 ** rng.uniform(-9, -1, 2) * rng.choice([-1, 1], 2)
        V = np.array([[c, g1, g2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        stars.append(Star(V, np.zeros((0, 2)), np.zeros((0, 1)),
                          -np.ones((2, 1)), np.ones((2, 1))))
    return stars


_SLOPES = {"float32_0.1": float(np.float32(0.1)), "0.125": 0.125,
           "0.25": 0.25}
_SLOPE_PRODUCT_TO_NEAREST = known_bug(
    "7b", "LeakyReLU multiplies the centre and the generators of a negative "
          "neuron by the slope with round-to-nearest (the slope product), so "
          "with a slope that is not dyadic the true output s x of a vertex "
          "is not in the returned set (one unit in the last place)")


# One marked cell per kernel (approx and exact) with the slope float32(0.1);
# the slope 0.3 ran the same products and its two marked cells were removed
# as excess expected failures.
@pytest.mark.parametrize("slope,method", [
    pytest.param(slope, method, marks=_SLOPE_PRODUCT_TO_NEAREST
                 if slope == "float32_0.1" else (),
                 id=f"{slope}-{method}")
    for slope in _SLOPES for method in ("approx", "exact")])
def test_leakyrelu_holds_the_exact_slope_product(slope, method):
    """Catches the slope product of LeakyReLU rounded to nearest (registry
    ROUND-9), a slope applied to the wrong side of zero or to the wrong
    entry, and an input set dropped from the loop over the sets: the ten
    Stars of a seeded family (about half of them negative) go to the layer
    in one call, and at every vertex of each Star's predicate box the exact
    output (rationals of the stored slope) must be a member of a returned
    set. The rows x1 = a1, x2 = a2 pin the predicate point, so no Star's
    output is held by another Star's set by chance. Neighbours: the dyadic
    slopes 0.125 and 0.25."""
    s_q = F(_SLOPES[slope])
    found = []
    stars = _slope_stars()
    sets = reach_layer(nn.LeakyReLU(_SLOPES[slope]), stars, method)
    holders = holders_of(sets)
    for k, s in enumerate(stars):
        c, G, _, _, _, _ = rational.star(s)
        for a in itertools.product((F(-1), F(1)), repeat=2):
            x = [ci + rational.dot(row, a) for ci, row in zip(c, G)]
            y = [v if v >= 0 else s_q * v for v in x]
            if not held(holders, y):
                found.append((k, tuple(map(int, a))))
    assert not found, (
        f"LeakyReLU slope {slope} {method}: the exact output is in no "
        f"returned set at {len(found)} (star, vertex), the first {found[:4]}")


# --------------------------------------------------------------------------- #
# 7. The range queries of a Box, a Zono, an ImageZono, a Hexatope, an Octatope #
# --------------------------------------------------------------------------- #
# A Box returns the bounds it stores, which its docstrings call exact; a Zono
# (and an ImageZono, which inherits the queries) sums the centre and the
# generator magnitudes and moves each bound outward by a bound on that
# rounding (``outward_interval_bounds``). The data make a sum computed to
# nearest land inside: the centre 1 with the generator 2**-60, centres up to
# 5e5 with generators from 1e-9 to 1, and the same numbers held in float32
# (a Zono keeps the float type it is given).
def _zono_family(family):
    if family == "one_ulp":
        return [Zono(np.array([[1.0], [-3.0], [0.1]]),
                     np.array([[TINY, 0.0], [0.0, TINY], [1 / 3, 0.7]]))]
    rng = np.random.default_rng(20261118)
    zonos = []
    for _ in range(6):
        c = rng.uniform(-5, 5, (3, 1)) * 10.0 ** rng.integers(-2, 6, (3, 1))
        V = rng.uniform(-1, 1, (3, 4)) * 10.0 ** rng.integers(-9, 1, (3, 1))
        zonos.append(Zono(c, V))
    if family == "float32":
        return [Zono(z.c.astype(np.float32), z.V.astype(np.float32))
                for z in zonos]
    return zonos


def _exact_zono_ranges(z):
    """The exact range c -+ sum |g| of every entry, in rationals, of the
    numbers the set stores (read by ``rational.star``)."""
    c, G = rational.star(z)[:2]
    radius = [sum(abs(g) for g in row) for row in G]
    return ([ci - r for ci, r in zip(c, radius)],
            [ci + r for ci, r in zip(c, radius)])


_ZONO_QUERIES = {
    "get_bounds": lambda z: z.get_bounds(),
    "estimate_ranges": lambda z: z.estimate_ranges(),
    "get_range": lambda z: tuple(zip(*[z.get_range(i) for i in range(z.dim)])),
    "get_box": _box_query,
}


def _zono_range_cells():
    cells = [pytest.param("Zono", family, query, id=f"Zono-{family}-{query}")
             for family in ("one_ulp", "random", "float32")
             for query in _ZONO_QUERIES]
    cells += [pytest.param("ImageZono", "random", query,
                           id=f"ImageZono-random-{query}")
              for query in ("get_bounds", "get_range")]
    cells += [pytest.param("Box", "random", query, id=f"Box-random-{query}")
              for query in ("get_range", "get_ranges", "estimate_ranges")]
    return cells


@pytest.mark.parametrize("kind,family,query", _zono_range_cells())
def test_a_range_query_of_a_box_or_a_zonotope_holds_the_exact_range(
        kind, family, query):
    """Catches a Zono or ImageZono range query that adds the generator
    magnitudes to the centre by round-to-nearest (a bound one unit in the
    last place inside the exact range c -+ sum |g|, rationals, zero slack),
    one that rounds a float32 set's bounds to float32, one that reads the
    range of another entry or loses a generator, and a Box whose stored
    bounds come back changed: the Box queries are documented as exact, so
    they must return the stored numbers exactly. Passes today."""
    if kind == "Box":
        rng = np.random.default_rng(20261119)
        lb = rng.uniform(-5, 5, 4) * 10.0 ** rng.integers(-3, 4, 4)
        ub = lb + rng.uniform(0, 1, 4) * 10.0 ** rng.integers(-12, 2, 4)
        lo, hi = getattr(Box(lb, ub), query)()
        assert (np.array_equal(np.ravel(lo), lb)
                and np.array_equal(np.ravel(hi), ub)), (
            f"Box.{query}: [{np.ravel(lo)}, {np.ravel(hi)}] is not the stored "
            f"[{lb}, {ub}]")
        return
    sets = _zono_family(family)
    if kind == "ImageZono":
        sets = [ImageZono(z.c, z.V, 1, 3, 1) for z in sets]
    found = []
    for k, z in enumerate(sets):
        lo, hi = _ZONO_QUERIES[query](z)
        found += [(k,) + miss
                  for miss in bounds_inside(_exact_zono_ranges(z), lo, hi)]
    assert not found, (
        f"{kind} {query} on the {family} family: {len(found)} bounds inside "
        f"the exact range (set, entry, side, how far inside), the first "
        f"{found[:4]}")


# The Hexatope and Octatope queries: the cut
# kernels of ``test_soundness_hexoct_sets.py`` (dyadic, rebuilt here), with
# every query and both solvers; an Octatope kernel with one cut, where the
# 'lp' queries hold the range today; and the one-ulp data (centres 1, -3 and
# 0.1, generators 2**-60 and dyadic ones, the box of the kernel), where a
# centre term added to nearest lands inside.
def _hexatope(center, generators, rows):
    from n2v.sets.hexatope import DifferenceConstraintSystem
    generators = np.asarray(generators, dtype=np.float64)
    dcs = DifferenceConstraintSystem(generators.shape[1])
    for i, j, b in rows:
        dcs.add_constraint(i, j, b)
    return Hexatope(np.asarray(center, dtype=np.float64), generators, dcs)


def _octatope(center, generators, rows):
    from n2v.sets.octatope import UTVPIConstraintSystem
    generators = np.asarray(generators, dtype=np.float64)
    utvpi = UTVPIConstraintSystem(generators.shape[1])
    for row in rows:
        utvpi.add_constraint(*row)
    return Octatope(np.asarray(center, dtype=np.float64), generators, utvpi)


_HEX_BOX_ROWS = [r for i in (1, 2, 3) for r in ((i, 0, 1.0), (0, i, 1.0))]
_OCT_BOX_ROWS = [r for i in range(3) for r in ((i, i, 1, 0, 1.0),
                                               (i, i, -1, 0, 1.0))]
_KERNEL_CENTRE = [-0.375, 0.5, -1.25]
_HEXOCT_FAMILIES = {
    ("Hexatope", "kernel"): lambda: _hexatope(
        _KERNEL_CENTRE, [[0.0, 0.5, -0.25, 0.75], [0.0, -1.0, 0.5, 0.25],
                         [0.0, 0.25, 0.25, -0.5]],
        _HEX_BOX_ROWS + [(1, 2, 0.25), (3, 0, 0.5), (0, 1, 0.75), (2, 3, 0.5)]),
    ("Octatope", "kernel"): lambda: _octatope(
        _KERNEL_CENTRE, [[0.5, -0.25, 0.75], [-1.0, 0.5, 0.25],
                         [0.25, 0.25, -0.5]],
        _OCT_BOX_ROWS + [(0, 1, -1, -1, 0.5), (1, 2, 1, 1, 0.75),
                         (0, 2, 1, -1, 0.25), (2, 1, -1, 1, 1.0),
                         (0, 2, 0, -1, 0.75), (1, 0, 0, 1, 0.5)]),
    ("Octatope", "one_cut"): lambda: _octatope(
        _KERNEL_CENTRE, [[0.5, -0.25, 0.75], [-1.0, 0.5, 0.25],
                         [0.25, 0.25, -0.5]],
        _OCT_BOX_ROWS + [(0, 1, 1, 1, 0.5)]),
    ("Hexatope", "one_ulp"): lambda: _hexatope(
        [1.0, -3.0, 0.1], [[0.0, TINY, 0.0, 0.0], [0.0, 0.0, TINY, 0.0],
                           [0.0, 0.25, 0.5, 0.75]], _HEX_BOX_ROWS),
    ("Octatope", "one_ulp"): lambda: _octatope(
        [1.0, -3.0, 0.1], [[TINY, 0.0, 0.0], [0.0, TINY, 0.0],
                           [0.25, 0.5, 0.75]], _OCT_BOX_ROWS),
    # A generator row whose entries (1/3, 0.1) do not sum exactly: the node
    # demands of the min-cost flow then miss 0 by a rounding.
    ("Hexatope", "inexact"): lambda: _hexatope(
        [0.0, 1e6 + 0.1, -3.0], [[0.0, 0.5, 0.0, 0.0], [0.0, 1 / 3, 0.1, 0.0],
                                 [0.0, 0.0, 0.0, 0.7]], _HEX_BOX_ROWS),
}

_HEXOCT_QUERIES = {
    "get_ranges": lambda s, solver: s.get_ranges(solver),
    "get_ranges_parallel": lambda s, solver: s.get_ranges(
        solver, parallel=True, n_workers=2),
    "get_range": lambda s, solver: tuple(zip(*[s.get_range(i, solver)
                                               for i in range(s.dim)])),
    "get_box": lambda s, solver: _box_query_with(s, solver),
    "get_bounds": lambda s, solver: s.get_bounds(solver),
    "estimate_ranges": lambda s, solver: s.estimate_ranges(),
}


# The exact ranges of the families above, computed once (the sets are
# rebuilt from the same numbers for every cell).
_HEXOCT_EXACT = {}


def _box_query_with(s, solver):
    box = s.get_box(solver)
    return box.lb, box.ub


_HEXOCT_LP_RANGE_UNPADDED = known_bug(
    9, "the 'lp' range queries of a Hexatope and an Octatope return the "
       "optimum of a CVXPY OSQP solve at a hard-coded 1e-7 (behind a bare "
       "except), the centre term included, with no pad: on the cut kernel "
       "of the Octatope the lower bounds of entries 0 and 1 lie 4.4e-8 and "
       "6.5e-8 inside the exact range, and the entry 1 + 2**-60 a of a "
       "Hexatope gets the range [1.0, 1.0]")
_MCF_NO_ANSWER = known_bug(
    9, "Hexatope.get_range(i, 'mcf') returns (None, None) for a bounded "
       "nonempty set when the objective's node demands do not sum to exactly "
       "0 in float64 (entry 1e6 + 0.1 + x1/3 + x2/10): _optimize_dcs_mcf "
       "accepts them by np.isclose, networkx's network_simplex raises "
       "NetworkXUnfeasible for a nonzero total, and the exception is turned "
       "into None, so the query gives no range where one is due")
_MCF_CENTRE_TO_NEAREST = known_bug(
    9, "the 'mcf' range queries of a Hexatope and an Octatope add the centre "
       "term f.c to the value of the min-cost flow by round-to-nearest with "
       "no outward step, so the entry 1 + 2**-60 a gets the range [1.0, 1.0] "
       "and the lower bound of the entry 0.1 + x1/4 + x2/2 + 3 x3/4 over the "
       "box lies 8.3e-17 inside the exact 0.1 - 3/2")


def _hexoct_range_cells():
    """The neighbours: every query on the Hexatope kernel with both solvers,
    on the one-cut Octatope with 'lp' and on the Octatope kernel with 'mcf',
    and the interval query on both kernels. The marked cells, one per line
    that adds the centre term or takes the unpadded optimum: 'lp' on the
    Octatope kernel (``_optimize_utvpi_lp``) and on the one-ulp Hexatope
    (``_optimize_dcs_lp``), 'mcf' on the one-ulp data of both types
    (``_optimize_dcs_mcf``, which the Octatope reaches through
    ``_optimize_utvpi_mcf``). Every query calls ``get_range``, so one query
    stands for the others in a marked cell. And ``get_range`` with 'mcf' on
    a generator row whose node demands do not sum exactly, which answers
    None; on the same set ``get_ranges`` falls back to 'lp' for that entry
    and the parallel ``get_ranges`` to ``estimate_range``, and both of these
    fallbacks are neighbours that hold the range today."""
    lp_queries = ("get_ranges", "get_ranges_parallel", "get_range", "get_box",
                  "get_bounds")
    mcf_queries = ("get_ranges", "get_ranges_parallel", "get_range")
    cells = []
    for cls in ("Hexatope", "Octatope"):
        lp_family = "kernel" if cls == "Hexatope" else "one_cut"
        cells += [pytest.param(cls, lp_family, "lp", q,
                               id=f"{cls}-{lp_family}-lp-{q}")
                  for q in lp_queries]
        cells += [pytest.param(cls, "kernel", "mcf", q,
                               id=f"{cls}-kernel-mcf-{q}")
                  for q in mcf_queries]
        cells.append(pytest.param(cls, "kernel", "none", "estimate_ranges",
                                  id=f"{cls}-kernel-estimate_ranges"))
        cells.append(pytest.param(cls, "one_ulp", "mcf", "get_ranges",
                                  marks=_MCF_CENTRE_TO_NEAREST,
                                  id=f"{cls}-one_ulp-mcf-get_ranges"))
    cells.append(pytest.param("Hexatope", "one_ulp", "lp", "get_ranges",
                              marks=_HEXOCT_LP_RANGE_UNPADDED,
                              id="Hexatope-one_ulp-lp-get_ranges"))
    cells.append(pytest.param("Hexatope", "inexact", "mcf", "get_range",
                              marks=_MCF_NO_ANSWER,
                              id="Hexatope-inexact-mcf-get_range"))
    cells += [pytest.param("Hexatope", "inexact", "mcf", q,
                           id=f"Hexatope-inexact-mcf-{q}")
              for q in ("get_ranges", "get_ranges_parallel")]
    cells.append(pytest.param("Octatope", "kernel", "lp", "get_ranges",
                              marks=_HEXOCT_LP_RANGE_UNPADDED,
                              id="Octatope-kernel-lp-get_ranges"))
    return cells


@pytest.mark.parametrize("cls,family,solver,query", _hexoct_range_cells())
def test_a_hexatope_or_octatope_range_query_holds_the_exact_range(
        cls, family, solver, query):
    """Catches a Hexatope or Octatope range query whose bound lies inside
    the exact range of its entry: the 'lp' optimum of OSQP taken with no pad
    (registry HEXOCT-4), the centre term added to nearest by the 'mcf'
    queries, a minimum and a maximum swapped, a centre term with the wrong
    sign, the parallel path or a fallback that reads another entry, and a
    ``get_box`` or ``get_bounds`` that changes the ranges. The exact range is
    the least and greatest value over the vertices of the set's own kernel
    polytope (anchor pinned to 0 for a Hexatope), in rationals, zero
    slack."""
    s = _HEXOCT_FAMILIES[(cls, family)]()
    if (cls, family) not in _HEXOCT_EXACT:
        _HEXOCT_EXACT[(cls, family)] = exact_ranges(s)
    exact = _HEXOCT_EXACT[(cls, family)]
    lo, hi = _HEXOCT_QUERIES[query](s, solver)
    found = bounds_inside(exact, lo, hi)
    assert not found, (
        f"{cls} {family} {query} solver={solver}: bounds inside the exact "
        f"range (entry, side, how far inside): {found}")


# --------------------------------------------------------------------------- #
# 8. Set arithmetic of a Zono and a Box other than the affine map              #
# --------------------------------------------------------------------------- #
def _zono_vertices(z):
    """``c + V s`` for every sign vector ``s``, in rationals of the stored
    numbers: every vertex of the zonotope is among these points."""
    c, G = rational.star(z)[:2]
    return sorted({tuple(ci + rational.dot(row, signs)
                         for ci, row in zip(c, G))
                   for signs in itertools.product((F(-1), F(1)),
                                                  repeat=len(G[0]))})


def _arithmetic_cells():
    return [pytest.param("zono_from_bounds", "inexact",
                         id="zono_from_bounds-inexact"),
            pytest.param("zono_order_reduction_box", "float64",
                         id="zono_order_reduction_box-float64"),
            pytest.param("zono_order_reduction_box", "float32",
                         id="zono_order_reduction_box-float32"),
            pytest.param("box_minkowski_sum", "inexact",
                         id="box_minkowski_sum-inexact")]


@pytest.mark.parametrize("op,family", _arithmetic_cells())
def test_zono_and_box_arithmetic_holds_the_exact_result(op, family):
    """Catches a set operation of a Zono or a Box whose result leaves out a
    point of the exact result by the rounding of a sum: a zonotope built
    from a box must hold every corner of the box; the order reduction of a
    zonotope (``order_reduction_box``, float64 and float32) must hold every
    vertex of the original; the Minkowski sum of two boxes must hold the
    sums of their bounds. Every point is exact (rationals of the stored
    numbers) and decided on the result's own rows with zero slack. The data
    round: ends such as 0.1 and 1/3, magnitudes far apart. Passes today.
    (``Zono.minkowski_sum`` adds the centres to nearest: that is the
    rounding of PLAN step 7b, whose radius covers the set-level
    ``minkowski_sum`` beside the affine map, and gets no cell here.)"""
    if op == "box_minkowski_sum":
        rng = np.random.default_rng(20261120)
        lb1, lb2 = rng.uniform(-5, 0, 4) / 3, rng.uniform(-5, 0, 4) * 0.1
        ub1 = lb1 + rng.uniform(0, 1, 4) * 10.0 ** rng.integers(-12, 3, 4)
        ub2 = lb2 + rng.uniform(0, 1, 4) / 7
        out = Box(lb1, ub1).minkowski_sum(Box(lb2, ub2))
        exact = ([F(a) + F(b) for a, b in zip(lb1, lb2)],
                 [F(a) + F(b) for a, b in zip(ub1, ub2)])
        found = bounds_inside(exact, out.lb, out.ub)
        assert not found, f"Box.minkowski_sum: bounds inside the sum: {found}"
        return
    if op == "zono_from_bounds":
        lb = np.array([0.1, -1 / 3, 1e-300, -7.0])
        ub = np.array([0.3, 2 / 3, 7.0, 1e6 + 0.1])
        result = Zono.from_bounds(lb, ub)
        points = list(itertools.product(*[(F(a), F(b))
                                          for a, b in zip(lb, ub)]))
    else:
        # Seeds on which a radius summed to nearest leaves out vertices of
        # the original (six of 64 in float64, two in float32).
        rng = np.random.default_rng(20261122 if family == "float64"
                                    else 20261121)
        z = Zono(np.array([[0.1], [1e3 / 3]]), rng.uniform(-1, 1, (2, 6)) / 3)
        if family == "float32":
            z = Zono(z.c.astype(np.float32), z.V.astype(np.float32))
        result = z.order_reduction_box(3)
        if result.V.shape[1] > 3 + z.dim:
            pytest.fail(f"order_reduction_box(3) kept {result.V.shape[1]} "
                        f"generators")
        points = _zono_vertices(z)
    holder = rational.Membership(rational.star(result))
    missed = [p for p in points if not holder(list(p))]
    assert not missed, (
        f"{op} ({family}): {len(missed)} of {len(points)} exact points are "
        f"not in the result, the first {[float(v) for v in missed[0]]}")
