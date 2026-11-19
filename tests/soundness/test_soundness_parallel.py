"""
Soundness tests for parallel LP solving.

These tests verify that parallel LP solving maintains mathematical correctness
and soundness properties.

Three groups: the parallel range LPs of a Star (``TestParallelLPSoundness``)
and the Hexatope and Octatope set operations (``TestHexatopeSoundness``,
``TestOctatopeSoundness``), which are set operations shared by every layer;
and, at the end, the soundness cells of the process-pool path of the exact
ReLU on Stars and ImageStars, which reuse the exact reference of
``test_soundness_relu_cells.py``.
"""

import pytest
import numpy as np

import n2v
from n2v.sets import Star, Box, Hexatope, Octatope
from n2v.config import set_parallel, config


class TestParallelLPSoundness:
    """The range queries of a Star on the paths the global settings select
    (``set_parallel``, the automatic threshold, the worker counts), against
    the exact range.

    Each star has three predicate variables in the box
    [-1, 2] x [-1/2, 1] x [-2, 1/2] and two constraint rows that cut it, and
    ``dim`` entries that are dyadic linear functions of the variables
    (coefficients up to 4, both signs). Its exact range is the least and
    greatest value over the vertices of the predicate polytope, enumerated
    in rationals (``tests/_oracles/rational.py``), never an n2v value. The
    LP queries are documented as exact: a returned bound must hold the exact
    range (pad 1e-12 relative for the centre added in round to nearest,
    registry SET-5) and lie within LP error of it (1e-9 relative), which a
    query that falls back to the interval estimate does not.

    ``setup_method`` and ``teardown_method`` reset ``n2v.config``."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def teardown_method(self):
        """These tests switch parallel LP on; leave the process as found."""
        config.reset()

    @staticmethod
    def tall_star(dim, dtype=np.float64):
        """``(star, exact_lo, exact_hi)`` for a star of ``dim`` entries (class
        docstring); ``dtype`` is the type of the arrays handed to the
        constructor."""
        from fractions import Fraction
        from tests._oracles import rational as Q
        rng = np.random.default_rng(100 + dim)
        G = rng.integers(-16, 17, size=(dim, 3)) / 4.0
        c = rng.integers(-16, 17, size=dim) / 8.0
        C = np.array([[1.0, 1.0, 0.5], [-0.5, 1.0, -1.0]])
        d = np.array([1.0, 1.0])
        lb, ub = np.array([-1.0, -0.5, -2.0]), np.array([2.0, 1.0, 0.5])
        corners = Q.vertices(Q.matrix(np.vstack([C, np.eye(3), -np.eye(3)])),
                             Q.vector(np.concatenate([d, ub, -lb])))
        values = [[Q.dot(Q.vector(g), a) + Fraction(ci) for a in corners]
                  for g, ci in zip(G, c)]
        star = Star(np.hstack([c[:, None], G]).astype(dtype), C.astype(dtype),
                    d.reshape(-1, 1).astype(dtype), lb.astype(dtype),
                    ub.astype(dtype))
        return star, [min(v) for v in values], [max(v) for v in values]

    @staticmethod
    def assert_exact(lo, hi, exact_lo, exact_hi, what, tight=True):
        """``[lo_i, hi_i]`` holds the exact range of every entry (pad 1e-12
        relative) and, with ``tight``, lies within 1e-9 relative of it."""
        from fractions import Fraction
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        if lo.size != len(exact_lo) or hi.size != len(exact_hi):
            pytest.fail(f"{what}: {lo.size} bounds for {len(exact_lo)} "
                        f"entries")
        bad = []
        for i, (l, h, el, eh) in enumerate(zip(lo, hi, exact_lo, exact_hi)):
            if not (np.isfinite(l) and np.isfinite(h)):
                bad.append((i, l, h, el, eh))
                continue
            pad_l = Fraction(1, 10 ** 12) * max(1, abs(el))
            pad_h = Fraction(1, 10 ** 12) * max(1, abs(eh))
            holds = Fraction(l) <= el + pad_l and Fraction(h) >= eh - pad_h
            close = (Fraction(l) >= el - 1000 * pad_l
                     and Fraction(h) <= eh + 1000 * pad_h)
            if not holds or (tight and not close):
                bad.append((i, l, h, el, eh))
        assert not bad, (
            f"{what}: {len(bad)} entries wrong, first: entry {bad[0][0]} "
            f"returned [{bad[0][1]!r}, {bad[0][2]!r}], exact range "
            f"[{float(bad[0][3])!r}, {float(bad[0][4])!r}]")

    def test_simple_bounds_soundness(self):
        """``set_parallel(True, n_workers=4)`` sends ``get_ranges()`` down the
        parallel path whatever the dimension (``config.parallel_lp``), with
        at most two workers below ten entries: a star of one entry and one of
        three entries given as float32 arrays. ``get_ranges(parallel=True)``
        of a star of one entry takes the batch path instead. Each result is
        the exact range. Catches a parallel path that drops an entry or
        writes a result at another entry, and a path the global switch
        selects that leaves out the centre."""
        set_parallel(True, n_workers=4)
        for dim, dtype in ((1, np.float64), (3, np.float32)):
            star, e_lo, e_hi = self.tall_star(dim, dtype)
            lo, hi = star.get_ranges()
            self.assert_exact(lo, hi, e_lo, e_hi,
                              f"set_parallel(True): get_ranges(), {dim} "
                              f"entries ({np.dtype(dtype).name})")
        config.reset()
        star, e_lo, e_hi = self.tall_star(1)
        lo, hi = star.get_ranges(parallel=True)
        self.assert_exact(lo, hi, e_lo, e_hi,
                          "get_ranges(parallel=True), one entry")

    def test_box_overapproximation_soundness(self):
        """``get_box()`` reads the global setting: after
        ``set_parallel(True, n_workers=4)`` it takes the parallel path (four
        workers at twelve entries), and the box is the exact range of every
        entry. Catches ``get_box`` building its box from the lower bounds
        alone and the parallel path losing its last entry."""
        set_parallel(True, n_workers=4)
        star, e_lo, e_hi = self.tall_star(12)
        box = star.get_box()
        self.assert_exact(box.lb, box.ub, e_lo, e_hi,
                          "set_parallel(True): get_box(), 12 entries")

    def test_high_dimensional_soundness(self):
        """From twenty entries the number of workers is the global
        ``n_workers`` itself: after ``set_parallel(True, n_workers=3)`` a
        star of 24 entries is ranged by ``get_ranges()`` and ``get_box()`` on
        the parallel path, and by ``get_ranges(parallel=False)`` on the batch
        path; every result is the exact range. Catches a maximum read from
        the LP result of another entry on the batch path, a centre left out
        of the upper bounds, and the parallel path dropping an entry."""
        set_parallel(True, n_workers=3)
        star, e_lo, e_hi = self.tall_star(24)
        lo, hi = star.get_ranges()
        self.assert_exact(lo, hi, e_lo, e_hi, "set_parallel(True, 3 workers): "
                          "get_ranges(), 24 entries")
        box = star.get_box()
        self.assert_exact(box.lb, box.ub, e_lo, e_hi,
                          "set_parallel(True, 3 workers): get_box()")
        lo, hi = star.get_ranges(parallel=False)
        self.assert_exact(lo, hi, e_lo, e_hi, "get_ranges(parallel=False), "
                          "24 entries")

    def test_auto_parallel_soundness(self):
        """``set_parallel('auto', threshold=10)``: ``get_ranges()`` of a star
        of 5 entries takes the batch path and one of 15 entries the parallel
        path; with the threshold lowered to 4 the star of 5 entries takes the
        parallel path. Every result is the exact range. Catches either path,
        as the automatic switch selects it, returning another range."""
        set_parallel('auto', threshold=10)
        small, s_lo, s_hi = self.tall_star(5)
        large, l_lo, l_hi = self.tall_star(15)
        lo, hi = small.get_ranges()
        self.assert_exact(lo, hi, s_lo, s_hi, "auto, threshold 10, 5 entries")
        lo, hi = large.get_ranges()
        self.assert_exact(lo, hi, l_lo, l_hi, "auto, threshold 10, 15 entries")
        set_parallel('auto', threshold=4)
        lo, hi = small.get_ranges()
        self.assert_exact(lo, hi, s_lo, s_hi, "auto, threshold 4, 5 entries")

    def test_repeated_calls_soundness(self):
        """One star queried again and again, in the order a reach does it:
        ``get_ranges`` on the parallel path twice, ``estimate_ranges`` (which
        stores state bounds on the star), then ``get_ranges`` on the batch
        path, ``get_box`` and ``get_ranges()`` under the global switch. Each
        LP query is the exact range and the estimate holds it. Catches a
        query that reads what an earlier query left on the star."""
        star, e_lo, e_hi = self.tall_star(12)
        for k in range(2):
            lo, hi = star.get_ranges(parallel=True, n_workers=4)
            self.assert_exact(lo, hi, e_lo, e_hi,
                              f"get_ranges(parallel=True), call {k + 1}")
        lo, hi = star.estimate_ranges()
        self.assert_exact(lo, hi, e_lo, e_hi, "estimate_ranges", tight=False)
        lo, hi = star.get_ranges(parallel=False)
        self.assert_exact(lo, hi, e_lo, e_hi, "get_ranges(parallel=False) "
                          "after estimate_ranges")
        box = star.get_box()
        self.assert_exact(box.lb, box.ub, e_lo, e_hi, "get_box after "
                          "estimate_ranges")
        set_parallel(True, n_workers=2)
        lo, hi = star.get_ranges()
        self.assert_exact(lo, hi, e_lo, e_hi, "get_ranges() under "
                          "set_parallel(True) after the other queries")

    def test_different_worker_counts_soundness(self):
        """The parallel path with 1, 2, 4 and 8 workers, given to
        ``get_ranges(parallel=True, n_workers=...)`` and through
        ``set_parallel(True, n_workers=...)`` (at sixteen entries at most
        four of them are used), on a star of 16 entries: every result is the
        exact range. Catches a thread pool that cannot run with one worker
        and a result that depends on the order in which the workers
        finish."""
        star, e_lo, e_hi = self.tall_star(16)
        for n_workers in (1, 2, 4, 8):
            lo, hi = star.get_ranges(parallel=True, n_workers=n_workers)
            self.assert_exact(lo, hi, e_lo, e_hi,
                              f"get_ranges(parallel=True, n_workers="
                              f"{n_workers})")
            set_parallel(True, n_workers=n_workers)
            lo, hi = star.get_ranges()
            self.assert_exact(lo, hi, e_lo, e_hi,
                              f"set_parallel(True, n_workers={n_workers})")


class TestHexatopeSoundness:
    """``Hexatope.from_bounds``, ``Hexatope.affine_map`` and
    ``Hexatope.estimate_ranges`` against exact references in rationals.

    The box is dyadic (``from_bounds`` rounds its centre and half-widths to
    nearest, registry HEXOCT-8, which dyadic bounds avoid) and has an entry
    above zero, one below, one across zero with ends beyond 1 and width
    above 2, and one of zero width; the map is a non-square dyadic W (4
    entries to 5, both signs, magnitudes above and below 1) with a bias. A
    set is read from its stored arrays (``rational.star``: the anchor
    variable pinned to 0), never through one of its methods. Checks are
    one-sided with no slack, except that the affine map, documented as
    exact, is also checked in the other direction."""

    LB = np.array([0.25, -3.5, -1.5, 2.0])
    UB = np.array([1.0, -1.25, 2.75, 2.0])
    W = np.array([[2.0, -0.5, 0.0, 1.0], [0.25, 1.0, -1.5, 0.0],
                  [-1.0, 0.0, 0.5, 0.75], [0.0, 2.0, 1.0, -0.25],
                  [0.5, -0.25, 0.0, 1.0]])
    B = np.array([[1.0], [-0.5], [0.25], [-2.0], [0.125]])

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    @staticmethod
    def corner_images(W, b, lb, ub):
        """The images ``W x + b`` of the corners of [lb, ub], in rationals."""
        from itertools import product
        from tests._oracles import rational as Q
        W, b = Q.matrix(W), Q.vector(np.asarray(b).reshape(-1))
        corners = [Q.vector(np.where(bits, ub, lb))
                   for bits in product((0, 1), repeat=len(lb))]
        return sorted({tuple(Q.dot(row, x) + bi for row, bi in zip(W, b))
                       for x in corners})

    @staticmethod
    def check_holds(s, points, what):
        """Every rational point is a member of ``s`` by its stored rows."""
        from tests._oracles import rational as Q
        holds = Q.Membership(Q.star(s))
        missed = [y for y in points if not holds(y)]
        assert not missed, (
            f"{what}: {len(missed)} of {len(points)} points of the true set "
            f"are not in the result, e.g. {[float(v) for v in missed[0]]}")

    @staticmethod
    def check_estimate(s, points, what):
        """``estimate_ranges`` of ``s``, returned and stored, holds the range
        of the rational points (no slack)."""
        from fractions import Fraction
        lo, hi = s.estimate_ranges()
        for kind, (l, h) in (("returned", (lo, hi)),
                             ("stored", (s.state_lb, s.state_ub))):
            l = np.asarray(l, dtype=np.float64).reshape(-1)
            h = np.asarray(h, dtype=np.float64).reshape(-1)
            if l.size != len(points[0]):
                pytest.fail(f"{what}: {l.size} {kind} bounds for "
                            f"{len(points[0])} entries")
            for i in range(l.size):
                e_lo = min(p[i] for p in points)
                e_hi = max(p[i] for p in points)
                assert (np.isfinite(l[i]) and np.isfinite(h[i])
                        and Fraction(l[i]) <= e_lo
                        and Fraction(h[i]) >= e_hi), (
                    f"{what}: estimate_ranges ({kind}) of entry {i} is "
                    f"[{l[i]!r}, {h[i]!r}], exact range "
                    f"[{float(e_lo)!r}, {float(e_hi)!r}]")

    @staticmethod
    def set_vertices(s):
        """Points whose hull is ``s``: the images of the vertices of its
        predicate polytope, read from its stored rows in rationals. The
        polytope must be bounded: no direction ``r`` with ``A r <= 0`` has a
        coordinate of size 1 (2 n exact feasibility problems, far cheaper
        than ``rational.set_vertices``' cone enumeration on five
        variables)."""
        from fractions import Fraction
        from tests._oracles import rational as Q
        q = Q.star(s)
        A, b = Q._predicate_rows(q)
        n = len(q[4])
        zeros = [Fraction(0)] * len(A)
        for j in range(n):
            for sign in (1, -1):
                row = [Fraction(-sign * int(k == j)) for k in range(n)]
                if Q.feasible((), (), A + [row],
                              zeros + [Fraction(-1)]) is not None:
                    raise AssertionError(
                        f"the {type(s).__name__} is unbounded along its "
                        f"variable {j}; the image of a box is bounded")
        return sorted({tuple(Q.dot(g, a) + ci for g, ci in zip(q[1], q[0]))
                       for a in Q.vertices(A, b)})

    @classmethod
    def check_from_bounds(cls, set_type, what):
        """``set_type.from_bounds`` of the box holds every corner, and its
        ``estimate_ranges`` holds the box."""
        s = set_type.from_bounds(cls.LB.reshape(-1, 1), cls.UB.reshape(-1, 1))
        corners = cls.corner_images(np.eye(cls.LB.size),
                                    np.zeros(cls.LB.size), cls.LB, cls.UB)
        cls.check_holds(s, corners, f"{what}: from_bounds")
        cls.check_estimate(s, corners, f"{what}: from_bounds")

    @classmethod
    def check_affine_map(cls, set_type, what):
        """``affine_map`` of the box set, with the bias as a column and with
        no bias, is the image exactly: every corner image is a member, every
        vertex of the result is the image of a point of the box, and
        ``estimate_ranges`` of the result holds the image's range."""
        from tests._oracles import rational as Q
        s = set_type.from_bounds(cls.LB.reshape(-1, 1), cls.UB.reshape(-1, 1))
        for b in (cls.B, None):
            bias = np.zeros(cls.W.shape[0]) if b is None else b
            label = f"{what}: affine_map ({'no bias' if b is None else 'bias'})"
            result = s.affine_map(cls.W, b)
            if result.dim != cls.W.shape[0]:
                pytest.fail(f"{label}: dimension {result.dim}, W has "
                            f"{cls.W.shape[0]} rows")
            images = cls.corner_images(cls.W, bias, cls.LB, cls.UB)
            cls.check_holds(result, images, label)
            image = (Q.vector(np.asarray(bias).reshape(-1)), Q.matrix(cls.W),
                     [], [], Q.vector(cls.LB), Q.vector(cls.UB))
            extra = [v for v in cls.set_vertices(result)
                     if not Q.member(image, v)]
            assert not extra, (
                f"{label}: {len(extra)} vertices of the result are not the "
                f"image of a point of the box, e.g. "
                f"{[float(x) for x in extra[0]]}")
            cls.check_estimate(result, images, label)

    def test_hexatope_bounds_computation(self):
        """``Hexatope.from_bounds`` of the dyadic box holds every corner of
        the box (exact membership by its stored rows), and
        ``estimate_ranges`` (returned and stored) holds the box with no
        slack. Catches a centre computed from the wrong combination of the
        bounds, a half-width that loses its factor, and an estimate that
        reads the wrong generators or returns one bound for both."""
        self.check_from_bounds(Hexatope, "Hexatope")

    def test_hexatope_affine_transformation(self):
        """``Hexatope.affine_map`` of the box set by a 5 x 4 dyadic W, with
        a bias column and with no bias, is the image exactly: the image of
        every corner is a member of the result, every vertex of the result
        is the image of a point of the box, and ``estimate_ranges`` of the
        result holds the image's range. Catches a bias that overwrites the
        mapped centre or is not added to it, a transposed W, and a result of
        the wrong dimension."""
        self.check_affine_map(Hexatope, "Hexatope")


class TestOctatopeSoundness:
    """``Octatope.from_bounds``, ``Octatope.affine_map`` and
    ``Octatope.estimate_ranges``: the same box, map and exact references as
    ``TestHexatopeSoundness`` (whose helpers these tests call); the set is
    read from its stored UTVPI rows."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def test_octatope_bounds_computation(self):
        """``Octatope.from_bounds`` of the dyadic box holds every corner of
        the box, and ``estimate_ranges`` (returned and stored) holds the box
        with no slack. Catches a centre computed as (ub - lb) / 2 (the box
        shifted wherever lb is not 0), a half-width that loses its factor,
        and an estimate that reads the wrong generators."""
        TestHexatopeSoundness.check_from_bounds(Octatope, "Octatope")

    def test_octatope_affine_transformation(self):
        """``Octatope.affine_map`` of the box set by a 5 x 4 dyadic W, with a
        bias column and with no bias, is the image exactly, and
        ``estimate_ranges`` of the result holds the image's range. Catches a
        bias that is computed but not added to the mapped centre, a
        transposed W, and a result of the wrong dimension."""
        TestHexatopeSoundness.check_affine_map(Octatope, "Octatope")


# =========================================================================== #
# Soundness cells: the process-pool path of the exact ReLU                    #
# =========================================================================== #
# The exact ReLU sends a list of two or more stars to a process pool when
# ``parallel=True`` (or the global ``parallel_lp`` setting) and runs them one
# by one when only one worker is available. Oracle, inputs and layout are those
# of ``test_soundness_relu_cells.py``: the exact image in rationals, checked in
# both directions, and replayed points checked by the LP oracle against the
# returned sets' rows; image sets are (C, H, W) = (2, 3, 5), compared in their
# stored (height, width, channels) order.
import torch.nn as _nn
from onnx import helper as _helper

from n2v.nn.layer_ops import relu_reach as _relu_reach
from n2v.utils.model_loader import load_onnx as _load_onnx
from tests.cell_support import (assert_covered, image_set, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx)
from tests.soundness.test_soundness_relu_cells import (
    _C, _H, _HI3, _LO3, _W, _assert_holds_image, _assert_no_extra_point, _chain, _chain_pieces, _family, _gemm_pieces,
    _gemm_weights, _image_box_few_live, _image_parts, _points,
    _relu_box_vertices, _relu_points)

_WORKERS = [pytest.param(1, id="one-worker"), pytest.param(2, id="pool")]


@pytest.mark.parametrize("n_workers", _WORKERS)
def test_parallel_exact_relu_on_several_stars_holds_the_exact_image(n_workers):
    """Exact ReLU with ``parallel=True`` on three Stars cut from a star with
    a binding row (each with crossing neurons, bounds of exactly 0 and
    LP-resolved neurons), through ``reach_layer``, by the process pool and by
    the one-worker fallback. Catches: the outputs of all but one input star
    lost (the result list overwritten as each job completes, or a return
    inside the fallback's loop after the first star; piece, regime), a star's
    bounds or results paired with another star (plumbing). Both directions
    are checked exactly. ``verbose=True`` runs the log lines of the pool and
    of the per-star split loop (the fallback hands it on positionally, the
    pool gives its workers None): the option must not change the answer
    (every other exact cell runs the default)."""
    sets, pieces, points = _family("several", "Star")
    what = (f"exact ReLU, parallel=True, n_workers={n_workers}, verbose=True, "
            f"on 3 Stars")
    out = reach_one_layer(_nn.ReLU(), sets, "exact", what, parallel=True,
                          n_workers=n_workers, verbose=True)
    _assert_holds_image(pieces, out, what)
    _assert_no_extra_point(pieces, out, what)
    assert_covered(out, _relu_points(points), what)


@pytest.mark.parametrize("n_workers", _WORKERS)
def test_parallel_exact_relu_on_several_image_stars_keeps_the_layout(
        n_workers):
    """Exact ReLU with ``parallel=True`` on three (2, 3, 5) ImageStars cut
    from one box, through ``reach_layer``. Every result is an ImageStar of
    the input's height, width and channels whose stored (height, width,
    channels) entries hold the exact image; a result returned as the
    flattened plain Star (stored order, no image shape) makes a following
    Flatten skip its reorder (layout). Catches the pool path and the
    one-worker fallback losing the image type, and either losing a star.
    (Exactness of the same kernel is checked on Stars above.)"""
    parts = _image_parts(True)
    sets = [image_set("ImageStar", a, b) for a, b in parts]
    what = (f"exact ReLU, parallel=True, n_workers={n_workers}, on 3 "
            f"ImageStars")
    out = reach_one_layer(_nn.ReLU(), sets, "exact", what, parallel=True,
                          n_workers=n_workers)
    for s in out:
        assert type(s).__name__ == "ImageStar", (
            f"{what}: returned a {type(s).__name__}, not an ImageStar")
    vertices = [v for a, b in parts for v in _relu_box_vertices(a, b)]
    assert_covered(out, vertices, what + " (vertices of the exact image)",
                   out_chw=(_C, _H, _W))
    lo, hi = _image_box_few_live()
    assert_covered(out, replay_torch(_nn.ReLU(), samples(lo, hi, 64),
                                     (_C, _H, _W)),
                   what, out_chw=(_C, _H, _W))


def test_parallel_exact_relu_chain_through_the_model_entry():
    """Linear, ReLU, Linear, ReLU, Linear through ``reach_pytorch_model``
    with ``parallel=True, n_workers=2``: the second ReLU receives the several
    pieces of the first and runs them in the process pool; the option travels
    through the model entry (plumbing). Exact in both directions, on a box
    Star and on a Star with a binding row."""
    from tests.cell_support import constrained_star, flat_set
    model = _chain(two_relus=True)
    for k, input_set in enumerate([flat_set("Star", _LO3, _HI3),
                                   constrained_star(_LO3, _HI3, seed=2)[0]]):
        what = f"two-ReLU chain, exact, parallel=True, on Star input {k}"
        out = reach_model(model, input_set, "exact", what, parallel=True,
                          n_workers=2)
        pieces = _chain_pieces(input_set, two_relus=True)
        _assert_holds_image(pieces, out, what)
        _assert_no_extra_point(pieces, out, what)
        assert_covered(out, replay_torch(model, _points(input_set, 64), (3,)),
                       what)


def test_parallel_exact_relu_relu_flatten_gemm_on_an_image_star(tmp_path):
    """ONNX Relu, Relu, Flatten, Gemm on a (1, 2, 3, 5) ImageStar through the
    model entry with ``parallel=True, n_workers=2``: the first Relu splits the
    image star at two crossing pixels, the second runs the pieces in the
    process pool, and Flatten reorders each result from its stored (height,
    width, channels) order using the image shape it carries. A pool result
    that comes back as a plain Star in the stored order is not reordered and
    Gemm reads the wrong pixels (layout). Reference: onnxruntime on the box's
    samples, and the exact image (Relu twice is Relu) in the network's order.
    Only containment is checked: the second Relu meets neurons whose LP
    minimum is exactly 0 and keeps each in one star with a slack variable of
    width about 1e-7 (the behaviour DESIGN section 18 approves for
    ``_step_relu``), so its result is exact only up to that slack."""
    weight, bias = _gemm_weights()
    nodes = [_helper.make_node("Relu", ["X"], ["R1"]),
             _helper.make_node("Relu", ["R1"], ["R2"]),
             _helper.make_node("Flatten", ["R2"], ["F"], axis=1),
             _helper.make_node("Gemm", ["F", "w", "b"], ["Y"], transB=1)]
    shape = [1, _C, _H, _W]
    path = save_onnx(tmp_path, nodes, shape, [1, 3],
                     {"w": weight.astype(np.float32),
                      "b": bias.astype(np.float32)}, name="relu_relu_gemm")
    lo, hi = _image_box_few_live()
    what = "Relu-Relu-Flatten-Gemm, exact, parallel=True, on an ImageStar"
    out = reach_model(_load_onnx(path), image_set("ImageStar", lo, hi),
                      "exact", what, parallel=True, n_workers=2)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what)
    _assert_holds_image(_gemm_pieces(), out, what)


def test_parallel_exact_relu_from_the_global_setting():
    """With ``parallel`` left unset, the global setting decides: after
    ``set_parallel(True, n_workers=2)`` a list of Stars goes to the process
    pool. The result holds the exact image in both directions (the setting
    must not change the answer)."""
    sets, pieces, points = _family("several", "Star")
    what = "exact ReLU on 3 Stars with the global parallel setting on"
    try:
        set_parallel(True, n_workers=2)
        assert _relu_reach._should_use_star_parallel(len(sets)), (
            "the global setting did not select the process pool, so this "
            "cell would not test it")
        out = reach_one_layer(_nn.ReLU(), sets, "exact", what)
    finally:
        config.reset()
    _assert_holds_image(pieces, out, what)
    _assert_no_extra_point(pieces, out, what)


class _FailingStar(Star):
    """A Star whose range estimate raises: an injected failure inside the
    exact ReLU of one star (a worker exception, as a crashed worker gives)."""

    def estimate_ranges(self, *args, **kwargs):
        raise RuntimeError("injected: this star's ReLU fails")


@pytest.mark.parametrize("n_workers", _WORKERS)
def test_parallel_exact_relu_never_drops_a_star_whose_job_fails(n_workers):
    """When the exact ReLU of one star of a list fails, the call raises; it
    never returns the other stars' results as if the failed star had no
    outputs (that empty part read as a proof is a false UNSAT). Checked for
    the process pool and the one-worker fallback; the neighbour without the
    failing star is the first cell of this block."""
    sets = list(_family("several", "Star")[0])
    bad = sets[1]
    sets[1] = _FailingStar(bad.V, bad.C, bad.d, bad.predicate_lb,
                           bad.predicate_ub)
    with pytest.raises(RuntimeError, match="injected"):
        reach_layer_out = _relu_reach.relu_star_exact(
            sets, parallel=True, n_workers=n_workers)
        pytest.fail(f"returned {len(reach_layer_out)} sets although the "
                    f"ReLU of one input star failed")
