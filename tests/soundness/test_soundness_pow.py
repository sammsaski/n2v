"""Soundness cells for ONNX Pow with a constant exponent.

What the file covers. Pow is reached only through the graph walker:
``_handle_onnx_pow`` in ``n2v/nn/reach.py`` reads the exponent from the
graph, answers exponent 1 with its input sets and exponent 0 with the
constant 1 (stars only), refuses an exponent that is not a non-negative whole
number, a non-uniform exponent and a computed exponent, and passes the whole
list of input sets to ``pow_star``, ``pow_zono`` or ``pow_box``
(``n2v/nn/layer_ops/pow_reach.py``) by set type. ``reach_layer`` with the
``OnnxPow`` module refuses every set type (the exponent is on the graph
node), and a torch model's ``x ** 2`` or ``torch.pow`` is refused by the
walker. The cells cover

- Star, Zono and Box built from a box, a Star whose rows tie coordinates
  together, a badly scaled Star under each LP solver, a Star with more
  predicate variables than entries and a Zono with more generators than
  entries;
- ImageStar and ImageZono of a (2, 3, 5) image with its own interval per
  pixel, an ImageStar whose rows tie pixels of two channels, the same
  ImageStar given by its rows alone with predicate variables over [-2, 2];
- the exponents 0 to 4 and every regime of ``x ** p``: an interval crossing
  zero with either end longer, an end exactly at 0, one-sided intervals and
  fixed coordinates, one of them placed before every varying coordinate;
- the chain affine, Pow, affine (shared predicate variables, rows consumed
  downstream) on float64 graphs and on a float32 image graph with a
  convolution after the Pow, the several pieces of an exact ReLU split (flat
  and image), the two bound pre-passes before a ReLU after the Pow, the
  exponent given as a scalar, a vector, an integer tensor, a Constant node
  and a Cast;
- the refusals: Hexatope, Octatope, the exponents 1/2, 5/2, 2 + 2**-30, -1
  and -2, a non-uniform or computed exponent, exponent 0 on a Zono or a Box,
  the torch operator ``**``.

Oracle. The models are ONNX graphs, float64 except the image chain, and every
input, weight and sample point is dyadic (a multiple of a small power of
two), so the onnxruntime replay of a member point is the exact value of the
network there (``x ** p`` of such a number needs far fewer than 24 bits).
Membership is the LP oracle against the result's own rows
(``assert_covered``, slack ``LP_TOL``); the points are the corners and the
quarter grid of the input box (``samples``) or the member points the Star
builders return (corners and points on the faces of their rows). Where the
input has few varying predicate variables, ``interval_proof.prove`` with
``power(p)`` proves in exact arithmetic that the result holds the image of
the whole input set, not only of the samples: the one-node cells on every
regime (all three set kinds, the Box exact because its dyadic bounds have
exact powers) and Gemm then Pow on Star and Zono. The other cells are checked
by replay only, for the reasons the docstrings give. Nothing here compares
two n2v paths with each other. The rounding of the bounds (registry rows
ROUND-2 and ROUND-3) is the subject of ``tests/contracts/test_smooth_rounding.py``.

Layout. A flat set holds the tensor in its row-major order; an image set
stores (height, width, channel), so image outputs are compared through
``assert_covered(..., out_chw=(C, H, W))``.
"""

import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from scipy.optimize import linprog

from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof as proof
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_star,
                                coupled_star, flat_set, image_set, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                samples, save_onnx)
from tests.known_bugs import known_bug

# --------------------------------------------------------------------------- #
# Inputs                                                                        #
# --------------------------------------------------------------------------- #
# One coordinate per regime of x ** p. The fixed coordinate comes first, so a
# kernel that looks up the generator row of a varying coordinate by its rank
# among the varying ones instead of by its index reads the wrong row. 1.5 and
# -0.75 are not fixed points of x ** p = p x, so a power computed as a product
# p * x is visible on them.
_REGIMES = (
    ("fixed", 1.5, 1.5),
    ("crossing zero, left end longer", -1.25, 0.75),
    ("crossing zero, right end longer", -0.5, 1.5),
    ("lower end at 0", 0.0, 1.25),
    ("upper end at 0", -1.75, 0.0),
    ("fixed, negative", -0.75, -0.75),
    ("positive", 0.5, 1.5),
    ("negative", -2.0, -0.75),
)
_LO = np.array([r[1] for r in _REGIMES])
_HI = np.array([r[2] for r in _REGIMES])
_N = _LO.size

# The chain affine, Pow, affine on a 2-D input box: the five hidden
# coordinates cross zero, start at 0, are negative, fixed (a zero row) and
# positive, and share the two predicate variables, so a wrong tangent or
# secant row moves the jointly reachable outputs while each range stays right.
# W2 is invertible, so the second layer hides nothing.
_IN_LO, _IN_HI = np.array([-1.0, -0.5]), np.array([0.5, 1.0])
_W1 = np.array([[1.0, -0.5], [0.5, 1.0], [-1.0, 0.25], [0.0, 0.0],
                [0.25, 0.5]])
_B1 = np.array([0.25, 1.0, -1.5, 1.5, 1.0])
_W2 = np.eye(5) + 0.5 * np.roll(np.eye(5), 1, axis=1)
_B2 = np.array([0.0, 0.5, -0.25, 0.0, 1.0])

# An image with more than one channel, height != width, a different interval
# per pixel (lo in [-2, 1.75], widths 1/4 to 5/4: all regimes occur).
_C, _H, _W = 2, 3, 5


# --------------------------------------------------------------------------- #
# Models                                                                        #
# --------------------------------------------------------------------------- #
def _pow_model(tmp_path, exponent, shape, name="pow"):
    """``Y = X ** e`` on a float64 tensor of ``shape``; ``e`` is the
    initializer ``exponent`` with its own dtype and shape (a float stands for
    a float64 scalar)."""
    if not isinstance(exponent, np.ndarray):
        exponent = np.array(float(exponent))
    out_shape = list(np.broadcast_shapes(tuple(shape), exponent.shape))
    return save_onnx(tmp_path, [helper.make_node("Pow", ["X", "e"], ["Y"])],
                     list(shape), out_shape, {"e": exponent}, name=name,
                     in_type="DOUBLE", out_type="DOUBLE")


def _chain_model(tmp_path, p, W1, b1, W2, b2, relu=False, name="chain"):
    """``Y = Gemm(Pow(Gemm(X)), p)``, with a ReLU between the first Gemm and
    the Pow when ``relu``; float64."""
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1)]
    base = "A"
    if relu:
        nodes.append(helper.make_node("Relu", ["A"], ["R"]))
        base = "R"
    nodes += [helper.make_node("Pow", [base, "e"], ["B"]),
              helper.make_node("Gemm", ["B", "W2", "b2"], ["Y"], transB=1)]
    inits = {"W1": np.asarray(W1, np.float64), "b1": np.asarray(b1, np.float64),
             "e": np.array(float(p)), "W2": np.asarray(W2, np.float64),
             "b2": np.asarray(b2, np.float64)}
    return save_onnx(tmp_path, nodes, [1, np.shape(W1)[1]],
                     [1, np.shape(W2)[0]], inits, name=name,
                     in_type="DOUBLE", out_type="DOUBLE")


def _flat_input(kind, lo, hi, seed=0):
    """``(input set, member points)`` for a flat kind, or for
    ``"Star with a binding row"`` (``constrained_star``)."""
    if kind == "Star with a binding row":
        return constrained_star(lo, hi, seed=seed)
    return flat_set(kind, lo, hi), samples(lo, hi)


def _assert_proved(enclose, lb, ub, sets, what, domain=None):
    """The interval proof that ``sets`` hold ``enclose`` of the box
    [lb, ub] (restricted to ``domain``), in exact arithmetic."""
    result = proof.prove(enclose, lb, ub, *[rational.star(s) for s in sets],
                         domain=domain)
    assert result.proved, f"{what}: no containment proof: {result}"


# --------------------------------------------------------------------------- #
# x ** p on a flat set, one node                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_pow_holds_the_image_of_every_regime(tmp_path, kind, p):
    """``x ** p`` for the square, the cube and an even exponent other than 2
    on one coordinate per regime (``_REGIMES``). Catches a power computed as
    a product, a wrong range for a crossing interval (the lower bound 0 of an
    even power, l ** p of an odd one), the even branch taken for an odd
    exponent or the odd one for an even exponent other than 2, an exponent
    fixed to 2, a centre of the Zono taken as the power of the midpoint, a
    tangent or secant row that cuts the graph, a fixed coordinate given
    p times its value, rows read from the wrong coordinate, and range LPs
    used without their outward pad. The replay sees the corners and the
    quarter grid; the proof holds for every point of the box."""
    path = _pow_model(tmp_path, p, [1, _N])
    what = f"Pow {p} on {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, _LO, _HI), "approx",
                       what)
    assert_covered(sets, replay_onnx(path, samples(_LO, _HI), [1, _N]), what)
    _assert_proved(proof.chain(proof.power(p)), _LO, _HI, sets, what)


# The regime box times 4 (ends from -8 to 6, widths up to 8): the predicate
# variables of a box star range over [-1, 1], and data that stays inside that
# range can hide a value read from the wrong variable.
_WIDE_LO, _WIDE_HI = 4 * _LO, 4 * _HI
# The positive and the negative coordinate of _REGIMES. Where x ** p is
# monotone on the interval, the tangent and secant rows meet the graph at
# both ends, so an output at an end is reached by the predicate value of that
# end only.
_TIED = [6, 7]


@pytest.mark.parametrize("p", [2, 3])
def test_pow_of_a_star_with_a_binding_row(tmp_path, p):
    """The regime box times 4 as a Star with two rows that tie coordinates
    together (``coupled_star``): a6 + a1 <= 0 ties the positive coordinate to
    the crossing one whose left end is longer, a7 - a6 <= 1/2 ties the
    negative coordinate to the positive one; the points include points on
    both faces and the corners each row allows. At the ends of the tied
    coordinates the output fixes the predicate variable, so the rows decide
    membership even after one node. Catches the kernel writing the input's
    own rows on the wrong predicate columns or with the wrong sign (member
    points' outputs are then cut off); dropping the rows only loosens the
    result and is not a soundness bug."""
    path = _pow_model(tmp_path, p, [1, _N])
    star, points = coupled_star("Star", _WIDE_LO, _WIDE_HI, kept=_TIED)
    what = f"Pow {p} on a Star whose rows tie coordinates"
    sets = reach_model(load_onnx(path), star, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, _N]), what)


# --------------------------------------------------------------------------- #
# affine, Pow, affine                                                           #
# --------------------------------------------------------------------------- #
def _grid(lo, hi, steps=8):
    """Every point of the grid with ``steps`` equal steps per axis of the box
    [lo, hi] (corners included); dyadic when the bounds are."""
    axes = [a + (b - a) * np.arange(steps + 1) / steps for a, b in zip(lo, hi)]
    return [np.array(x) for x in itertools.product(*axes)]


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box",
                                  "Star with a binding row"])
def test_pow_between_two_affine_layers(tmp_path, kind, p):
    """Gemm, Pow, Gemm through the model entry. After the first Gemm the
    hidden coordinates share the input's predicate variables, so the tangent
    and secant rows tie them together; the second Gemm (invertible, so it
    hides nothing) consumes those rows. Catches a secant whose slope has the
    wrong sign, a tangent with the derivative off by the factor p or anchored
    at 0, the concave rows of a negative interval on the wrong side, the
    convex rows on an odd power that crosses zero, and a fixed hidden
    coordinate given the wrong value. The replay checks the corners, the
    quarter grid and the eighth grid of the input box (for the binding row,
    corners, half-grid points and points on the cut face); the proof of the
    whole set is ``test_pow_after_an_affine_layer_is_proved``, because no
    enclosure fits where the second Gemm mixes the face z >= 0 of an even
    power that crosses zero."""
    path = _chain_model(tmp_path, p, _W1, _B1, _W2, _B2)
    input_set, points = _flat_input(kind, _IN_LO, _IN_HI)
    if kind != "Star with a binding row":
        points = points + _grid(_IN_LO, _IN_HI)
    what = f"Gemm, Pow {p}, Gemm on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


# Gemm then Pow from one input variable to three coordinates: x + 1/4 crosses
# zero, x / 2 + 1/2 has its lower end exactly at 0, -x - 3/2 is negative.
# With one shared variable the proof needs about 70 boxes for the square;
# with two it needed about 370 (8 s), all near the corners where a tangent
# meets the graph.
_PROOF_W = np.array([[1.0], [0.5], [-1.0]])
_PROOF_B = np.array([0.25, 0.5, -1.5])
_PROOF_LO, _PROOF_HI = np.array([-1.0]), np.array([1.0])


def _affine_pow_model(tmp_path, p, W, b, name="affine_pow"):
    """``Y = Gemm(X) ** p``, float64."""
    nodes = [helper.make_node("Gemm", ["X", "W", "b"], ["A"], transB=1),
             helper.make_node("Pow", ["A", "e"], ["Y"])]
    return save_onnx(tmp_path, nodes, [1, np.shape(W)[1]], [1, np.shape(W)[0]],
                     {"W": np.asarray(W, np.float64),
                      "b": np.asarray(b, np.float64), "e": np.array(float(p))},
                     name=name, in_type="DOUBLE", out_type="DOUBLE")


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_pow_after_an_affine_layer_is_proved(tmp_path, kind, p):
    """Gemm then Pow: three coordinates that share one input predicate
    variable (crossing zero, lower end at 0, negative). The interval proof
    shows in exact arithmetic that the result holds the image of every point
    of the input interval, so a tangent or secant row that cuts the graph
    anywhere between the sample points fails, as does a secant whose slope
    has the wrong sign (the coordinates move together) and a predicate bound
    narrower than the range of the power. Box is left out: the Box after a
    Gemm has outward-padded bounds that are not dyadic, and their power
    rounded to nearest (registry row ROUND-2) is another file's subject."""
    path = _affine_pow_model(tmp_path, p, _PROOF_W, _PROOF_B)
    what = f"Gemm, Pow {p} on {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, _PROOF_LO, _PROOF_HI),
                       "approx", what)
    assert_covered(sets, replay_onnx(path, samples(_PROOF_LO, _PROOF_HI)
                                     + _grid(_PROOF_LO, _PROOF_HI, 64),
                                     [1, 1]), what)
    _assert_proved(proof.chain(proof.linear(_PROOF_W, _PROOF_B),
                               proof.power(p)), _PROOF_LO, _PROOF_HI, sets,
                   what)


# Gemm from four inputs to two coordinates, one crossing zero and one
# positive: the Pow kernel gets a Star with four predicate variables for two
# entries and a Zono with four generators for two entries, the shape a ReLU
# relaxation leaves and not the one ``from_bounds`` gives.
_WIDE_IN_LO = np.array([-1.0, -0.5, 0.25, -0.75])
_WIDE_IN_HI = np.array([0.5, 1.0, 1.25, 0.0])
_FOLD_W = np.array([[1.0, -0.5, 0.25, 0.5], [-0.25, 1.0, 0.5, -1.0]])
_FOLD_B = np.array([0.25, 2.0])
_FOLD_W2 = np.array([[1.0, 0.5], [-0.5, 1.0]])
_FOLD_B2 = np.array([0.0, 0.5])


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_pow_of_a_set_with_more_predicate_variables_than_entries(tmp_path,
                                                                 kind):
    """Gemm (four inputs to two), Pow 2, Gemm. Catches a kernel that sizes
    the generator row, the input's rows or the predicate bounds by the number
    of entries instead of the number of predicate variables (or
    generators), which only a set with more variables than entries can
    show: every other cell has at most as many."""
    path = _chain_model(tmp_path, 2, _FOLD_W, _FOLD_B, _FOLD_W2, _FOLD_B2,
                        name="fold")
    what = f"Gemm 4 to 2, Pow 2, Gemm on {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, _WIDE_IN_LO,
                                                 _WIDE_IN_HI), "approx", what)
    assert_covered(sets, replay_onnx(path, samples(_WIDE_IN_LO, _WIDE_IN_HI),
                                     [1, 4]), what)


# Gemm, Pow 2, Gemm, Relu, Gemm. The first coordinate after the first Gemm
# ranges over [1/4, 5/2] and its square over [1/16, 25/4], so the ReLU neuron
# x0 ** 2 - 3 crosses zero, while on the range of x0 itself it is negative:
# a bound pre-pass that hands the ReLU the range of x for x ** 2 calls it
# inactive. The second neuron x1 ** 2 - x0 ** 2 / 2 crosses zero too.
_PRE_W1 = np.array([[1.0, 0.5], [-1.0, 1.0]])
_PRE_B1 = np.array([1.5, 0.0])
_PRE_W2 = np.array([[1.0, 0.0], [-0.5, 1.0]])
_PRE_B2 = np.array([-3.0, 0.0])
_PRE_W3 = np.array([[1.0, 0.5], [-0.5, 1.0]])
_PRE_B3 = np.array([0.0, 0.25])


@pytest.mark.parametrize("pre_pass", ["ibp", "zono"])
def test_pow_before_a_relu_under_the_bound_pre_pass(tmp_path, pre_pass):
    """Gemm, Pow 2, Gemm, Relu, Gemm on a Star under ``precompute_bounds``.
    The IBP pre-pass has no rule for Pow and stops trusting every node after
    it, so the ReLU computes its own ranges; the zonotope pre-pass runs the
    graph on a Zono, so the ranges the ReLU of this Star reach uses come from
    ``pow_zono``. Catches a pre-pass that passes Pow on as the identity (the
    ReLU then takes a crossing neuron for an inactive one) and a ``pow_zono``
    range that is too narrow, which reaches the Star result through the
    ReLU's bounds. The Zono and Box ReLU read no precomputed bounds."""
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1),
             helper.make_node("Pow", ["A", "e"], ["P"]),
             helper.make_node("Gemm", ["P", "W2", "b2"], ["Q"], transB=1),
             helper.make_node("Relu", ["Q"], ["R"]),
             helper.make_node("Gemm", ["R", "W3", "b3"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2],
                     {"W1": _PRE_W1, "b1": _PRE_B1, "e": np.array(2.0),
                      "W2": _PRE_W2, "b2": _PRE_B2, "W3": _PRE_W3,
                      "b3": _PRE_B3}, name="pre_pass", in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"Gemm, Pow 2, Gemm, Relu, Gemm under precompute_bounds={pre_pass}"
    sets = reach_model(load_onnx(path), flat_set("Star", _IN_LO, _IN_HI),
                       "approx", what, precompute_bounds=pre_pass)
    assert_covered(sets, replay_onnx(path, samples(_IN_LO, _IN_HI)
                                     + _grid(_IN_LO, _IN_HI), [1, 2]), what)


# --------------------------------------------------------------------------- #
# The LP solver of the range LPs                                                #
# --------------------------------------------------------------------------- #
def _scaled_star(n, seed):
    """A valid, badly scaled Star: ``x = b + W a`` with large random ``W``,
    ``a`` in [-1, 1]^n cut by three rows ``H W a <= 1/2`` that keep ``a = 0``
    strictly inside, and member points computed with scipy alone: for every
    coordinate the two LP extremes, pulled by one part in a million towards
    ``a = 0``, plus random members of the predicate polytope.
    Returns ``(star, points)``."""
    rng = np.random.default_rng(seed)
    W = 3.0 * rng.normal(size=(n, n))
    b = rng.normal(size=n)
    H = rng.normal(size=(3, n)) @ W
    C = np.vstack([np.eye(n), -np.eye(n), H])
    d = np.concatenate([np.ones(2 * n), 0.5 * np.ones(3)])
    star = Star(np.hstack([b[:, None], W]), C, d[:, None], -np.ones((n, 1)),
                np.ones((n, 1)))
    points = [b.copy()]
    for i in range(n):
        for sign in (1.0, -1.0):
            res = linprog(sign * W[i], A_ub=H, b_ub=0.5 * np.ones(3),
                          bounds=[(-1.0, 1.0)] * n, method="highs")
            if res.status != 0:
                raise RuntimeError(f"extreme point LP: {res.message}")
            a = res.x * (1.0 - 1e-6)
            if np.any(H @ a > 0.5) or np.any(np.abs(a) > 1):
                raise RuntimeError("the pulled extreme point left the set")
            points.append(b + W @ a)
    while len(points) < 64:
        a = rng.uniform(-1.0, 1.0, size=n)
        if np.all(H @ a <= 0.5):
            points.append(b + W @ a)
    return star, points


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("lp_solver", [None, "linprog", "SCS"])
def test_pow_of_a_badly_scaled_star_under_each_lp_solver(tmp_path, lp_solver,
                                                         p):
    """The range LPs of the Star kernel are bounds: they must run on HiGHS
    whatever ``lp_solver`` asks for and be padded outward. On this star SCS
    misses the extreme of a coordinate by about 1e-3, so a kernel that hands
    the caller's solver to its range LPs loses the outputs of the extreme
    points (the points the replay includes)."""
    star, points = _scaled_star(4, seed=79)
    path = _pow_model(tmp_path, p, [1, 4])
    options = {} if lp_solver is None else {"lp_solver": lp_solver}
    what = f"Pow {p} on a badly scaled Star, lp_solver={lp_solver}"
    sets = reach_model(load_onnx(path), star, "approx", what, **options)
    assert_covered(sets, replay_onnx(path, points, [1, 4]), what)


# --------------------------------------------------------------------------- #
# Image sets                                                                    #
# --------------------------------------------------------------------------- #
# Two pixels of pixel_box(2, 3, 5) with one-sided intervals, in two channels
# and two rows of the image: (0, 1, 0) in [-2, -5/4] and (1, 2, 0) in
# [5/4, 5/2], as row-major (C, H, W) indices. ``coupled_star`` ties the first
# to the first other pixel (0, 0, 0), in [5/4, 5/2], and the two to each other.
_IMAGE_TIED = [5, 25]


def _image_input(kind):
    """``(input set, member points as (C, H, W) tensors)`` for an image kind
    on the per-pixel box ``pixel_box(2, 3, 5)``."""
    lo, hi = pixel_box(_C, _H, _W)
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi), samples(lo, hi)
    star, points = coupled_star("ImageStar", lo, hi, kept=_IMAGE_TIED)
    if kind == "ImageStar with a binding row":
        return star, points
    if kind == "ImageStar without predicate bounds":
        # The same set over b = 2 a: the generators and the rows are halved,
        # so the rows alone bound each variable to [-2, 2], and a kernel that
        # assumes [-1, 1] for variables without bounds cuts the set.
        V = np.array(star.V, dtype=np.float64)
        V[..., 1:] /= 2.0
        return ImageStar(V, np.asarray(star.C) / 2.0, star.d, None, None, _H,
                         _W, _C), points
    raise ValueError(kind)


_IMAGE_KINDS = ["ImageStar", "ImageZono", "ImageStar with a binding row",
                "ImageStar without predicate bounds"]


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("kind", _IMAGE_KINDS)
def test_pow_on_an_image_set(tmp_path, kind, p):
    """Pow on a (1, 2, 3, 5) tensor reached with an image set. The kernel
    flattens the ImageStar, relaxes it and rebuilds the image; the ImageZono
    is rebuilt from a centre and radii. Catches an image rebuilt with height,
    width or channels confused or its entries in the wrong order (each pixel
    has its own interval), rows written on the wrong predicate columns or
    with the wrong sign (the rows tie one-sided pixels, whose end outputs fix
    their predicate variables), and a kernel that gives predicate variables
    bounded only by the rows a box of its own."""
    path = _pow_model(tmp_path, p, [1, _C, _H, _W])
    input_set, points = _image_input(kind)
    what = f"Pow {p} on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what,
                   out_chw=(_C, _H, _W))


# A 2 x 3 kernel over both channels, two output channels; every weight a
# multiple of 1/4.
_CONV_K = ((np.arange(2 * _C * 2 * 3).reshape(2, _C, 2, 3) % 5) - 2) / 4.0
_CONV_B = np.array([0.25, -0.5])


@pytest.mark.parametrize("kind", _IMAGE_KINDS)
def test_pow_on_an_image_set_between_two_affine_layers(tmp_path, kind):
    """Add a per-channel constant, Pow 3, a convolution with a 2 x 3 kernel
    over both channels, Flatten, as a float32 graph (the corpus is float32;
    every number here is dyadic and needs fewer than 24 bits, so float32 is
    exact and the replay needs no slack). The per-channel constant reads the
    channel of every pixel before the Pow, the convolution reads each pixel's
    neighbours in both channels after it, and the Flatten converts the stored
    (height, width, channel) order to the tensor's. Catches a Pow result in
    the wrong layout or with height and width swapped, which the convolution
    then reads at the wrong places, and an exponent or constant read wrongly
    from a float32 graph."""
    shift = np.array([0.5, -0.25], dtype=np.float32).reshape(1, _C, 1, 1)
    nodes = [helper.make_node("Add", ["X", "shift"], ["A"]),
             helper.make_node("Pow", ["A", "e"], ["B"]),
             helper.make_node("Conv", ["B", "K", "kb"], ["Q"],
                              kernel_shape=[2, 3]),
             helper.make_node("Flatten", ["Q"], ["Y"])]
    n_out = 2 * (_H - 1) * (_W - 2)
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, n_out],
                     {"shift": shift, "e": np.array(3.0, dtype=np.float32),
                      "K": _CONV_K.astype(np.float32),
                      "kb": _CONV_B.astype(np.float32)}, name="image_chain")
    input_set, points = _image_input(kind)
    what = f"Add, Pow 3, Conv, Flatten on {kind} (float32 graph)"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what)


# --------------------------------------------------------------------------- #
# Several input stars: the pieces of an exact split                             #
# --------------------------------------------------------------------------- #
# Two of the four ReLU neurons cross zero on the input box, one is active and
# one inactive: up to four pieces, each with points among the samples.
_SPLIT_LO, _SPLIT_HI = np.array([-1.0, -0.5, 0.25, -0.75]), \
    np.array([0.5, 1.0, 1.25, 0.0])
_SPLIT_W1 = np.array([[1.0, 0.5, 0.0, 0.0], [-0.5, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
_SPLIT_B1 = np.array([0.0, 0.0, 0.25, -0.5])
_SPLIT_W2 = np.eye(4) - 0.5 * np.roll(np.eye(4), 1, axis=1)
_SPLIT_B2 = np.array([0.5, 0.0, -0.25, 0.0])


def _split_model(tmp_path, p):
    return _chain_model(tmp_path, p, _SPLIT_W1, _SPLIT_B1, _SPLIT_W2,
                        _SPLIT_B2, relu=True, name=f"split{p}")


@pytest.mark.parametrize("p", [1, 2, 3])
def test_pow_after_an_exact_relu_split_holds_every_piece(tmp_path, p):
    """Gemm, ReLU, Pow, Gemm under ``method='exact'``: the ReLU splits the
    input star into pieces and Pow must be applied to every piece in the
    list. Pow 1 is the identity, which is exact, so it must answer. Pow 2 and
    3 have no exact rule, so the call may refuse (``NotImplementedError``,
    registry REACH-4; the pieces then reach Pow under approx through the
    split option REACH-4 names); an answer must hold the outputs of every
    piece. The same graph with Pow 1 must answer with several pieces, so a
    refusal by the ReLU, or a split that no longer happens, fails the cell
    instead of passing it unchecked. Catches a handler or kernel that passes
    on only the first set of the list, the identity branch of exponent 1
    included, or reuses its bounds for the others."""
    path = _split_model(tmp_path, p)
    points = samples(_SPLIT_LO, _SPLIT_HI)
    patterns = {tuple(_SPLIT_W1 @ x + _SPLIT_B1 > 0) for x in points}
    if len(patterns) < 3:
        pytest.fail(f"the samples meet only {len(patterns)} activation "
                    f"patterns; the cell needs points in several pieces")
    star = flat_set("Star", _SPLIT_LO, _SPLIT_HI)
    what = f"Gemm, ReLU, Pow {p}, Gemm under exact"
    try:
        sets = reach_model(load_onnx(path), star, "exact", what)
    except NotImplementedError:
        if p == 1:
            raise
        sets = None
    if sets is not None:
        assert_covered(sets, replay_onnx(path, points, [1, 4]), what)
    pieces = sets if p == 1 else reach_model(
        load_onnx(_split_model(tmp_path, 1)), star, "exact",
        "Gemm, ReLU, Pow 1, Gemm under exact")
    if len(pieces) < 2:
        pytest.fail(f"the exact ReLU gave {len(pieces)} piece(s); the cell "
                    f"needs several to reach the Pow node")


@pytest.mark.parametrize("p", [1, 2])
def test_pow_after_an_exact_relu_split_of_an_image_holds_every_piece(
        tmp_path, p):
    """Relu then Pow on a (2, 1, 3) ImageStar (one pixel crosses zero) under
    ``method='exact'``: two ImageStar pieces reach the handler, which today
    relaxes them as under approx (registry REACH-4; Pow 1 is exact). Refusing
    Pow 2 is allowed, an answer must hold every output. Catches the image
    path of ``pow_star`` (flatten, relax, rebuild) applied to only the first
    piece, or a piece rebuilt in the wrong shape."""
    channels, height, width = 2, 1, 3
    lo, hi = pixel_box(channels, height, width)
    shape = [1, channels, height, width]
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Pow", ["R", "e"], ["Y"])]
    path = save_onnx(tmp_path, nodes, shape, shape, {"e": np.array(float(p))},
                     name="image_split", in_type="DOUBLE", out_type="DOUBLE")
    what = f"Relu, Pow {p} on an ImageStar under exact"
    try:
        sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                           "exact", what)
    except NotImplementedError:
        if p == 1:
            raise
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), shape), what,
                   out_chw=(channels, height, width))
    if p == 1 and len(sets) < 2:
        pytest.fail(f"the exact ReLU gave {len(sets)} piece(s); the cell "
                    f"needs several")


# --------------------------------------------------------------------------- #
# The exponent: 1, 0, its tensor form, and the refusals                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box",
                                  "Star with a binding row"])
def test_pow_with_exponent_one_between_two_affine_layers(tmp_path, kind):
    """``x ** 1`` is answered with the input sets themselves. Catches an
    identity that returns the wrong node's sets or drops the list."""
    path = _chain_model(tmp_path, 1, _W1, _B1, _W2, _B2, name="one")
    input_set, points = _flat_input(kind, _IN_LO, _IN_HI)
    what = f"Gemm, Pow 1, Gemm on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_pow_with_exponent_one_on_an_image_set(tmp_path, kind):
    """``x ** 1`` on a (2, 3, 5) image set is answered with the input sets.
    Catches an identity that hands on a flat set in the wrong order, the sets
    of another node, or an image whose height, width or channels changed."""
    path = _pow_model(tmp_path, 1, [1, _C, _H, _W])
    input_set, points = _image_input(kind)
    what = f"Pow 1 on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what,
                   out_chw=(_C, _H, _W))


@pytest.mark.parametrize("kind", ["Star", "Star with a binding row",
                                  "ImageStar"])
def test_pow_with_exponent_zero_is_one(tmp_path, kind):
    """``x ** 0`` is 1 everywhere, 0 ** 0 included (onnxruntime). The handler
    builds a star with every generator zero and centre 1 on the input's
    predicates. Catches a centre left at 0 or at x, and an image rebuilt in
    the wrong shape."""
    if kind == "ImageStar":
        shape = [1, _C, _H, _W]
        input_set, points = _image_input(kind)
        out_chw = (_C, _H, _W)
    else:
        shape = [1, _N]
        input_set, points = _flat_input(kind, _LO, _HI)
        out_chw = None
    path = _pow_model(tmp_path, 0, shape)
    what = f"Pow 0 on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(sets, replay_onnx(path, points, shape), what,
                   out_chw=out_chw)


@pytest.mark.parametrize("kind", ["Zono", "Box", "ImageZono"])
def test_pow_with_exponent_zero_refuses_a_zono_or_box(tmp_path, kind):
    """Exponent 0 has a rule for stars only; the other set types refuse."""
    if kind == "ImageZono":
        shape, (input_set, _) = [1, _C, _H, _W], _image_input(kind)
    else:
        shape, input_set = [1, _N], flat_set(kind, _LO, _HI)
    model = load_onnx(_pow_model(tmp_path, 0, shape))
    with pytest.raises(NotImplementedError):
        reach_model(model, input_set, "approx", f"Pow 0 on {kind}")


_BROADCAST_EXPONENT = known_bug(
    7, "Pow with a uniform constant exponent whose shape broadcasts the base "
       "to a larger tensor returns a set of the base's size: the model's "
       "output has three times as many entries as the result")


# Each form of the exponent once, on a rotating set type: the handler reads
# the exponent before it looks at the sets, so the set type does not change
# how the exponent is read.
@pytest.mark.parametrize("kind,form", [
    ("Star", "float64 of shape (1,)"),
    ("Zono", "float64 of shape (1, 8)"),
    ("Box", "float64 of shape (1, 1, 8)"),
    ("Star", "int64 scalar"),
    ("Zono", "Constant node"),
    ("Box", "Cast of an int64 initializer"),
    pytest.param("Star", "float64 of shape (3, 8)", marks=_BROADCAST_EXPONENT),
])
def test_pow_reads_its_exponent_tensor(tmp_path, kind, form):
    """The exponent 3 stored as a vector, a row, a tensor of higher rank, an
    integer tensor, the output of a Constant node and a Cast of an integer
    initializer (ONNX lets the exponent have its own type and shape, and the
    handler reads it through an initializer, an ``OnnxConstant`` module or a
    constant subgraph evaluated with torch). Catches an exponent read as 2 or
    as the wrong element, a refusal of a legal form, and a form whose value
    the handler cannot read. A shape that broadcasts the base to a larger
    output must be answered at that size or refused."""
    if form == "Constant node":
        nodes = [helper.make_node("Constant", [], ["e"],
                                  value=helper.make_tensor(
                                      "c", TensorProto.DOUBLE, [], [3.0])),
                 helper.make_node("Pow", ["X", "e"], ["Y"])]
        path = save_onnx(tmp_path, nodes, [1, _N], [1, _N], {}, name="pow",
                         in_type="DOUBLE", out_type="DOUBLE")
        exponent_shape = ()
    elif form == "Cast of an int64 initializer":
        nodes = [helper.make_node("Cast", ["k"], ["e"], to=TensorProto.DOUBLE),
                 helper.make_node("Pow", ["X", "e"], ["Y"])]
        path = save_onnx(tmp_path, nodes, [1, _N], [1, _N],
                         {"k": np.array(3, dtype=np.int64)}, name="pow",
                         in_type="DOUBLE", out_type="DOUBLE")
        exponent_shape = ()
    else:
        exponent = {
            "float64 of shape (1,)": np.array([3.0]),
            "float64 of shape (1, 8)": np.full((1, _N), 3.0),
            "float64 of shape (1, 1, 8)": np.full((1, 1, _N), 3.0),
            "int64 scalar": np.array(3, dtype=np.int64),
            "float64 of shape (3, 8)": np.full((3, _N), 3.0),
        }[form]
        path = _pow_model(tmp_path, exponent, [1, _N])
        exponent_shape = exponent.shape
    outputs = replay_onnx(path, samples(_LO, _HI), [1, _N])
    if outputs[0].size != int(np.prod(np.broadcast_shapes((1, _N),
                                                          exponent_shape))):
        pytest.fail("the reference has the wrong number of entries")
    what = f"Pow with the exponent as {form} on {kind}"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, _LO, _HI),
                           "approx", what)
    except NotImplementedError:
        if int(np.prod(exponent_shape)) > 1 and outputs[0].size > _N:
            return                       # refusing the broadcast is allowed
        raise
    assert_covered(sets, outputs, what)


# The handler refuses before it looks at the set type, so one set type runs
# every exponent.
@pytest.mark.parametrize("exponent", [
    0.5, 2.5, -1.0, -2.0,
    pytest.param(2.0 + 2.0 ** -30, id="2 + 2**-30")])
@pytest.mark.parametrize("kind", ["Star"])
def test_pow_refuses_an_exponent_that_is_not_a_non_negative_whole_number(
        tmp_path, kind, exponent):
    """A square root, fractional powers and negative powers have no rule (a
    negative power is also undefined at 0, which the box contains).
    2 + 2**-30 is a float64 number that rounds to 2 in float32: an exponent
    read through float32 would be answered as a square, and for bases above
    1 the true power lies above it."""
    model = load_onnx(_pow_model(tmp_path, exponent, [1, _N]))
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set(kind, _LO, _HI), "approx",
                    f"Pow {exponent} on {kind}")


@pytest.mark.parametrize("form", ["non-uniform exponent", "x ** x",
                                  "constant ** x"])
def test_pow_refuses_an_exponent_it_cannot_read_as_one_number(tmp_path, form):
    """An exponent that differs between entries, or that is computed from the
    input, has no rule; the input as the exponent of a constant base has
    none either."""
    if form == "non-uniform exponent":
        path = _pow_model(tmp_path, np.array([[2.0, 3.0] * (_N // 2)]),
                          [1, _N])
    else:
        node = (helper.make_node("Pow", ["X", "X"], ["Y"]) if form == "x ** x"
                else helper.make_node("Pow", ["k", "X"], ["Y"]))
        inits = {} if form == "x ** x" else {"k": np.full((1, _N), 2.0)}
        path = save_onnx(tmp_path, [node], [1, _N], [1, _N], inits,
                         name="computed", in_type="DOUBLE", out_type="DOUBLE")
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set("Star", _LO, _HI), "approx",
                    form)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_pow_refuses_a_hexatope_or_octatope(tmp_path, kind):
    """No Pow rule exists for these two set types; the handler must raise
    rather than pass the input through or answer with another type's
    rule."""
    model = load_onnx(_pow_model(tmp_path, 2, [1, _N]))
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set(kind, _LO, _HI), "approx",
                    f"Pow 2 on {kind}")


def test_reach_layer_refuses_the_pow_module(tmp_path):
    """``reach_layer`` with the ``OnnxPow`` module of a Pow 3 graph refuses
    every set type: the exponent is an input of the graph node, not an
    attribute of the module, so a dispatcher rule for the bare module can
    only guess it. Catches a dispatcher that answers the module with an
    assumed exponent (a cube answered as a square) or passes the sets
    through."""
    model = load_onnx(_pow_model(tmp_path, 3, [1, _N]))
    module = next(m for m in model.modules()
                  if type(m).__name__ == "OnnxPow")
    lo, hi = pixel_box(_C, _H, _W)
    inputs = [flat_set(kind, _LO, _HI) for kind in
              ("Star", "Zono", "Box", "Hexatope", "Octatope")]
    inputs += [image_set(kind, lo, hi) for kind in ("ImageStar", "ImageZono")]
    for input_set in inputs:
        with pytest.raises(NotImplementedError):
            reach_one_layer(module, [input_set], "approx",
                            f"OnnxPow module on {type(input_set).__name__}")


class _Square(nn.Module):
    def forward(self, x):
        return x ** 2


class _Cube(nn.Module):
    def forward(self, x):
        return torch.pow(x, 3)


@pytest.mark.parametrize("module", [_Square, _Cube], ids=["x ** 2",
                                                          "torch.pow(x, 3)"])
def test_pow_in_a_torch_model_is_refused(module):
    """The torch power operator traces to a function node with no reach
    rule; it must raise, not pass its input through."""
    with pytest.raises(NotImplementedError):
        reach_model(module(), flat_set("Star", _LO, _HI), "approx",
                    module.__name__)
