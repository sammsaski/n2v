"""Soundness cells for Sin and Cos.

Both are ONNX operators; ``load_onnx`` turns each node into onnx2torch's
``OnnxFunction`` wrapping ``torch.sin`` or ``torch.cos``, which is what the
cells pass to ``reach_layer`` (one layer) and what ``reach_pytorch_model``
meets inside a graph. There is no torch module for either: ``torch.sin`` in a
torch forward is a function node, whose refusal is the cell of
``tests/soundness/test_soundness_function_nodes.py``. The kernels are
``trig_star``, ``trig_zono`` and ``trig_box`` in
``n2v/nn/layer_ops/trig_reach.py``: on a Star, a neuron whose range holds no
extremum and no inflection point gets two tangent rows and a secant row
(convex where the function is negative, concave where it is positive), any
other neuron a free variable boxed to the function's exact range over its
interval; Zono and Box take that range. Sin and Cos share every line except
``_spec``, which picks the function, its derivative and where its extrema and
inflection points lie; Hexatope and Octatope have no rule and refuse.

Sin and Cos have no finite exact image, so the references are:

- a containment proof (``tests/_oracles/interval_proof.py``): the input
  set's predicate box is covered by boxes, the function is enclosed on each
  in interval arithmetic at 100 bits, and every vertex of the enclosure must
  be in one returned set, decided in fractions. Used on Stars of one
  predicate variable (one-sided ranges are where the tangent and secant rows
  bind);
- the exact value of the function where it is known exactly: 1 at a maximum
  of Sin, decided by mpmath at 200 bits to lie inside the input interval;
- on every cell, sample points evaluated in float64 (torch, or numpy for a
  graph written from its definition), each a member of a returned set by one
  LP on the set's own rows at ``LP_TOL`` (``assert_covered``). The points
  are the corners and the quarter grid of the input box, or the half grid
  and the vertices of the predicate polytope of a coupled set.

Ranges whose claim hinges on the last bit of a float64 function value (a
point input, an end exactly at an extremum) are the cells of
``tests/contracts/test_smooth_rounding.py`` (registry ROUND-2, ROUND-3); here
they are only checked at ``LP_TOL``. ``method='exact'`` on a Star is refused
by the contract ``tests/contracts/test_reach_coverage.py`` (REACH-4); here a
model reach under ``exact`` must refuse or hold every output.

Layout: a network tensor is (C, H, W); an image set stores (H, W, C), and
image results are compared after the conversion ``assert_covered`` makes
with ``out_chw``.
"""
from itertools import product as _cartesian

import mpmath
import numpy as np
import pytest
from onnx import helper
from onnx2torch.node_converters.functions import OnnxFunction

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof as P
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, flat_set, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_torch, samples,
                                save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_OPS = ("Sin", "Cos")
_PROOF = {"Sin": P.SIN, "Cos": P.COS}
_NUMPY = {"Sin": np.sin, "Cos": np.cos}

# Coordinates: a point first (a constant entry before varying ones; 0, where
# both are exact in float64), [1, 2] (Sin: its maximum; Cos: an inflection
# point), [-2, -1] (Sin: its minimum), [1/4, 5/4] and [-5/4, -1/4] (one
# curvature, monotone, no extremum: the tangent and secant rows), [-1/2,
# 3/4] (Sin: an inflection point; Cos: its maximum), [-4, 3] (several of
# each, wider than a period), [5/2, 4] (Cos: its minimum), [17/4, 9/2]
# (narrow, both negative and convex there) and the point 3/2.
_LO = np.array([0.0, 1.0, -2.0, 0.25, -1.25, -0.5, -4.0, 2.5, 4.25, 1.5])
_HI = np.array([0.0, 2.0, -1.0, 1.25, -0.25, 0.75, 3.0, 4.0, 4.5, 1.5])
# For several input sets the widest coordinate (the one ``split_cover`` cuts)
# is made monotone for the operation, so the parts' images are disjoint
# there and a result for one part alone cannot cover another part's outputs.
_MONOTONE = {"Sin": (-1.5, 1.5), "Cos": (0.0625, 3.0625)}
_N = 64


def _box(op, parts):
    """``(lo, hi)`` of the product cells: the coordinates above, with the
    widest replaced by a monotone stretch of the same kind when the box is
    split."""
    lo, hi = _LO.copy(), _HI.copy()
    if parts > 1:
        lo[6], hi[6] = _MONOTONE[op]
    return lo, hi


def _grid(k, steps=5):
    values = np.linspace(-1.0, 1.0, steps)
    return [np.array(a) for a in _cartesian(*[values] * k)]


def _prove(stages, star, what, max_boxes=3000):
    """Prove that the returned sets contain the image of a Star with a
    finite predicate box under ``stages``: the Star's own map from its
    predicate variables, then the stages."""
    c, G, C, d, lb, ub = R.star(star)
    if any(v is None for v in list(lb) + list(ub)):
        raise ValueError("the input Star needs a finite predicate box")
    return lambda sets: P.prove(
        P.chain(P.linear(G, c), *stages), lb, ub,
        *[R.star(s) for s in sets],
        domain=(C, d) if C else None, max_boxes=max_boxes)


# --------------------------------------------------------------------------- #
# One layer through reach_layer                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parts", [1, 3])
@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("op", _OPS)
def test_trig_on_a_box_covers_float64(op, kind, parts):
    """Sin and Cos through ``reach_layer`` on a Box, a Star and a Zono built
    from bounds, one set and three parts of a split box at once: every
    float64 output of the corners and the quarter grid is a member by LP.
    The intervals hold a maximum, a minimum, both, neither, an inflection
    point, one is wider than a period, and the first coordinate is a point,
    so a constant entry precedes varying ones. Catches a range that misses
    an interior extremum (the last one, or one of the wrong kind), a tangent
    or secant row on the wrong side, a curvature decided on the wrong side,
    a neuron relaxed over another entry's interval, a row built from another
    coordinate's generators (a constant entry before it shifts the varying
    ones), and the outputs of all but one input set dropped (for three parts
    the cut coordinate is monotone, so no part's result covers another's)."""
    lo, hi = _box(op, parts)
    sets = ([flat_set(kind, lo, hi)] if parts == 1
            else split_cover(kind, lo, hi, parts))
    layer = OnnxFunction(op)
    what = f"{op} on {kind} ({len(sets)} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, samples(lo, hi, n=_N),
                                     (1, lo.size)), what)


@pytest.mark.parametrize("kind", ["Star"])
@pytest.mark.parametrize("op", _OPS)
def test_trig_on_a_set_of_constants_covers_float64(op, kind):
    """Sin and Cos on a Star whose every entry is one value (0, where both
    are exact in float64, and three others): the Star kernel's branch with
    no varying entry, where no new variable is made and each constant is
    written straight into the result. Every float64 output is a member by LP
    (whether f of a constant is rounded outward is the cell of
    ``tests/contracts/test_smooth_rounding.py``). Catches that branch
    returning its input unchanged and a constant written with the other
    function. A Zono and a Box have no such branch: their constant entries
    take the interval code, which the box cell's constant coordinates run."""
    values = np.array([0.0, 0.375, -1.5, 2.5])
    sets = [flat_set(kind, values, values)]
    layer = OnnxFunction(op)
    what = f"{op} on a {kind} of constants"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, [values], (1, values.size)),
                   what)


# One predicate variable a in [-1, 1], three entries tied through it. Sin:
# [1/4, 5/4] (concave, rising), [-5/4, -1/4] (convex, rising), [1, 2] (its
# maximum: a boxed variable). Cos: [1/4, 5/4] (concave, falling), [2, 3]
# (convex, falling), [-1/2, 1/2] (its maximum).
_CURVE = {"Sin": (np.array([0.75, -0.75, 1.5]), np.array([[0.5], [0.5],
                                                          [0.5]])),
          "Cos": (np.array([0.75, 2.5, 0.0]), np.array([[0.5], [0.5],
                                                        [0.5]]))}


def _curve_star(op):
    c, G = _CURVE[op]
    return Star(np.hstack([c[:, None], G]), np.array([[1.0], [-1.0]]),
                np.ones((2, 1)), -np.ones((1, 1)), np.ones((1, 1)))


@pytest.mark.parametrize("op", _OPS)
def test_trig_on_a_curve_star_is_proved(op):
    """Sin and Cos on a Star of one predicate variable whose entries are on
    a concave stretch, a convex stretch and across an extremum: the interval
    proof shows that the returned set contains the image of every input, in
    fractions; float64 samples by LP as well. Catches a tangent or secant
    row with a flipped sign or slope, or relating the output to another
    entry (each entry keeps its range, so only the curve shows it), and a
    curvature decided on the wrong side."""
    star = _curve_star(op)
    c, G = _CURVE[op]
    layer = OnnxFunction(op)
    what = f"{op} on a Star of one predicate variable"
    out = reach_one_layer(layer, [star], "approx", what)
    result = _prove([_PROOF[op]], star, what)(out)
    assert result.proved, f"{what}: {result}"
    points = [c + G @ a for a in _grid(1, 33)]
    assert_covered(out, replay_torch(layer, points, (1, 3)), what)


# Six entries on two predicate variables a in [-1, 1]^2: on a concave
# stretch, on a convex stretch, across an extremum, across an inflection
# point, the constant 0, and narrow.
_SC = np.array([0.75, -0.75, 1.5, 3.0, 0.0, 4.375])
_SG = np.array([[0.25, 0.25], [0.25, -0.25], [0.5, 0.25], [0.25, -0.5],
                [0.0, 0.0], [0.0625, 0.03125]])
# Three entries on five predicate variables (more variables than entries):
# each entry's range is its centre plus or minus 9/16, so for Sin [3/16,
# 21/16] is concave, [-21/16, -3/16] convex and [15/16, 33/16] holds the
# maximum; for Cos the first two are concave and the third holds an
# inflection point. The row a0 + ... + a4 <= 1 cuts the corners with four
# or five variables at 1 off, and binds every range from above.
_WC = np.array([0.75, -0.75, 1.5])
_WG = np.array([[0.25, 0.125, 0.0, 0.0625, 0.125],
                [0.125, -0.25, 0.0625, 0.0, 0.125],
                [0.25, 0.0, 0.125, 0.125, -0.0625]])


def _coupled(extra=(), box=True, scale=1.0):
    """The six-entry Star over a in [-1, 1]^2 cut by the rows ``extra``.
    With ``scale`` s its variables are b = s a, bounded by the rows
    ``|b_i| <= s`` and generators divided by s: the same set. ``box`` gives
    it the predicate bounds [-s, s], ``"infinite"`` gives it -inf and +inf
    (the rows alone bound it), False none."""
    rows = list(extra)
    C = np.vstack([np.eye(2), -np.eye(2)] + [r for r, _ in rows])
    d = np.concatenate([np.full(4, scale), [scale * v for _, v in rows]])
    V = np.hstack([_SC[:, None], _SG / scale])
    if box is True:
        star = Star(V, C, d.reshape(-1, 1), np.full((2, 1), -scale),
                    np.full((2, 1), scale))
    elif box == "infinite":
        star = Star(V, C, d.reshape(-1, 1), np.full((2, 1), -np.inf),
                    np.full((2, 1), np.inf))
    else:
        star = Star(V, C, d.reshape(-1, 1))
    points = [_SC + _SG @ a for a in _grid(2, 9)
              if all(r @ a <= v for r, v in rows)]
    return star, points


def _wide_points(n=64, seed=0):
    """Member points of the five-variable set: every vertex of its
    predicate polytope (the corners of [-1, 1]^5 with at most three
    variables at 1) and ``n`` points of the half grid that keep the row."""
    corners = [np.array(a) for a in _cartesian(*[(-1.0, 1.0)] * 5)
               if sum(a) <= 1]
    rng = np.random.default_rng(seed)
    grid = []
    while len(grid) < n:
        a = rng.integers(-2, 3, size=5) / 2.0
        if a.sum() <= 1:
            grid.append(a)
    return [_WC + _WG @ a for a in corners + grid]


def _family(name):
    if name == "Star with a binding row":
        # a0 + a1 <= 1/2 cuts off the corner (1, 1)
        star, points = _coupled([(np.array([1.0, 1.0]), 0.5)])
        return [star], points
    if name == "Star with no predicate bounds":
        star, points = _coupled(box=False, scale=2.0)
        return [star], points
    if name == "Star with infinite predicate bounds":
        star, points = _coupled(box="infinite", scale=2.0)
        return [star], points
    if name == "two halves of a split":
        low, p1 = _coupled([(_SG[3], 0.0)])
        high, p2 = _coupled([(-_SG[3], 0.0)])
        return [low, high], p1 + p2
    if name == "coupled Zono":
        zono = Zono(_SC.reshape(-1, 1), _SG.copy())
        return [zono], [_SC + _SG @ a for a in _grid(2, 9)]
    if name == "Star with more variables than entries":
        C = np.vstack([np.eye(5), -np.eye(5), np.ones((1, 5))])
        d = np.ones((11, 1))
        star = Star(np.hstack([_WC[:, None], _WG]), C, d, -np.ones((5, 1)),
                    np.ones((5, 1)))
        return [star], _wide_points()
    if name == "Zono with more generators than entries":
        zono = Zono(_WC.reshape(-1, 1), _WG.copy())
        corners = [np.array(a) for a in _cartesian(*[(-1.0, 1.0)] * 5)]
        return [zono], [_WC + _WG @ a for a in corners]
    raise ValueError(name)


@pytest.mark.parametrize("family", [
    "Star with a binding row", "Star with no predicate bounds",
    "Star with infinite predicate bounds", "two halves of a split",
    "coupled Zono", "Star with more variables than entries",
    "Zono with more generators than entries"])
@pytest.mark.parametrize("op", _OPS)
def test_trig_on_a_coupled_set_covers_float64(op, family):
    """Sin and Cos on sets whose entries share predicate variables: a Star
    with a binding row (its LP ranges are inside its interval bounds), the
    same set given by rows that bound its variables to [-2, 2] with no
    predicate bounds and with infinite ones (``reach_layer`` hands both to
    the kernel as they are), the two halves of the Star cut through entry 3
    (two input sets at once), a coupled Zono, and a Star and a Zono with
    more variables than entries. Every float64 output of the half grid (and
    of every vertex of the predicate polytope where the set has more
    variables than entries) is a member by LP. Catches the old constraint
    rows dropped or put in the wrong columns, a relaxation row tied to
    another entry's generators or with a sign flipped, ranges taken from the
    interval bounds where the rows bind, predicate bounds of [-1, 1] given
    to a Star that has none, and the outputs of one input set dropped."""
    sets, points = _family(family)
    layer = OnnxFunction(op)
    what = f"{op} on {family}"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, points, (1, points[0].size)),
                   what)


@pytest.mark.parametrize("lp_solver", ["SCS"])
@pytest.mark.parametrize("op", ["Sin"])
def test_trig_with_another_lp_solver_covers_float64(request, op, lp_solver):
    """Sin on the Star with a binding row with ``lp_solver='SCS'`` named;
    the ``inexact_scs`` fixture makes every SCS optimum 2e-5 inward, as SCS
    is at its default tolerance. The row binds the ranges of entries 0 and
    2, and the points include every vertex of the predicate polytope. The
    neuron ranges are the box of the relaxation, so they must come from
    HiGHS (padded) whatever the caller names. Catches the range LPs run on
    the caller's inexact solver. Other spellings ('default', 'linprog',
    'highs') are HiGHS solvers the kernel keeps, the code of the default
    cells."""
    request.getfixturevalue("inexact_scs")
    sets, points = _family("Star with a binding row")
    layer = OnnxFunction(op)
    what = f"{op} on a Star with a binding row, lp_solver={lp_solver}"
    out = reach_one_layer(layer, sets, "approx", what, lp_solver=lp_solver)
    assert_covered(out, replay_torch(layer, points, (1, 6)), what)


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("op", _OPS)
def test_trig_on_an_image_set_keeps_the_image(op, kind, parts):
    """Sin and Cos on an ImageStar and an ImageZono of shape (2, 3, 5) with
    their own interval per pixel, one set and two parts at once (the two
    parts on the box scaled by 4, so intervals reach 5 wide and bounds 8):
    every result is an image set of the input's type, height, width and
    channels, and every float64 output is a member in the stored order.
    Catches a result converted back with height and width exchanged or left
    flat, the outputs of the second part dropped, and the image paths of the
    kernels, which no other cell of this file reaches."""
    lo, hi = pixel_box(2, 3, 5)
    if parts > 1:
        lo, hi = 4 * lo, 4 * hi
    sets = ([image_set(kind, lo, hi)] if parts == 1
            else image_split_cover(kind, lo, hi, parts))
    layer = OnnxFunction(op)
    what = f"{op} on {kind} ({parts} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    wanted = ImageStar if kind == "ImageStar" else ImageZono
    shapes = {(type(s), s.height, s.width, s.num_channels) for s in out}
    assert shapes == {(wanted, 3, 5, 2)}, (
        f"{what}: results {shapes}, expected {wanted.__name__} 3x5x2")
    assert_covered(out, replay_torch(layer, samples(lo, hi, n=_N),
                                     (1, 2, 3, 5)), what, out_chw=(2, 3, 5))


@pytest.mark.parametrize("op,kind,method", [
    pytest.param("Sin", "Hexatope", "approx", id="Hexatope"),
    pytest.param("Sin", "Octatope", "approx", id="Octatope"),
    pytest.param("Cos", "Hexatope", "approx", id="Cos-Hexatope"),
    pytest.param("Cos", "Octatope", "approx", id="Cos-Octatope"),
    pytest.param("Sin", "Hexatope", "exact", id="Hexatope-exact"),
    pytest.param("Sin", "Octatope", "exact", id="Octatope-exact"),
    pytest.param("Cos", "Hexatope", "exact", id="Cos-Hexatope-exact"),
    pytest.param("Cos", "Octatope", "exact", id="Cos-Octatope-exact"),
])
def test_trig_on_a_hexatope_or_octatope_is_refused(op, kind, method):
    """No Sin or Cos rule exists for these set types: ``reach_layer`` must
    raise ``NotImplementedError`` under either method (the model entry
    raises ``ValueError`` for ``exact`` on them before any node, the cell of
    ``tests/soundness/test_soundness_graph_walk.py``). Catches a placeholder
    branch that answers one function or one method with the input set."""
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxFunction(op),
                    [flat_set(kind, _LO[1:4], _HI[1:4])], method)


@known_bug(
    18, "reach_layer with method='exact' on a Zono (or an ImageZono) runs "
        "the Sin and Cos interval rule and returns it with no warning; a "
        "zonotope with a generator per entry is not the image of a coupled "
        "zonotope (REACH-8, with the Sigmoid, Tanh, LeakyReLU "
        "and MaxPool2d branches of the same lane)",
    raises=DID_NOT_RAISE)
def test_trig_exact_on_a_zono_is_refused():
    """``method="exact"`` on the coupled Zono through ``reach_layer``: Sin
    has no exact rule, so the call must raise ``NotImplementedError``. The
    trig branch of the Zono lane does not read the method (an ImageZono
    takes the same branch, Cos the same line) and answers with the interval
    box, whose corner (0.95, -0.25, 1.0) lies 0.16 from the image. The model
    entry refuses exact on a Zono before any node. Neighbour that passes
    today with the same assertion:
    ``test_trig_on_a_hexatope_or_octatope_is_refused[Hexatope-exact]``."""
    sets, _ = _family("coupled Zono")
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxFunction("Sin"), sets, "exact")


# --------------------------------------------------------------------------- #
# Lists that mix set families, through reach_layer                              #
# --------------------------------------------------------------------------- #
# A (2, 3, 5) tensor with a narrow interval per pixel, the step differing
# between the tensor's order and the stored (height, width, channel) one.
# The first set lies on [-3/2, -1/4], the second on [1/2, 7/4]: Sin rises
# on both and their images do not meet, so neither result can hold the
# other set's outputs.
_STEP = (np.arange(30).reshape(2, 3, 5) % 7) * 0.125
_FIRST = (-1.5 + _STEP, -1.5 + _STEP + 0.0625)
_SECOND = (0.5 + _STEP, 0.5 + _STEP + 0.0625)

_IMAGE_STAR_AFTER_A_ZONO_IS_FLATTENED = known_bug(
    7, "reach_layer of Sin or Cos on a list led by a Zono or an ImageZono "
       "sends a later ImageStar through trig_zono, which builds a flat Zono "
       "of its ranges in the stored (height, width, channel) order: the "
       "image shape is lost and every output of that set lies outside the "
       "result read in the tensor's own order (registry REACH-6, the "
       "dispatcher picks the kernel by the first set)")


@pytest.mark.parametrize("first,second", [
    pytest.param("Zono", "ImageStar", id="Zono-then-ImageStar",
                 marks=_IMAGE_STAR_AFTER_A_ZONO_IS_FLATTENED),
    pytest.param("Zono", "ImageZono", id="Zono-then-ImageZono"),
])
def test_trig_on_a_list_mixing_set_families_refuses_or_holds_every_output(
        first, second):
    """Sin through ``reach_layer`` on a list of a flat Zono and an image set
    of the same (2, 3, 5) tensor: the call refuses (``NotImplementedError``
    or ``ValueError``), or the outputs of each set are members of the
    result, image results compared in their stored order and flat results
    in the tensor's own. Catches an image set returned flat in its stored
    order. The other orders crash today (``AttributeError``, ``TypeError``;
    registry REACH-6, held by the dispatcher cell planned for every layer)."""
    def make(kind, lo, hi):
        if kind.startswith("Image"):
            return image_set(kind, lo, hi)
        return flat_set(kind, lo.reshape(-1), hi.reshape(-1))
    what = f"Sin on [{first}, {second}]"
    try:
        out = reach_one_layer(OnnxFunction("Sin"),
                              [make(first, *_FIRST), make(second, *_SECOND)],
                              "approx", what)
    except (NotImplementedError, ValueError):
        return
    for name, (lo, hi) in (("first", _FIRST), ("second", _SECOND)):
        assert_covered(out, [np.sin(p) for p in samples(lo, hi, n=32)],
                       f"{what}, outputs of the {name} set", out_chw=(2, 3, 5))


# --------------------------------------------------------------------------- #
# Inputs at the edges of the float range, through the model entry point         #
# --------------------------------------------------------------------------- #
def _one_node(tmp_path, op, size):
    return save_onnx(tmp_path, [helper.make_node(op, ["X"], ["Y"])],
                     [1, size], [1, size], name=op)


@pytest.mark.parametrize("case", [
    pytest.param("an infinite lower bound", marks=known_bug(
        23, "trig_box (through _trig_range and _grid_in) converts the "
            "infinite end of a Box to an integer and raises OverflowError "
            "where the range [-1, 1] or a refusal is due",
        raises=OverflowError)),
    "finite bounds",
])
def test_trig_on_a_box_with_an_infinite_end_refuses_or_holds_every_output(
        tmp_path, case):
    """Sin through ``reach_pytorch_model`` on an ONNX graph, on a Box whose
    first coordinate is [-inf, 1/2] (a Box may hold an infinite bound) or,
    the neighbour that passes today, [-64, 1/2]; the second is [1/4, 3/4].
    The call refuses with ``NotImplementedError``, or the result has no NaN
    bound and holds the float64 outputs of the finite box [-64, 1/2] x
    [1/4, 3/4] (members of the input either way). Catches a crash or a NaN
    bound on an infinite end."""
    lo = np.array([-np.inf if case.startswith("an infinite") else -64.0,
                   0.25])
    hi = np.array([0.5, 0.75])
    what = f"Sin on a Box with {case}"
    try:
        sets = reach_model(load_onnx(_one_node(tmp_path, "Sin", 2)),
                           Box(lo, hi), "approx", what)
    except NotImplementedError:
        return
    for s in sets:
        assert not (np.isnan(s.lb).any() or np.isnan(s.ub).any()), (
            f"{what}: the result has a NaN bound, which holds no point")
    finite_lo = np.array([-64.0, 0.25])
    assert_covered(sets, [np.sin(p) for p in samples(finite_lo, hi)], what)


@pytest.mark.parametrize("start", [
    pytest.param(1e14, id="1e14", marks=known_bug(
        7, "_grid_in places the critical points of Sin and Cos at offset + "
           "k * float64(pi), which drifts from the true extremum by about k "
           "times 1.2e-16 plus the rounding of the product; far from 0 the "
           "range misses the extremum's value: the upper bound of Sin over "
           "[1e14, 1e14 + 8] is 0.999998993 where the true maximum is 1 "
           "(4e-16 short at 1e9, 1.7e-7 at 1e13, 0.076 at 1e16)")),
    pytest.param(1e6, id="1e6"),
])
def test_sin_far_from_zero_holds_the_extrema_inside_its_interval(tmp_path,
                                                                 start):
    """Sin through ``reach_pytorch_model`` on a Box [s, s + 8], which holds
    a maximum pi/2 + 2 pi k and a minimum 3 pi/2 + 2 pi k strictly inside
    (shown by mpmath at 200 bits): the true outputs 1 and -1 are reached, so
    the returned box must hold both, with no slack. Catches a range whose
    critical points are not where the extrema are, and one that drops the
    last critical point of the interval. The neighbour at 1e6 passes today
    with the same reference."""
    lo, hi = float(start), float(start) + 8.0
    with mpmath.workprec(200):
        for offset in (mpmath.pi / 2, 3 * mpmath.pi / 2):
            k = mpmath.ceil((mpmath.mpf(lo) - offset) / (2 * mpmath.pi))
            extremum = offset + 2 * mpmath.pi * k
            if not mpmath.mpf(lo) < extremum < mpmath.mpf(hi):
                pytest.fail(f"setup: the extremum {extremum} is not inside "
                            f"[{lo!r}, {hi!r}]")
    what = f"Sin on the Box [{lo!r}, {hi!r}]"
    sets = reach_model(load_onnx(_one_node(tmp_path, "Sin", 1)),
                       Box(np.array([lo]), np.array([hi])), "approx", what)
    assert_covered(sets, [np.ones(1), -np.ones(1)], what, tol=0.0)


# --------------------------------------------------------------------------- #
# Between two affine layers, through the model entry point                      #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0], [-0.5], [2.0]])
_B1 = np.array([0.5, -1.0, -0.25])
_W2 = np.array([[1.0, -1.0, 0.5], [0.5, 0.5, -1.0]])
_B2 = np.array([0.0, 0.25])


def _onnx_chain(tmp_path, op):
    f32 = lambda a: np.asarray(a, dtype=np.float32)      # noqa: E731
    return save_onnx(
        tmp_path,
        [helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], transB=1),
         helper.make_node(op, ["H"], ["A"]),
         helper.make_node("Gemm", ["A", "W2", "B2"], ["Y"], transB=1)],
        [1, 1], [1, 2], {"W1": f32(_W1), "B1": f32(_B1), "W2": f32(_W2),
                         "B2": f32(_B2)}, name=f"{op}_chain")


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
@pytest.mark.parametrize("op", _OPS)
def test_trig_between_two_affine_layers_holds_every_output(tmp_path, op,
                                                           kind):
    """Gemm, op, Gemm loaded with ``load_onnx`` and reached through
    ``reach_pytorch_model`` on the input interval [-1, 1/2] (hidden neurons
    on [-1/2, 1], [-5/4, -1/2] and [-9/4, 3/4]: one-sided, across an
    inflection point, across an extremum, all functions of the one input):
    the network's outputs, computed in float64 from its definition (the
    weights are dyadic, so the float32 graph is the same network), are
    members by LP, and on a Star the interval proof shows the whole image is
    contained. Catches a relaxation row that keeps every hidden range but
    not the relation the next layer reads, and Sin and Cos sent to each
    other's kernel on the model route."""
    lo, hi = np.array([-1.0]), np.array([0.5])
    input_set = flat_set(kind, lo, hi)
    what = f"Gemm, {op}, Gemm on {kind}"
    points = [np.array([v]) for v in np.linspace(-1.0, 0.5, 41)]
    f = _NUMPY[op]
    outputs = [_W2 @ f(_W1 @ x + _B1) + _B2 for x in points]
    out = reach_model(load_onnx(_onnx_chain(tmp_path, op)), input_set,
                      "approx", what)
    assert_covered(out, outputs, what)
    if kind == "Star":
        result = _prove([P.linear(_W1, _B1), _PROOF[op], P.linear(_W2, _B2)],
                        input_set, what)(out)
        assert result.proved, f"{what}: {result}"


@pytest.mark.parametrize("op,kind", [("Sin", "Star"), ("Cos", "ImageStar")],
                         ids=["Sin-Star", "Cos-ImageStar"])
def test_trig_under_exact_refuses_or_holds_every_output(tmp_path, op, kind):
    """``method="exact"`` through ``reach_pytorch_model`` on the one-node
    graph of a (2, 3, 5) tensor, a flat Star or an ImageStar with its own
    interval per pixel: Sin and Cos have no exact rule, so the call refuses
    with ``NotImplementedError`` (the settled rule, registry REACH-4,
    asserted by ``tests/contracts/test_reach_coverage.py``) or, as today,
    returns the approx relaxation, which must hold every float64 output.
    Catches an exact branch that answers with something other than an
    enclosure (the input returned as a placeholder)."""
    lo, hi = pixel_box(2, 3, 5, seed=7)
    path = save_onnx(tmp_path, [helper.make_node(op, ["X"], ["Y"])],
                     [1, 2, 3, 5], [1, 2, 3, 5], name=op)
    input_set = (image_set(kind, lo, hi) if kind == "ImageStar"
                 else flat_set(kind, lo.reshape(-1), hi.reshape(-1)))
    what = f"{op} under exact on a {kind}"
    try:
        sets = reach_model(load_onnx(path), input_set, "exact", what)
    except NotImplementedError:
        return
    outputs = [_NUMPY[op](p) for p in samples(lo, hi)]
    if kind == "ImageStar":
        assert_covered(sets, outputs, what, out_chw=(2, 3, 5))
    else:
        assert_covered(sets, [y.reshape(-1) for y in outputs], what)
