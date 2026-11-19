"""Soundness of Sigmoid reachability: every true output of every member point
of the input set is a member of a returned set.

Sigmoid has no exact rule. The approx rules are a relaxation by tangent and
secant rows on a Star and an ImageStar (``_s_curve_single_star_approx``, which
Tanh shares), the interval [f(lb), f(ub)] on a Zono and an ImageZono, and the
interval of the end values moved outward on a Box. Hexatope and Octatope have
no rule and must refuse. The cells below say which of these each one checks
and which bug it catches; the references are described above them. One-ulp
rounding of the bounds is held by ``tests/contracts/test_smooth_rounding.py``,
``method="exact"`` on a Star by
``tests/contracts/test_reach_coverage.py::test_exact_refuses_a_layer_with_no_exact_rule``.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono
from n2v.nn.layer_ops.dispatcher import reach_layer


# =========================================================================== #
# Soundness cells                                                              #
#                                                                              #
# Sigmoid has no finite exact image, so the references are:                       #
#                                                                              #
# - a containment proof (tests/_oracles/interval_proof.py): the input set's    #
#   predicate box is covered by boxes, Sigmoid is enclosed on each in interval    #
#   arithmetic at 100 bits, and every vertex of the enclosure must be in one   #
#   returned set, decided in fractions. Used on Stars of one predicate         #
#   variable whose ranges keep away from the regimes where a relaxation row    #
#   lies within float64 rounding of the graph (an end at the inflection point, #
#   saturation, a range of a few thousandths): those are the cells of          #
#   tests/contracts/test_smooth_rounding.py's author;                          #
# - on every cell, sample points replayed in float64 (torch, or onnxruntime   #
#   on the float64 copy of an ONNX graph), each a member of a returned set by  #
#   one LP on the set's own rows at ``LP_TOL`` (tests/_oracles/membership.py   #
#   through                                                                    #
#   ``assert_covered``). The points are corners and the quarter grid of the    #
#   input box, or the half grid of the predicate box of a coupled set.         #
#                                                                              #
# Layout: a network tensor is (C, H, W); an image set stores (H, W, C), and    #
# image results are compared after the conversion ``assert_covered`` makes     #
# with ``out_chw``.                                                            #
# =========================================================================== #
from itertools import product as _cartesian

import pytest
from onnx import helper

from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof as P
from tests._oracles import rational as R
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.cell_support import (assert_covered, flat_set, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)

# Coordinates: a point first (a constant entry before varying ones: 0, where
# sigmoid is exact in float64), below 0, above 0, crossing 0 asymmetrically,
# lower bound exactly 0, upper bound exactly 0, a narrow interval, a point
# that is not 0, and a wide crossing one.
_LO = np.array([0.0, -3.0, 0.5, -0.75, 0.0, -1.5, 1.25, 0.375, -4.0])
_HI = np.array([0.0, -0.5, 2.0, 2.5, 1.0, 0.0, 1.375, 0.375, 3.0])
_N = 64


def _grid(k, steps=5):
    values = np.linspace(-1.0, 1.0, steps)
    return [np.array(a) for a in _cartesian(*[values] * k)]


def _prove(stages, star, what, max_boxes=3000):
    """Prove that the returned sets contain the image of a Star with a
    finite predicate box (and possibly more rows) under ``stages``: the
    Star's own map from its predicate variables, then the stages."""
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
def test_sigmoid_on_a_box_covers_float64(kind, parts):
    """Sigmoid through ``reach_layer`` on a Box, a Star and a Zono built from
    bounds, one set and three parts of a split box at once: every float64
    output of the corners and the quarter grid is a member by LP. The first
    coordinate is a point, so a constant entry precedes varying ones; the
    others are below 0, above 0, crossing, bounded by 0 on either side,
    narrow and wide. Catches a bound evaluated at the wrong end or of the
    wrong coordinate, a relaxation row that moves an output's range (on a
    product of intervals a row that keeps every range is invisible; the
    curve and coupled cells check the relation), a constant entry placed in
    another row, and the outputs of all but the first (or last) input set
    dropped."""
    sets = ([flat_set(kind, _LO, _HI)] if parts == 1
            else split_cover(kind, _LO, _HI, parts))
    layer = nn.Sigmoid()
    what = f"Sigmoid on {kind} ({len(sets)} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, samples(_LO, _HI, n=_N),
                                     (1, _LO.size)), what)


@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
def test_sigmoid_on_a_set_of_constants_covers_float64(kind):
    """Sigmoid on a set whose every entry is one value (0, where Sigmoid is exact
    in float64, and three others): the branch that writes f of each constant
    straight into the result, with no new variable. Every float64 output
    is a member by LP (whether f of a constant is rounded outward is not
    tested here). Catches a constant evaluated at the wrong entry
    or at the wrong end, and the branch returning the input."""
    values = np.array([0.0, 0.375, -1.5, 2.5])
    sets = [flat_set(kind, values, values)]
    layer = nn.Sigmoid()
    what = f"Sigmoid on a {kind} of constants"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, [values], (1, values.size)),
                   what)


@known_bug(
    7, "the Sigmoid Star kernel returns no set for a Star of dimension 0 (its "
       "one member, the empty vector, has the empty output), so the result "
       "says the input is empty (registry EMPTY-6)")
def test_sigmoid_on_a_star_of_dimension_zero_is_not_dropped():
    """A Star of dimension 0 over a non-empty predicate polytope is not empty,
    so ``reach_layer`` must return a set for it, or refuse it with
    ``ValueError``; no sets at all is the answer "empty", which a
    specification check reads as a proof. Today the shared Star kernel
    returns None for it and ``sigmoid_star_approx`` drops it. Catches a kernel
    that drops an input set it has not certified empty. Neighbour that
    passes today: ``test_sigmoid_on_a_set_of_constants_covers_float64[Star]``."""
    star = Star(np.zeros((0, 2)), np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                -np.ones((1, 1)), np.ones((1, 1)))
    if star.dim != 0:
        pytest.fail(f"the input Star has dimension {star.dim}, not 0")
    try:
        out = reach_layer(nn.Sigmoid(), [star], "approx")
    except ValueError:
        return
    assert len(out) >= 1, (
        "Sigmoid on a Star of dimension 0: no set returned for a non-empty "
        "input")


# One predicate variable a in [-1, 1]: entry 0 on [1, 2] (above 0), entry 1
# on [-3/2, -1] (below 0), entry 2 on [-1/2, 1] (crossing). The entries are
# tied to each other through a, so a relaxation row that relates an output
# to the input with the wrong sign or slope cuts the curve they trace.
_CURVE_C = np.array([1.5, -1.25, 0.25])
_CURVE_G = np.array([[0.5], [0.25], [0.75]])


def _curve_star():
    return Star(np.hstack([_CURVE_C[:, None], _CURVE_G]),
                np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                -np.ones((1, 1)), np.ones((1, 1)))


def test_sigmoid_on_a_curve_star_is_proved():
    """Sigmoid on a Star of one predicate variable whose three entries lie
    above 0, below 0 and across 0: the interval proof shows that the
    returned set contains the image of every input, in fractions; float64
    samples by LP as well. Catches a tangent or secant row with a flipped
    sign or slope (each entry keeps its range [f(l), f(u)], so only the
    relation between the entries shows it), rows of one neuron written for
    another, and a regime (convex side, concave side, crossing) handled as
    another."""
    star = _curve_star()
    layer = nn.Sigmoid()
    what = "Sigmoid on a Star of one predicate variable"
    out = reach_one_layer(layer, [star], "approx", what)
    result = _prove([P.SIGMOID], star, what)(out)
    assert result.proved, f"{what}: {result}"
    points = [_CURVE_C + _CURVE_G @ a for a in _grid(1, 33)]
    assert_covered(out, replay_torch(layer, points, (1, 3)), what)


# Six entries on two predicate variables a in [-1, 1]^2: entry 0 above 0,
# entry 1 below 0, entries 2 and 3 crossing, entry 4 the constant 0, entry 5
# above 0 and narrow.
_SC = np.array([1.5, -1.25, 0.25, -0.5, 0.0, 0.75])
_SG = np.array([[0.5, 0.25], [0.25, -0.5], [0.75, 0.5], [0.5, -0.75],
                [0.0, 0.0], [0.0625, 0.03125]])


def _coupled(extra=(), box="unit"):
    """The Star of the six entries, cut by the extra rows ``r a <= v``, and
    its member points (the half grid of a in [-1, 1]^2 that keeps every
    extra row). ``box`` is how the predicate variables are bounded:
    "unit", the predicate box [-1, 1]^2 (and the rows of that box, as
    ``Star.from_bounds`` writes them); "infinite", predicate bounds -inf and
    +inf with the same rows; "none", no predicate bounds and rows that bound
    the variables b = 2 a to [-2, 2], the generators halved, so the set is
    the same and a kernel that gives the missing box the value [-1, 1] cuts
    it in half."""
    rows = list(extra)
    scale = 2.0 if box == "none" else 1.0
    C = np.vstack([np.eye(2), -np.eye(2)] + [r for r, _ in rows])
    d = scale * np.concatenate([np.ones(4), [v for _, v in rows]]).reshape(
        -1, 1)
    V = np.hstack([_SC[:, None], _SG / scale])
    if box == "unit":
        star = Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))
    elif box == "infinite":
        star = Star(V, C, d, np.full((2, 1), -np.inf), np.full((2, 1), np.inf))
    else:
        star = Star(V, C, d)
    points = [_SC + _SG @ a for a in _grid(2, 9)
              if all(r @ a <= v for r, v in rows)]
    return star, points


def _family(name):
    if name == "coupled Star":
        star, points = _coupled()
        return [star], points
    if name == "Star with a binding row":
        # a0 + a1 <= 1/2 cuts off the corner (1, 1): the LP ranges of
        # entries 0, 2 and 5 are below their interval bounds
        star, points = _coupled([(np.array([1.0, 1.0]), 0.5)])
        return [star], points
    if name == "Star with no predicate bounds":
        star, points = _coupled(box="none")
        return [star], points
    if name == "Star with infinite predicate bounds":
        star, points = _coupled(box="infinite")
        return [star], points
    if name == "two halves of a split":
        low, p1 = _coupled([(_SG[2], -_SC[2])])
        high, p2 = _coupled([(-_SG[2], _SC[2])])
        return [low, high], p1 + p2
    if name == "coupled Zono":
        zono = Zono(_SC.reshape(-1, 1), _SG.copy())
        return [zono], [_SC + _SG @ a for a in _grid(2, 9)]
    raise ValueError(name)


@pytest.mark.parametrize("family", ["coupled Star", "Star with a binding row",
                                    "two halves of a split", "coupled Zono",
                                    "Star with no predicate bounds",
                                    "Star with infinite predicate bounds"])
def test_sigmoid_on_a_coupled_set_covers_float64(family):
    """Sigmoid on sets whose entries share predicate variables: a coupled Star,
    one with a binding row (its LP ranges are inside its interval bounds),
    the two halves of the Star cut at entry 2 = 0 (two input sets at once),
    a coupled Zono, and the coupled Star with no predicate bounds (its rows
    let the variables reach -2 and 2) and with predicate bounds -inf and
    +inf, both only through ``reach_layer`` (the model entry replaces such
    bounds before any layer runs). Every float64 output of the half grid is
    a member by LP. Catches the old constraint rows dropped or put in the
    wrong columns, a relaxation row tied to another entry's generators,
    ranges taken from the interval bounds where the rows bind, the outputs
    of one input set dropped, and a kernel that bounds the variables of a
    Star without a predicate box by [-1, 1]."""
    sets, points = _family(family)
    layer = nn.Sigmoid()
    what = f"Sigmoid on {family}"
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, points, (1, 6)), what)


@pytest.mark.parametrize("lp_solver", ["linprog", "SCS"])
def test_sigmoid_with_another_lp_solver_covers_float64(request, lp_solver):
    """Sigmoid on the Star with a binding row with ``lp_solver`` named; for SCS
    the ``inexact_scs`` fixture makes every SCS optimum 2e-5 inward, as SCS
    is at its default tolerance. The neuron ranges are the box of the
    relaxation, so they must come from HiGHS (padded) whatever the caller
    names: every float64 output must still be a member. Catches the range
    LPs run on the caller's inexact solver."""
    if lp_solver == "SCS":
        request.getfixturevalue("inexact_scs")
    sets, points = _family("Star with a binding row")
    layer = nn.Sigmoid()
    what = f"Sigmoid on a Star with a binding row, lp_solver={lp_solver}"
    out = reach_one_layer(layer, sets, "approx", what, lp_solver=lp_solver)
    assert_covered(out, replay_torch(layer, points, (1, 6)), what)


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_sigmoid_on_an_image_set_keeps_the_image(kind, parts):
    """Sigmoid on an ImageStar and an ImageZono of shape (2, 3, 5) with their
    own interval per pixel, one set and two parts at once: every result is
    an image set of the input's type, height, width and channels, and every
    float64 output is a member in the stored order. Catches a result
    converted back with height and width exchanged or left flat, and the
    image paths of the kernels, which no other cell reaches."""
    lo, hi = pixel_box(2, 3, 5)
    sets = ([image_set(kind, lo, hi)] if parts == 1
            else image_split_cover(kind, lo, hi, parts))
    layer = nn.Sigmoid()
    what = f"Sigmoid on {kind} ({parts} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    wanted = ImageStar if kind == "ImageStar" else ImageZono
    shapes = {(type(s), s.height, s.width, s.num_channels) for s in out}
    assert shapes == {(wanted, 3, 5, 2)}, (
        f"{what}: results {shapes}, expected {wanted.__name__} 3x5x2")
    assert_covered(out, replay_torch(layer, samples(lo, hi, n=_N),
                                     (1, 2, 3, 5)), what, out_chw=(2, 3, 5))


@pytest.mark.parametrize("kind,method", [
    pytest.param("Hexatope", "approx", id="Hexatope"),
    pytest.param("Octatope", "approx", id="Octatope"),
    pytest.param("Hexatope", "exact", id="Hexatope-exact"),
    pytest.param("Octatope", "exact", id="Octatope-exact")])
def test_sigmoid_on_a_hexatope_or_octatope_is_refused(kind, method):
    """No Sigmoid rule exists for these set types, under either method: the
    call must raise ``NotImplementedError``. Catches a placeholder branch
    that answers (the input returned as the output, say)."""
    with pytest.raises(NotImplementedError):
        reach_layer(nn.Sigmoid(), [flat_set(kind, _LO[1:4], _HI[1:4])],
                    method)


@known_bug(
    18, "reach_layer with method='exact' on a Zono (or an ImageZono) runs "
        "the Sigmoid interval rule and returns it with no warning, although "
        "a zonotope is not the image of a zonotope under Sigmoid (registry "
        "REACH-8, the Zono lane of REACH-4)",
    raises=DID_NOT_RAISE)
def test_sigmoid_exact_on_a_zono_is_refused():
    """``method="exact"`` on the coupled Zono through ``reach_layer``: Sigmoid
    has no exact rule, so the call must raise ``NotImplementedError``. The
    Zono lane of the dispatcher does not read the method (an ImageZono takes
    the same lane) and answers with the over-approximating interval rule
    under the name exact. The Star lane of the same defect is marked by
    ``tests/contracts/test_reach_coverage.py::test_exact_refuses_a_layer_with_no_exact_rule``.
    Neighbour that passes today with the same assertion:
    ``test_sigmoid_on_a_hexatope_or_octatope_is_refused[Hexatope-exact]``."""
    sets, _ = _family("coupled Zono")
    with pytest.raises(NotImplementedError):
        reach_layer(nn.Sigmoid(), sets, "exact")


# --------------------------------------------------------------------------- #
# Between two affine layers, through the model entry points                     #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0], [-0.5], [2.0]])
_B1 = np.array([1.5, -1.0, -0.25])
_W2 = np.array([[1.0, -1.0, 0.5], [0.5, 0.5, -1.0]])
_B2 = np.array([0.0, 0.25])


def _torch_chain():
    model = nn.Sequential(nn.Linear(1, 3), nn.Sigmoid(), nn.Linear(3, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(_W1))
        model[0].bias.copy_(torch.tensor(_B1))
        model[2].weight.copy_(torch.tensor(_W2))
        model[2].bias.copy_(torch.tensor(_B2))
    return model


def _onnx_chain(tmp_path, element="FLOAT"):
    """The chain as an ONNX graph of float32 ("FLOAT") or float64 ("DOUBLE")
    tensors; the weights are dyadic, so both graphs hold the same network."""
    dtype = np.float32 if element == "FLOAT" else np.float64
    cast = lambda a: np.asarray(a, dtype=dtype)      # noqa: E731
    return save_onnx(
        tmp_path,
        [helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], transB=1),
         helper.make_node("Sigmoid", ["H"], ["A"]),
         helper.make_node("Gemm", ["A", "W2", "B2"], ["Y"], transB=1)],
        [1, 1], [1, 2], {"W1": cast(_W1), "B1": cast(_B1), "W2": cast(_W2),
                         "B2": cast(_B2)},
        name=f"sigmoid_chain_{element.lower()}", in_type=element,
        out_type=element)


@pytest.mark.parametrize("entry,kind", [
    ("torch", "Star"), ("torch", "Zono"), ("torch", "Box"),
    ("onnx", "Star"), ("onnx", "Zono"), ("onnx", "Box")])
def test_sigmoid_between_two_affine_layers_holds_every_output(tmp_path, entry,
                                                           kind):
    """Linear, Sigmoid, Linear through ``reach_pytorch_model``, as a torch
    Sequential and as an ONNX graph loaded with ``load_onnx``, on the input
    interval [-1, 1.5] (hidden neurons on [1/2, 3], [-7/4, -1/2] and
    [-9/4, 11/4]: above, below and across 0, all functions of the one
    input): the float64 outputs (torch, or onnxruntime on the float64 copy
    of the graph) are members by LP, and on a Star the interval proof shows
    the whole image is contained. Catches a relaxation row that keeps every
    hidden range but not the relation the next layer reads, and the
    activation's result handed to the wrong node."""
    lo, hi = np.array([-1.0]), np.array([1.5])
    input_set = flat_set(kind, lo, hi)
    what = f"Linear, Sigmoid, Linear ({entry}) on {kind}"
    points = [np.array([v]) for v in np.linspace(-1.0, 1.5, 41)]
    if entry == "torch":
        model = _torch_chain()
        out = reach_model(model, input_set, "approx", what)
        truth = replay_torch(model, points, (1, 1))
    else:
        out = reach_model(load_onnx(_onnx_chain(tmp_path)), input_set,
                          "approx", what)
        # n2v answers for the real-valued network; a float32 replay lies a
        # float32 rounding away from it (DESIGN section 1), so the reference
        # is the float64 copy of the graph.
        truth = replay_onnx(_onnx_chain(tmp_path, "DOUBLE"), points, [1, 1])
    assert_covered(out, truth, what)
    if kind == "Star":
        result = _prove([P.linear(_W1, _B1), P.SIGMOID, P.linear(_W2, _B2)],
                        input_set, what)(out)
        assert result.proved, f"{what}: {result}"


# --------------------------------------------------------------------------- #
# Far below 0 (registry ROUND-7)                                                #
# --------------------------------------------------------------------------- #
_UNDERFLOW_TO_ZERO = known_bug(
    7, "the approx Sigmoid reach of a Star whose inputs all lie below about "
       "-745 returns the single point 0 (float64 exp underflows there), "
       "while the real sigmoid is positive, so every true output is outside "
       "the result (registry ROUND-7)")


@pytest.mark.parametrize("lo,hi", [
    pytest.param(-30.0, -20.0, id="from -30 to -20"),
    pytest.param(-800.0, -750.0, marks=_UNDERFLOW_TO_ZERO,
                 id="from -800 to -750"),
])
def test_sigmoid_on_a_star_far_below_zero_holds_its_value(lo, hi):
    """Sigmoid on a Star of one entry far below 0: the interval proof shows
    the returned set contains sigmoid of every input, in fractions (the
    real values, about 1e-326 at -750, are below the smallest float64, so
    no float replay can see this). Passes at [-30, -20]; at [-800, -750]
    the result today is the point 0. Catches an upper bound or row that
    rounds a positive function value down to 0."""
    star = Star.from_bounds(np.array([[lo]]), np.array([[hi]]))
    what = f"Sigmoid on a Star over [{lo}, {hi}]"
    # The kernel's sigmoid evaluates both branches of ``np.where``; exp
    # overflows in the branch it then discards. That float warning is not
    # the question here (warnings are errors in this suite), the answer is.
    with np.errstate(over="ignore"):
        out = reach_one_layer(nn.Sigmoid(), [star], "approx", what)
    result = _prove([P.SIGMOID], star, what, max_boxes=200)(out)
    assert result.proved, f"{what}: {result}"
