"""Smooth operations: the reach set of one neuron holds every value of the
function, to the last bit (DESIGN sections 1 and 17, PROTOCOL section 3.2).

Sigmoid, Tanh, Sin, Cos and Pow have no exact rule, so "approx" returns an
enclosure: an interval for a Box or a Zono, a polygon of rows for a Star. The
enclosure touches the function wherever a bound is a function value: at the
ends of the range, and for a Star wherever a tangent or a chord meets the
curve. A bound computed there by round-to-nearest lands inside the function
about half the time, by one unit in the last place. A sound kernel moves each
such bound outward by more than its own rounding error.

The reference is ``tests/_oracles/interval_proof.py``: a proof, for every
point of the range and not for samples, that the value is in the computed set,
in rationals with the function enclosed by mpmath at 100 bits. A range counts
only when it is proved. A refutation shows the point that is left out. An
undecided range is one where the set meets the function with no margin at all,
so no box around the contact can be shown inside, and it is a failure too.

Each test takes one operation on one set type through a family of ranges and
asserts that every one is proved. A single range would pass or fail by the
luck of one last bit, on this processor's ``exp`` and ``sin``. A family fails
as a whole when the kernel does not round outward: it has ranges whose bound
is rounded inward on every IEEE machine (Tanh from 20 up and Sigmoid from 40
up are 1.0 in float64 and below 1 in fact; cos(2**-30) is 1.0 and sin(2**-30)
is 2**-30), ranges that end at an inflection point, where a Star's tangent row
has no curvature to hide behind, and ranges with ends that are not dyadic.

For a Box and a Zono the result is an interval and the claim is about the
values: every f(x) is in it. For a Star the claim is about the rows: every
point (x, f(x)) satisfies them, read on the predicate variables the output
star shares with its input, which is what the next layer consumes.

A neuron whose input is one value takes another path through each kernel: no
relaxation is built and the output is ``f`` of the constant. The second test
gives every operation such inputs. A point is the right answer only where the
function's value is a float64 number (``f(0)``, a power of a dyadic number).

Each marked cell has a passing neighbour with the same operation, set type,
entry and reference, on inputs where today's kernel holds the function
(ranges and values whose bounds are float64 numbers, or whose rows have a
margin), so a kernel that fails for another reason than the rounding is
seen. Softmax keeps one marked cell per kernel that computes its bound and
every route keeps its neighbours. The last test is a case of ROUND-13 (Sin
far from 0 on less than a period).
"""

from fractions import Fraction
from itertools import product

import numpy as np
import onnx
import pytest
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, Star, Zono
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof as proof
from tests._oracles import rational
from tests.known_bugs import known_bug

# A failing test names this many ranges and stops; one is enough to fail it.
SHOWN = 3
# A range is proved in about a hundred boxes and given up as undecided in
# under a thousand. A proof that is still running after this many is posed
# beyond the oracle's precision; it raises, which no ``xfail`` here absorbs.
BOXES = 3000

_TINY = 2.0 ** -30

# Ranges in saturation, then around and ending at the inflection point at 0,
# then ordinary ones. The first range of each list is the one where float64
# gives 1.0 at both ends. It stops where it does because the proof works at
# 100 bits: Tanh above 40 is closer to 1 than that, and a set that leaves it
# out could not be told from one that holds it.
_S_CURVE = [
    (3.0, 8.0), (-9.0, -5.0),
    (-2.0 ** -10, 2.0 ** -10), (-2.0 ** -20, 2.0 ** -20),
    (0.0, 0.5), (0.0, 1.0), (0.0, 4.0), (-1.0, 0.0), (-4.0, 0.0),
    (-2.0, 2.0), (1.0, 3.0), (-3.0, -1.0), (-0.5, 0.25), (0.1, 0.3),
]
_SIGMOID = [(40.0, 80.0)] + _S_CURVE
_TANH = [(20.0, 40.0)] + _S_CURVE
# Ranges with an extremum inside, at an end, and none; with an inflection
# point inside and none; longer than a period.
_TRIG = [
    (_TINY, 1.0), (-_TINY, _TINY), (0.0, 1.0), (-0.5, 0.25), (0.125, 0.375),
    (0.1, 0.3), (0.5, 1.5), (1.0, 3.0), (-3.0, -1.0), (-2.0, 2.0),
    (-1.0, 3.0), (-4.0, 0.5), (4.0, 6.0), (5.0, 6.0), (2.5, 7.25),
    (-10.0, 10.0),
]
# A power of a dyadic number is exact in float64, so most ends are not dyadic.
_POWER = [
    (0.1, 0.3), (1 / 3, 2 / 3), (0.7, 1.1), (0.01, 0.07), (2.5, 7.3),
    (-1.3, -0.6), (-0.001, 0.001), (-0.7, 1.9), (-1 / 7, 1 / 3), (-0.9, 0.2),
    (0.0, 1.0), (1.0, 3.0), (-3.0, -1.0), (-2.0, 2.0), (100.0, 1000.0),
]

# op: (ONNX node type, initializers, the function for the proof)
_OPS = {
    "Sigmoid": ("Sigmoid", {}, proof.SIGMOID),
    "Tanh": ("Tanh", {}, proof.TANH),
    "Sin": ("Sin", {}, proof.SIN),
    "Cos": ("Cos", {}, proof.COS),
    "Pow2": ("Pow", {"p": np.array(2.0, dtype=np.float32)}, proof.power(2)),
    "Pow3": ("Pow", {"p": np.array(3.0, dtype=np.float32)}, proof.power(3)),
}
_RANGES = {"Sigmoid": _SIGMOID, "Tanh": _TANH, "Sin": _TRIG, "Cos": _TRIG,
           "Pow2": _POWER, "Pow3": _POWER}
# One value per neuron. The first of each list is rounded inward on every IEEE
# machine, the last is a value float64 holds exactly, the others are ordinary.
_CONSTANTS = {
    "Sigmoid": [40.0, 0.5, -3.0, 0.1, 0.0],
    "Tanh": [20.0, 0.5, -3.0, 0.1, 0.0],
    "Sin": [_TINY, 1.0, -2.5, 0.1, 0.0],
    "Cos": [_TINY, 1.0, -2.5, 0.1, 0.0],
    "Pow2": [0.1, 1 / 3, -0.7, 1.9, 2.0],
    "Pow3": [0.1, 1 / 3, -0.7, 1.9, 2.0],
}

_STAR_ROW_AT_ITS_COMPUTED_VALUE = known_bug(
    7, "the Sigmoid and Tanh star relaxation pads the neuron's range and "
       "leaves each row at its float64 value: where the padded range gives "
       "the row no margin (an end at the inflection point, a range in "
       "saturation, a range of a few thousandths) the row cuts into the "
       "graph or touches it")
_INTERVAL_AT_ITS_COMPUTED_VALUES = known_bug(
    7, "the interval is [f(lb), f(ub)] (with the extrema inside the range "
       "for Sin and Cos) as float64 computes them, not rounded outward, so "
       "an end can lie inside the function's range")
_CONSTANT_AT_ITS_COMPUTED_VALUE = known_bug(
    7, "a neuron whose input is one value gets f of it as float64 computes "
       "it: a point, which is not the function's value unless that value is a "
       "float64 number")


def _model(tmp_path, op):
    """A float32 opset-13 graph with the one node, [1, 1] to [1, 1]."""
    node, inits, _ = _OPS[op]
    graph = helper.make_graph(
        [helper.make_node(node, ["X"] + list(inits), ["Y"])], op,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        initializer=[numpy_helper.from_array(a, k) for k, a in inits.items()])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = str(tmp_path / f"{op}.onnx")
    onnx.save(model, path)
    return load_onnx(path)


def _input(kind, lo, hi):
    lo, hi = np.array([[lo]]), np.array([[hi]])
    if kind == "Box":
        return Box(lo, hi)
    return {"Star": Star, "Zono": Zono}[kind].from_bounds(lo, hi)


def _ends(input_set):
    """The exact ends of a one-dimensional input set built from bounds. They
    are read from the set: the centre and half-width that ``from_bounds``
    stores for bounds that are not dyadic give an interval one unit in the
    last place wider than the bounds."""
    if isinstance(input_set, Box):
        return (rational.frac(input_set.lb.item()),
                rational.frac(input_set.ub.item()))
    (c,), (g,), C, d, lb, ub = rational.star(input_set)
    for row, bound in zip(C, d):
        if sum(max(a * low, a * high)
               for a, low, high in zip(row, lb, ub)) > bound:
            pytest.fail("a constraint row of the input set cuts into the box "
                        "of its predicate bounds, so its ends are not its "
                        "centre and generators over that box")
    reach = [sorted((a * low, a * high)) for a, low, high in zip(g, lb, ub)]
    return (c + sum(r[0] for r in reach), c + sum(r[1] for r in reach))


def _with_its_input(input_star, output_star):
    """The set of (x, y) over the predicate variables of ``output_star``, in
    the form of ``rational.star``. The output star must keep its input's
    predicate system in front; if it does not, the pairing means nothing and
    the test stops with an error no ``xfail`` absorbs."""
    c_in, G_in, C_in, d_in, lb_in, ub_in = rational.star(input_star)
    c, G, C, d, lb, ub = rational.star(output_star)
    n, m = len(lb_in), len(C_in)
    if not (lb[:n] == lb_in and ub[:n] == ub_in and d[:m] == d_in
            and [row[:n] for row in C[:m]] == C_in
            and all(v == 0 for row in C[:m] for v in row[n:])):
        pytest.fail("the output star does not keep its input's predicate "
                    "variables and rows in front, so it cannot be read "
                    "together with its input")
    G_in = [row + [Fraction(0)] * (len(lb) - n) for row in G_in]
    return (c_in + c, G_in + G, C, d, lb, ub)


def unproved(model, op, kind, ranges, limit=SHOWN):
    """The ranges that are not proved, at most ``limit`` of them, each with
    what the proof found."""
    function = _OPS[op][2]
    found = []
    for lo, hi in ranges:
        input_set = _input(kind, lo, hi)
        sets = reach_pytorch_model(model, input_set, method="approx")
        if not sets:
            pytest.fail(f"{op} on {kind} [{lo}, {hi}]: no reach sets; an "
                        f"empty result is not the defect a marker names")
        if kind == "Star":
            enclose = proof.graph(proof.chain(function))
            computed = [_with_its_input(input_set, s) for s in sets]
        elif kind == "Zono":
            enclose, computed = proof.chain(function), [
                rational.star(s) for s in sets]
        else:
            enclose, computed = proof.chain(function), [
                rational.interval(s.lb.ravel(), s.ub.ravel()) for s in sets]
        low, high = _ends(input_set)
        result = proof.prove(enclose, [low], [high], *computed,
                             max_boxes=BOXES)
        if result.proved:
            continue
        if result.refuted is not None:
            x = float(result.refuted[0])
            found.append(f"[{lo!r}, {hi!r}]: the value at x = {x!r} is "
                         f"outside the set")
        else:
            a, b = result.undecided[0][0]
            found.append(f"[{lo!r}, {hi!r}]: the set has no margin around "
                         f"x = {float(a)!r} (width {float(b - a):.1e})")
        if limit is not None and len(found) >= limit:
            break
    return found


@pytest.mark.parametrize("op,kind", [
    pytest.param("Sigmoid", "Star", marks=_STAR_ROW_AT_ITS_COMPUTED_VALUE),
    pytest.param("Sigmoid", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    ("Sigmoid", "Box"),
    pytest.param("Tanh", "Star", marks=_STAR_ROW_AT_ITS_COMPUTED_VALUE),
    pytest.param("Tanh", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    ("Tanh", "Box"),
    ("Sin", "Star"),
    pytest.param("Sin", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    pytest.param("Sin", "Box", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    ("Cos", "Star"),
    pytest.param("Cos", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    pytest.param("Cos", "Box", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    ("Pow2", "Star"),
    pytest.param("Pow2", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    pytest.param("Pow2", "Box", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    ("Pow3", "Star"),
    pytest.param("Pow3", "Zono", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
    pytest.param("Pow3", "Box", marks=_INTERVAL_AT_ITS_COMPUTED_VALUES),
])
def test_a_smooth_operation_encloses_its_function_on_every_range(
        tmp_path, op, kind):
    """Catches an interval or a relaxation row of a smooth operation that a
    computation by round-to-nearest leaves on the inner side of the
    function (registry ROUND-1 for the Sigmoid and Tanh Star rows, ROUND-2
    for the intervals of the Zono and Box rules), and a bound placed at the
    wrong end, a missed extremum or a row with the wrong slope, on the
    family of ranges of each operation, through ``reach_pytorch_model`` of a
    one-node ONNX graph. Sigmoid and Tanh on a Box move their bounds outward
    and pass. The Star rules for Sin, Cos and Pow pass because the padded
    range leaves every row a margin far above its rounding error."""
    found = unproved(_model(tmp_path, op), op, kind, _RANGES[op])
    assert not found, (
        f"{op} on {kind}: not proved on these ranges (the search stops at "
        f"{SHOWN}):\n  " + "\n  ".join(found))


@pytest.mark.parametrize("op,kind", [
    pytest.param("Sigmoid", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Sigmoid", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    ("Sigmoid", "Box"),
    pytest.param("Tanh", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Tanh", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    ("Tanh", "Box"),
    pytest.param("Sin", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Sin", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Sin", "Box", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Cos", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Cos", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Cos", "Box", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow2", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow2", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow2", "Box", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow3", "Star", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow3", "Zono", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
    pytest.param("Pow3", "Box", marks=_CONSTANT_AT_ITS_COMPUTED_VALUE),
])
def test_a_smooth_operation_on_one_value_holds_the_functions_value(
        tmp_path, op, kind):
    """Catches a neuron whose input is one value and whose output is the
    point f(x) as float64 computes it (registry ROUND-3), on values whose
    image is not a float64 number. Sigmoid and Tanh on a Box move the two
    equal bounds apart and pass."""
    points = [(v, v) for v in _CONSTANTS[op]]
    found = unproved(_model(tmp_path, op), op, kind, points)
    assert not found, (
        f"{op} on {kind}: not proved on these values (the search stops at "
        f"{SHOWN}):\n  " + "\n  ".join(found))


# The neighbours of the marked cells above: the same operations, set types,
# entry and reference, on inputs where today's kernels hold the function.
# On one value: values whose image is a float64 number (f(0) for Sigmoid,
# Tanh, Sin and Cos; the powers of 2 and -1/2). On a range: for a Zono or a
# Box, ranges whose bounds are float64 numbers (Sigmoid from 0, where 1/2 is
# exact, into saturation, where the computed 1 is the asymptote; Tanh
# saturated at both ends; Sin and Cos over both of their extrema; powers of
# dyadic ends; Sigmoid below about -709 overflows the exp of the branch it
# discards, a RuntimeWarning that the suite raises); for a Star, ranges
# whose relaxation rows have a margin (none of the three regimes of
# ROUND-1).
_FLOAT_IMAGE_VALUES = {"Sigmoid": [0.0], "Tanh": [0.0], "Sin": [0.0],
                       "Cos": [0.0], "Pow2": [2.0, -0.5],
                       "Pow3": [2.0, -0.5]}
_HELD_RANGES = {
    ("Sigmoid", "Zono"): [(0.0, 40.0)],
    ("Tanh", "Zono"): [(-20.0, 20.0)],
    ("Sigmoid", "Star"): [(1.0, 3.0), (-2.0, 2.0)],
    ("Tanh", "Star"): [(1.0, 3.0), (-2.0, 2.0)],
}
for _op in ("Sin", "Cos"):
    for _kind in ("Zono", "Box"):
        _HELD_RANGES[(_op, _kind)] = [(-4.0, 4.0)]
for _op in ("Pow2", "Pow3"):
    for _kind in ("Zono", "Box"):
        _HELD_RANGES[(_op, _kind)] = [(0.5, 2.0), (-2.0, 1.5)]


@pytest.mark.parametrize("op,kind", sorted(_HELD_RANGES))
def test_a_smooth_operation_encloses_its_function_where_today_it_holds(
        tmp_path, op, kind):
    """The passing neighbour of each marked cell of
    ``test_a_smooth_operation_encloses_its_function_on_every_range``.
    Catches an interval or a relaxation row that leaves the function out by
    more than rounding: a bound at the wrong end, a centre or a radius
    computed wrongly, a missed extremum, a row with the wrong slope or on
    the wrong side; the ends of these ranges are float64 numbers, so the
    rounding of ROUND-1 and ROUND-2 does not show."""
    found = unproved(_model(tmp_path, op), op, kind, _HELD_RANGES[(op, kind)],
                     limit=None)
    assert not found, (f"{op} on {kind}: not proved:\n  " + "\n  ".join(found))


@pytest.mark.parametrize("op,kind", [
    (op, kind) for op in _OPS for kind in ("Star", "Zono", "Box")
    if not (op in ("Sigmoid", "Tanh") and kind == "Box")])
def test_a_smooth_operation_on_one_value_with_a_float_image_holds_it(
        tmp_path, op, kind):
    """The passing neighbour of each marked cell of
    ``test_a_smooth_operation_on_one_value_holds_the_functions_value``, on
    values whose image is a float64 number, so the rounding of ROUND-3 does
    not show. Catches a constant neuron that is given its input instead of
    f of it, the value of another neuron, or no value (the constant branch
    of the Sigmoid and Tanh Star kernel, and the zero-width intervals of
    the other rules)."""
    points = [(v, v) for v in _FLOAT_IMAGE_VALUES[op]]
    found = unproved(_model(tmp_path, op), op, kind, points, limit=None)
    assert not found, (f"{op} on {kind}: not proved on the values "
                       f"{_FLOAT_IMAGE_VALUES[op]}:\n  " + "\n  ".join(found))


# A constant neuron beside a varying one takes the other constant branch of
# the Sigmoid and Tanh Star kernel (the varying neurons are relaxed, the
# constant ones get f of their value). ROUND-3 on that branch: x = 1/2, whose
# image is not a float64 number; the neighbour x = 0, whose image 1/2 (Sigmoid)
# or 0 (Tanh) is.
@pytest.mark.parametrize("op,value", [
    pytest.param(op, value, id=f"{op}-{value}",
                 marks=_CONSTANT_AT_ITS_COMPUTED_VALUE if value else ())
    for op in ("Sigmoid", "Tanh") for value in (0.0, 0.5)])
def test_a_constant_neuron_beside_a_varying_one_holds_the_functions_value(
        op, value):
    """``reach_pytorch_model`` of ``nn.Sequential`` of the activation on the
    Star of the box [value, value] x [1, 3]: entry 0 is constant and entry 1
    varies. At x1 = 1, 2 and 3 the enclosure (100 bits) of the true output
    (f(value), f(x1)) must be inside one returned set, decided in rationals
    on its rows. Catches a constant neuron given its input, another neuron's
    value or f of it rounded to nearest (ROUND-3 at 1/2) on the branch where
    the other neurons are relaxed."""
    module, function = {"Sigmoid": (nn.Sigmoid, proof.SIGMOID),
                        "Tanh": (nn.Tanh, proof.TANH)}[op]
    star = Star.from_bounds(np.array([[value], [1.0]]),
                            np.array([[value], [3.0]]))
    if star.nVar != 1:
        pytest.fail(f"the input Star has {star.nVar} predicate variables, "
                    f"so entry 0 is not a constant neuron")
    sets = reach_pytorch_model(nn.Sequential(module()), star, method="approx")
    if not sets:
        pytest.fail(f"{op}: no reach sets")
    holders = [rational.Membership(rational.star(s)) for s in sets]
    enclose = proof.chain(function)
    for x1 in (1, 2, 3):
        low, high = proof.value(enclose, [Fraction(value), Fraction(x1)])
        corners = list(product(*[sorted({a, b}) for a, b in zip(low, high)]))
        assert any(all(holder(list(c)) for c in corners)
                   for holder in holders), (
            f"{op} of the constant {value} beside x1 = {x1}: the true output "
            f"{[float(v) for v in low]} is in none of the {len(sets)} sets")


# Sin and Cos place their critical points at offset + k * float64(pi), which
# drifts from the true extremum by about k * 1.2e-16 (registry ROUND-13).
# Far from 0 an interval narrower than a period that holds an extremum gets
# a bound short of it. The case of ``test_soundness_trig.py`` is 8 wide,
# more than 2 pi, so a fix that gives every interval of 2 pi or more the
# range [-1, 1] passes it; this one is 3 wide.
_ROUND13_SHORT_OF_THE_EXTREMUM = known_bug(
    7, "_grid_in places the critical points of Sin at offset + k * "
       "float64(pi), which drifts from the true extremum by about k times "
       "1.2e-16, and _trig_range takes the value at the computed point as "
       "the extremum: on an interval of width 3 near 1e14 that holds a "
       "maximum of Sin the upper bound is short of 1 by about 1e-6 on a Box "
       "and on a Zono")


@pytest.mark.parametrize("kind,start", [
    pytest.param(kind, start, id=f"{kind}-{start:g}",
                 marks=_ROUND13_SHORT_OF_THE_EXTREMUM if start > 1e7 else ())
    for kind in ("Box", "Zono") for start in (1e14, 1e6)])
def test_sin_far_from_zero_on_less_than_a_period_holds_its_maximum(
        tmp_path, kind, start):
    """Sin through ``reach_pytorch_model`` of a one-node ONNX graph on
    [s, s + 3], where s is the integer below the first maximum
    pi / 2 + 2 pi k after ``start`` minus 1 (located by mpmath at 200 bits,
    so the maximum lies inside, 1 from the left end): the true output 1 is
    reached, so it must be a member of a returned set, in rationals with no
    slack. Catches a critical point placed where the extremum is not
    (ROUND-13 at 1e14) and a range that drops an interior extremum. The
    neighbour at 1e6 passes today with the same reference."""
    import mpmath
    with mpmath.workprec(200):
        k = mpmath.ceil((mpmath.mpf(start) - mpmath.pi / 2) / (2 * mpmath.pi))
        maximum = mpmath.pi / 2 + 2 * mpmath.pi * k
        lo = float(mpmath.floor(maximum)) - 1.0
        hi = lo + 3.0
        if not (mpmath.mpf(lo) < maximum < mpmath.mpf(hi)
                and hi - lo < 2 * mpmath.pi):
            pytest.fail(f"setup: the maximum {maximum} is not inside "
                        f"[{lo!r}, {hi!r}] or the interval is a period wide")
    sets = reach_pytorch_model(_model(tmp_path, "Sin"),
                               _input(kind, lo, hi), method="approx")
    if not sets:
        pytest.fail(f"Sin on the {kind} [{lo!r}, {hi!r}]: no reach sets")
    holders = [rational.Membership(rational.star(s)) for s in sets]
    assert any(holder([Fraction(1)]) for holder in holders), (
        f"Sin on the {kind} [{lo!r}, {hi!r}]: the true maximum 1 is in none "
        f"of the {len(sets)} returned sets")


# --------------------------------------------------------------------------- #
# Softmax                                                                       #
# --------------------------------------------------------------------------- #
# Softmax couples its entries, so the one-neuron proof above does not apply.
# It is increasing in its own logit and decreasing in every other one, so over
# a box each entry takes its least and its greatest value at a corner. The
# claim checked here is the one the corners decide: at every vertex of the
# input set (read from the set's own rows, so a constructor that widens the
# box gives the true corners) the exact softmax, enclosed at 100 bits by the
# proof's interval arithmetic, is a member of one returned set, decided in
# rationals on that set's rows. The kernels return a set shaped like a box (a
# Box, a Zono of a box, a Star over fresh variables), so for them this is the
# claim that every bound holds the least or the greatest value of its entry.
#
# Inputs, each family a list of boxes: random boxes of two to four logits with
# widths from 1e-6 to 1 (a bound computed by round-to-nearest lands inside
# about half the time); saturated logits, where float64 gives 1.0 for a value
# below 1 on every IEEE machine; one point, where no input range is padded;
# and equal logits, where every value is 1/n, a float64 number, so a correct
# kernel returns it exactly (the neighbour that shows the reference right).
def _softmax(xs):
    """Softmax of a list of intervals in the proof's interval arithmetic:
    entry i is 1 / sum_j exp(x_j - x_i)."""
    return [1 / sum(proof.IV.exp(xj - xi) for xj in xs) for xi in xs]


_SOFTMAX = proof.chain(proof.function(_softmax))


def _softmax_boxes(family):
    """The input boxes of a family, as pairs of float64 arrays."""
    if family == "random":
        rng = np.random.default_rng(20261113)
        boxes = []
        for _ in range(5):
            n = int(rng.integers(2, 5))
            centre = rng.uniform(-4.0, 4.0, n)
            radius = 10.0 ** rng.uniform(-6.0, 0.0, n)
            boxes.append((centre - radius, centre + radius))
        return boxes
    if family == "saturated":
        return [(np.array(lo), np.array(hi)) for lo, hi in (
            ([40.0, 0.0], [80.0, 0.0]), ([40.0, -1.0], [80.0, 1.0]),
            ([37.0, 0.0], [38.0, 0.0]), ([30.0, 0.0], [30.5, 0.0]),
            ([0.0, 45.0, -1.0], [0.0, 60.0, 0.5]))]
    if family == "point":
        rng = np.random.default_rng(20261114)
        points = [rng.uniform(-4.0, 4.0, int(n)) for n in (2, 3, 4, 3)]
        return [(x, x.copy()) for x in points]
    if family == "equal":
        return [(np.zeros(2), np.zeros(2)), (np.full(4, 0.75), np.full(4, 0.75)),
                (np.full(4, -2.5), np.full(4, -2.5))]
    raise ValueError(family)


def _softmax_set(kind, lo, hi):
    lo, hi = lo.reshape(-1, 1), hi.reshape(-1, 1)
    if kind == "Box":
        return Box(lo, hi)
    return {"Star": Star, "Zono": Zono}[kind].from_bounds(lo, hi)


def _softmax_model(tmp_path, opset, n):
    """A float32 graph with one Softmax node over the last axis of [1, n]:
    ``axis=1`` at opset 11 (softmax of the flattened row), ``axis=-1`` at
    opset 13."""
    graph = helper.make_graph(
        [helper.make_node("Softmax", ["X"], ["Y"],
                          axis=1 if opset < 13 else -1)], "softmax",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, n])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = str(tmp_path / f"softmax_{opset}_{n}.onnx")
    onnx.save(model, path)
    return load_onnx(path)


def _softmax_reach(tmp_path, route, input_set, n):
    if route == "reach_layer":
        return reach_layer(nn.Softmax(dim=-1), [input_set], "approx")
    if route == "Sequential":
        return reach_pytorch_model(nn.Sequential(nn.Softmax(dim=-1)),
                                   input_set, method="approx")
    if route == "NeuralNetwork":
        return NeuralNetwork(nn.Sequential(nn.Softmax(dim=-1))).reach(
            input_set, method="approx")
    opset = {"onnx11": 11, "onnx13": 13}[route]
    return reach_pytorch_model(_softmax_model(tmp_path, opset, n), input_set,
                               method="approx")


def softmax_misses(tmp_path, route, kind, family, limit=SHOWN):
    """The vertices of the input sets whose exact softmax is in no returned
    set, at most ``limit`` of them, each with the input box. An input the
    route refuses (``NotImplementedError``, which keeps the promise) is
    skipped; None when the route refuses every input of the family."""
    found, answered = [], 0
    for lo, hi in _softmax_boxes(family):
        input_set = _softmax_set(kind, lo, hi)
        try:
            sets = _softmax_reach(tmp_path, route, input_set, lo.size)
        except NotImplementedError:
            continue
        answered += 1
        if not sets:
            pytest.fail(f"Softmax on {kind} {lo}..{hi}: no reach sets; an "
                        f"empty result is not the defect a marker names")
        holders = [rational.Membership(rational.star(s)) for s in sets]
        for x in rational.set_vertices(rational.star(input_set)):
            low, high = proof.value(_SOFTMAX, list(x))
            corners = list(product(*[sorted({a, b}) for a, b in zip(low, high)]))
            if any(all(holder(list(y)) for y in corners) for holder in holders):
                continue
            found.append(f"box [{lo.tolist()}, {hi.tolist()}]: softmax at the "
                         f"vertex {[float(v) for v in x]} = "
                         f"{[float(v) for v in low]} is in no returned set")
            if limit is not None and len(found) >= limit:
                return found
    return found if answered else None


_SOFTMAX_BOUND_AT_ITS_COMPUTED_VALUE = known_bug(
    7, "the Softmax bounds are computed from exp and division by "
       "round-to-nearest with no outward step (softmax_reach._softmax_bounds "
       "for reach_layer on a Box and on a Star, correlated_row_softmax_bounds "
       "for every model reach of a Softmax node), so a bound can lie inside "
       "the exact least or greatest value of its entry by up to 2.75e-16: "
       "about half the bounds on random boxes, and on saturated logits the "
       "least value of the leading entry, which float64 rounds to 1.0")

_SOFTMAX_CELLS = []
for _route in ("reach_layer", "Sequential", "NeuralNetwork", "onnx11",
               "onnx13"):
    for _kind in ("Box", "Zono", "Star"):
        for _family in ("random", "saturated", "point", "equal"):
            # Passing today: equal logits everywhere (exact values); the
            # dispatcher's Star kernel on random boxes, whose input ranges it
            # pads by 1e-7 before the bounds; reach_layer on a Zono, which
            # refuses.
            sound = (_family == "equal"
                     or (_route == "reach_layer" and _kind == "Zono")
                     or (_route == "reach_layer" and _kind == "Star"
                         and _family == "random"))
            # One marked cell per kernel that computes the bound, on the
            # saturated family (inside on every IEEE machine): softmax_box
            # and softmax_star through reach_layer, softmax_attn_box, _zono
            # and _star through the ONNX graph at opset 13. The other routes
            # reach the same three kernels, and the random and point
            # families run the same lines; their marked cells were removed
            # as excess expected failures, and every route keeps its
            # neighbours.
            if not sound and not (_family == "saturated"
                                  and _route in ("reach_layer", "onnx13")):
                continue
            _SOFTMAX_CELLS.append(pytest.param(
                _route, _kind, _family,
                marks=() if sound else _SOFTMAX_BOUND_AT_ITS_COMPUTED_VALUE,
                id=f"{_route}-{_kind}-{_family}"))


@pytest.mark.parametrize("route,kind,family", _SOFTMAX_CELLS)
def test_softmax_holds_its_value_at_every_vertex_of_its_input(
        tmp_path, route, kind, family):
    """Catches a Softmax bound that is not moved outward (registry ROUND-6),
    on every entry point: ``reach_layer``, ``reach_pytorch_model`` of
    ``nn.Sequential(nn.Softmax(dim=-1))`` and of an ONNX Softmax at opsets 11
    and 13, and ``NeuralNetwork.reach``; and a bound, a centre or a row of
    the returned set that leaves out the value at any vertex. A refusal
    (``NotImplementedError``, today ``reach_layer`` on a Zono) keeps the
    promise and passes."""
    found = softmax_misses(tmp_path, route, kind, family)
    if found is None:
        return
    assert not found, (
        f"Softmax through {route} on {kind}, {family} inputs: (the search "
        f"stops at {SHOWN}):\n  " + "\n  ".join(found))


@pytest.mark.parametrize("route,kind", [
    (route, kind)
    for route in ("reach_layer", "Sequential", "NeuralNetwork", "onnx11",
                  "onnx13")
    for kind in ("Box", "Zono", "Star")])
def test_softmax_holds_its_value_near_every_corner_of_its_input(tmp_path,
                                                                route, kind):
    """The neighbour of the random-family cells above with the same boxes,
    builders, routes and reference, at points where no bound is touched:
    each vertex of the input set moved 1/32 of the way to the centre. A
    bound one unit in the last place inside (ROUND-6) does not reach these
    points, so the cells pass today on every route; a bound taken from the
    wrong corner (the least value of an entry read with its own logit low
    and every other logit low too, instead of high) or placed on the wrong
    entry leaves them out. A refusal (``NotImplementedError``, today
    ``reach_layer`` on a Zono) keeps the promise."""
    found = []
    for lo, hi in _softmax_boxes("random"):
        input_set = _softmax_set(kind, lo, hi)
        try:
            sets = _softmax_reach(tmp_path, route, input_set, lo.size)
        except NotImplementedError:
            continue
        if not sets:
            pytest.fail(f"Softmax on {kind} through {route}: no reach sets")
        holders = [rational.Membership(rational.star(s)) for s in sets]
        vertices = rational.set_vertices(rational.star(input_set))
        centre = [sum(v[i] for v in vertices) / len(vertices)
                  for i in range(lo.size)]
        for v in vertices:
            x = [vi + (ci - vi) / 32 for vi, ci in zip(v, centre)]
            low, high = proof.value(_SOFTMAX, x)
            corners = list(product(*[sorted({a, b})
                                     for a, b in zip(low, high)]))
            if not any(all(holder(list(y)) for y in corners)
                       for holder in holders):
                found.append(f"box [{lo.tolist()}, {hi.tolist()}]: softmax "
                             f"at {[float(t) for t in x]} is in no set")
                break
    assert not found, (f"Softmax through {route} on {kind}:\n  "
                       + "\n  ".join(found))
