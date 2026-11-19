"""Self-test of the interval proof oracle (``tests/_oracles/interval_proof.py``).

An oracle that proves everything proves nothing. The reference here shares no
code with the oracle and does not use mpmath: Exp, Sin and Cos are summed from
their series in ``fractions.Fraction`` with a bound on the tail, and Sigmoid
and Tanh are built from Exp. The enclosures of the oracle are checked against
that reference. Then sound relaxations are built by hand in rationals, the
oracle must prove them, and it must refute each of them once a bug is planted:
a line moved by one unit in the last place, a slope too steep, a range cut
short, a flipped row or generator. Every witness is checked against the
reference.
"""

import math
import random
from fractions import Fraction as F
from itertools import product

import pytest

from tests._oracles import interval_proof as P
from tests._oracles import rational as R

ULP = F(1, 2 ** 53)                     # the gap below 1.0 in float64
CLOSE = F(1, 2 ** 80)                   # far below anything a float resolves


# --------------------------------------------------------------------------- #
# The reference: series in rationals                                            #
# --------------------------------------------------------------------------- #
TINY = F(1, 2 ** 210)


def _snap(lo, hi, bits=200):
    """The bracket widened to multiples of ``2**-bits``, to keep it small."""
    scale = 2 ** bits
    return F(math.floor(lo * scale), scale), F(math.ceil(hi * scale), scale)


def _series(x, power):
    """Bounds of ``x**k / k!`` summed over ``k = power, power + 2, ...`` with
    alternating signs (``power`` 0: cos; 1: sin)."""
    x = F(x)
    total, term, sign, k = F(0), x ** power, 1, power
    while k <= 2 * abs(x) or abs(term) > TINY:
        total += sign * term
        term = term * x * x / ((k + 1) * (k + 2))
        sign, k = -sign, k + 2
    # ``term`` is the first one left out, and each later one is under a
    # quarter of the one before it.
    return _snap(total - 2 * abs(term), total + 2 * abs(term))


def ref_exp(x):
    x = F(x)
    total, term, k = F(0), F(1), 0
    while k <= 2 * abs(x) or abs(term) > TINY:
        total += term
        k += 1
        term = term * x / k
    # ``term`` is the first one left out, and each later one is under half of
    # the one before it.
    return _snap(total - 2 * abs(term), total + 2 * abs(term))


def ref_sin(x):
    return _series(x, 1)


def ref_cos(x):
    return _series(x, 0)


def ref_sigmoid(x):
    lo, hi = ref_exp(-F(x))
    return 1 / (1 + hi), 1 / (1 + lo)


def ref_tanh(x):
    lo, hi = ref_exp(2 * F(x))
    return 1 - 2 / (lo + 1), 1 - 2 / (hi + 1)


def ref_dsigmoid(x):
    lo, hi = ref_sigmoid(x)
    return lo * (1 - hi), hi * (1 - lo)


def ref_dtanh(x):
    lo, hi = ref_tanh(x)
    squares = (lo * lo, hi * hi)
    least = F(0) if lo <= 0 <= hi else min(squares)
    return 1 - max(squares), 1 - least


def _meet(a, b):
    """Two brackets of one number overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


def _increasing(ref, bracket):
    """The bracket of an increasing function of a bracketed number."""
    lo, hi = _snap(*bracket, bits=64)
    return ref(lo)[0], ref(hi)[1]


def _decreasing(ref, bracket):
    """The bracket of a decreasing function of a bracketed number."""
    lo, hi = _snap(*bracket, bits=64)
    return ref(hi)[0], ref(lo)[1]


def _combine(row, brackets, offset=0):
    """The bracket of ``row . values + offset``."""
    ends = [sorted((w * lo, w * hi)) for w, (lo, hi) in zip(row, brackets)]
    return (F(offset) + sum(lo for lo, _ in ends),
            F(offset) + sum(hi for _, hi in ends))


def test_the_reference_agrees_with_known_values_and_identities():
    known = {ref_exp: "2.718281828459045235360287",
             ref_sin: "0.841470984807896506652502",
             ref_cos: "0.540302305868139717400936",
             ref_tanh: "0.761594155955764888119458",
             ref_sigmoid: "0.731058578630004879251159"}
    for ref, digits in known.items():
        lo, hi = ref(1)
        assert hi - lo < F(1, 2 ** 190)
        assert abs(lo - F(digits)) < F(1, 10 ** 23)
    for x in [F(-7, 2), F(-1, 3), F(0), F(5, 8), F(9, 4), F(6)]:
        for ref, f in [(ref_exp, math.exp), (ref_sin, math.sin),
                       (ref_cos, math.cos), (ref_tanh, math.tanh)]:
            lo, hi = ref(x)
            assert lo <= hi and hi - lo < F(1, 2 ** 150)
            assert float(lo) == pytest.approx(f(float(x)), rel=1e-14, abs=1e-15)
        one = (1, 1)
        (e, E), (m, M) = ref_exp(x), ref_exp(-x)
        assert _meet((e * m, E * M), one)                  # e^x e^-x = 1
        (s, S), (c, C) = ref_sin(x), ref_cos(x)
        squares = [v * v for v in (s, S, c, C)]
        assert min(squares[:2]) + min(squares[2:]) <= 1 <= \
            max(squares[:2]) + max(squares[2:])            # sin^2 + cos^2 = 1
        double = sorted(2 * a * b for a in (s, S) for b in (c, C))
        assert _meet((double[0], double[-1]), ref_sin(2 * x))
        assert ref_exp(0)[0] <= 1 <= ref_exp(0)[1]


# --------------------------------------------------------------------------- #
# Values at points                                                              #
# --------------------------------------------------------------------------- #
STAGES = {
    "sigmoid": (P.SIGMOID, ref_sigmoid, ref_dsigmoid),
    "tanh": (P.TANH, ref_tanh, ref_dtanh),
    "exp": (P.EXP, ref_exp, ref_exp),
    "sin": (P.SIN, ref_sin, ref_cos),
    "cos": (P.COS, ref_cos, lambda x: tuple(-v for v in ref_sin(x)[::-1])),
    "square": (P.power(2), lambda x: (x * x, x * x),
               lambda x: (2 * x, 2 * x)),
    "cube": (P.power(3), lambda x: (x ** 3, x ** 3),
             lambda x: (3 * x * x, 3 * x * x)),
}
POINTS = [F(k, 8) for k in range(-40, 41, 3)] + [
    F(0), F(1, 3), F(-22, 7), F(0.1), F(-2.7), F(1, 2 ** 30)]


@pytest.mark.parametrize("name", STAGES)
def test_the_value_at_a_point_is_a_thin_bracket_of_the_reference(name):
    stage, ref, _ = STAGES[name]
    for x in POINTS:
        (low,), (high,) = P.value(P.chain(stage), [x])
        assert low <= high and high - low < CLOSE
        assert _meet((low, high), ref(x)), float(x)


def test_rationals_become_intervals_that_hold_them():
    third, fifth = F(1, 3), F(1, 5)
    lo, hi = P._endpoints(P._interval(third, third))
    assert lo < third < hi and hi - lo < CLOSE       # not a binary fraction
    lo, hi = P._endpoints(P._interval(-third, fifth))
    assert lo < -third and fifth < hi
    assert P._endpoints(P._interval(F(3, 8), F(5, 2))) == (F(3, 8), F(5, 2))
    assert P._times((F(-2), F(3)), (F(-5), F(1))) == (F(-15), F(10))
    assert P._times((F(-2), F(-1)), (F(-5), F(-3))) == (F(3), F(10))


def test_an_enclosure_that_is_not_finite_is_an_error():
    with pytest.raises(ArithmeticError, match="not finite"):
        P._endpoints(1 / P.IV.mpf([0, 1]))
    with pytest.raises(ArithmeticError, match="not finite"):
        P.value(P.chain(P.function(lambda q: [1 / q[0]])), [0])


# --------------------------------------------------------------------------- #
# Forms on boxes                                                                #
# --------------------------------------------------------------------------- #
W = [[1, F(1, 2)], [F(-3, 4), 1]]
OFFSET = [F(1, 4), F(-1, 2)]
V = [[2, -1], [F(1, 3), F(1, 2)], [-1, 0]]
SHIFT = [0, F(1, 7), -2]


def _layer(ref):
    return lambda x: [ref(R.dot(row, x) + c) for row, c in zip(W, OFFSET)]


def _two_layers(x):
    hidden = _layer(ref_sigmoid)(x)
    return [_combine(row, hidden, c) for row, c in zip(V, SHIFT)]


def _exp_of_tanh(x):
    lo, hi = ref_tanh(x[0])
    return [_increasing(ref_exp, (F(1, 2) - 3 * hi, F(1, 2) - 3 * lo))]


def _cos_of_sigmoid(x):
    lo, hi = ref_sigmoid(x[0])             # 3 sigmoid is in (0, 3): cos falls
    return [_decreasing(ref_cos, (3 * lo, 3 * hi))]


def _product_of_sigmoids(x):
    (a, A), (b, B) = ref_sigmoid(x[0]), ref_sigmoid(x[1])
    return [(a * b, A * B)]


CASES = {
    "sigmoid": (P.chain(P.SIGMOID), [(-3, 1)],
                lambda x: [ref_sigmoid(x[0])]),
    "tanh": (P.chain(P.TANH), [(F(-1, 2), 2)], lambda x: [ref_tanh(x[0])]),
    "exp": (P.chain(P.EXP), [(-1, 2)], lambda x: [ref_exp(x[0])]),
    "sin": (P.chain(P.SIN), [(1, 4)], lambda x: [ref_sin(x[0])]),
    "cos": (P.chain(P.COS), [(-1, 3)], lambda x: [ref_cos(x[0])]),
    "cube": (P.chain(P.power(3)), [(-2, 1)],
             lambda x: [(x[0] ** 3, x[0] ** 3)]),
    "sigmoid-layer": (P.chain(P.linear(W, OFFSET), P.SIGMOID),
                      [(-1, 1), (0, 2)], _layer(ref_sigmoid)),
    "tanh-layer": (P.chain(P.linear(W, OFFSET), P.TANH),
                   [(-1, 1), (-2, 1)], _layer(ref_tanh)),
    "two-layers": (P.chain(P.linear(W, OFFSET), P.SIGMOID, P.linear(V, SHIFT)),
                   [(-1, 1), (-1, 1)], _two_layers),
    "graph": (P.graph(P.chain(P.SIGMOID)), [(-2, 2)],
              lambda x: [(x[0], x[0]), ref_sigmoid(x[0])]),
    "tanh-of-sigmoid": (P.chain(P.SIGMOID, P.TANH), [(-2, 3)],
                        lambda x: [_increasing(ref_tanh, ref_sigmoid(x[0]))]),
    "exp-of-tanh": (P.chain(P.TANH, P.linear([[-3]], [F(1, 2)]), P.EXP),
                    [(-1, 1)], _exp_of_tanh),
    "cos-of-sigmoid": (P.chain(P.SIGMOID, P.linear([[3]]), P.COS), [(-2, 3)],
                       _cos_of_sigmoid),
    "sin-of-a-sum": (P.chain(P.linear([[1, 2]], [F(1, 4)]), P.SIN),
                     [(-1, 1), (0, 1)],
                     lambda x: [ref_sin(x[0] + 2 * x[1] + F(1, 4))]),
    "product": (P.chain(P.function(lambda q: [q[0] * q[1], q[0] - q[1]])),
                [(0, 1), (-1, 2)],
                lambda x: [(x[0] * x[1],) * 2, (x[0] - x[1],) * 2]),
    "product-of-sigmoids": (
        P.chain(P.SIGMOID, P.function(lambda q: [q[0] * q[1]])),
        [(-1, 2), (-2, 1)], _product_of_sigmoids),
}


def _boxes_and_points(box, rng):
    """The box and three boxes inside it, each with its corners, its middle
    and random points."""
    box = [(F(lo), F(hi)) for lo, hi in box]
    boxes = [box]
    for _ in range(3):
        inner = []
        for lo, hi in box:
            a, b = sorted(rng.sample(range(65), 2))
            inner.append((lo + (hi - lo) * F(a, 64), lo + (hi - lo) * F(b, 64)))
        boxes.append(inner)
    for inner in boxes:
        points = [list(p) for p in product(*inner)]
        points.append([(lo + hi) / 2 for lo, hi in inner])
        points += [[lo + (hi - lo) * F(rng.randrange(33), 32)
                    for lo, hi in inner] for _ in range(12)]
        yield inner, points


@pytest.mark.parametrize("name", CASES)
def test_both_enclosures_of_a_form_hold_the_reference_on_the_box(name):
    enclose, box, ref = CASES[name]
    rng = random.Random(11)
    for inner, points in _boxes_and_points(box, rng):
        forms = enclose(inner)
        radii = P._radii(inner)
        middle = [(lo + hi) / 2 for lo, hi in inner]
        for x in points:
            truth = ref(x)
            assert len(truth) == len(forms)
            step = [xi - mi for xi, mi in zip(x, middle)]
            for form, (lo, hi) in zip(forms, truth):
                line = form.centre + R.dot(form.slopes, step)
                assert line + form.low <= hi and lo <= line + form.high
                assert form.span[0] <= hi and lo <= form.span[1]
                low, high = form.bounds(radii)
                assert low <= hi and lo <= high
                wide = form.range(radii)
                assert wide[0] <= low and high <= wide[1]
                assert form.span[0] <= low and high <= form.span[1]


DERIVATIVE_RANGES = [
    ("sigmoid", 1, 2), ("sigmoid", -3, -1), ("sigmoid", -1, 3),
    ("sigmoid", 0, 1), ("tanh", F(1, 2), 2), ("tanh", -2, F(1, 4)),
    ("tanh", -1, 0), ("exp", -1, 2), ("cube", 1, 2), ("square", -1, 3),
    ("sin", F(1, 2), F(3, 2)), ("cos", F(-1, 2), 1),
]


@pytest.mark.parametrize("name, lo, hi", DERIVATIVE_RANGES)
def test_the_remainder_is_the_spread_of_the_derivative_and_no_more(
        name, lo, hi):
    """On ``[lo, hi]`` the linear part has the middle of the derivative's
    range for its slope, and the remainder is half the spread of that range
    times the half-width, each way. The range is the true one: for Sigmoid
    and Tanh the derivative falls with the distance from zero."""
    stage, ref, dref = STAGES[name]
    lo, hi = F(lo), F(hi)
    (form,) = P.chain(stage)([(lo, hi)])
    middle, radius = (lo + hi) / 2, (hi - lo) / 2
    ends = sorted([dref(lo), dref(hi)])
    if name in ("sigmoid", "tanh") and lo < 0 < hi:
        ends.append(dref(0))               # the largest, inside the interval
    if name == "cos" and lo < 0 < hi:
        ends = sorted(ends + [dref(F(1))])   # -sin on [-1/2, 1]: it falls
        ends = [dref(hi), dref(lo)]
    least, most = ends[0], ends[-1]
    assert _meet((form.centre + form.low, form.centre + form.high),
                 ref(middle))
    assert abs(form.slopes[0] - (least[0] + most[0]) / 2) < CLOSE
    spread = (most[0] - least[0]) * radius
    assert abs((form.high - form.low) - spread) < CLOSE
    assert abs(form.high + form.low) < CLOSE           # the same each way


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "exp", "sin", "cube"])
def test_the_remainder_of_a_smooth_stage_falls_with_the_square_of_the_width(
        name):
    stage = STAGES[name][0]
    at = F(7, 10)
    wide, narrow = [P.chain(stage)([(at - r, at + r)])[0]
                    for r in (F(1, 16), F(1, 32))]
    assert 3.6 < (wide.high - wide.low) / (narrow.high - narrow.low) < 4.4


def test_the_remainder_of_a_general_function_falls_with_the_width():
    enclose = P.chain(P.function(lambda q: [q[0] * q[0]]))
    at = F(7, 10)
    wide, narrow = [enclose([(at - r, at + r)])[0]
                    for r in (F(1, 16), F(1, 32))]
    assert wide.slopes == [0]
    assert 1.8 < (wide.high - wide.low) / (narrow.high - narrow.low) < 2.2


def test_a_linear_stage_adds_no_remainder():
    box = [(F(-1), F(2)), (F(0), F(3))]
    forms = P.chain(P.linear(W, OFFSET), P.linear(V, SHIFT))(box)
    for k, form in enumerate(forms):
        assert form.low == form.high == 0
        row = [sum(V[k][j] * W[j][i] for j in range(2)) for i in range(2)]
        assert form.slopes == row
        ends = [R.dot(row, x) + R.dot(V[k], OFFSET) + SHIFT[k]
                for x in product(*box)]
        exact = (min(ends), max(ends))
        assert form.range(P._radii(box)) == form.bounds(P._radii(box)) == exact
        # The interval is taken stage by stage, so it is wider after two.
        assert form.span[0] <= exact[0] and exact[1] <= form.span[1]
    for k, form in enumerate(P.chain(P.linear(V, SHIFT))(box)):
        ends = [R.dot(V[k], x) + SHIFT[k] for x in product(*box)]
        assert form.span == (min(ends), max(ends)) == form.range(P._radii(box))


def test_the_derivative_is_enclosed_all_the_way_to_the_expansion_point():
    """A form can say more through its remainder than through its centre:
    this quantity is ``0 + e`` with ``e`` in ``[99/100, 1]``. Its cube is
    expanded at 0, so the slope ranges over ``[0, 3]``, not only over the
    values the quantity takes."""
    quantity = P.Form(F(0), [F(0)], F(99, 100), F(1), (F(99, 100), F(1)))
    (cube,) = P.power(3)([(F(0), F(0))], [quantity])
    for value in (F(99, 100), F(199, 200), F(1)):
        assert cube.centre + cube.low <= value ** 3 <= cube.centre + cube.high
        assert cube.span[0] <= value ** 3 <= cube.span[1]


def test_a_remainder_is_carried_through_a_stage_at_its_slope():
    """The quantity is ``2 + e`` with ``e`` in ``[0, 1]``. Its cube changes
    nineteen times as fast on average, and the remainder with it; a falling
    function turns the remainder over."""
    quantity = P.Form(F(2), [F(0)], F(0), F(1), (F(2), F(3)))
    (cube,) = P.power(3)([(F(0), F(0))], [quantity])
    falling = P.smooth(lambda t: -t ** 3, lambda t: -3 * t ** 2)
    (minus,) = falling([(F(0), F(0))], [quantity])
    for value in (F(2), F(5, 2), F(3)):
        assert cube.centre + cube.low <= value ** 3 <= cube.centre + cube.high
        assert minus.centre + minus.low <= -value ** 3 \
            <= minus.centre + minus.high


def test_a_stage_reads_the_tighter_of_the_two_enclosures():
    """The linear part knows that ``x - x`` is zero and the interval does
    not; the interval knows the image of a sigmoid on a wide box and the
    linear part does not. The stage that follows is given both."""
    box = [(F(-4), F(4))]
    radii = P._radii(box)
    twice, minus = P.linear([[1], [1]]), P.linear([[1, -1]])
    (zero,) = P.chain(twice, minus)(box)
    assert zero.span == (-8, 8) and zero.range(radii) == (0, 0)
    (half,) = P.chain(twice, minus, P.SIGMOID)(box)
    for lo, hi in (half.span, half.range(radii)):
        assert lo <= F(1, 2) <= hi and hi - lo < CLOSE
    (inner,) = P.chain(P.SIGMOID)(box)
    assert inner.range(radii)[0] < 0 < inner.span[0]
    (outer,) = P.chain(P.SIGMOID, P.EXP)(box)
    bottom = _increasing(ref_exp, ref_sigmoid(F(-4)))
    top = _increasing(ref_exp, ref_sigmoid(F(4)))
    near = F(1, 2 ** 60)                # the reference is snapped to 64 bits
    assert bottom[0] - near < outer.span[0] <= bottom[1]
    assert top[0] <= outer.span[1] < top[1] + near


def test_power_takes_whole_exponents_from_two():
    for p in (1, 0, -2, 1.5, 2.5, 3.5):
        with pytest.raises(ValueError, match="whole number"):
            P.power(p)
    (low,), (high,) = P.value(P.chain(P.power(5.0)), [F(-3, 2)])
    assert low == high == F(-243, 32)


# --------------------------------------------------------------------------- #
# One neuron: the relaxation polygon and the graph                              #
# --------------------------------------------------------------------------- #
def _parallelogram(ref, dref, l, u, pad, steeper=1):
    """Rows ``a . (x, y) <= rhs`` around the graph of an S-shaped function on
    ``[l, u]``: the lines through the two ends of the graph with the smaller
    of the two end slopes, moved apart by ``pad``, and the range of ``x``.
    Sound for ``pad >= 0``: the function less that slope times ``x`` does not
    fall on the interval."""
    l, u = F(l), F(u)
    slope = min(dref(l)[0], dref(u)[0]) * steeper
    below, above = ref(l)[0] - pad, ref(u)[1] + pad
    return [([slope, -1], slope * l - below),      # y >= slope (x - l) + below
            ([-slope, 1], above - slope * u),      # y <= slope (x - u) + above
            ([1, 0], u), ([-1, 0], -l)]


def _polygon(rows):
    return R.halfspaces([a for a, _ in rows], [rhs for _, rhs in rows])


def _breaks_a_row(rows, x, truth):
    """Is ``(x, y)`` outside one row for every ``y`` of the bracket?"""
    return any(all(a[0] * x + a[1] * y > rhs for y in truth)
               for a, rhs in rows)


NEURONS = {
    "sigmoid-symmetric": (P.SIGMOID, ref_sigmoid, ref_dsigmoid, -2, 2),
    "sigmoid-lopsided": (P.SIGMOID, ref_sigmoid, ref_dsigmoid, -1, 3),
    "sigmoid-one-sided": (P.SIGMOID, ref_sigmoid, ref_dsigmoid, 1, 3),
    "tanh-lopsided": (P.TANH, ref_tanh, ref_dtanh, F(-3, 2), F(1, 2)),
}


def _prove_graph(stage, l, u, rows, **options):
    return P.prove(P.graph(P.chain(stage)), [l], [u], _polygon(rows),
                   **options)


@pytest.mark.parametrize("name", NEURONS)
def test_a_relaxation_one_ulp_outside_the_graph_is_proved(name):
    stage, ref, dref, l, u = NEURONS[name]
    result = _prove_graph(stage, l, u, _parallelogram(ref, dref, l, u, ULP))
    assert result.proved, result
    assert result.refuted is None and result.undecided == []
    assert result.boxes < 400                # the tangents were not ground down


@pytest.mark.parametrize("name", NEURONS)
def test_a_relaxation_that_touches_the_graph_is_not_proved(name):
    """With no pad each line passes through an end of the graph. That is the
    kernel without outward rounding of PROTOCOL 3.2: it must not pass, and it
    must not be called unsound either."""
    stage, ref, dref, l, u = NEURONS[name]
    width = F(1, 2 ** 20)
    result = _prove_graph(stage, l, u, _parallelogram(ref, dref, l, u, 0),
                          min_width=width)
    assert not result.proved and result.refuted is None
    assert result.undecided
    for ((lo, hi),) in result.undecided:
        assert lo == l or hi == u            # only at the two touching ends
        assert width / 2 < hi - lo <= width  # given up at the stated width
    assert {lo == l for ((lo, hi),) in result.undecided} == {True, False}


PLANTS = {
    "one ulp inside": dict(pad=-ULP),
    "slope too steep": dict(pad=ULP, steeper=F(3, 2)),
}


@pytest.mark.parametrize("plant", PLANTS)
@pytest.mark.parametrize("name", NEURONS)
def test_a_relaxation_with_a_planted_bug_is_refuted_with_a_witness(
        name, plant):
    stage, ref, dref, l, u = NEURONS[name]
    rows = _parallelogram(ref, dref, l, u, **PLANTS[plant])
    result = _prove_graph(stage, l, u, rows)
    assert not result.proved and result.refuted is not None, result
    (x,) = result.refuted
    assert l <= x <= u
    assert _breaks_a_row(rows, x, ref(x))


@pytest.mark.parametrize("name", NEURONS)
def test_a_relaxation_for_a_shorter_range_is_refuted(name):
    stage, ref, dref, l, u = NEURONS[name]
    for short, cut in [((l, u - F(1, 8)), 2), ((l + F(1, 8), u), 3)]:
        rows = _parallelogram(ref, dref, *short, ULP)
        result = _prove_graph(stage, l, u, rows)
        assert result.refuted is not None, result
        (x,) = result.refuted
        assert l <= x <= u and _breaks_a_row(rows, x, ref(x))
        # The range itself is short, whatever the lines do out there.
        result = _prove_graph(stage, l, u, [rows[cut]])
        (x,) = result.refuted
        assert _breaks_a_row([rows[cut]], x, ref(x))


@pytest.mark.parametrize("flipped", [0, 1])
@pytest.mark.parametrize("name", NEURONS)
def test_a_relaxation_with_a_flipped_row_is_refuted(name, flipped):
    stage, ref, dref, l, u = NEURONS[name]
    rows = _parallelogram(ref, dref, l, u, ULP)
    a, rhs = rows[flipped]
    rows[flipped] = ([-v for v in a], -rhs)
    result = _prove_graph(stage, l, u, rows)
    assert result.refuted is not None, result
    (x,) = result.refuted
    assert _breaks_a_row(rows, x, ref(x))


def test_a_bug_at_either_end_of_the_range_is_found():
    """Each half of a cut box is examined: the one line that is wrong is
    wrong at one end only, the left or the right."""
    stage, ref, dref, l, u = NEURONS["sigmoid-symmetric"]
    for wrong in (0, 1):
        rows = _parallelogram(ref, dref, l, u, ULP)
        a, rhs = rows[wrong]
        rows[wrong] = (a, rhs - 4 * ULP)
        result = _prove_graph(stage, l, u, rows)
        assert result.refuted is not None, result
        (x,) = result.refuted
        assert _breaks_a_row(rows, x, ref(x))
        assert (x < 0) == (wrong == 0)


def test_the_tangent_at_the_inflection_point_stays_undecided():
    """The documented limit. ``y <= x / 4 + 1 / 2`` holds for Sigmoid on
    ``[0, 1]`` and meets the graph at ``(0, 1/2)`` with no room at all. No
    enclosure fits under it, and it is not unsound."""
    rows = [([F(-1, 4), 1], F(1, 2)), ([0, -1], F(-1, 2)),
            ([1, 0], 1), ([-1, 0], 0)]
    width = F(1, 2 ** 12)
    result = _prove_graph(P.SIGMOID, 0, 1, rows, min_width=width)
    assert not result.proved and result.refuted is None
    # The room grows with the cube of the distance: next to the point only.
    assert result.undecided[0] == [(0, width)]
    assert all(hi <= 4 * width for ((lo, hi),) in result.undecided)
    # One ulp of room on the two rows that meet there is enough, and one ulp
    # the other way is a bug.
    rows[0], rows[1] = (rows[0][0], F(1, 2) + ULP), (rows[1][0], ULP - F(1, 2))
    assert _prove_graph(P.SIGMOID, 0, 1, rows).proved
    rows[0] = (rows[0][0], F(1, 2) - ULP)
    assert _prove_graph(P.SIGMOID, 0, 1, rows).refuted == [0]


# --------------------------------------------------------------------------- #
# One neuron: the range of the output                                           #
# --------------------------------------------------------------------------- #
def test_a_range_that_ends_exactly_on_the_image_is_proved():
    square, sigmoid, sine = P.chain(P.power(2)), P.chain(P.SIGMOID), \
        P.chain(P.SIN)
    assert P.prove(square, [1], [3], R.interval([1], [9])).proved
    assert P.prove(square, [-1], [3], R.interval([0], [9])).proved
    top = ref_sigmoid(1)[1] + ULP
    assert P.prove(sigmoid, [0], [1], R.interval([F(1, 2)], [top])).proved
    low = min(ref_sin(1)[0], ref_sin(2)[0]) - ULP
    assert P.prove(sine, [1], [2], R.interval([low], [1])).proved


def test_one_output_is_taken_by_its_interval_and_another_by_its_slope():
    """Two neurons on one input in ``[1, 3]``, with the input kept. The
    square lies in exactly ``[1, 9]``, which only its interval fits at the two
    ends. The sigmoid lies between two lines one ulp from its graph, which
    only its linear part fits. Neither enclosure alone passes the end boxes."""
    rows = _parallelogram(ref_sigmoid, ref_dsigmoid, 1, 3, ULP)
    A = [[a[0], 0, a[1]] for a, _ in rows] + [[0, 1, 0], [0, -1, 0]]
    b = [rhs for _, rhs in rows] + [9, -1]
    both = P.graph(
        lambda box: P.chain(P.power(2))(box) + P.chain(P.SIGMOID)(box))
    result = P.prove(both, [1], [3], R.halfspaces(A, b))
    assert result.proved and result.boxes < 400, result
    # One ulp short on the square is still found, at the end it is short at.
    b[-2] = 9 - 8 * ULP
    assert P.prove(both, [1], [3], R.halfspaces(A, b)).refuted == [3]


def test_a_witness_in_the_middle_of_a_box_is_found_before_it_is_cut():
    """The square of ``[-1, 1]`` is ``[0, 1]``. A set that starts at 1/2
    holds the values at both ends of the input and not the one in the
    middle."""
    square, short = P.chain(P.power(2)), R.interval([F(1, 2)], [2])
    result = P.prove(square, [-1], [1], short)
    assert result.refuted == [0] and result.boxes == 1
    # The same when the box is at the minimum width and is never cut.
    result = P.prove(square, [-1], [1], short, min_width=2)
    assert result.refuted == [0] and result.boxes == 1


@pytest.mark.parametrize("lb, ub, at", [
    ([1 + ULP], [9], 1), ([1], [9 - 8 * ULP], 3)])
def test_a_range_one_ulp_short_is_refuted_at_its_end(lb, ub, at):
    result = P.prove(P.chain(P.power(2)), [1], [3], R.interval(lb, ub))
    assert result.refuted == [at], result


def test_a_range_one_ulp_short_of_an_irrational_end_is_refuted():
    top = ref_sigmoid(1)[0] - ULP
    result = P.prove(P.chain(P.SIGMOID), [0], [1],
                     R.interval([F(1, 2)], [top]))
    assert result.refuted == [1], result
    result = P.prove(P.chain(P.SIN), [1], [2],
                     R.interval([F(4, 5)], [1 - ULP]))
    (x,) = result.refuted
    assert ref_sin(x)[0] > 1 - ULP             # near pi / 2, inside the range


def test_a_value_that_straddles_the_boundary_is_neither_proved_nor_refuted():
    """The oracle cannot tell on which side of this bound Sigmoid(1) lies,
    and says so."""
    (low,), (high,) = P.value(P.chain(P.SIGMOID), [1])
    width = high - low
    assert width > 0

    def with_lower_bound(bound):
        return P.prove(P.chain(P.SIGMOID), [1], [2],
                       R.interval([bound], [1]), min_width=F(1, 2 ** 8))

    for inside in [low + width / 4, low + width / 2, high - width / 4, high]:
        result = with_lower_bound(inside)
        assert not result.proved and result.refuted is None
        assert result.undecided[0][0][0] == 1
    assert with_lower_bound(high + width / 4).refuted == [1]
    assert with_lower_bound(low).proved
    # The same with a second set far away, and on an input that is one point.
    far = R.interval([5], [6])
    inside = R.interval([high - width / 4], [1])
    for sets in [(inside, far), (far, inside)]:
        result = P.prove(P.chain(P.SIGMOID), [1], [2], *sets,
                         min_width=F(1, 2 ** 8))
        assert not result.proved and result.refuted is None
    result = P.prove(P.chain(P.SIGMOID), [1], [1], inside)
    assert not result.proved and result.refuted is None
    assert result.undecided == [[(1, 1)]] and result.boxes == 1


def test_several_sets_are_a_union_and_each_box_needs_one_of_them():
    square = P.chain(P.power(2))
    small = R.interval([F(-1, 10)], [F(3, 2)])
    large = R.interval([F(6, 5)], [F(41, 10)])
    result = P.prove(square, [-1], [2], small, large)
    assert result.proved and result.boxes > 1
    assert P.prove(square, [-1], [2], large, small).proved
    for alone in (small, large):
        result = P.prove(square, [-1], [2], alone)
        (x,) = result.refuted
        assert not R.member(alone, [x * x])
    apart = R.interval([F(8, 5)], [F(41, 10)])
    (x,) = P.prove(square, [-1], [2], small, apart).refuted
    assert F(3, 2) < x * x < F(8, 5)


# --------------------------------------------------------------------------- #
# A layer of two neurons                                                        #
# --------------------------------------------------------------------------- #
def _layer_set(ref, dref, pad, half=1):
    """A sound set for ``ref(W x + OFFSET)`` on the square of half-width
    ``half``, as a zonotope: each neuron is its slope times its input plus an
    offset that ranges over the gap between the two lines of
    :func:`_parallelogram`."""
    centre, generators = [], []
    for k, (row, c) in enumerate(zip(W, OFFSET)):
        reach = sum(abs(F(w)) for w in row) * half
        l, u = c - reach, c + reach
        slope = min(dref(l)[0], dref(u)[0])
        low = ref(l)[0] - slope * l - pad
        high = ref(u)[1] - slope * u + pad
        centre.append(slope * c + (low + high) / 2)
        own = [(high - low) / 2 if j == k else F(0) for j in range(2)]
        generators.append([slope * F(w) * half for w in row] + own)
    return (centre, generators, [], [], [F(-1)] * 4, [F(1)] * 4)


def _prove_layer(stage, computed, half=1, **options):
    f = P.chain(P.linear(W, OFFSET), stage)
    return P.prove(f, [-half, -half], [half, half], computed, **options)


def _is_outside(computed, truth):
    """Is every point of the bracketed value outside the set?"""
    return not R.meets_box(computed, [lo for lo, _ in truth],
                           [hi for _, hi in truth])


LAYERS = {"sigmoid": (P.SIGMOID, ref_sigmoid, ref_dsigmoid),
          "tanh": (P.TANH, ref_tanh, ref_dtanh)}


@pytest.mark.parametrize("name", LAYERS)
def test_a_sound_set_for_a_layer_is_proved(name):
    stage, ref, dref = LAYERS[name]
    result = _prove_layer(stage, _layer_set(ref, dref, F(1, 2 ** 30)))
    assert result.proved, result


def _move_the_centre(computed):
    computed[0][0] += F(1, 2 ** 20)


def _tighten_a_bound(computed):
    computed[5][2] = 1 - F(1, 2 ** 20)


@pytest.mark.parametrize("plant", [_move_the_centre, _tighten_a_bound, "pad"])
@pytest.mark.parametrize("name", LAYERS)
def test_a_set_for_a_layer_with_a_planted_bug_is_refuted(name, plant):
    stage, ref, dref = LAYERS[name]
    if plant == "pad":
        computed = _layer_set(ref, dref, -F(1, 2 ** 30))
    else:
        computed = _layer_set(ref, dref, F(1, 2 ** 30))
        plant(computed)
    result = _prove_layer(stage, computed)
    assert result.refuted is not None, result
    x = result.refuted
    assert all(-1 <= v <= 1 for v in x)
    assert _is_outside(computed, _layer(ref)(x))


@pytest.mark.parametrize("name", LAYERS)
def test_a_generator_with_the_wrong_sign_is_refuted_on_a_narrow_box(name):
    """On a narrow box the neurons are close to linear and the set is thin
    across the image, so the direction of every generator matters. (On a wide
    box the same flip can leave a set that is still sound.)"""
    stage, ref, dref = LAYERS[name]
    half = F(1, 8)
    assert _prove_layer(
        stage, _layer_set(ref, dref, F(1, 2 ** 30), half), half).proved
    for k, j in product(range(2), repeat=2):
        computed = _layer_set(ref, dref, F(1, 2 ** 30), half)
        computed[1][k][j] = -computed[1][k][j]
        result = _prove_layer(stage, computed, half)
        assert result.refuted is not None, result
        x = result.refuted
        assert all(-half <= v <= half for v in x)
        assert _is_outside(computed, _layer(ref)(x))


def test_the_image_of_a_linear_map_is_its_own_enclosure():
    exact = ([F(c) for c in OFFSET], R.matrix(W), [], [], [F(-1)] * 2,
             [F(1)] * 2)
    f = P.chain(P.linear(W, OFFSET))
    result = P.prove(f, [-1, -1], [1, 1], exact)
    assert result.proved and result.boxes == 1
    for k, j in product(range(2), repeat=2):
        wrong = (exact[0], R.matrix(W), [], [], exact[4], exact[5])
        wrong[1][k][j] = -wrong[1][k][j]
        x = P.prove(f, [-1, -1], [1, 1], wrong).refuted
        image = [R.dot(row, x) + c for row, c in zip(W, OFFSET)]
        assert not R.member(wrong, image)


def test_a_face_that_is_too_tight_is_refuted_without_grinding():
    """Two neurons that share no input: the bound of one is short along a
    whole face of the input square, by less than any box will be cut to.
    The witness is a corner of the first box."""
    f = P.chain(P.SIGMOID)
    top = ref_sigmoid(1)
    sound = R.interval([F(1, 4), F(1, 4)], [top[1] + ULP, top[1] + ULP])
    assert P.prove(f, [-1, -1], [1, 1], sound).proved
    short = R.interval([F(1, 4), F(1, 4)], [top[0] - ULP, top[1] + ULP])
    result = P.prove(f, [-1, -1], [1, 1], short)
    assert result.refuted is not None and result.refuted[0] == 1
    assert result.boxes == 1


# --------------------------------------------------------------------------- #
# Part of a box, general functions, and the limits                              #
# --------------------------------------------------------------------------- #
def test_a_domain_restricts_the_input_to_part_of_the_box():
    """``exp(x1 + x2)`` where ``x1 + x2 <= 0`` is at most 1; on the whole
    square it reaches ``e^2``."""
    f = P.chain(P.linear([[1, 1]]), P.EXP)
    computed = R.interval([F(1, 8)], [F(11, 10)])
    half = ([[1, 1]], [0])
    result = P.prove(f, [-1, -1], [1, 1], computed, domain=half)
    assert result.proved, result
    whole = P.prove(f, [-1, -1], [1, 1], computed)
    assert whole.refuted is not None
    assert sum(whole.refuted) > 0              # in the part that was left out
    other = P.prove(f, [-1, -1], [1, 1], computed, domain=([[-1, -1]], [0]))
    assert other.refuted is not None and sum(other.refuted) >= 0
    # A witness is a point of the domain, not of a box that reaches into it.
    tight = R.interval([F(1, 8)], [F(1)])
    result = P.prove(f, [-1, -1], [1, 1], tight, domain=half,
                     min_width=F(1, 2 ** 3))
    assert not result.proved and result.refuted is None


def test_a_function_written_in_intervals_is_proved_with_a_margin():
    f = P.chain(P.function(lambda q: [q[0] * q[1]]))
    assert P.prove(f, [0, 0], [1, 2],
                   R.interval([F(-1, 20)], [F(41, 20)])).proved
    result = P.prove(f, [0, 0], [1, 2], R.interval([F(1, 10)], [F(41, 20)]))
    x = result.refuted
    assert x[0] * x[1] < F(1, 10)
    result = P.prove(f, [0, 0], [1, 2], R.interval([F(-1, 20)], [F(19, 10)]))
    x = result.refuted
    assert x[0] * x[1] > F(19, 10)


def test_an_axis_the_function_ignores_is_never_cut():
    stage, ref, dref, l, u = NEURONS["sigmoid-lopsided"]
    f = P.chain(P.linear([[1, 0]]), stage)
    assert P._axis_to_cut(f, [(F(l), F(u)), (F(-50), F(50))]) == 0
    assert P._axis_to_cut(f, [(F(1), F(1)), (F(-50), F(50))]) == 1
    assert P._axis_to_cut(f, [(F(1), F(1)), (F(2), F(2))]) is None
    rows = _parallelogram(ref, dref, l, u, ULP)
    alone = _prove_graph(stage, l, u, rows)
    beside = P.prove(lambda box: P._inputs(box)[:1] + f(box),
                     [l, -50], [u, 50], _polygon(rows))
    assert beside.proved and beside.boxes == alone.boxes


def test_an_input_that_is_a_single_point_on_one_axis_is_handled():
    f = P.chain(P.linear([[1, 1]]), P.TANH)
    lo, hi = ref_tanh(-1)[0] - ULP, ref_tanh(F(3, 2))[1] + ULP
    assert P.prove(f, [-2, 1], [F(1, 2), 1], R.interval([lo], [hi])).proved
    result = P.prove(f, [-2, 1], [F(1, 2), 1],
                     R.interval([lo], [ref_tanh(F(3, 2))[0] - ULP]))
    assert result.refuted == [F(1, 2), 1]


def test_the_empty_chain_is_the_identity():
    square = R.interval([0, 0], [1, 1])
    result = P.prove(P.chain(), [0, 0], [1, 1], square)
    assert result.proved and result.boxes == 1
    result = P.prove(P.chain(), [0, 0], [1, 1],
                     R.interval([0, 0], [1, 1 - ULP]))
    assert result.refuted is not None and result.refuted[1] == 1


def test_a_proof_that_costs_too_much_raises_and_none_is_an_error():
    stage, ref, dref, l, u = NEURONS["sigmoid-symmetric"]
    rows = _parallelogram(ref, dref, l, u, ULP)
    with pytest.raises(RuntimeError, match="no answer after 5 boxes"):
        _prove_graph(stage, l, u, rows, max_boxes=5)
    needed = _prove_graph(stage, l, u, rows).boxes
    assert _prove_graph(stage, l, u, rows, max_boxes=needed).proved
    with pytest.raises(RuntimeError, match="no answer after"):
        _prove_graph(stage, l, u, rows, max_boxes=needed - 1)
    with pytest.raises(ValueError, match="no computed set"):
        P.prove(P.chain(stage), [l], [u])


def test_a_result_reads_as_what_happened():
    square = P.chain(P.power(2))
    proved = P.prove(square, [1], [3], R.interval([1], [9]))
    assert str(proved) == "proved with 1 boxes"
    refuted = P.prove(square, [1], [3], R.interval([1], [8]))
    assert str(refuted) == "refuted at the input (3) after 1 boxes"
    stage, ref, dref, l, u = NEURONS["sigmoid-symmetric"]
    undecided = _prove_graph(stage, l, u, _parallelogram(ref, dref, l, u, 0),
                             min_width=F(1, 4))
    count = len(undecided.undecided)
    assert str(undecided) == (
        f"undecided: {count} box(es) failed below the minimum width, "
        "the first at [-2, -1.75]")
