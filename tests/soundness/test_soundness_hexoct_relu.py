"""Soundness cells of ReLU on Hexatope and Octatope sets.

The two set types have one rule, ``method='approx'``: the kernels
(``_relu_single_hexatope``, ``_relu_single_octatope``) zero the neurons that
are stably inactive and split every crossing neuron into an active and an
inactive piece, each cut by ``intersect_half_space`` (the bounding box of the
difference or UTVPI system, a direct row where the cut is one), and decide
the phases from the interval estimate on the kernel box (no solver) or from a
solver's range (``solver='lp'`` or ``'mcf'``). The model entry refuses
``exact`` for these set types; ``reach_layer`` does not read the method
(marked below).

Oracle: the exact image of the input set in rationals (the set read from its
stored arrays by ``tests/_oracles/rational.star``, which pins the anchor of a
Hexatope to 0), cut into pieces by the activation pattern; all vertices of
each piece must lie in one returned set. Membership of a vertex is decided by
the LP oracle (``membership.distance``) with the slack ``LP_TOL``: the split
pieces carry bounds that a solver computed with no outward pad (registry
HEXOCT-4, marked by
``test_soundness_range_rounding.py::test_hexoct_relu_pieces_hold_the_exact_image``).
Member points of the input set are also replayed on torch in float64. The
cells of a crossing point that is not a float return sets with no row a
solver computed and are decided in rationals with zero slack.

Inputs: every number is dyadic and every regime is decided by a wide margin,
apart from bounds that are exactly 0 by construction. The families are a box
with a zero-width coordinate; affine images of the kernel box whose neurons
share kernel variables (crossing neurons of different widths, an upper bound
and a lower bound of exactly 0, a stably inactive neuron, constants, and
stably active neurons that pin every kernel variable, so that an output
point fixes the input point); neurons that are a difference or a sum of two
kernel variables (the direct rows of both bounding boxes); a set built with
the public constructor whose rows cut a corner of the kernel box, with
neurons that only a solver's range proves stable; exactly one stably
inactive neuron; a list of three cut sets; a kernel variable that ranges
beyond [-1, 1]. Composition: Linear, ReLU, Linear through the model entry,
and Linear, ReLU, Linear, ReLU, Linear through ``NeuralNetwork.reach``, whose
second ReLU receives the pieces of the first. ONNX: the Clip cell of
``test_soundness_relu_cells.py`` loads a Clip with min 0 for both set types.

Set data in float32 cannot occur: both constructors store their arrays in
float64.
"""
import copy
import functools
from fractions import Fraction

import numpy as np
import pytest
import torch.nn as _nn

from n2v.nn import NeuralNetwork
from n2v.nn.layer_ops.dispatcher import reach_layer as _reach_layer
from n2v.nn.reach import reach_pytorch_model as _reach_pytorch_model
from n2v.sets import Hexatope, Octatope
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.octatope import UTVPIConstraintSystem
from tests._oracles import rational as _R
from tests.cell_support import (assert_covered, flat_box, flat_set,
                                reach_model, reach_one_layer, replay_torch)
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.soundness.test_soundness_relu_cells import (
    _CUTS, _HI3, _LO3, _MIXED, _STABLE, _assert_holds_image_lp, _chain,
    _chain_pieces, _maximal, _pieces, _points, _relu_points)

_SOLVERS = [None, "lp", "mcf"]
_HEXOCT = {"Hexatope": Hexatope, "Octatope": Octatope}


def _coupled(kind, c, G, cut=None):
    """``c + G a`` for the kernel box of ``from_bounds`` on [-1, 1]^3, or on
    the box whose first coordinate is cut to ``cut``: an affine image, so
    neurons share kernel variables."""
    lo, hi = -np.ones(3), np.ones(3)
    if cut is not None:
        lo[0], hi[0] = cut
    return _HEXOCT[kind].from_bounds(lo, hi).affine_map(G, c.reshape(-1, 1))


@functools.lru_cache(maxsize=None)
def _hexoct_reference(name, kind):
    """``(input sets, exact pieces of their ReLU image, member points)`` of a
    family, computed once for the three solver choices (the reach does not
    change its input sets)."""
    sets = _hexoct_family(name, kind)
    pieces = [p for s in sets for p in _R.relu(_pieces(s))]
    points = [x for s in sets for x in _points(s, 24)]
    return sets, pieces, points


# Exactly one stably inactive neuron (a guard on the number of inactive
# neurons, "more than one" for "at least one", leaves it unzeroed), coupled
# with a crossing neuron through a0, and pins of every kernel variable.
_ONE_INACTIVE = (np.array([-0.5, -1.5, 3.0, 3.0, 3.0]),
                 np.array([[1.0, 0.5, 0.0],      # crosses zero: [-2, 1]
                           [0.5, 0.0, 0.25],     # stably inactive: [-2.25, -0.75]
                           [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
# Neurons that are a difference or a sum of two kernel variables: n0 = 1/4 +
# a0 - a1 in [-7/4, 9/4] and n2 = -3/4 - a0 + a2 in [-11/4, 5/4] cross zero
# with a generator row of two opposite equal entries, which both bounding
# boxes add as a row as it is (a difference row of a Hexatope, a UTVPI row of
# an Octatope); n1 = -1/2 + a1 + a2 in [-5/2, 3/2], a sum, is a UTVPI row of
# an Octatope and goes through the general bounding box of a Hexatope. The
# second and third splits cut pieces that already carry such rows. Three
# neurons pin a0, a1, a2.
_DIFFERENCES = (np.array([0.25, -0.5, -0.75, 3.0, 3.0, 3.0]),
                np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0],
                          [-1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def _cut_corner(kind):
    """A set built with the public constructor: the kernel box [-1, 1]^3 and
    one more row that cuts a corner of it, a0 - a1 <= 1/2 (a difference row
    of a Hexatope, whose anchor is variable 0) or a0 + a1 <= 1/2 (a UTVPI
    row of an Octatope). Neuron 0 = -1/2 + 2 a0 + a1 crosses zero under the
    row. Neurons 1 and 2, -3/4 + (a0 - a1) and 3/4 - (a0 - a1) (with a0 + a1
    for the Octatope), cross zero by the interval estimate on the kernel box
    and are stable under the row (maximum -1/4, minimum 1/4): without a
    solver they are split, with one they are zeroed and kept. Three neurons
    pin a0, a1, a2."""
    sign = -1.0 if kind == "Hexatope" else 1.0
    c = np.array([-0.5, -0.75, 0.75, 3.0, 3.0, 3.0])
    G = np.array([[2.0, 1.0, 0.0], [1.0, sign, 0.0], [-1.0, -sign, 0.0],
                  [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if kind == "Hexatope":
        dcs = DifferenceConstraintSystem(4)
        for i in range(1, 4):
            dcs.add_constraint(i, 0, 1.0)
            dcs.add_constraint(0, i, 1.0)
        dcs.add_constraint(1, 2, 0.5)
        return Hexatope(c, np.hstack([np.zeros((6, 1)), G]), dcs)
    utvpi = UTVPIConstraintSystem(3)
    for i in range(3):
        utvpi.add_constraint(i, i, 1, 0, 1.0)
        utvpi.add_constraint(i, i, -1, 0, 1.0)
    utvpi.add_constraint(0, 1, 1, 1, 0.5)
    return Octatope(c, G, utvpi)


def _hexoct_family(name, kind):
    if name == "box":
        lo, hi = flat_box(5, seed=4, zero_width=True)
        return [flat_set(kind, lo, hi)]
    if name == "coupled":
        return [_coupled(kind, *_MIXED)]
    if name == "stable":
        return [_coupled(kind, *_STABLE)]
    if name == "one-inactive":
        return [_coupled(kind, *_ONE_INACTIVE)]
    if name == "several":
        return [_coupled(kind, *_MIXED, cut=cut) for cut in _CUTS]
    if name == "differences":
        return [_coupled(kind, *_DIFFERENCES)]
    if name == "cut-corner":
        return [_cut_corner(kind)]
    raise ValueError(name)


@pytest.mark.parametrize("solver", _SOLVERS, ids=lambda s: f"solver={s}")
@pytest.mark.parametrize("name", ["box", "coupled", "stable", "one-inactive",
                                  "several", "differences", "cut-corner"])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_relu_hexoct_holds_the_exact_image(kind, name, solver):
    """ReLU on a Hexatope or an Octatope through ``reach_layer``, with the
    phases decided by interval bounds (no solver) or by solver ranges ('lp',
    'mcf'). The coupled family has crossing neurons of different widths, a
    neuron whose upper bound is exactly 0 and one whose lower bound is
    exactly 0 (each sharing its kernel variable with a stably active neuron
    that pins it), a stably inactive neuron and constant neurons; its data
    go well beyond [-1, 1] (a neuron over [-7/2, 5/2], pins over [2, 4]) while
    the kernel variables stay in it, so a cut stated in output units instead
    of kernel units misses outputs. The one-inactive family has exactly one
    stably inactive neuron; the differences family crosses zero with neurons
    whose cut is added as a row as it is; the cut-corner family is given by
    a row that is not a box row and has neurons that only a solver's range
    proves stable; the several family is a list of three sets.

    Catches: the half-space of the active or inactive piece of a split with
    the wrong sign, or the inactive piece not zeroed (range); the pieces of
    one neuron's split collected per set instead of per neuron, so pieces
    are lost (piece); a neuron with an upper bound of exactly 0 neither
    zeroed nor split, or a single stably inactive neuron left as it is
    (regime); the solver's lower and upper ranges swapped, or clipped by the
    wrong interval bound (plumbing, range); a direct difference or UTVPI row
    stored with its variables exchanged, its signs wrong or its bound not
    divided by the scale; general bounding-box bounds from the wrong
    optimum or on the exchanged pair; a list of input sets of which one is
    dropped."""
    sets, pieces, points = _hexoct_reference(name, kind)
    kw = {} if solver is None else {"solver": solver}
    what = f"ReLU approx on the {kind} family {name!r} with solver={solver}"
    out = reach_one_layer(_nn.ReLU(), [copy.deepcopy(s) for s in sets],
                          "approx", what, **kw)
    _assert_holds_image_lp(pieces, out, what)
    assert_covered(out, _relu_points(points), what)


@pytest.mark.parametrize("solver", _SOLVERS, ids=lambda s: f"solver={s}")
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_relu_hexoct_chain_through_the_model_entry(kind, solver):
    """Linear, ReLU, Linear through ``reach_pytorch_model`` on a box: the
    pieces the ReLU returns are consumed by the next Linear, so a row of a
    piece that is wrong while its bounding box is right shows downstream.
    The solver option is handed to the model entry; a dropped option is not
    visible here (the kernel without a solver is sound too), and each
    solver's path of the kernel is checked by
    ``test_relu_hexoct_holds_the_exact_image``."""
    input_set = flat_set(kind, _LO3, _HI3)
    kw = {} if solver is None else {"solver": solver}
    what = f"Linear-ReLU-Linear approx on a {kind} with solver={solver}"
    out = reach_model(_chain(), input_set, "approx", what, **kw)
    _assert_holds_image_lp(_chain_pieces(input_set), out, what)
    points = _points(input_set, 64)
    assert_covered(out, replay_torch(_chain(), points, (3,)), what)


def _contents(s):
    """What a Hexatope or an Octatope holds: its arrays and its rows."""
    rows = s.dcs.constraints if isinstance(s, Hexatope) else s.utvpi.constraints
    return (s.center.tobytes(), s.generators.tobytes(), s.generators.shape,
            tuple(tuple(vars(r).values()) for r in rows))


@pytest.mark.parametrize("solver", [None, "lp"], ids=lambda s: f"solver={s}")
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_relu_hexoct_two_relu_chain_through_neural_network(kind, solver):
    """Linear, ReLU, Linear, ReLU, Linear through ``NeuralNetwork(model)
    .reach`` on a box. The second ReLU receives the pieces of the first, a
    list of sets whose kernels carry the rows the first split added (direct
    and bounding-box rows, not box rows), so its phase decision (the
    interval estimate, or the solver's range over those rows) and its own
    cuts run on such sets. Every vertex of the exact image must lie in one
    returned set (LP oracle) and the torch float64 outputs of member points
    must be covered; the reach must answer and leave its input set as it
    was. Catches a second split that reads or rewrites the rows of the
    first wrongly, and a wrong piece of either ReLU that the last Linear
    does not hide. The 'mcf' variant is left out: on such pieces the
    min-cost flow does not terminate (HEXOCT-10, marked by
    ``test_soundness_hexoct_sets.py::test_relu_with_min_cost_flow_phases_answers_and_holds_the_image``)."""
    input_set = flat_set(kind, _LO3, _HI3)
    kw = {} if solver is None else {"solver": solver}
    what = (f"Linear-ReLU-Linear-ReLU-Linear approx on a {kind} with "
            f"solver={solver} through NeuralNetwork.reach")
    model = _chain(two_relus=True)
    before = _contents(input_set)
    out = NeuralNetwork(model).reach(input_set, method="approx", **kw)
    if _contents(input_set) != before:
        pytest.fail(f"{what}: the reach changed its input set")
    if not out:
        raise AssertionError(f"{what}: reach returned no sets")
    _assert_holds_image_lp(_chain_pieces(input_set, two_relus=True), out,
                           what)
    assert_covered(out, replay_torch(model, _points(input_set, 64), (3,)),
                   what)


def _wide_kernel(kind, width):
    """The set ``relu`` is applied to in the cells below:
    ``(1.5 + t, 5 + t)`` for the kernel variable t in [-width, width],
    built with the public constructors (for a Hexatope, the anchor x0 has a
    zero generator and rows x1 - x0 <= width, x0 - x1 <= width; for an
    Octatope, rows x0 <= width and -x0 <= width)."""
    if kind == "Hexatope":
        dcs = DifferenceConstraintSystem(2)
        dcs.add_constraint(1, 0, width)
        dcs.add_constraint(0, 1, width)
        return Hexatope(np.array([1.5, 5.0]), np.array([[0.0, 1.0], [0.0, 1.0]]),
                        dcs)
    utvpi = UTVPIConstraintSystem(1)
    utvpi.add_constraint(0, 0, 1, 0, width)
    utvpi.add_constraint(0, 0, -1, 0, width)
    return Octatope(np.array([1.5, 5.0]), np.array([[1.0], [1.0]]), utvpi)


_KERNEL_BOX_ASSUMED = known_bug(
    9, "the Hexatope and Octatope ReLU decide phases from bounds that assume "
       "every kernel variable lies in [-1, 1] (estimate_ranges, and the box "
       "of _kernel_phase_ranges when a solver is given); for a set whose rows "
       "allow a variable beyond 1 a crossing neuron is taken for stably "
       "active and its outputs 0 are lost (by up to 0.75)")


@pytest.mark.parametrize("solver", _SOLVERS, ids=lambda s: f"solver={s}")
@pytest.mark.parametrize("width", [
    pytest.param(3.0, marks=_KERNEL_BOX_ASSUMED, id="variable-in-[-3,3]"),
    pytest.param(1.0, id="variable-in-[-1,1]")])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_relu_hexoct_with_a_kernel_variable_beyond_one(kind, width, solver):
    """A legal set built with the public constructor whose kernel variable
    ranges over [-3, 3]: neuron 0 = 1.5 + t crosses zero. Every true output
    ``relu(1.5 + t, 5 + t)`` must lie in a returned set. The neighbour with
    the variable in [-1, 1] (neuron 0 stably active there) passes with the
    same reference. 'lp' and 'mcf' share the line that intersects their
    range with the [-1, 1] box but compute the range differently, so each
    keeps its marked case."""
    s = _wide_kernel(kind, width)
    kw = {} if solver is None else {"solver": solver}
    what = f"ReLU on a {kind} whose kernel variable is in [-{width}, {width}]"
    out = reach_one_layer(_nn.ReLU(), [s], "approx", what, **kw)
    pieces = _R.relu(_pieces(s))
    expected = 2 if width > 1.5 else 1      # neuron 0 = 1.5 + t crosses zero
    if len(_maximal(pieces)[0]) != expected:
        pytest.fail(f"{what}: the reference has {len(_maximal(pieces)[0])} "
                    f"pieces, not {expected}")
    _assert_holds_image_lp(pieces, out, what)


def _row_cut_set(kind, slope):
    """A set whose crossing neuron is cut by a row added as it is, with the
    crossing point at -1/slope in kernel units: for an Octatope the box
    [-1, 2] (slope 3, centre 1/2, generator 3/2) or [-1, 3] (slope 2), a
    one-entry row; for a Hexatope the neuron 1 + slope (a0 - a1), a
    difference row, with neurons 3 + a0 and 3 + a1 that pin the kernel
    variables. Returns ``(set, true output of the kernel point d)``, where d
    is the neuron's kernel coordinate (x0, or a0 - a1 with a1 = 0)."""
    if kind == "Octatope":
        s = Octatope.from_bounds(np.array([-1.0]), np.array([slope - 1.0]))
        c, g = Fraction(s.center[0]), Fraction(s.generators[0, 0])
        return s, lambda d: [max(c + g * d, Fraction(0))]
    G = np.array([[slope, -slope], [1.0, 0.0], [0.0, 1.0]])
    s = Hexatope.from_bounds(-np.ones(2), np.ones(2)).affine_map(
        G, np.array([[1.0], [3.0], [3.0]]))
    return s, lambda d: [max(1 + Fraction(slope) * d, Fraction(0)), 3 + d,
                         Fraction(3)]


_ROW_CUT_ROUNDED = known_bug(
    9, "the ReLU split adds the cut of a neuron whose generator row is one "
       "entry or two entries of equal magnitude (Octatope) or a difference "
       "of two kernel variables (Hexatope) as a row whose bound is divided "
       "by the entry in round to nearest; when the crossing point is not a "
       "float and the quotient rounds into the active side, the member "
       "points between the two have a positive neuron value, which the "
       "active piece excludes and the inactive piece zeroes, so their true "
       "outputs (neuron value up to about 5e-17) are in no returned set")


@pytest.mark.parametrize("kind,slope", [
    pytest.param("Octatope", 3.0, marks=_ROW_CUT_ROUNDED,
                 id="Octatope-crossing-at-minus-1/3"),
    pytest.param("Octatope", 2.0, id="Octatope-crossing-at-minus-1/2"),
    pytest.param("Hexatope", 3.0, marks=_ROW_CUT_ROUNDED,
                 id="Hexatope-crossing-at-minus-1/3"),
    pytest.param("Hexatope", 2.0, id="Hexatope-crossing-at-minus-1/2")])
def test_relu_split_by_a_row_holds_the_outputs_at_the_crossing(kind, slope):
    """The cut of a crossing neuron that the bounding box adds as a row as
    it is (no solver) must keep every true output at the crossing point:
    the kernel points d = -1/slope + k 2^-60 for k in -16..16, at most
    1.4e-17 from the crossing point, have true outputs that must be members
    of a returned set, decided in rationals with zero slack (the returned
    sets carry no row a solver computed). With the crossing point -1/2 the
    quotient is exact and the neighbour passes with the same reference;
    with -1/3 the stored quotient lies about 1.9e-17 inside the active side
    (ROUND-26, PLAN step 9). Catches as well a direct row that
    leaves a gap between the two pieces, or a piece that is not cut."""
    s, true_output = _row_cut_set(kind, slope)
    what = (f"ReLU on {'an' if kind == 'Octatope' else 'a'} {kind} "
            f"whose crossing point is -1/{slope:g}")
    out = reach_one_layer(_nn.ReLU(), [s], "approx", what)
    members = [_R.Membership(_R.star(o)) for o in out]
    crossing = Fraction(-1) / Fraction(slope)
    missed = [k for k in range(-16, 17)
              if not any(m(true_output(crossing + Fraction(k, 2 ** 60)))
                         for m in members)]
    assert not missed, (
        f"{what}: the true outputs of the kernel points -1/{slope:g} + k "
        f"2^-60 for k in {missed} are in none of the {len(out)} returned "
        f"sets (largest neuron value "
        f"{float(true_output(crossing + Fraction(max(missed), 2 ** 60))[0]):.3g})")


@pytest.mark.parametrize("kind", [
    pytest.param("Hexatope", marks=known_bug(
        18, "reach_layer with method='exact' on a Hexatope runs the "
            "over-approximating ReLU split (bounding boxes of the pieces) "
            "and returns it as the exact result", raises=DID_NOT_RAISE)),
    pytest.param("Octatope", marks=known_bug(
        18, "reach_layer with method='exact' on an Octatope runs the "
            "over-approximating ReLU split (bounding boxes of the pieces) "
            "and returns it as the exact result", raises=DID_NOT_RAISE))])
def test_exact_relu_on_a_hexatope_or_octatope_refuses(kind):
    """The Hexatope and Octatope ReLU intersects each piece with a half-space
    by a bounding box of the difference or UTVPI system, which is not the
    exact image, so ``method='exact'`` must refuse and name ``"approx"``
    (DESIGN section 7). The passing neighbour is the model entry below,
    which refuses."""
    s = _coupled(kind, *_MIXED)
    with pytest.raises(NotImplementedError):
        _reach_layer(_nn.ReLU(), [s], "exact")


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_exact_hexoct_through_the_model_entry_refuses(kind):
    """The model entry refuses ``method='exact'`` for a Hexatope and an
    Octatope (``ValueError`` today, ``UnsupportedError`` later); the refusal
    names ``'approx'``, which the match checks, so that an unrelated
    ``ValueError`` raised on the way does not pass. Catches a model entry
    that runs the walk under ``exact``
    for these sets (the kernel would return its relaxation as exact)."""
    with pytest.raises((ValueError, NotImplementedError), match="'approx'"):
        _reach_pytorch_model(_chain(), flat_set(kind, _LO3, _HI3),
                             method="exact")
