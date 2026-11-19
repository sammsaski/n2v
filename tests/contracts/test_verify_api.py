"""Verification entry points: what they refuse and what they never return
(DESIGN sections 4, 7, 8, 9 and 17).

A verdict of UNSAT is a proof. Three inputs carry no proof and must be refused
instead of answered: an unsafe region with nothing in it, a reach result with
nothing in it that no emptiness certificate backs, and an option the call
cannot honour. Counterexample search must work on the dtype the model is in.

The last section is the model handle ``n2v.nn.NeuralNetwork`` (DESIGN section
9), the entry most callers reach through: wrapping a module leaves the
caller's module as it was; the function analysed is the module's inference
function in its own dtype; ``forward`` returns what that function returns;
``reach`` hands its method and every option on unchanged and refuses an option
value it cannot honour; ``layers`` replayed in order computes the model, or the
access is refused, and a model it cannot read is never listed as a shorter
one.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

import n2v
from n2v.nn import NeuralNetwork, ReachConfig
from n2v.nn import reach as reach_module
from n2v.nn.bab import verify_bab
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.probabilistic import ConformalReachConfig
from n2v.sets import Box, HalfSpace, Star, predicate_box
from n2v.utils import lpsolver
from n2v.utils.falsify import falsify
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_nonlinear import verify_nonlinear_reach
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_star, flat_box,
                                flat_set, replay_onnx, replay_torch, samples,
                                save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug


def _unit_star(n=1):
    return Star.from_bounds(np.zeros((n, 1)), np.ones((n, 1)))


def _affine(weight, bias=None, dtype=torch.float64):
    """One Linear layer with the given weight (rows = outputs)."""
    weight = torch.as_tensor(weight, dtype=dtype)
    layer = nn.Linear(weight.shape[1], weight.shape[0]).to(dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.zero_()
        if bias is not None:
            layer.bias.copy_(torch.as_tensor(bias, dtype=dtype))
    return layer


# --------------------------------------------------------------------------- #
# An unsafe region with nothing in it                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [
    pytest.param([], id="empty-list"),
    pytest.param({"Hg": []}, id="dict-with-empty-Hg"),
    pytest.param([{"Hg": []}], id="group-with-empty-Hg"),
])
@known_bug(6, "an unsafe region given as an empty list of half-spaces is "
              "answered UNSAT instead of refused", raises=DID_NOT_RAISE)
def test_empty_unsafe_region_is_refused(spec):
    """No half-space means no statement of what is unsafe. Every reach set is
    'disjoint' from nothing, which today reads as a proof."""
    with pytest.raises(ValueError):
        verify_specification([_unit_star()], spec)


def test_unsafe_region_with_no_rows_is_never_unsat():
    """A half-space object with zero rows is the whole output space: every
    output is unsafe, so UNSAT would be wrong for any non-empty reach set."""
    whole_space = HalfSpace(np.zeros((0, 1)), np.zeros((0, 1)))
    try:
        result = verify_specification([_unit_star()], whole_space)
    except ValueError:
        return                                  # refusing it is also correct
    assert result.verdict != "UNSAT"


# --------------------------------------------------------------------------- #
# A reach result with nothing in it                                             #
# --------------------------------------------------------------------------- #
_UNSAFE = HalfSpace(np.array([[1.0]]), np.array([[5.0]]))       # y <= 5


@known_bug(16, "a bare empty list of reach sets carries no emptiness "
               "certificate, yet it is answered UNSAT", raises=DID_NOT_RAISE)
def test_empty_list_of_reach_sets_is_refused():
    """Replaces ``test_verify_specification.py::test_empty_reach_set``, which
    asserted the UNSAT. A kernel that loses every set by mistake would turn
    into a proof here."""
    with pytest.raises(ValueError):
        verify_specification([], _UNSAFE)


@known_bug(16, "reach_layer passes an empty list of input sets through as an "
               "empty result", raises=DID_NOT_RAISE)
def test_reach_layer_refuses_an_empty_list_of_input_sets():
    with pytest.raises(ValueError):
        reach_layer(nn.ReLU(), [], "exact")


@pytest.mark.parametrize("method", ["exact", "approx"])
@known_bug(16, "reach returns no sets on the finitizer's emptiness report "
               "without confirming it")
def test_reach_never_returns_nothing_for_a_feasible_input(monkeypatch, method):
    """The input star is feasible and the predicate-box finitizer is made to
    report it empty. Reach may refuse, or carry on with the star; it must not
    return zero sets, which the specification check reads as UNSAT. The refine
    drivers have this guard (``test_unconfirmed_empty_input_is_unknown``).

    This demands more than the decided sentence of DESIGN section 8 (reach
    returns no sets only when ``confirm_infeasible`` confirmed emptiness):
    the report itself is false here, so reach has to check it again or
    refuse. ``test_reach_returns_no_sets_only_after_the_confirmation`` is the
    test at the level of the sentence.

    The finitizer is replaced where it is defined and, when ``n2v.nn.reach``
    has the name (an import at the top of the module), there too, so the
    test reaches reach wherever the import sits; a replacement that is never
    called stops the test as broken, outside the marker."""
    calls = []

    def reports_empty(star, lp_solver=None):
        calls.append(star)
        return star, True

    monkeypatch.setattr(predicate_box, "ensure_finite_predicate_box",
                        reports_empty)
    if hasattr(reach_module, "ensure_finite_predicate_box"):
        monkeypatch.setattr(reach_module, "ensure_finite_predicate_box",
                            reports_empty)
    net = NeuralNetwork(nn.Sequential(_affine([[1.0, -1.0], [0.5, 1.0]]),
                                      nn.ReLU()))
    star = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    if star.is_empty_set():
        # pytest.fail, not an assert: the marker absorbs AssertionError only.
        pytest.fail("setup: the input star is reported empty")
    try:
        sets = net.reach(star, method=method)
    except ValueError:
        sets = None
    if not calls:
        pytest.fail("reach never called the replaced finitizer: the test no "
                    "longer reaches the emptiness report it is about")
    if sets is None:
        return                                  # refusing is also correct
    assert len(sets) >= 1, f"reach(method={method!r}) returned no sets"


def _feasible_star_with_an_unbounded_variable():
    """alpha_0 has the box (-inf, inf) and is held to [-1, 1] only by two
    rows; a third row cuts a corner. alpha = (0, 0) meets every row, so the
    star is not empty (checked in fractions by the caller)."""
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0]])
    d = np.array([[1.0], [1.0], [1.5]])
    return Star(V, C, d, np.array([[-np.inf], [-1.0]]),
                np.array([[np.inf], [1.0]]))


_LP_BACKENDS = ("solve_lp_no_presolve", "_solve_lp_highspy", "_solve_lp_scipy",
                "_solve_lp_cvxpy")


@pytest.mark.parametrize("method", ["exact", "approx"])
def test_reach_returns_no_sets_only_after_the_confirmation(monkeypatch,
                                                           method):
    """The decided sentence (DESIGN section 8): reach returns no sets only when
    ``confirm_infeasible`` confirmed that the input is empty. The star is
    feasible and has an unbounded predicate variable, so the real finitizer
    solves bound LPs; every LP solver backend below it (the seam under
    ``solve_lp`` and ``_solve_lp_careful``, which ``confirm_infeasible`` does
    not use) is made to report 'infeasible', while ``confirm_infeasible``
    runs as it is and finds the star feasible. Reach may refuse
    (``ValueError``) or carry on; it must not return zero sets. Catches a
    finitizer that drops a star on a solver's 'infeasible' without the
    confirmation. Today the finitizer refuses here, so no later step of reach
    runs; the later steps (the phase LPs of the ReLU kernels) are the cell
    ``test_reach_keeps_a_star_whose_phase_lps_only_reported_infeasible``."""
    star = _feasible_star_with_an_unbounded_variable()
    alpha = [Fraction(0), Fraction(0)]
    if not all(sum(Fraction(float(cij)) * aj for cij, aj in zip(row, alpha))
               <= Fraction(float(di)) for row, di in zip(star.C, star.d[:, 0])):
        pytest.fail("the input star is not shown feasible")
    calls = []

    def reports_infeasible(*args, **kwargs):
        calls.append(args)
        return None, None, "infeasible", {"solver": "stub"}

    for name in _LP_BACKENDS:
        monkeypatch.setattr(lpsolver, name, reports_infeasible)
    net = NeuralNetwork(nn.Sequential(_affine([[1.0, -1.0], [0.5, 1.0]]),
                                      nn.ReLU()))
    try:
        sets = net.reach(star, method=method)
    except ValueError:
        sets = None
    if not calls:
        pytest.fail("no LP solver backend was asked: the test no longer "
                    "reaches the finitizer's bound LPs")
    if sets is None:
        return                                  # refusing is correct
    assert len(sets) >= 1, (
        f"reach(method={method!r}) returned no sets for a feasible input "
        f"whose LPs only reported 'infeasible'")


@pytest.mark.parametrize("method,options", [
    pytest.param("exact", {}, id="exact"),
    pytest.param("approx", {"relax_method": "area", "relax_factor": 0.5},
                 id="approx-area"),
])
def test_reach_keeps_a_star_whose_phase_lps_only_reported_infeasible(
        monkeypatch, method, options):
    """The step after the finitizer.
    The star has a finite predicate box, so the finitizer solves no LP, and a
    binding row that is not a box row (``constrained_star``, bounds wider than
    [-1, 1]); both neurons of Linear then ReLU cross zero. Every LP backend,
    the batched HiGHS path that the exact ReLU's phase LPs take among them,
    reports 'infeasible', while ``confirm_infeasible`` runs as it is and finds
    the star feasible. Reach may refuse (``ValueError``); an answer must hold
    the torch float64 outputs of the star's member points (corners, grid
    points and points on the cut face). Catches a ReLU kernel that drops the
    star, or a neuron's side, on an LP's 'infeasible' without the
    confirmation (exact: the phase LP of ``_phase_range``; approx with the
    area relaxation: its bound LPs; standard approx asks no LP)."""
    star, points = constrained_star(np.array([-3.0, -1.0]),
                                    np.array([1.0, 2.0]))
    calls = []

    def reports_infeasible(*args, **kwargs):
        calls.append(args)
        return None, None, "infeasible", {"solver": "stub"}

    def reports_infeasible_batch(objectives, *args, **kwargs):
        calls.append(objectives)
        return [None] * len(objectives)

    for name in _LP_BACKENDS:
        monkeypatch.setattr(lpsolver, name, reports_infeasible)
    monkeypatch.setattr(lpsolver, "_solve_batch_highspy",
                        reports_infeasible_batch)
    model = _relu_net()
    try:
        sets = NeuralNetwork(model).reach(star, method=method, **options)
    except ValueError:
        sets = None
    if not calls:
        pytest.fail("no LP solver backend was asked: the test no longer "
                    "reaches an LP of the ReLU kernel")
    if sets is None:
        return                                  # refusing is correct
    assert_covered(sets, replay_torch(model, points, (1, 2)),
                   f"reach(method={method!r}, {options}) with every LP "
                   f"reporting 'infeasible'")


# A certified-empty input set: the model and the three stars of
# verify/decisions-rows-and-empty.md section 5.1. A is the
# feasible box [-1, 1]^2; B adds the rows a0 + a1 >= 1.5 and a0 + a1 <= 1
# (their sum reads 0 <= -0.5) with a finite predicate box; C is B with the box
# of a0 unbounded. Today exact reach returns [] for B and C, approx returns
# one (empty) set for B and [] for C, and verify_specification answers UNSAT
# on all four. When verify_specification starts refusing a bare [] (PLAN step
# 16), reach has to hand over a result that still carries the emptiness.
_EMPTY_V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_EMPTY_C = np.array([[-1.0, -1.0], [1.0, 1.0]])
_EMPTY_D = np.array([[-1.5], [1.0]])


def _case_star(case):
    if case == "A-feasible":
        return Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    if case == "B-certified-empty":
        return Star(_EMPTY_V, _EMPTY_C, _EMPTY_D, -np.ones((2, 1)),
                    np.ones((2, 1)))
    return Star(_EMPTY_V, _EMPTY_C, _EMPTY_D, np.array([[-np.inf], [-1.0]]),
                np.array([[np.inf], [1.0]]))


def _relu_twice():
    """Linear(identity), ReLU, Linear(identity), ReLU in float32."""
    layers = []
    for _ in range(2):
        layer = nn.Linear(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        layers += [layer, nn.ReLU()]
    return nn.Sequential(*layers)


def _gemm_relu_twice(tmp_path):
    eye, zero = np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W0", "B0"], ["a"], transB=1),
             helper.make_node("Relu", ["a"], ["r"]),
             helper.make_node("Gemm", ["r", "W1", "B1"], ["b"], transB=1),
             helper.make_node("Relu", ["b"], ["Y"])]
    return save_onnx(tmp_path, nodes, [1, 2], [1, 2],
                     {"W0": eye, "B0": zero, "W1": eye, "B1": zero})


_Y0_AT_MOST_ZERO = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))


def _reach_case(entry, case, method, tmp_path):
    if entry == "NeuralNetwork.reach":
        return NeuralNetwork(_relu_twice()).reach(_case_star(case),
                                                   method=method)
    return reach_pytorch_model(load_onnx(_gemm_relu_twice(tmp_path)),
                               _case_star(case), method=method)


def _rows_contradict():
    """The two rows of B and C add up to 0 <= -0.5, in fractions."""
    total = [Fraction(float(a)) + Fraction(float(b))
             for a, b in zip(_EMPTY_C[0], _EMPTY_C[1])]
    return total == [0, 0] and (Fraction(float(_EMPTY_D[0, 0]))
                                + Fraction(float(_EMPTY_D[1, 0])) < 0)


@pytest.mark.parametrize("entry", ["NeuralNetwork.reach",
                                   "reach_pytorch_model(load_onnx)"])
@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.parametrize("case", ["B-certified-empty", "C-unbounded-variable"])
def test_a_certified_empty_input_stays_unsat_end_to_end(case, method, entry,
                                                        tmp_path):
    """A star with no point, certified empty (B, finite predicate box) or
    confirmed empty by ``confirm_infeasible`` (C, an unbounded predicate
    variable): reach, then ``verify_specification`` on whatever reach
    returned, must answer UNSAT and raise nothing. Guards PLAN step 16: a
    refusal of the bare [] that reach hands over today, without reach
    returning a result that carries the emptiness, turns this UNSAT into an
    error. Catches also a kernel or a model loop that raises on an empty
    star."""
    if not _rows_contradict():
        pytest.fail("the rows of the empty star do not contradict")
    result = _reach_case(entry, case, method, tmp_path)
    assert verify_specification(result, _Y0_AT_MOST_ZERO).verdict == "UNSAT"


@pytest.mark.parametrize("entry", ["NeuralNetwork.reach",
                                   "reach_pytorch_model(load_onnx)"])
@pytest.mark.parametrize("method", ["exact", "approx"])
def test_a_feasible_input_beside_the_empty_ones_is_not_unsat(method, entry,
                                                             tmp_path):
    """The neighbour of the cell above (case A, the box [-1, 1]^2, same model
    and entries): reach returns sets that hold 128 torch float64 outputs of
    the box, and y0 <= 0, which holds the true outputs at x0 <= 0, is not
    answered UNSAT."""
    result = _reach_case(entry, "A-feasible", method, tmp_path)
    assert len(result) >= 1
    lo, hi = -np.ones(2), np.ones(2)
    assert_covered(result, replay_torch(_relu_twice(), samples(lo, hi), (1, 2)),
                   f"{entry} {method} on the feasible box")
    assert verify_specification(result, _Y0_AT_MOST_ZERO).verdict != "UNSAT"


def _relu_then(tail, tmp_path):
    """The graph X (1, 2) -> Relu -> tail,
    and the tail's value at a zero input (the least output, which y0 <= that
    value reaches at x0 <= 0)."""
    nodes = [helper.make_node("Relu", ["X"], ["r"])]
    inits = {}
    if tail == "Gemm":
        inits = {"w": np.array([[1.0, 0.5], [-0.25, 1.0]], dtype=np.float32),
                 "b": np.array([0.25, 0.0], dtype=np.float32)}
        nodes.append(helper.make_node("Gemm", ["r", "w", "b"], ["Y"]))
        at_zero = 0.25
    elif tail == "Sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["r"], ["Y"]))
        at_zero = 0.5
    elif tail == "Pow":
        inits = {"p": np.array(2.0, dtype=np.float32)}
        nodes.append(helper.make_node("Pow", ["r", "p"], ["Y"]))
        at_zero = 0.0
    elif tail in ("Sin", "Cos"):
        nodes.append(helper.make_node(tail, ["r"], ["Y"]))
        at_zero = 0.0 if tail == "Sin" else 1.0
    elif tail == "Add":
        nodes.append(helper.make_node("Add", ["r", "r"], ["Y"]))
        at_zero = 0.0
    else:
        inits = {"k": np.array([[2.0, 3.0]], dtype=np.float32)}
        nodes.append(helper.make_node("Mul", ["r", "k"], ["Y"]))
        at_zero = 0.0
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits, name=tail)
    return load_onnx(path), HalfSpace(np.array([[1.0, 0.0]]),
                                      np.array([[at_zero]]))


_RANGE_LP_CRASHES_ON_AN_EMPTY_STAR = known_bug(
    7, "the approx Pow, Sin and Cos kernels raise ValueError on an empty star "
       "(their range LP finds no optimum and they read that as a failure), so "
       "a certified-empty input reached through Relu then Pow, Sin or Cos is "
       "an error instead of UNSAT",
    raises=ValueError)
_TAILS = ("Gemm", "Sigmoid", "Pow", "Add", "Mul", "Sin", "Cos")
_CRASHING_TAILS = ("Pow", "Sin", "Cos")
# Operations with no exact rule: DESIGN section 17 and registry REACH-4 say
# method='exact' on them raises NotImplementedError (UnsupportedError), which
# is a correct outcome here; every
# assertion still holds when an answer comes back.
_NO_EXACT_RULE = ("Sigmoid", "Pow", "Sin", "Cos")


def _reach_tail(model, star, method, tail):
    """The reach sets, or None for the refusal of an exact rule that does not
    exist."""
    try:
        return reach_pytorch_model(model, star, method=method)
    except NotImplementedError:
        if method == "exact" and tail in _NO_EXACT_RULE:
            return None
        raise


@pytest.mark.parametrize("method,tail", [
    (method, tail) for method in ("exact", "approx") for tail in _TAILS
    if method == "exact" or tail not in _CRASHING_TAILS] + [
    pytest.param("approx", tail, marks=_RANGE_LP_CRASHES_ON_AN_EMPTY_STAR)
    for tail in _CRASHING_TAILS])
def test_a_certified_empty_star_stays_unsat_through_the_rest_of_a_graph(
        method, tail, tmp_path):
    """The certified-empty star B through Relu followed by Gemm, Sigmoid, Pow,
    the Add of two branches, a Mul by a constant, Sin or Cos, loaded from
    ONNX: the verdict on
    whatever reach returns is UNSAT, and nothing raises. An empty result is
    carried through the rest of the graph today; this guards the loop over
    model nodes and the kernels after the empty one. Under
    ``method='exact'`` a Sigmoid, Pow, Sin or Cos tail may be refused with
    ``NotImplementedError`` (no exact rule, REACH-4)."""
    if not _rows_contradict():
        pytest.fail("the rows of the empty star do not contradict")
    model, unsafe = _relu_then(tail, tmp_path)
    result = _reach_tail(model, _case_star("B-certified-empty"), method, tail)
    if result is None:
        return                                  # the refusal is correct
    assert verify_specification(result, unsafe).verdict == "UNSAT"


@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.parametrize("tail", _TAILS)
def test_a_feasible_star_through_the_rest_of_a_graph_is_not_unsat(
        method, tail, tmp_path):
    """The neighbour: the feasible box A through the same graphs gives sets,
    and the unsafe region y0 <= (the output at x = 0), which every input with
    x0 <= 0 reaches, is not answered UNSAT. Under ``method='exact'`` a
    Sigmoid, Pow, Sin or Cos tail may be refused with ``NotImplementedError``
    (no exact rule, REACH-4)."""
    model, unsafe = _relu_then(tail, tmp_path)
    result = _reach_tail(model, _case_star("A-feasible"), method, tail)
    if result is None:
        return                                  # the refusal is correct
    assert len(result) >= 1
    assert verify_specification(result, unsafe).verdict != "UNSAT"


# The two other places that read a bare empty list as proved (owner decision
# 3 of 2026-11-13, registry rows EMPTY-4 and EMPTY-5). The promise is the one
# ``test_empty_list_of_reach_sets_is_refused`` states for
# verify_specification: a bare [] carries no certificate and is refused.
def _nonlinear_spec(tmp_path, threshold, lo=1.0, hi=2.0, op=">="):
    """x in [lo, hi], unsafe ``y * y op threshold`` (a VNN-LIB 2.0 nonlinear
    spec). The setup check is a plain ``assert``: the DID_NOT_RAISE marker of
    the test below absorbs ``pytest.fail``, not an AssertionError."""
    path = tmp_path / f"nonlinear_{lo}_{hi}_{op}_{threshold}.vnnlib"
    path.write_text(
        "(vnnlib-version <2.0>)\n"
        "(declare-network f\n"
        "    (declare-input X real [1,1])\n"
        "    (declare-output Y real [1,1])\n"
        ")\n"
        f"(assert (and (>= X[0,0] {lo}) (<= X[0,0] {hi})))\n"
        f"(assert ({op} (* Y[0,0] Y[0,0]) {threshold}))\n")
    spec = load_vnnlib(str(path))
    assert spec.get("format") == "nonlinear", (
        f"setup: the spec was not read as nonlinear: {spec.get('format')}")
    return spec


def _nonlinear_input(spec):
    lb = np.asarray(spec["lb"], dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(spec["ub"], dtype=np.float64).reshape(-1, 1)
    return Star.from_bounds(lb, ub)


def _identity_reach(input_set):
    """The reach sets of y = x on ``input_set`` (approx)."""
    return NeuralNetwork(nn.Sequential(_affine([[1.0]]))).reach(
        input_set, method="approx")


@known_bug(16, "verify_nonlinear_reach answers UNSAT for a bare empty list of "
               "reach sets, which carries no emptiness certificate",
           raises=DID_NOT_RAISE)
def test_nonlinear_verify_refuses_an_empty_list_of_reach_sets(tmp_path):
    """``verify_nonlinear_reach([], ...)``: the only caller (the competition
    runner) passes the result of ``NeuralNetwork.reach`` on a box, which is
    never empty on a legal input, so the function itself is the most public
    entry that gets a bare []. y = x on [1, 2] reaches y * y >= 2, so UNSAT
    is also wrong for the network. Setup, outside the marker (a plain
    ``assert``): the same input set and the real reach sets prove y * y >= 4.5
    unreachable, so the input set passes the checks that come before the
    empty-list branch and an answer of UNKNOWN from those checks cannot pass
    for the documented failure."""
    spec = _nonlinear_spec(tmp_path, 2.0)
    input_set = _nonlinear_input(spec)
    assert verify_nonlinear_reach(
        _identity_reach(input_set), input_set,
        _nonlinear_spec(tmp_path, 4.5)) == "UNSAT", (
        "setup: the input set is not read by verify_nonlinear_reach")
    with pytest.raises(ValueError):
        verify_nonlinear_reach([], input_set, spec)


@pytest.mark.parametrize("lo,hi,op,threshold", [
    pytest.param(1.0, 2.0, ">=", 2.0, id="inside"),
    pytest.param(1.0, 2.0, ">=", 4.0, id="upper-boundary"),
    pytest.param(1.0, 2.0, "<=", 1.0, id="lower-boundary"),
    pytest.param(-1.0, 2.0, "<=", 0.0, id="zero-crossing"),
])
def test_nonlinear_verify_on_a_real_reach_result_is_not_unsat(
        tmp_path, lo, hi, op, threshold):
    """The neighbour: y = x on [lo, hi] and the reach sets of y = x, with an
    unsafe region ``y * y op threshold`` that the network reaches (inside;
    on the boundary, at x = 2 where y * y = 4, at x = 1 where y * y = 1, and
    at x = 0 where y * y = 0 on an input that crosses zero), is not UNSAT.
    The true range of y * y is [lo^2, hi^2] or, across zero, [0, hi^2]. The
    function answers only UNSAT or UNKNOWN, and UNKNOWN is also its answer
    for an input set it cannot read, so the same sets must prove y * y >= 4.5
    unreachable: that shows the sets were read. Catches a check that drops a
    star or turns a non-strict comparison strict, and a product range that
    looks only at two of the four corner products (it misses y * y = 0 on
    the zero-crossing input)."""
    spec = _nonlinear_spec(tmp_path, threshold, lo, hi, op)
    input_set = _nonlinear_input(spec)
    sets = _identity_reach(input_set)
    assert len(sets) >= 1
    assert verify_nonlinear_reach(sets, input_set, spec) != "UNSAT"
    assert verify_nonlinear_reach(
        sets, input_set, _nonlinear_spec(tmp_path, 4.5, lo, hi)) == "UNSAT"


@pytest.mark.parametrize("order", ["reaching-star-last", "reaching-star-first"])
def test_nonlinear_verify_reads_every_reach_star(tmp_path, order):
    """Two reach stars: y = x on
    [1, 2] reached as the two pieces [1, 1.5] and [1.5, 2]. The unsafe region
    y * y >= 3 holds exactly for x in [sqrt(3), 2], inside the second piece
    only, so the verdict is not UNSAT; the same two stars prove y * y >= 4.5
    unreachable, which shows they are read. Catches a check that reads only
    the first star (the reaching star last) or only the last one (the
    reaching star first)."""
    spec = _nonlinear_spec(tmp_path, 3.0)
    input_set = _nonlinear_input(spec)
    low = _identity_reach(Star.from_bounds(np.array([[1.0]]),
                                           np.array([[1.5]])))
    high = _identity_reach(Star.from_bounds(np.array([[1.5]]),
                                            np.array([[2.0]])))
    sets = low + high if order == "reaching-star-last" else high + low
    assert verify_nonlinear_reach(sets, input_set, spec) != "UNSAT"
    assert verify_nonlinear_reach(
        sets, input_set, _nonlinear_spec(tmp_path, 4.5)) == "UNSAT"


_UNIT_BOX = (np.zeros(2), np.ones(2))
_Y0_AT_LEAST_HALF = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))


@known_bug(16, "verify_bab reads a bare empty list of reach sets from its "
               "reach_fn as a proved subdomain and answers VERIFIED",
           raises=DID_NOT_RAISE)
def test_bab_refuses_an_empty_list_of_reach_sets():
    """``verify_bab`` with a ``reach_fn`` that returns [] (the subdomain check
    ``_default_bound_fn`` reads it through ``verify_specification``).
    ``verify_bab_model`` builds its reach_fn from ``NeuralNetwork.reach`` on
    a box, which is never empty on a legal input, so ``verify_bab`` is the
    most public entry that gets a bare []. y0 >= 0.5 holds on most of the
    box, so VERIFIED is also wrong for the network of the neighbour."""
    with pytest.raises(ValueError):
        verify_bab(lambda lb, ub: [], *_UNIT_BOX, _Y0_AT_LEAST_HALF,
                   max_nodes=8)


def test_bab_on_a_real_reach_result_is_not_verified():
    """The neighbour: the same search with the reach sets of y0 = x0 + x1 / 2
    on each subdomain does not answer VERIFIED for y0 >= 0.5, which the
    network reaches, nor for y0 >= 1.5, which it reaches only at the corner
    (1, 1). The same reach_fn proves y0 >= 1.5 + 2^-10 unreachable, which
    shows that the subdomain check reads the sets. Catches a subdomain check
    that prunes a subdomain whose sets meet the unsafe region."""
    net = NeuralNetwork(nn.Sequential(_affine([[1.0, 0.5], [-0.25, 1.0]])))

    def reach_fn(lb, ub):
        return net.reach(Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)),
                         method="approx")

    result = verify_bab(reach_fn, *_UNIT_BOX, _Y0_AT_LEAST_HALF, max_nodes=8)
    assert result.verdict != "VERIFIED"
    corner = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-1.5]]))
    assert verify_bab(reach_fn, *_UNIT_BOX, corner,
                      max_nodes=8).verdict != "VERIFIED"
    beyond = HalfSpace(np.array([[-1.0, 0.0]]),
                       np.array([[-(1.5 + 2.0 ** -10)]]))
    assert verify_bab(reach_fn, *_UNIT_BOX, beyond,
                      max_nodes=8).verdict == "VERIFIED"


# --------------------------------------------------------------------------- #
# Removed reach methods                                                         #
# --------------------------------------------------------------------------- #
class _Residual(nn.Module):
    """y = relu(x) + x."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x) + x


@known_bug(4, "method='hybrid' still runs, and reads only the module calls of "
              "the graph: a skip connection is dropped", raises=DID_NOT_RAISE)
def test_hybrid_reach_method_is_removed():
    """Replaces the two ``TestHybridMethod`` tests of
    ``tests/unit/probabilistic/test_integration.py``, which asserted only that
    something came back. On this model the true outputs are [0, 2] and the
    hybrid result is [0, 1]: it computes relu(x) and forgets the ``+ x``."""
    net = NeuralNetwork(_Residual())
    with pytest.raises(ValueError, match="conformal"):
        net.reach(_unit_star(), method="hybrid")


@known_bug(4, "the method name 'probabilistic' is still accepted as an alias "
              "of 'conformal'", raises=DID_NOT_RAISE)
def test_probabilistic_reach_alias_is_removed():
    net = NeuralNetwork(nn.Sequential(_affine([[1.0, 0.5]], dtype=torch.float32)))
    with pytest.raises(ValueError, match="conformal"):
        net.reach(Box(np.zeros(2), np.ones(2)), method="probabilistic",
                  m=50, epsilon=0.1, seed=0)


# --------------------------------------------------------------------------- #
# Options of the top-level reach (DESIGN sections 4 and 7)                      #
# --------------------------------------------------------------------------- #
# Today's entry (NeuralNetwork.reach) takes **kwargs: a misspelled keyword is
# dropped, an unknown relax_method runs the standard relaxation and
# relax_factor=0 runs exact splitting under "approx". PLAN step 16 keeps that
# entry as a forward with today's meaning and puts the validation on
# ``n2v.reach``, so these three tests are written against that name and fail
# today because it does not exist.
_NO_TOP_LEVEL_REACH = known_bug(
    16, "n2v.reach, the entry that validates its options, does not exist; "
        "today's entry accepts this call silently", raises=AttributeError)


def _relu_net():
    return nn.Sequential(_affine([[1.0, -1.0], [0.5, 1.0]]), nn.ReLU())


@_NO_TOP_LEVEL_REACH
def test_reach_rejects_a_misspelled_option():
    with pytest.raises(TypeError):
        n2v.reach(_relu_net(), _unit_star(2), "approx", relax_mehtod="area")


@_NO_TOP_LEVEL_REACH
def test_reach_rejects_an_unknown_relax_method():
    with pytest.raises(ValueError, match="relax_method"):
        n2v.reach(_relu_net(), _unit_star(2), "approx",
                  config=n2v.ReachConfig(relax_method="bogus",
                                         relax_factor=0.5))


@_NO_TOP_LEVEL_REACH
def test_reach_rejects_relax_factor_zero():
    """Zero means exact splitting today. That is not an over-approximation
    setting, so it is refused with a pointer to the honest option."""
    with pytest.raises(ValueError, match="split_piecewise_linear"):
        n2v.reach(_relu_net(), _unit_star(2), "approx",
                  config=n2v.ReachConfig(relax_method="area", relax_factor=0))


# --------------------------------------------------------------------------- #
# Counterexample search on a float64 model                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["random", "pgd", "random+pgd", "apgd",
                                    "square"])
@known_bug(5, "every attack builds float32 tensors, so a float64 model fails "
              "with a dtype mismatch", raises=RuntimeError)
def test_falsify_finds_a_counterexample_on_a_float64_model(method):
    """y0 = x0 + x1 / 2 on the unit box; unsafe is y0 >= 0.5, which holds on
    more than half of the box, so every attack finds a point. The only other
    float64 falsify test replaces the attack with a stub."""
    model = nn.Sequential(_affine([[1.0, 0.5], [-0.25, 1.0]]))
    lb, ub = np.zeros(2), np.ones(2)
    unsafe = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))
    status, witness = falsify(model, lb, ub, unsafe, method=method, seed=0)
    assert status == 0
    x = np.asarray(witness[0], dtype=np.float64).ravel()
    assert np.all(lb <= x) and np.all(x <= ub)
    with torch.no_grad():
        y = model(torch.as_tensor(x).unsqueeze(0)).numpy().ravel()
    assert y[0] >= 0.5


def test_falsify_finds_a_counterexample_on_a_float32_model():
    """The passing neighbour of the float64 cell: the same network in float32
    (its weights are float32 numbers), the default attack. The witness lies
    in the box and violates the property in exact arithmetic,
    x0 + x1 / 2 >= 1/2, read from the text of the property and not from the
    library's checker. Catches a witness outside the box or one that does not
    violate the property."""
    model = nn.Sequential(_affine([[1.0, 0.5], [-0.25, 1.0]],
                                  dtype=torch.float32))
    lb, ub = np.zeros(2), np.ones(2)
    unsafe = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))
    status, witness = falsify(model, lb, ub, unsafe, seed=0)
    assert status == 0
    x = [Fraction(float(v)) for v in np.asarray(witness[0]).ravel()]
    assert len(x) == 2 and all(0 <= v <= 1 for v in x)
    assert x[0] + x[1] / 2 >= Fraction(1, 2)


# --------------------------------------------------------------------------- #
# The model handle: NeuralNetwork (DESIGN section 9)                            #
# --------------------------------------------------------------------------- #
# Weights, biases and statistics below are dyadic unless a case says it needs
# a value float32 cannot hold, so float32 evaluation, float64 evaluation and
# exact arithmetic agree and an exact comparison is meaningful.
_W1, _B1 = [[1.0, -1.0], [0.5, 1.0]], [0.25, -0.5]
_W2, _B2 = [[1.0, 0.5], [-0.25, 1.0]], [0.0, 0.125]
_LO, _HI = np.array([-1.0, -1.0]), np.array([1.0, 2.0])   # both neurons cross 0


def _two_layer_net(dtype=torch.float64):
    """Linear, ReLU, Linear on two inputs; on the box [_LO, _HI] both hidden
    neurons take both signs."""
    return nn.Sequential(_affine(_W1, _B1, dtype), nn.ReLU(),
                         _affine(_W2, _B2, dtype))


def test_model_handle_refuses_an_object_that_is_not_a_module():
    """A plain function is not a module n2v can trace or reach: refused at
    construction with the deliberate ``TypeError`` of
    ``NeuralNetwork.__init__``. Catches a handle that accepts it and fails
    later, or never."""
    with pytest.raises(TypeError):
        NeuralNetwork(lambda x: x)


def _batchnorm_net(dtype):
    """Linear(2, 3), BatchNorm1d(3), ReLU, Linear(3, 1) with running
    statistics that are not the defaults and, in float64, weights float32
    cannot hold (0.1 and 2^25 + 1)."""
    bn = nn.BatchNorm1d(3).to(dtype)
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([0.5, -1.0, 2.0], dtype=dtype))
        bn.running_var.copy_(torch.tensor([4.0, 0.25, 9.0], dtype=dtype))
        bn.weight.copy_(torch.tensor([1.5, -2.0, 0.1], dtype=dtype))
        bn.bias.copy_(torch.tensor([0.25, 0.0, -0.1], dtype=dtype))
    return nn.Sequential(
        _affine([[1.0, 0.1], [2.0 ** 25 + 1.0, -1.0], [-0.5, 0.25]],
                [0.1, 0.0, -0.3], dtype),
        bn, nn.ReLU(), _affine([[1.0, -0.1, 3.0]], [0.7], dtype))


def _state_bytes(model):
    return {k: (v.dtype, v.detach().cpu().numpy().tobytes())
            for k, v in model.state_dict().items()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["float32", "float64"])
def test_the_callers_model_keeps_its_dtype_and_values(dtype):
    """``NeuralNetwork(model)`` and the calls on it (``layers``, ``forward``,
    ``reach`` on a Star, exact and approx, and on a Box) leave every parameter
    and buffer of the caller's module as it was: the same dtype and the same
    bytes (DESIGN section 9: the caller's module is never modified; registry
    SHAPE-4 names a float64 conversion in place). Catches a handle that
    converts the module in place (``model.float()`` rounds the float64
    weights 0.1 and 2^25 + 1, ``model.double()`` turns a float32 model into
    a float64 one) and a reach that folds BatchNorm into the caller's own
    layers. The module is in inference mode before it is wrapped: the
    training flag is the marked case below."""
    model = _batchnorm_net(dtype).eval()
    before = _state_bytes(model)
    net = NeuralNetwork(model)
    net.layers
    net.forward(torch.ones((2, 2), dtype=dtype))
    star = Star.from_bounds(_LO.reshape(-1, 1), _HI.reshape(-1, 1))
    net.reach(star, method="exact")
    net.reach(star, method="approx")
    net.reach(Box(_LO, _HI), method="approx")
    after = _state_bytes(model)
    assert sorted(after) == sorted(before)
    changed = [k for k in before if after[k] != before[k]]
    assert not changed, f"the caller's {changed} changed dtype or value"
    assert not any(m.training for m in model.modules())


@known_bug(12, "NeuralNetwork(model) calls eval() on the caller's module "
               "instead of on a private copy, so a model the caller is "
               "training leaves training mode (registry SHAPE-4)")
def test_wrapping_a_model_leaves_its_training_flag_unchanged():
    """A module in training mode (torch's default for a new module), with a
    BatchNorm and a Dropout whose forward depends on the flag, keeps the
    flag of every submodule after ``NeuralNetwork(model)`` (DESIGN section 9:
    a private copy is put in inference mode)."""
    model = nn.Sequential(_batchnorm_net(torch.float64), nn.Dropout(0.5))
    flags = [m.training for m in model.modules()]
    if not all(flags):
        pytest.fail("setup: the model is not in training mode")
    NeuralNetwork(model)
    assert [m.training for m in model.modules()] == flags


def _training_mode_net(dtype=torch.float64):
    """x -> Linear(2, 2) -> BatchNorm1d(2, eps=0) -> Dropout(0.5), left in
    training mode. Its inference function is
    y = (W x + b - mean) / sqrt(var) * gamma + beta, every value dyadic for a
    dyadic x (var = (4, 1/4)), so float32 and float64 compute it exactly. In
    training mode BatchNorm normalises by the batch and Dropout zeroes and
    scales entries at random."""
    bn = nn.BatchNorm1d(2, eps=0.0).to(dtype)
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor([-5.0, 3.0], dtype=dtype))
        bn.running_var.copy_(torch.tensor([4.0, 0.25], dtype=dtype))
        bn.weight.copy_(torch.tensor([2.0, -1.0], dtype=dtype))
        bn.bias.copy_(torch.tensor([0.25, 0.5], dtype=dtype))
    return nn.Sequential(
        _affine([[1.0, -0.5], [0.25, 2.0]], [0.5, -1.0], dtype),
        bn, nn.Dropout(0.5))


def _training_mode_net_inference(x):
    """The inference function of ``_training_mode_net`` in numpy, rows of x
    are points."""
    W, b = np.array([[1.0, -0.5], [0.25, 2.0]]), np.array([0.5, -1.0])
    z = np.asarray(x, dtype=np.float64) @ W.T + b
    return ((z - np.array([-5.0, 3.0])) / np.sqrt(np.array([4.0, 0.25]))
            * np.array([2.0, -1.0]) + np.array([0.25, 0.5]))


def test_forward_of_a_model_wrapped_in_training_mode_is_its_inference_function():
    """``forward`` of a handle around a module left in training mode returns
    the inference function, computed independently in numpy (exact: dyadic
    data). Catches a handle that does not put its model in inference mode:
    BatchNorm then normalises by the batch of three points and Dropout zeroes
    entries."""
    x = np.array([[0.0, 0.0], [1.0, 0.5], [-0.25, 1.0]])
    y = NeuralNetwork(_training_mode_net()).forward(torch.as_tensor(x))
    assert np.array_equal(y.numpy(), _training_mode_net_inference(x))


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float32, id="float32"),
    pytest.param(torch.float64, id="float64", marks=known_bug(
        13, "conformal_reach feeds an nn.Module float32 inputs whatever its "
            "dtype, so the conformal reach of a float64 model fails with "
            "'mat1 and mat2 must have the same dtype'", raises=RuntimeError)),
])
def test_conformal_reach_of_a_model_wrapped_in_training_mode_covers_its_inference_function(
        dtype):
    """``reach(method='conformal')`` through a handle around a module left in
    training mode covers the inference function, for a float32 and a float64
    module. On [0, 1]^2 the inference
    outputs lie in [5.25, 6.75] x [4, 8.5]; the batch-normalised outputs of
    training mode lie near [-2, 2]. Of 256 fresh uniform points (the
    distribution the calibration draws from), more than half outside the
    returned box fails the case. The box claims miscoverage at most
    epsilon = 0.01 with the coverage of a fresh score distributed as
    Beta(ell, m + 1 - ell) = Beta(999, 2) (m = 1000, ell = m - 1), so a
    correct implementation fails only if the coverage is below 0.9
    (probability 2.0e-44) or if 128 or more of 256 points fall outside at a
    miscoverage of at most 0.1 (probability 9.0e-59). Catches a handle that
    leaves the model in training mode (every fresh output outside)."""
    net = NeuralNetwork(_training_mode_net(dtype))
    box = net.reach(Box(np.zeros(2), np.ones(2)), method="conformal",
                    config=ConformalReachConfig(m=1000, epsilon=0.01,
                                                surrogate="naive", seed=3))
    lo = np.asarray(box.lb, dtype=np.float64).ravel()
    hi = np.asarray(box.ub, dtype=np.float64).ravel()
    x = np.random.default_rng(1).random((256, 2))
    y = _training_mode_net_inference(x)
    outside = int(np.sum(np.any((y < lo) | (y > hi), axis=1)))
    assert outside <= 128, (
        f"{outside} of 256 inference outputs lie outside the conformal box "
        f"[{lo}, {hi}]")


@pytest.mark.parametrize("make,input_size,output_size", [
    pytest.param(lambda: nn.Linear(3, 5), (3,), (5,), id="linear"),
    pytest.param(lambda: nn.Conv2d(2, 4, (2, 3)), (2, 3, 5), (4, 2, 3),
                 id="conv-two-channels-height-not-width"),
])
def test_input_size_gives_the_output_shape_without_the_batch_axis(
        make, input_size, output_size):
    """``NeuralNetwork(model, input_size=...)`` runs the model once and
    records ``output_size``, the output's shape without the batch axis
    (``input_size`` excludes it too), computed by hand: Linear(3, 5) gives
    (5,); a 2 x 3 kernel over a (2, 3, 5) image with 4 output channels gives
    (4, 2, 3). Without ``input_size`` both stay None. Catches an output size
    that keeps the batch axis or drops another one, and a handle that loses
    the ``input_size`` it was given."""
    net = NeuralNetwork(make(), input_size=input_size)
    assert net.input_size == input_size
    assert net.output_size == output_size
    bare = NeuralNetwork(make())
    assert bare.input_size is None and bare.output_size is None


@pytest.mark.parametrize("input_size", [
    pytest.param((2,), id="too-few-entries"),
    pytest.param((), id="empty-tuple"),
])
def test_an_input_size_the_model_cannot_take_is_refused(input_size):
    """Linear(3, 5) takes three entries per sample: an ``input_size`` of two
    and an empty one (a value Python reads as false, which must not skip the
    check) are refused with ``ValueError``. Catches a check that is skipped
    for a size Python reads as false, or that lets a wrong size through."""
    with pytest.raises(ValueError):
        NeuralNetwork(nn.Linear(3, 5), input_size=input_size)


@known_bug(12, "NeuralNetwork(model, input_size=...) checks the size with a "
               "float32 input whatever the model's dtype, so a legal float64 "
               "model is refused with ValueError ('mat1 and mat2 must have "
               "the same dtype')", raises=ValueError)
def test_input_size_of_a_float64_model_gives_its_output_shape():
    """The float64 sibling of the linear case above: Linear(3, 5) in float64
    with ``input_size=(3,)`` has the output size (5,)."""
    net = NeuralNetwork(nn.Linear(3, 5).double(), input_size=(3,))
    assert net.output_size == (5,)


@pytest.mark.parametrize("batch,dtype", [(1, torch.float64),
                                         (3, torch.float32)],
                         ids=["batch-of-one-float64", "batch-of-three-float32"])
def test_forward_returns_the_models_output_with_its_batch_axis(batch, dtype):
    """``forward`` returns what the model returns: the output with its batch
    axis (shape (batch, 2), also for a batch of one), in the model's dtype,
    without a gradient graph even for an input that requires one, and the
    values of y = W2 relu(W1 x + b1) + b2 computed in numpy (exact: dyadic
    data). Catches an output whose batch axis of one is squeezed away, an
    input or output cast to another dtype, and a forward outside no_grad."""
    x = np.array([[0.5, -1.0], [-2.0, 0.25], [1.5, 3.0]])[:batch]
    model = _two_layer_net(dtype)
    y = NeuralNetwork(model).forward(
        torch.as_tensor(x, dtype=dtype).requires_grad_(True))
    want = (np.maximum(x @ np.array(_W1).T + _B1, 0.0) @ np.array(_W2).T
            + _B2)
    assert tuple(y.shape) == (batch, 2)
    assert y.dtype == dtype and not y.requires_grad
    assert np.array_equal(y.numpy().astype(np.float64), want)


class _Wrap(nn.Module):
    """A module whose forward is ``fn(self, x)``, so each case below states
    its forward in one line."""

    def __init__(self, fn, **children):
        super().__init__()
        self.fn = fn
        for name, child in children.items():
            self.add_module(name, child)

    def forward(self, x):
        return self.fn(self, x)


def _nested():
    return _Wrap(lambda m, x: m.head(m.block(x)),
                 block=nn.Sequential(nn.Linear(3, 4), nn.ReLU(),
                                     nn.Sequential(nn.Linear(4, 4),
                                                   nn.Tanh())),
                 head=nn.Linear(4, 2))


def _functional():
    """Every functional activation ``_function_node_to_module`` maps, the
    LeakyReLU slope given positionally, by keyword and left out."""
    def forward(m, x):
        x = torch.tanh(F.relu(m.a(x)))
        x = m.b(F.leaky_relu(x, 0.25))
        x = torch.relu(F.leaky_relu(x, negative_slope=0.125))
        return torch.sigmoid(F.leaky_relu(m.c(x)))
    return _Wrap(forward, a=nn.Linear(3, 4), b=nn.Linear(4, 4),
                 c=nn.Linear(4, 2))


def _onnx_chain(tmp_path):
    w0 = np.array([[1.0, 0.5], [-0.25, 1.0], [2.0, -1.0]], dtype=np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W0", "B0"], ["a"], transB=1),
             helper.make_node("Relu", ["a"], ["r"]),
             helper.make_node("Gemm", ["r", "W1", "B1"], ["Y"], transB=1)]
    return load_onnx(save_onnx(
        tmp_path, nodes, [1, 2], [1, 1],
        {"W0": w0, "B0": np.array([0.5, 0.0, -0.25], dtype=np.float32),
         "W1": np.array([[1.0, -2.0, 0.5]], dtype=np.float32),
         "B1": np.array([0.125], dtype=np.float32)}, name="chain"))


def _onnx_reshape_tail(tmp_path):
    """Flatten, Gemm, Relu, then a Reshape whose target is a second input of
    the converter's module (``OnnxReshape(x, shape)``)."""
    nodes = [helper.make_node("Flatten", ["X"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "W", "B"], ["g"], transB=1),
             helper.make_node("Relu", ["g"], ["r"]),
             helper.make_node("Reshape", ["r", "S"], ["Y"])]
    return load_onnx(save_onnx(
        tmp_path, nodes, [1, 2, 3], [1, 2, 2],
        {"W": (np.arange(24, dtype=np.float32).reshape(4, 6) / 8 - 1),
         "B": np.array([0.5, 0.0, -0.25, 1.0], dtype=np.float32),
         "S": np.array([1, 2, 2], dtype=np.int64)}, name="reshape_tail"))


_LAYERS_DROP_A_FUNCTION = known_bug(
    12, "NeuralNetwork.layers drops every function call it cannot map to a "
        "layer (the + of a residual connection, torch.cat, F.gelu), so a "
        "model it cannot read is listed as a shorter chain instead of "
        "refused")
_LAYERS_DROP_A_METHOD = known_bug(
    12, "NeuralNetwork.layers drops every tensor method call but flatten "
        "(x.relu(), x.sigmoid(), x.view), so the model is listed as a "
        "shorter chain instead of refused or mapped")
_LAYERS_MODULE_WITH_A_SECOND_INPUT = known_bug(
    12, "NeuralNetwork.layers lists a module call that takes more than the "
        "set (an ONNX Reshape and its shape input) as a one-input layer, so "
        "replaying the list in order fails", raises=TypeError)
_LAYERS_FLATTEN_START = known_bug(
    12, "NeuralNetwork.layers maps a flatten whose start_dim is left out "
        "(x.flatten(), torch.flatten(x)) to nn.Flatten(start_dim=1), while "
        "torch flattens from axis 0, so the listed chain computes another "
        "shape")


@pytest.mark.parametrize("make,in_shape,may_refuse", [
    pytest.param(lambda tmp: nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Sigmoid()),
        (4, 3), False, id="sequential"),
    pytest.param(lambda tmp: _nested(), (4, 3), False, id="nested-containers"),
    pytest.param(lambda tmp: _functional(), (4, 3), False,
                 id="functional-activations"),
    pytest.param(lambda tmp: _Wrap(lambda m, x: x.flatten(1, 2)),
                 (2, 2, 3, 4), False, id="flatten-method-start-and-end"),
    pytest.param(lambda tmp: _Wrap(lambda m, x: x.flatten(2)),
                 (2, 2, 3, 4), False, id="flatten-method-start"),
    pytest.param(lambda tmp: _Wrap(
        lambda m, x: x.flatten(start_dim=2, end_dim=3)),
        (2, 2, 3, 4), False, id="flatten-method-keywords"),
    pytest.param(lambda tmp: _Wrap(lambda m, x: torch.flatten(x, 1, 2)),
                 (2, 2, 3, 4), False, id="torch-flatten-start-and-end"),
    pytest.param(_onnx_chain, (3, 2), False, id="graph-module-from-onnx"),
    pytest.param(_onnx_reshape_tail, (1, 2, 3), True,
                 id="graph-module-module-with-a-second-input",
                 marks=_LAYERS_MODULE_WITH_A_SECOND_INPUT),
    pytest.param(lambda tmp: _Wrap(lambda m, x: F.relu(m.a(x)) + x,
                                   a=nn.Linear(3, 3)),
                 (4, 3), True, id="residual-add",
                 marks=_LAYERS_DROP_A_FUNCTION),
    pytest.param(lambda tmp: _Wrap(lambda m, x: m.b(m.a(x).relu()),
                                   a=nn.Linear(3, 4), b=nn.Linear(4, 2)),
                 (4, 3), True, id="tensor-method-relu",
                 marks=_LAYERS_DROP_A_METHOD),
    pytest.param(lambda tmp: _Wrap(lambda m, x: x.flatten()),
                 (2, 2, 3, 4), True, id="flatten-method-no-argument",
                 marks=_LAYERS_FLATTEN_START),
    pytest.param(lambda tmp: _Wrap(lambda m, x: torch.flatten(x)),
                 (2, 2, 3, 4), True, id="torch-flatten-no-start",
                 marks=_LAYERS_FLATTEN_START),
])
def test_layers_replayed_in_order_compute_the_model(make, in_shape, may_refuse,
                                                    tmp_path):
    """``NeuralNetwork(model).layers`` applied one after the other to a batch
    gives the model's own output, shape and values bit for bit (the same
    torch operations in the same order). A model the list cannot express (a
    residual connection) must be refused instead (``TypeError`` or
    ``NotImplementedError``); for the marked cases either outcome is right.
    Catches a cache that never traces (an empty list), modules of nested
    containers looked up by node name, functional activations that are not
    collected, a LeakyReLU slope read from the wrong argument, a flatten
    whose start or end dimension is read from the wrong argument or default,
    an operation dropped from the list without notice, and a loaded model
    whose converter modules are traced into (a Reshape then drops out)."""
    model = make(tmp_path)
    if not isinstance(model, torch.fx.GraphModule):
        model = model.double()
    dtype = next(model.parameters(), torch.zeros((), dtype=torch.float64)).dtype
    x = torch.randn(in_shape, generator=torch.Generator().manual_seed(0),
                    dtype=torch.float64).mul(3.0).to(dtype)
    try:
        layers = NeuralNetwork(model).layers
    except (TypeError, NotImplementedError):
        if may_refuse:
            return
        raise
    with torch.no_grad():
        want = model(x)
        got = x
        for layer in layers:
            got = layer(got)
    assert got.shape == want.shape, (
        f"layers {[type(layer).__name__ for layer in layers]} give shape "
        f"{tuple(got.shape)}, the model {tuple(want.shape)}")
    assert torch.equal(got, want), (
        f"layers {[type(layer).__name__ for layer in layers]} compute "
        f"another function")


class _DataDependent(nn.Module):
    """``forward`` branches on the data, so ``torch.fx`` cannot trace it."""

    def __init__(self):
        super().__init__()
        self.lin = _affine([[1.0, -0.5], [0.25, 2.0]], [0.5, -1.0],
                           torch.float32)

    def forward(self, x):
        if x.sum() > 0:
            return self.lin(x)
        return x


def test_layers_of_an_untraceable_model_are_refused_on_every_access():
    """Wrapping a model ``torch.fx`` cannot trace is legal (the class
    docstring: the probabilistic methods never read ``layers``); reading
    ``layers`` is refused on the first and on every later access, and a
    sound reach is refused too (today ``TypeError``, DESIGN section 9 names
    ``UnsupportedError``), while a conformal reach answers and ``repr`` still
    shows no layer count. Catches a refusal that is returned (and cached)
    instead of raised, a handle that traces when it is built, and a reach or
    a repr that reads ``layers``."""
    net = NeuralNetwork(_DataDependent())
    for _ in range(2):
        with pytest.raises((TypeError, NotImplementedError)):
            net.layers
    assert repr(net) == "NeuralNetwork(layers=?, input_size=None)"
    with pytest.raises((TypeError, NotImplementedError)):
        net.reach(Star.from_bounds(np.full((2, 1), 0.5), np.ones((2, 1))),
                  method="approx")
    box = net.reach(Box(np.full(2, 0.5), np.ones(2)), method="conformal",
                    config=ConformalReachConfig(m=200, epsilon=0.05,
                                                surrogate="naive", seed=0))
    assert np.asarray(box.lb).size == 2 and np.asarray(box.ub).size == 2


def test_layers_are_traced_once_and_repr_never_traces(monkeypatch):
    """The trace runs on the first read of ``layers`` and never again (the
    second read returns the same list), never for a ``GraphModule`` (its own
    graph is read), and never for ``repr``, which shows ``layers=?`` until
    the trace has run and the count after it, beside ``input_size`` as given.
    Catches a cache that is never filled or never read, a repr that traces,
    and a GraphModule traced again."""
    traced = []
    real_trace = torch.fx.symbolic_trace

    def counting_trace(module, *args, **kwargs):
        traced.append(type(module).__name__)
        return real_trace(module, *args, **kwargs)

    monkeypatch.setattr(torch.fx, "symbolic_trace", counting_trace)
    model = nn.Sequential(_affine(_W1, _B1, torch.float32), nn.ReLU(),
                          _affine([[1.0, -0.5]], None, torch.float32))
    net = NeuralNetwork(model, input_size=(2,))      # output size (1,)
    assert repr(net) == "NeuralNetwork(layers=?, input_size=(2,))"
    assert traced == []
    first = net.layers
    second = net.layers
    assert traced == ["Sequential"] and second is first
    assert [type(layer).__name__ for layer in first] == [
        "Linear", "ReLU", "Linear"]
    assert repr(net) == "NeuralNetwork(layers=3, input_size=(2,))"
    graph_net = NeuralNetwork(real_trace(model))
    assert len(graph_net.layers) == 3 and traced == ["Sequential"]


def test_reach_without_a_method_is_the_exact_image():
    """``reach(star)`` with no method runs the documented default
    ``'exact'``: compared with the exact image in rationals (Linear, ReLU,
    Linear with both neurons crossing zero), every vertex of every exact
    piece lies in one returned star, and every vertex of every returned star
    is a true output. Catches a default that is not ``'exact'`` (the
    relaxation of ``'approx'`` holds points no input reaches)."""
    star = Star.from_bounds(_LO.reshape(-1, 1), _HI.reshape(-1, 1))
    sets = NeuralNetwork(_two_layer_net()).reach(star)
    if not (1 <= len(sets) <= 4 and all(s.nVar <= 4 for s in sets)):
        pytest.fail(f"{len(sets)} sets, predicate counts "
                    f"{[s.nVar for s in sets]}: too many to enumerate")
    pieces = rational.affine(rational.relu(rational.affine(
        rational.box(_LO, _HI), _W1, _B1)), _W2, _B2)
    computed = [rational.star(s) for s in sets]
    assert rational.escapes(pieces, *computed) == []
    assert rational.straddlers(pieces, *computed) == []
    vertices = [v for c in computed for v in rational.set_vertices(c)]
    assert rational.unreached(pieces, vertices) == []


def test_reach_hands_input_shape_on_to_the_model_entry(tmp_path):
    """An ONNX model on a (1, 2, 3) input computes x - ReduceMean(x, axis 1);
    a flat Star given with ``input_shape=(2, 3)`` through the handle holds
    128 onnxruntime outputs (exact: dyadic bounds, wider than 2 and beyond
    +-1). Without the shape the flat set is read as a (1, 6) tensor and the
    mean runs over the wrong axis. Catches a reach that does not hand its
    keyword options on."""
    nodes = [helper.make_node("ReduceMean", ["X"], ["M"], axes=[1],
                              keepdims=1),
             helper.make_node("Sub", ["X", "M"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2, 3], [1, 2, 3], name="centre")
    lo, hi = (8.0 * v for v in flat_box(6))
    sets = NeuralNetwork(load_onnx(path)).reach(
        Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1)),
        method="approx", input_shape=(2, 3))
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 2, 3]),
                   "x - ReduceMean(x, axis 1) with input_shape (2, 3)")


_BARE_OPTION_NOT_VALIDATED = known_bug(
    16, "a reach option given as a bare keyword is not validated (registry "
        "REACH-3): a relaxation factor outside [0, 1] under the standard "
        "rule, an unknown relax_method and n_workers below 1 are answered",
    raises=DID_NOT_RAISE)
_LP_SOLVER_NAME_NOT_CHECKED = known_bug(
    16, "an unknown LP solver name is refused only by the Star lanes, which "
        "resolve it; the Box, Zono, Hexatope and Octatope lanes never read "
        "it and answer, as a bare keyword and in a ReachConfig alike",
    raises=DID_NOT_RAISE)


def _star_of_the_box():
    return Star.from_bounds(_LO.reshape(-1, 1), _HI.reshape(-1, 1))


@pytest.mark.parametrize("call,expected", [
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="aprox"),
                 ValueError, id="misspelt-method"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method=None),
                 ValueError, id="method-none"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       lp_solver="higsh"),
                 ValueError, id="unknown-lp-solver"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       precompute_bounds="symbolic"),
                 ValueError, id="unknown-bound-prepass"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       config={"relax_factor": 0.5}),
                 TypeError, id="config-of-another-type"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       config=ReachConfig(method="approx"),
                                       lp_solver="default"),
                 TypeError, id="config-beside-keywords"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       config=ReachConfig(method="exact")),
                 TypeError, id="config-method-disagrees"),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       config=ReachConfig(method="approx",
                                                          relax_factor=1.25)),
                 ValueError, id="config-relax-factor-above-one"),
    pytest.param(lambda net: net.reach(Box(_LO, _HI), method="approx",
                                       config=ReachConfig(method="approx",
                                                          lp_solver="higsh")),
                 ValueError, id="config-unknown-lp-solver-on-a-box",
                 marks=_LP_SOLVER_NAME_NOT_CHECKED),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="approx",
                                       input_shape=(3,)),
                 ValueError, id="input-shape-of-another-size",
                 marks=known_bug(
                     12, "an input_shape whose number of entries is not the "
                         "set's dimension is checked only by an operation "
                         "that reads the tracked shape; on a model with "
                         "none it is answered", raises=DID_NOT_RAISE)),
    pytest.param(lambda net: net.reach(_star_of_the_box(), method="conformal",
                                       m=50, epsilon=0.1, seed=0),
                 (TypeError, NotImplementedError),
                 id="probabilistic-method-on-a-star"),
])
def test_reach_refuses_an_option_value_it_cannot_honour(call, expected):
    """Through the handle, an option value the call cannot honour is refused:
    a misspelt method and ``None`` (a value Python reads as false, which must
    not become the default), an unknown LP solver or bound pre-pass name
    (``ValueError``); a ``config`` of another type, beside keywords, or whose
    method disagrees (``TypeError``, as the docstring states); a relaxation
    factor above one in a ``ReachConfig``; a probabilistic method on a Star
    (today ``TypeError``, the settled type is ``NotImplementedError``). The
    marked cases: an unknown LP solver name in a ``ReachConfig`` on a Box
    (DESIGN section 10: ``ValueError``), whose lane solves no LP and never
    reads the name; an ``input_shape`` of three entries for a set of two,
    a shape that does not fit (``ValueError``), on Linear, ReLU, Linear,
    which reads no tracked shape. Catches a reach that does not hand its
    method or its options on (the value is then never read and the call
    answers)."""
    with pytest.raises(expected):
        call(NeuralNetwork(_two_layer_net()))


@known_bug(16, "unknown reach keywords are accepted silently: a misspelt "
               "option is dropped and the default runs (registry REACH-3)",
           raises=DID_NOT_RAISE)
def test_reach_refuses_a_keyword_no_part_of_the_call_reads():
    """``relax_mehtod='area'`` (misspelt) is read by nothing: the call runs
    the standard relaxation as if it had not been given. The sibling through
    ``n2v.reach`` is ``test_reach_rejects_a_misspelled_option``."""
    with pytest.raises(TypeError):
        NeuralNetwork(_two_layer_net()).reach(_star_of_the_box(),
                                              method="approx",
                                              relax_mehtod="area")


@pytest.mark.parametrize("kind,method,values", [
    pytest.param("Star", "approx",
                 ({"relax_factor": 1.25}, {"relax_factor": -0.25}),
                 id="relax-factor-outside-0-1",
                 marks=_BARE_OPTION_NOT_VALIDATED),
    pytest.param("Star", "approx", ({"relax_method": "bogus"},),
                 id="unknown-relax-method", marks=_BARE_OPTION_NOT_VALIDATED),
    pytest.param("Star", "exact", ({"parallel": True, "n_workers": 0},),
                 id="n-workers-below-one", marks=_BARE_OPTION_NOT_VALIDATED),
    pytest.param("Box", "approx", ({"lp_solver": "higsh"},),
                 id="unknown-lp-solver-on-a-box",
                 marks=_LP_SOLVER_NAME_NOT_CHECKED),
])
def test_reach_refuses_an_out_of_range_option_given_as_a_keyword(kind, method,
                                                                 values):
    """The handle's docstring offers bare keywords and ``config=`` as two
    spellings of the same options; ``ReachConfig`` refuses a relaxation factor
    outside [0, 1] and ``n_workers`` below 1 (the neighbour
    ``test_reach_refuses_an_option_value_it_cannot_honour[config-relax-factor-above-one]``),
    and DESIGN section 10 says an unknown LP solver name raises ``ValueError``
    (the neighbour ``[unknown-lp-solver]`` on a Star). Given as bare keywords,
    each must be refused with ``ValueError`` too, as must an unknown
    ``relax_method``: -0.25 and 1.25 with the standard rule, which reads the
    factor only to compare it with
    0; ``n_workers=0`` with ``parallel=True``; ``lp_solver='higsh'`` on a Box,
    whose lane solves no LP. One case per code path that fails today, so a
    fix of one of them shows as a pass."""
    net = NeuralNetwork(_two_layer_net())
    for options in values:
        with pytest.raises(ValueError):
            net.reach(flat_set(kind, _LO, _HI), method=method, **options)


def _float64_net():
    """y0 = (2^25 + 1) relu(x0) - 2^25 relu(x0) and y1 = 0.1 relu(x1) in
    float64: neither 2^25 + 1 nor 0.1 is a float32 number. On x0 in [1, 2]
    the first output is x0 itself."""
    return nn.Sequential(
        _affine([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), nn.ReLU(),
        _affine([[2.0 ** 25 + 1.0, -(2.0 ** 25), 0.0], [0.0, 0.0, 0.1]]))


def test_one_handle_of_a_float64_model_answers_every_reach_for_its_weights():
    """One handle, its ``layers`` read first, reached four times in a row
    (Star exact and approx, Zono, Box): each result holds the true outputs
    (y0 = x0, y1 = 0.1 relu(x1), computed in numpy before the model is
    wrapped) of 128 points of [1, 2] x [-3, 5], and y0 >= 2, reached at
    x0 = 2, is not answered UNSAT; approx on the Star gives one set (the
    third hidden neuron crosses zero, so exact splits it in two). Catches a
    handle that analyses a float32 copy of the model (2^25 + 1 becomes 2^25,
    so y0 is read as 0 and y0 >= 2 as unreachable), a method that is not
    handed on (a Zono and a Box are refused under the default 'exact', a Star
    is split) and a reach that leaves state on the handle for the next
    one."""
    lo, hi = np.array([1.0, -3.0]), np.array([2.0, 5.0])
    points = samples(lo, hi)
    outputs = [np.array([x[0], 0.1 * max(x[1], 0.0)]) for x in points]
    net = NeuralNetwork(_float64_net())
    net.layers
    y0_at_least_2 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-2.0]]))
    for kind, method in (("Star", "exact"), ("Star", "approx"),
                         ("Zono", "approx"), ("Box", "approx")):
        sets = net.reach(flat_set(kind, lo, hi), method=method)
        what = f"{kind} {method}"
        assert all(type(s).__name__ == kind for s in sets), what
        if (kind, method) == ("Star", "approx"):
            assert len(sets) == 1, f"approx returned {len(sets)} stars for one"
        assert_covered(sets, outputs, what)
        assert verify_specification(sets, y0_at_least_2).verdict != "UNSAT", (
            what)
