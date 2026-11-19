"""Probabilistic results: what a stated guarantee means and when an answer is
one (DESIGN section 14, DEFERRED-BUGS.md).

A probabilistic result is a statement of the form "with confidence c, the
probability of the bad event is at most epsilon". It is never a proof, so it
must not read as one, and the stated epsilon must hold for the network that
was asked about. Two rules follow and every test here is one of them:

* a stated epsilon is not below the true probability more often than the
  stated confidence allows;
* an answer of "safe" comes with an epsilon. A search that found nothing has
  no epsilon and answers UNKNOWN.

Four defects against these rules were deferred by the owner (DEFERRED-BUGS.md,
bugs 1 to 4): the tests are written to the rule and carry ``deferred_bug``.
The other two (the sum of two probabilistic boxes, a probabilistic box read as
a sound one) are fixed in PLAN step 13.

The last two sections test the probabilistic lane of ``verify_specification``
itself (``_verify_specification_probabilistic`` and the validation of
``ProbVerifyConfig``). A flow set whose flow is the identity (y = z, the
latent ball of radius ``threshold``) makes every certify method's answer a
closed form, so each field of the result is compared with the rule that
defines it: ``epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)`` with the
epsilon_2 the method's own test supports, ``delta_total`` = the set's
confidence times 1 - beta, a detector (``amls``, ``is_tilted``) states no
epsilon, and a method that could not certify answers UNKNOWN.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import logging
import math
import re
import warnings
from fractions import Fraction
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import beta as beta_distribution
from scipy.stats import binom, norm

from n2v.nn import NeuralNetwork
from n2v.probabilistic import FlowReachConfig, conformal_reach, flow_reach
from n2v.probabilistic import ConformalReachConfig
from n2v.probabilistic.flow.importance_sampling import (
    is_tilted_estimate_halfspace_mass)
from n2v.probabilistic.flow.scores import HyperrectScore
from n2v.probabilistic.flow.sets import AffineTransform, ProbabilisticSet
from n2v.sets import Box, HalfSpace, ProbabilisticBox
from n2v.utils.verify_specification import (ProbVerifyConfig,
                                            verify_specification)
from tests.known_bugs import DID_NOT_RAISE, deferred_bug, known_bug


def _at_least(row, bound):
    """The unsafe region ``row @ y >= bound`` as a half-space ``G y <= g``."""
    return HalfSpace(-np.array([row], dtype=float), -np.array([[bound]], dtype=float))


# --------------------------------------------------------------------------- #
# The sum of two probabilistic boxes                                            #
# --------------------------------------------------------------------------- #
@known_bug(13, "ProbabilisticBox.minkowski_sum states the larger of the two "
               "miss probabilities instead of their sum")
def test_sum_of_two_probabilistic_boxes_adds_the_miss_probabilities():
    """The sum holds the true output only when both operands do, so with no
    assumption about how the two draws depend on each other it misses with
    probability up to epsilon_a + epsilon_b = 0.03 (the union bound, DESIGN
    section 14): it may state no smaller epsilon. One-sided: a larger stated
    epsilon is sound. The operands have different m and ell (1000, 999 and
    500, 490), so a combination of the two calibrations that is not a valid
    one (a rank above the sample size) fails here too. Replaces
    ``test_probabilistic_box.py::test_minkowski_sum_with_probabilistic_box``,
    which asserted the maximum."""
    a = ProbabilisticBox([0, 0], [1, 1], m=1000, ell=999, epsilon=0.01)
    b = ProbabilisticBox([0, 0], [1, 1], m=500, ell=490, epsilon=0.02)
    total = a.minkowski_sum(b)
    union_bound = Fraction(a.epsilon) + Fraction(b.epsilon)
    assert Fraction(total.epsilon) >= union_bound - _UNION_BOUND_SLACK, (
        f"the sum states epsilon {total.epsilon}, the union bound is "
        f"{float(union_bound)}")


@known_bug(13, "ProbabilisticBox.minkowski_sum states a miss probability "
               "below the exact one")
def test_stated_miss_probability_of_a_sum_covers_the_exact_one():
    """u and v are independent and uniform on [0, 1]. The box [0, 0.95] x {0}
    holds (u, 0) with probability exactly 0.95, and {0} x [0, 0.95] holds
    (0, v) with probability exactly 0.95. Their sum is [0, 0.95]^2, which
    holds (u, v) with probability 0.95^2: it misses with probability 0.0975,
    and the sum states 0.05."""
    a = ProbabilisticBox([0, 0], [0.95, 0], m=1000, ell=999, epsilon=0.05)
    b = ProbabilisticBox([0, 0], [0, 0.95], m=1000, ell=999, epsilon=0.05)
    total = a.minkowski_sum(b)
    np.testing.assert_allclose(np.asarray(total.lb).ravel(), [0, 0])
    np.testing.assert_allclose(np.asarray(total.ub).ravel(), [0.95, 0.95])
    exact_miss = 1 - 0.95 ** 2
    assert total.epsilon >= exact_miss - 1e-12


_UNION_BOUND_SLACK = Fraction(1, 10**12)   # rounding of a correct sum


def _union_bound(*values):
    """The confidence (or coverage) two statements support together with no
    assumption about how their draws depend on each other: each fails with
    probability 1 - value, so both hold with probability at least
    ``sum(values) - (len(values) - 1)``. Exact, from the stated floats."""
    return sum(Fraction(v) for v in values) - (len(values) - 1)


@known_bug(13, "ProbabilisticBox.minkowski_sum states a confidence above "
               "a.confidence + b.confidence - 1, the value that needs no "
               "assumption about how the two draws depend on each other: "
               "it recomputes the confidence from the larger epsilon")
def test_confidence_of_a_sum_is_at_most_the_union_bound_of_its_operands():
    """The statement about the sum is derived from both operands' statements
    holding together. Each fails with probability 1 - confidence, so both
    hold with probability at least a.confidence + b.confidence - 1, and
    without an assumption about how the two draws depend on each other the
    sum may state no more (owner decision 2026-11-13). The operands state
    0.9995207555 and 0.9999999640, so the limit is 0.9995207195. Today the
    sum recomputes the confidence from the larger epsilon, which raises it to
    0.9999999640. The limit is computed in fractions from the operands; the
    slack of 1e-12 forgives only the rounding of a correct sum."""
    a = ProbabilisticBox([0, 0], [1, 1], m=1000, ell=999, epsilon=0.01)
    b = ProbabilisticBox([0, 0], [1, 1], m=1000, ell=999, epsilon=0.02)
    if not a.confidence < b.confidence:         # the case the test relies on
        pytest.fail("the operands no longer have different confidences")
    total = a.minkowski_sum(b)
    limit = _union_bound(a.confidence, b.confidence)
    assert Fraction(total.confidence) <= limit + _UNION_BOUND_SLACK, (
        f"the sum states confidence {total.confidence}, the operands "
        f"support at most {float(limit)}")


def _stated(result):
    """(coverage, confidence) a set states. A set that states none, such as a
    plain ``Box``, reads as sound: every output inside, with certainty."""
    if hasattr(result, "coverage") and hasattr(result, "confidence"):
        return result.coverage, result.confidence
    return 1.0, 1.0


def _within_union_bound(result, operands):
    """Both fields at most the union bound of the operands' fields."""
    coverage, confidence = _stated(result)
    stated = [_stated(o) for o in operands]
    return (Fraction(coverage) <= _union_bound(*(c for c, _ in stated))
            + _UNION_BOUND_SLACK
            and Fraction(confidence) <= _union_bound(*(d for _, d in stated))
            + _UNION_BOUND_SLACK)


def _within_one_operand(result, operands):
    """A hull contains each operand, so it may state either operand's own
    (coverage, confidence) pair, but not a pair no operand states."""
    coverage, confidence = _stated(result)
    return any(coverage <= c + 1e-12 and confidence <= d + 1e-12
               for c, d in map(_stated, operands))


def _two_probabilistic_boxes():
    """Overlapping boxes with different guarantees: coverage 0.99 at
    confidence 0.99952 and coverage 0.98 at confidence 0.99999996."""
    return (ProbabilisticBox([0, 0], [1, 1], m=1000, ell=999, epsilon=0.01),
            ProbabilisticBox([0.5, -0.5], [1.5, 0.5], m=1000, ell=999,
                             epsilon=0.02))


_INHERITED_BOX_METHOD = known_bug(
    13, "ProbabilisticBox inherits intersect, convex_hull and union from Box, "
        "and they return a plain Box: the combination of two probabilistic "
        "boxes reads as a sound set and states no coverage or confidence")
_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND = known_bug(
    13, "a sound Box operation (Box.minkowski_sum, intersect, convex_hull, "
        "union, Box.box_hull) given a ProbabilisticBox as its other operand "
        "returns a plain Box: the result reads as a sound set and states no "
        "coverage or confidence")

_NOT_OFFERED, _REFUSED = object(), object()


def _operation(receiver, name, *operands):
    """``receiver.name(*operands)``. ``_NOT_OFFERED`` when the receiver has no
    such method (decided before the call, so an ``AttributeError`` raised
    inside the operation is a crash, not an operation that is not offered);
    ``_REFUSED`` for a refusal: ``NotImplementedError``, or a ``TypeError``
    whose message says that a probabilistic set is not accepted (DESIGN
    section 14 keeps the two families apart that way). Any other exception
    is a crash and fails the test."""
    method = getattr(receiver, name, None)
    if method is None:
        return _NOT_OFFERED
    try:
        return method(*operands)
    except NotImplementedError:
        return _REFUSED
    except TypeError as error:
        if re.search(r"\bprobabilistic\b|certify_set", str(error), re.I):
            return _REFUSED
        raise


def _first_operand_only(limit):
    """A limit that counts only the probabilistic operand. A sound ``Box``
    operand is a fixed set and states nothing about the quantity the
    probabilistic operand covers, so a mixed operation may state at most what
    the probabilistic operand states."""
    return lambda result, operands: limit(result, operands[:1])


@pytest.mark.parametrize("receiver, name, other, limit", [
    pytest.param("a", "minkowski_sum", "sound b", _first_operand_only(
        _within_union_bound), id="sum-with-a-sound-box"),
    pytest.param("sound b", "minkowski_sum", "a", _first_operand_only(
        _within_union_bound), id="sound-box-plus-probabilistic-box",
        marks=_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND),
    pytest.param("sound b", "intersect", "a", _first_operand_only(
        _within_union_bound), id="sound-box-intersect-probabilistic-box",
        marks=_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND),
    pytest.param("sound b", "convex_hull", "a", _first_operand_only(
        _within_one_operand), id="sound-box-convex-hull-probabilistic-box",
        marks=_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND),
    pytest.param("sound b", "union", "a", _first_operand_only(
        _within_one_operand), id="sound-box-union-probabilistic-box",
        marks=_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND),
    pytest.param("Box", "box_hull", "[a, b]", _within_one_operand,
                 id="box-hull",
                 marks=_BOX_OPERATION_TAKES_A_PROBABILISTIC_OPERAND),
    pytest.param("a", "intersect", "b", _within_union_bound, id="intersect",
                 marks=_INHERITED_BOX_METHOD),
    pytest.param("a", "convex_hull", "b", _within_one_operand,
                 id="convex-hull", marks=_INHERITED_BOX_METHOD),
    pytest.param("a", "union", "b", _within_one_operand, id="union",
                 marks=_INHERITED_BOX_METHOD),
])
def test_a_combination_of_probabilistic_boxes_states_no_more_than_its_operands_support(
        receiver, name, other, limit):
    """The result of an operation on two sets, at least one of them
    probabilistic, may state no more coverage and no more confidence than the
    operands' statements support, or the operation is refused, or it is not
    offered. A sum or an intersection needs both statements to hold: each
    field is at most the union bound of the operands' fields. A hull contains
    each operand, so it may keep one operand's pair, not a better one. A
    sound ``Box`` operand states nothing about the quantity the probabilistic
    operand covers, so with one the result may state at most the
    probabilistic operand's pair. A plain ``Box`` result states coverage 1 at
    confidence 1, which no probabilistic operand supports: it is the harm of
    reading a probabilistic box as a sound one. Catches a Box operation that
    takes a probabilistic operand without notice (PROB-14), an inherited Box
    method on a probabilistic box (PROB-4), and, in the passing case
    ``[sum-with-a-sound-box]``, a sum with a sound box that changes the
    probabilistic operand's guarantee or crashes. The sum of two
    probabilistic boxes is the three tests above."""
    a, b = _two_probabilistic_boxes()
    sets = {"a": a, "b": b, "sound b": Box(b.lb, b.ub), "Box": Box,
            "[a, b]": [a, b]}
    result = _operation(sets[receiver], name, sets[other])
    if result is _NOT_OFFERED or result is _REFUSED:
        return
    assert result is not None, "the boxes overlap; the result is not empty"
    operands = (a, b)
    assert limit(result, operands), (
        f"{type(result).__name__} states (coverage, confidence) = "
        f"{_stated(result)}; operands state {[_stated(o) for o in operands]}")


# --------------------------------------------------------------------------- #
# A stated guarantee: its name and when it is vacuous                           #
# --------------------------------------------------------------------------- #
def _joint_confidence(pset, beta):
    """The joint confidence of a probabilistic answer from a flow set: the
    set's own confidence (Beta-binomial, from scipy) times 1 - beta of the
    scenario check, as ``verify_specification`` documents it."""
    own = 1 - Fraction(beta_distribution.cdf(1 - pset.epsilon, pset.ell,
                                             pset.m + 1 - pset.ell))
    return own * (1 - Fraction(beta))


def _scenario_answer():
    config = ProbVerifyConfig(method="scenario", n_samples=2000, seed=0)
    pset = _standard_normal_set()
    with warnings.catch_warnings():
        # From PLAN step 13 on, verify_specification warns that a
        # probabilistic answer is not a proof; that warning is another
        # test's. Only it is ignored: every other warning stays an error
        # (pytest.ini).
        warnings.filterwarnings("ignore", message=r".*\bprobabilistic\b")
        result = verify_specification(pset, _at_least([1.0], 5.0),
                                      config=config)
    return result, _joint_confidence(pset, config.beta)


def test_delta_total_keeps_the_joint_confidence():
    """DESIGN section 14: the old ``delta_total`` keeps its name and its
    value, the joint confidence, until 0.3. The reference is scipy's
    Beta-binomial confidence of the set times 1 - beta. Catches a value
    turned into a failure probability, or a joint confidence that leaves out
    one of the two draws. Neighbour of the next test (same answer, same
    reference)."""
    result, joint = _scenario_answer()
    if result.verdict != "UNSAT":
        pytest.fail(f"the case needs an UNSAT answer, got {result.verdict}")
    assert abs(Fraction(result.delta_total) - joint) <= Fraction(1, 10**12)


@known_bug(13, "VerificationResult.delta_total holds the joint confidence "
               "(0.9985 here) under a failure-probability name, and the "
               "result has no guarantee with a field named confidence")
def test_probabilistic_answer_states_its_confidence_in_a_field_named_confidence():
    """DESIGN section 14: ``Guarantee.confidence`` holds the confidence of a
    probabilistic answer. A reader who takes ``delta`` for a failure
    probability, its usual meaning, reads 0.9985 as near certain failure.
    The field may state less than the joint confidence, never more, and it
    holds a confidence, not the failure probability 0.0015. The field is read
    with a default, so a missing field is an assertion and any other error
    (a library ``AttributeError`` among them) fails the test."""
    result, joint = _scenario_answer()
    if result.verdict != "UNSAT":
        pytest.fail(f"the case needs an UNSAT answer, got {result.verdict}")
    confidence = getattr(getattr(result, "guarantee", None), "confidence",
                         None)
    assert confidence is not None, (
        "the result states no guarantee with a field named confidence")
    assert 0.5 < confidence and (
        Fraction(confidence) <= joint + Fraction(1, 10**12)), (
        f"guarantee.confidence = {confidence}, joint confidence {float(joint)}")


_VACUOUS_M, _VACUOUS_EPSILON = 500, 0.001


def _beta_binomial_confidence(m, epsilon):
    """Confidence of a conformal box with rank ell = m - 1, from scipy."""
    return 1 - beta_distribution.cdf(1 - epsilon, m - 1, 2)


def _m_that_reaches(confidence, epsilon):
    """The smallest m (rank m - 1) whose confidence reaches ``confidence``,
    by bisection: the confidence grows with m (each added calibration sample
    can only raise the probability that the rank-(m - 1) score exceeds the
    1 - epsilon quantile). The caller checks m - 1 as well."""
    low, high = 2, 4
    while _beta_binomial_confidence(high, epsilon) < confidence:
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if _beta_binomial_confidence(middle, epsilon) < confidence:
            low = middle
        else:
            high = middle
    return high


def test_the_m_named_by_the_vacuous_guarantee_warning_reaches_099():
    """The reference of the next two tests: m = 500 at epsilon 0.001 gives
    confidence 0.09, and 6636 is the smallest m that reaches 0.99."""
    assert _beta_binomial_confidence(_VACUOUS_M, _VACUOUS_EPSILON) < 0.1
    m = _m_that_reaches(0.99, _VACUOUS_EPSILON)
    assert m == 6636
    assert _beta_binomial_confidence(m - 1, _VACUOUS_EPSILON) < 0.99


def _small_network():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


_ENTRY_POINTS = {
    "conformal_reach": lambda model, box, config: conformal_reach(
        model, box, config),
    "NeuralNetwork.reach": lambda model, box, config: NeuralNetwork(
        model).reach(box, method="conformal", config=config),
}


@pytest.mark.parametrize("entry", _ENTRY_POINTS)
def test_stated_confidence_of_a_conformal_box_is_the_beta_binomial_value(
        entry):
    """The confidence the box states with m = 500 is scipy's Beta-binomial
    value, 0.09: the vacuous guarantee the next test is about is real.
    Neighbour of the next test (same calls, same reference). Catches a
    confidence computed with the wrong rank, size or epsilon."""
    config = ConformalReachConfig(m=_VACUOUS_M, epsilon=_VACUOUS_EPSILON,
                                  surrogate="naive", seed=0)
    box = _ENTRY_POINTS[entry](_small_network(),
                               Box(-np.ones(2), np.ones(2)), config)
    expected = _beta_binomial_confidence(_VACUOUS_M, _VACUOUS_EPSILON)
    assert abs(box.confidence - expected) <= 1e-12


@pytest.mark.parametrize("entry", _ENTRY_POINTS)
@known_bug(13, "a conformal box whose confidence is far below 0.9 (0.09 for "
               "m = 500 at epsilon 0.001) is returned without a warning")
def test_a_vacuous_guarantee_is_logged_as_a_warning_naming_the_m_that_reaches_099(
        entry, caplog):
    """DESIGN section 14: a confidence below 0.9 is logged as a warning
    naming the m that reaches 0.99 (6636 here, from scipy). Without it a
    caller reads coverage 0.999 and does not look at a confidence of 0.09.
    A Python warning with the same text is accepted as well."""
    config = ConformalReachConfig(m=_VACUOUS_M, epsilon=_VACUOUS_EPSILON,
                                  surrogate="naive", seed=0)
    needed = str(_m_that_reaches(0.99, _VACUOUS_EPSILON))
    with caplog.at_level(logging.WARNING), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ENTRY_POINTS[entry](_small_network(), Box(-np.ones(2), np.ones(2)),
                             config)
    messages = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    messages += [str(w.message) for w in caught]
    assert any(needed in message for message in messages), (
        f"no warning names m = {needed}; warnings: {messages}")


# --------------------------------------------------------------------------- #
# A probabilistic box is not a sound reach set                                  #
# --------------------------------------------------------------------------- #
def _probabilistic_box():
    """[0, 0.3] x [0.5, 1], claimed to hold the output with probability 0.99."""
    return ProbabilisticBox([0.0, 0.5], [0.3, 1.0], m=1000, ell=999, epsilon=0.01)


@known_bug(13, "ProbabilisticBox is a subclass of Box, so every function "
               "that takes a sound box takes it without notice")
def test_probabilistic_box_is_not_a_box():
    """Replaces ``test_integration.py::
    test_probabilistic_box_is_instance_of_box``, which asserted the
    subclassing."""
    assert not isinstance(_probabilistic_box(), Box)


_PROBABILISTIC_BOX_ROUTES = {
    "hand-built": _probabilistic_box,
    "NeuralNetwork.reach-conformal": lambda: NeuralNetwork(
        _small_network()).reach(
            Box(-np.ones(2), np.ones(2)), method="conformal",
            config=ConformalReachConfig(m=500, epsilon=0.05,
                                        surrogate="naive", seed=0)),
}


def _stated_epsilon(result):
    """The miss probability a result states: ``guarantee.epsilon`` (DESIGN
    section 14) or the old ``epsilon_total``; None when it states none."""
    epsilon = getattr(getattr(result, "guarantee", None), "epsilon", None)
    return epsilon if epsilon is not None else getattr(
        result, "epsilon_total", None)


@pytest.mark.parametrize("route", _PROBABILISTIC_BOX_ROUTES)
@known_bug(13, "a probabilistic box that misses the unsafe region is answered "
               "UNSAT with no warning that the answer is probabilistic and "
               "no epsilon")
def test_unsat_from_a_probabilistic_box_says_it_is_probabilistic(route):
    """A probabilistic box that misses the unsafe region (y0 at least 1 above
    the box) supports only "safe with probability": a fraction epsilon of the
    outputs may lie outside the box. DESIGN section 14 and PLAN step 13: such
    an answer warns that it is probabilistic and states its epsilon, at
    least the box's own. Fails for a plain 'UNSAT' (no warning) and for an
    'UNSAT' with the epsilon unset; an UNKNOWN states nothing and passes.
    Routes: a box built by hand and the box ``NeuralNetwork.reach(method=
    'conformal')`` returns. Replaces ``test_verify_specification.py::
    test_probabilistic_box_uses_box_path``, which asserted the bare UNSAT."""
    box = _PROBABILISTIC_BOX_ROUTES[route]()
    unsafe = _at_least([1.0, 0.0], float(np.ravel(box.ub)[0]) + 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = verify_specification([box], unsafe)
    verdict = str(getattr(result.verdict, "value", result.verdict))
    if verdict.upper() == "UNKNOWN":
        return
    messages = [str(w.message) for w in caught]
    assert any(re.search(r"\bprobabilistic\b", m, re.I) for m in messages), (
        f"verdict {verdict} with no warning that it is probabilistic; "
        f"warnings: {messages}")
    stated = _stated_epsilon(result)
    assert stated is not None and stated >= box.epsilon, (
        f"verdict {verdict} states epsilon {stated}; the box misses with "
        f"probability up to {box.epsilon}")


@pytest.mark.parametrize("unsafe", [
    pytest.param(HalfSpace(np.array([[1.0, -1.0]]), np.array([[0.0]])),
                 id="box-inside-the-unsafe-region"),          # y0 <= y1
    pytest.param(HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.1]])),
                 id="box-crosses-the-unsafe-region"),         # y0 <= 0.1
])
def test_probabilistic_box_that_meets_the_unsafe_region_is_not_unsat(unsafe):
    """Replaces ``test_integration.py::
    test_conformal_reach_specification_accepts_probabilistic_box``, which put
    the whole box inside the unsafe region and accepted any verdict."""
    result = verify_specification([_probabilistic_box()], unsafe)
    assert result.verdict == "UNKNOWN"


# --------------------------------------------------------------------------- #
# Deferred bug 1: coverage of the conformal box                                 #
# --------------------------------------------------------------------------- #
_M, _EPSILON, _FRESH = 2000, 0.003, 40_000
# Misses allowed among the fresh outputs when the true miss probability is
# epsilon, at a false-alarm probability of 1e-3 (155 of 40,000).
_ALLOWED_MISSES = int(binom.ppf(1 - 1e-3, _FRESH, _EPSILON))


def _misses_of_the_conformal_box(n_outputs, surrogate, seed):
    """Fresh outputs of a random 4-32-n ReLU network outside its conformal box."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(),
                          nn.Linear(32, n_outputs)).eval()
    lb, ub = -np.ones(4), np.ones(4)
    box = conformal_reach(model, Box(lb, ub), m=_M, epsilon=_EPSILON,
                          surrogate=surrogate, seed=seed)
    assert box.confidence > 0.98
    x = np.random.default_rng(10_000 + seed).uniform(lb, ub, size=(_FRESH, 4))
    with torch.no_grad():
        y = model(torch.from_numpy(x.astype(np.float32))).numpy()
    lo, hi = np.asarray(box.lb).reshape(-1), np.asarray(box.ub).reshape(-1)
    return int(np.any((y < lo) | (y > hi), axis=1).sum())


_UNDER_COVERS = "the conformal box under-covers: each output's scale is the " \
                "largest error over the calibration samples themselves"


@pytest.mark.parametrize("surrogate, n_outputs", [
    pytest.param("naive", 50, marks=deferred_bug(1, _UNDER_COVERS)),
    pytest.param("naive", 200, marks=deferred_bug(1, _UNDER_COVERS)),
    pytest.param("clipping_block", 50, marks=pytest.mark.slow),
    pytest.param("clipping_block", 200, marks=pytest.mark.slow),
])
def test_conformal_box_misses_no_more_often_than_it_states(surrogate, n_outputs):
    """The box states: with confidence 0.983, a fresh output is outside with
    probability at most 0.3%. Three independent runs; a run fails when its
    count of misses among 40,000 fresh outputs is above the binomial allowance.
    One failing run in three is within the stated confidence, two are not (a
    correct method fails this test about once in a thousand runs).

    Today the naive surrogate misses 2 to 7 times the allowance in every run.
    The clipping-block surrogate is within it at these sizes; it has the same
    structure and is not shown correct in general (DEFERRED-BUGS.md)."""
    misses = [_misses_of_the_conformal_box(n_outputs, surrogate, seed)
              for seed in range(3)]
    failing = [count for count in misses if count > _ALLOWED_MISSES]
    assert len(failing) <= 1, (
        f"misses per run {misses}, allowed {_ALLOWED_MISSES} of {_FRESH}")


# --------------------------------------------------------------------------- #
# Deferred bug 2: the epsilon of a flow result                                  #
# --------------------------------------------------------------------------- #
def _two_cluster_network():
    """y0 = 4 clip(50 x1 - 47, 0, 1) + 0.3 x0 and y1 = x2 on [-1, 1]^3.

    For x1 >= 0.96 (probability 0.02) y0 is in [3.7, 4.3], a small cluster
    apart from the main one, where y0 is in [-0.3, 0.3]. The band
    0.94 <= x1 <= 0.96 between them has probability 0.01."""
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(
            [[0, 50, 0], [0, 50, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
             [0, 0, -1]], dtype=torch.float32))
        model[0].bias.copy_(torch.tensor([-47.0, -48.0, 0, 0, 0, 0]))
        model[2].weight.copy_(torch.tensor(
            [[4, -4, 0.3, -0.3, 0, 0], [0, 0, 0, 0, 1, -1]],
            dtype=torch.float32))
        model[2].bias.zero_()
    return model


# P[y0 >= 3.5]: the whole small cluster (0.02) and the part of the band with
# 4 c + 0.3 x0 >= 3.5 for c uniform on [0, 1], which is 1/8 of it.
_P_UNSAFE = 0.02 + 0.01 / 8


def test_two_cluster_network_is_unsafe_with_the_probability_used_below():
    """The reference value of the next test, checked by sampling."""
    x = np.random.default_rng(0).uniform(-1, 1, size=(400_000, 3))
    with torch.no_grad():
        y = _two_cluster_network()(torch.from_numpy(x.astype(np.float32))).numpy()
    unsafe = int((y[:, 0] >= 3.5).sum())
    low, high = binom.ppf([1e-4, 1 - 1e-4], len(x), _P_UNSAFE)
    assert low <= unsafe <= high


@deferred_bug(2, "epsilon_total of a flow result is stated for the network but "
                 "half of it is measured on the flow model, which leaves out "
                 "the small cluster")
def test_flow_result_states_no_epsilon_below_the_true_unsafe_probability():
    """The small cluster is unsafe and holds 2.1% of the outputs. The flow
    model fitted to the outputs gives it almost no mass, the scenario check
    samples the flow model, and the result reads UNSAT with epsilon_total
    0.0084 at a stated confidence of 0.9985: a bound 2.5 times too small.
    One flow training (seed 0): seeds 1 and 2 give the same answer, and each
    training is most of this file's run time."""
    model = _two_cluster_network()
    unsafe = _at_least([1.0, 0.0], 3.5)
    reach_set = flow_reach(model, Box(-np.ones(3), np.ones(3)),
                           FlowReachConfig(m=2000, epsilon=0.005,
                                           n_train=1000, flow_epochs=50,
                                           seed=0))
    result = verify_specification(
        reach_set, unsafe,
        config=ProbVerifyConfig(method="scenario", n_samples=2000))
    if result.verdict == "UNSAT":
        assert result.epsilon_total >= _P_UNSAFE, (
            f"UNSAT with epsilon_total {result.epsilon_total}; the true "
            f"probability is {_P_UNSAFE}")


# --------------------------------------------------------------------------- #
# Deferred bugs 3 and 4: a search that found nothing                            #
# --------------------------------------------------------------------------- #
class _IdentityFlow:
    """A flow whose outputs are its latent samples: y is standard normal."""

    velocity_field = None

    def inverse(self, z, **kwargs):
        return z

    def forward(self, y, **kwargs):
        return y

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


def _standard_normal_set():
    """A flow reach set over one standard normal output."""
    return ProbabilisticSet(SimpleNamespace(flow_model=_IdentityFlow()),
                            threshold=1.0, m=1000, ell=999, epsilon=0.01, dim=1)


def _unsat_without_a_bound(config_for_seed, seeds):
    """The seeds whose answer is UNSAT with no epsilon behind it. The unsafe
    region is y >= 5: not empty, probability 2.9e-7."""
    unsafe = _at_least([1.0], 5.0)
    results = {seed: verify_specification(_standard_normal_set(), unsafe,
                                          config=config_for_seed(seed))
               for seed in seeds}
    return [seed for seed, result in results.items()
            if result.verdict == "UNSAT" and result.epsilon_total is None]


@deferred_bug(3, "amls answers UNSAT when it detected nothing, with no "
                 "epsilon")
def test_amls_never_answers_unsat_without_a_bound():
    """amls is a detector. Its own result type says so: finding nothing in a
    region that is rare but not empty is UNKNOWN."""
    bare = _unsat_without_a_bound(
        lambda seed: ProbVerifyConfig(method="amls", n_samples=100,
                                      amls_max_levels=2, seed=seed),
        seeds=range(5))
    assert bare == []


@deferred_bug(4, "is_tilted answers UNSAT when no sample was unsafe, with no "
                 "epsilon")
def test_is_tilted_never_answers_unsat_without_a_bound():
    bare = _unsat_without_a_bound(
        lambda seed: ProbVerifyConfig(method="is_tilted", n_samples=200,
                                      seed=seed),
        seeds=range(5))
    assert bare == []


@deferred_bug(4, "is_tilted samples the untilted prior, so the tilt changes "
                 "nothing about what it detects")
def test_is_tilted_detects_a_rare_region_more_often_with_a_larger_tilt():
    """The unsafe region y >= 3 has probability 1.35e-3, so 200 plain samples
    reach it in about one run in four. Sampling tilted toward the region
    reaches it more often. Today both counts are the same number: the same
    samples are drawn for every tilt and only reweighted afterwards."""
    unsafe = _at_least([1.0], 3.0)

    def detections(tilt):
        return sum(is_tilted_estimate_halfspace_mass(
            _IdentityFlow(), unsafe, n_samples=200, lambda_tilt=tilt,
            seed=seed).detected_unsafe for seed in range(40))

    assert detections(10.0) > detections(0.0)


@deferred_bug(4, "the self-normalised is_tilted estimate shrinks the weight "
                 "of the safe samples only, so it over-estimates")
def test_is_tilted_estimate_is_unbiased():
    """P[y >= 1] = 0.1587 for a standard normal y. The mean of 50 estimates
    must be within 4 standard errors of it; today it is 0.73 at the default
    tilt. Replaces ``test_importance_sampling.py::
    test_is_self_normalised_estimates_q_conditional_mass``, which asserted
    that the tilted estimate is the larger one."""
    unsafe = _at_least([1.0], 1.0)
    estimates = np.array([is_tilted_estimate_halfspace_mass(
        _IdentityFlow(), unsafe, n_samples=2000, lambda_tilt=5.0,
        seed=seed).pi_hat for seed in range(50)])
    standard_error = estimates.std(ddof=1) / np.sqrt(len(estimates))
    assert abs(estimates.mean() - norm.sf(1.0)) <= 4 * standard_error


# --------------------------------------------------------------------------- #
# The configuration of the probabilistic lane                                   #
# --------------------------------------------------------------------------- #
_NEEDS_A_TARGET = ("amls_bounded", "amls_bounded_union", "raw_mc_uniform")


def test_prob_verify_config_accepts_every_documented_value():
    """Every method the class documents, with the target where it needs
    one, and the smallest legal sample, step and level counts with the
    'qmc' sampling are accepted and kept. Catches a validation that rejects
    a documented method (such as 'amls_bounded_union', which then cannot be
    selected at all) or a bound check that is off by one on the accepting
    side."""
    for method in ("scenario", "amls", "amls_bounded", "amls_bounded_union",
                   "is_tilted", "raw_mc_uniform"):
        target = 0.05 if method in _NEEDS_A_TARGET else None
        config = ProbVerifyConfig(method=method,
                                  amls_bounded_eps_2_target=target)
        assert (config.method, config.amls_bounded_eps_2_target) == (
            method, target)
    smallest = {"n_samples": 1, "n_ode_steps": 1, "amls_max_levels": 1,
                "sampling_strategy": "qmc"}
    config = ProbVerifyConfig(**smallest)
    assert {name: getattr(config, name) for name in smallest} == smallest


@pytest.mark.parametrize("fields", [
    pytest.param({"method": "monte_carlo"}, id="unknown-method"),
    pytest.param({"n_samples": 0}, id="no-samples"),
    pytest.param({"beta": 0.0}, id="beta-zero"),
    pytest.param({"beta": 1.0}, id="beta-one"),
    pytest.param({"n_ode_steps": 0}, id="no-ode-steps"),
    pytest.param({"sampling_strategy": "sobol"}, id="unknown-sampling"),
    pytest.param({"amls_quantile": 0.0}, id="quantile-zero"),
    pytest.param({"amls_quantile": 1.0}, id="quantile-one"),
    pytest.param({"amls_max_levels": 0}, id="no-levels"),
] + [
    pytest.param({"method": method}, id=f"{method}-without-a-target")
    for method in _NEEDS_A_TARGET
])
def test_prob_verify_config_refuses_a_value_outside_its_range(fields):
    """A value outside the documented range is refused with ``ValueError``
    when the configuration is built, not deep inside a certify method (or
    never: a raw Monte Carlo run with no target has no gate to certify
    against). One case per check, at the boundary where there is one.
    Catches a check that is dropped, inverted or off by one (a sample budget
    of 0 accepted), and a method dropped from the list of those that need
    ``amls_bounded_eps_2_target``."""
    with pytest.raises(ValueError):
        ProbVerifyConfig(**fields)


@known_bug(13, "ProbVerifyConfig accepts an amls_bounded_eps_2_target outside "
               "(0, 1): with 1.5 a raw Monte Carlo run answers UNSAT with "
               "epsilon_total 1.495, with -0.1 it states epsilon_total -0.089",
           raises=DID_NOT_RAISE)
def test_prob_verify_config_refuses_a_target_that_is_not_a_probability():
    """``amls_bounded_eps_2_target`` is a miscoverage of the verification
    layer, a probability, like ``FlowReachConfig.epsilon`` and ``beta``, which
    are checked to lie in (0, 1). A target of 1.5 lets every check that
    detects nothing answer UNSAT with an epsilon_total above 1, a statement
    with no content under a verdict that reads as safe; a negative target
    gives a negative epsilon_total. Both are refused when the configuration
    is built."""
    for target in (1.5, -0.1):
        with pytest.raises(ValueError):
            ProbVerifyConfig(method="raw_mc_uniform",
                             amls_bounded_eps_2_target=target)


# --------------------------------------------------------------------------- #
# The probabilistic lane of verify_specification                                #
# --------------------------------------------------------------------------- #
_SET_EPSILON, _RADIUS = 0.01, 2.0


def _disc_set(affine_transform=None):
    """A flow set over two outputs whose flow is the identity: the set is the
    disc of radius 2 in the frame of the flow, the samples of every certify
    method are the latent samples themselves, and the set states coverage
    0.99 (m = 1000, ell = 999)."""
    return ProbabilisticSet(SimpleNamespace(flow_model=_IdentityFlow()),
                            threshold=_RADIUS, m=1000, ell=999,
                            epsilon=_SET_EPSILON, dim=2,
                            affine_transform=affine_transform)


def _far():
    """y0 >= 5: no point of the disc, and no sample of a chain that stays in
    or near it."""
    return _at_least([1.0, 0.0], 5.0)


def _inside():
    """y0 >= 0: half of the disc; the first sample of any method meets it."""
    return _at_least([1.0, 0.0], 0.0)


def _joint(epsilon_2):
    """1 - (1 - epsilon_1)(1 - epsilon_2) in 50-digit arithmetic, epsilon_1
    the set's own (the exact value of its float)."""
    with mpmath.workdps(50):
        return 1 - (1 - mpmath.mpf(_SET_EPSILON)) * (1 - mpmath.mpf(epsilon_2))


def _assert_documented_epsilon_total(result, epsilon_2):
    """The stated epsilon_total is the documented 1 - (1 - epsilon_1)(1 -
    epsilon_2) for the method's documented epsilon_2, up to the rounding of
    its float evaluation (a few ulps). A unit check of the documented value;
    the callers add the one-sided soundness check against the smallest
    miscoverage the method's test supports, with no slack."""
    assert result.epsilon_total is not None, "the answer states no epsilon"
    documented = float(_joint(epsilon_2))
    assert abs(result.epsilon_total - documented) <= 4 * math.ulp(1.0), (
        f"epsilon_total {result.epsilon_total}, documented {documented}")


def _scenario_bound(beta, n):
    """The smallest miscoverage a scenario test of n samples with no sample
    in the region supports at confidence 1 - beta: 1 - beta^(1/n) (exact,
    50 digits); the documented log(1/beta)/n lies above it."""
    with mpmath.workdps(50):
        return 1 - mpmath.mpf(beta) ** (1 / mpmath.mpf(n))


def _amls_upper(beta, quantile=0.1, levels=5, n=200):
    """The bounded AMLS upper bound when no chain reaches the region (the
    region lies beyond the ball the chains are kept in): every level keeps
    the fraction ``quantile``, so pi_hat = quantile^levels, and the
    asymptotic (1 - beta) bound is pi_hat exp(z_{1-beta} sqrt(levels (1 -
    quantile) / (quantile n))) (Cerou and Guyader). From scipy and math."""
    return min(1.0, quantile ** levels * math.exp(
        norm.ppf(1 - beta) * math.sqrt(levels * (1 - quantile)
                                       / (quantile * n))))


def _assert_joint_confidence(result, pset, beta):
    """delta_total is the set's confidence (Beta-binomial, from scipy) times
    1 - beta."""
    assert abs(Fraction(result.delta_total)
               - _joint_confidence(pset, beta)) <= Fraction(1, 10**12)


def test_scenario_unsat_states_the_union_bound_over_every_test_it_ran():
    """Two groups (ANDed): y0 >= 1 holds part of the disc, so its test finds
    a sample there; the OR of y0 >= 5 and y1 >= 5 misses it, so both its
    tests find none and the answer is UNSAT. The stated joint miscoverage
    rests on the three tests that ran: epsilon_total is at least 1 - (1 -
    0.01)(1 - 3 (1 - beta^(1/N))) (exact) and is the documented 1 - (1 -
    0.01)(1 - 3 ln(1/beta)/N); delta_total is the set's confidence times
    1 - beta; n_samples_used is N; q is the radius; the AMLS fields are
    unset. beta = 1e-4 is not the default. Catches a beta handed on as
    1 - beta, an epsilon_total that is not the product rule (the larger of
    the two epsilons), the epsilon of the certifying group stated alone, a
    joint confidence that is not the product, and a set without a whitening
    transform read as if it had one."""
    beta, n = 1e-4, 500
    pset = _disc_set()
    spec = [{"Hg": _at_least([1.0, 0.0], 1.0)},
            {"Hg": [_far(), _at_least([0.0, 1.0], 5.0)]}]
    result = verify_specification(pset, spec, config=ProbVerifyConfig(
        method="scenario", n_samples=n, beta=beta, seed=0))
    assert result.verdict == "UNSAT"
    assert mpmath.mpf(result.epsilon_total) >= _joint(
        3 * _scenario_bound(beta, n))
    _assert_documented_epsilon_total(result, 3 * math.log(1 / beta) / n)
    _assert_joint_confidence(result, pset, beta)
    assert result.n_samples_used == n
    assert result.q == _RADIUS
    assert (result.amls_levels_used, result.amls_bounded_eps_2_upper,
            result.amls_bounded_detected_unsafe) == (None, None, None)


def test_omitted_config_runs_the_documented_default_check():
    """``verify_specification(flow_set, spec)`` with no config runs the
    default ``ProbVerifyConfig()``: the scenario check with 10000 samples at
    beta 0.001, so a region the disc misses is UNSAT with the documented
    epsilon_total and n_samples_used 10000. Catches a default that is not
    the documented one or that is not handed on."""
    pset = _disc_set()
    result = verify_specification(pset, _far())
    assert result.verdict == "UNSAT"
    assert mpmath.mpf(result.epsilon_total) >= _joint(
        _scenario_bound(0.001, 10_000))
    _assert_documented_epsilon_total(result, math.log(1000) / 10_000)
    _assert_joint_confidence(result, pset, 0.001)
    assert result.n_samples_used == 10_000


# The set below lives in a whitened frame: raw outputs are mean + std * z for
# z in the disc of radius 2, so y0 is in [-1, 7] and y1 in [-2, 0].
_MEAN, _STD = np.array([3.0, -1.0]), np.array([2.0, 0.5])


@pytest.mark.parametrize("spec, verdict", [
    # Raw y1 >= 2.5 misses the set (whitened z1 >= 7); raw y0 >= 4 holds part
    # of it (whitened z0 >= 0.5). Read without the transform both miss the
    # disc (z1 >= 2.5, z0 >= 4): a false UNSAT.
    pytest.param([_at_least([0.0, 1.0], 2.5), _at_least([1.0, 0.0], 4.0)],
                 "UNKNOWN", id="second-member-holds-part-of-the-set"),
    # Raw y0 <= -1.5 misses the set (whitened z0 <= -2.25); read without the
    # transform it cuts the disc (z0 <= -1.5).
    pytest.param(_at_least([-1.0, 0.0], 1.5), "UNSAT",
                 id="region-misses-the-set"),
])
def test_a_raw_spec_is_whitened_member_by_member(spec, verdict):
    """The set's score works in whitened coordinates and the spec is written
    in raw ones, so every member of every group is moved into the whitened
    frame before the check. Catches a transform that is skipped (the first
    case answers a false UNSAT, the second a needless UNKNOWN) or applied
    with the first member's rows for every member (a false UNSAT)."""
    pset = _disc_set(AffineTransform(mean=_MEAN, std=_STD))
    result = verify_specification(pset, spec, config=ProbVerifyConfig(
        method="scenario", n_samples=500, seed=0))
    assert result.verdict == verdict
    if verdict == "UNSAT":
        _assert_documented_epsilon_total(result, math.log(1000) / 500)


@pytest.mark.parametrize("target, region, verdict", [
    pytest.param(0.02, _far, "UNKNOWN", id="bound-above-the-target"),
    pytest.param(0.03, _far, "UNSAT", id="bound-below-the-target"),
    pytest.param(0.03, lambda: _at_least([1.0, 0.0], 1.5), "UNKNOWN",
                 id="region-near-the-edge-of-the-disc"),
])
def test_raw_monte_carlo_certifies_only_at_the_stated_confidence(
        target, region, verdict):
    """Raw Monte Carlo with N = 500 samples of the disc and beta = 1e-6: with
    no sample in the region, its one-sided Clopper-Pearson upper bound at
    confidence 1 - beta is 1 - beta^(1/N) = 0.02725, so a target of 0.02 is
    not met (UNKNOWN) and 0.03 is (UNSAT, stating epsilon_total 1 - (1 -
    0.01)(1 - 0.03) and delta_total = the set's confidence times 1 - beta).
    y0 >= 1.5 holds 3.3 % of the disc, between radius 1.5 and 2, so the
    answer is UNKNOWN (no sample there has probability 4.4e-8). Catches beta
    not handed on (at the default 0.001 the bound is 0.0137 and 0.02 is met:
    an UNSAT whose stated confidence the test did not reach), a sampling
    ball of the wrong radius, and the target or the sample count not handed
    on."""
    beta, n = 1e-6, 500
    pset = _disc_set()
    result = verify_specification(pset, region(), config=ProbVerifyConfig(
        method="raw_mc_uniform", n_samples=n, beta=beta, seed=0,
        amls_bounded_eps_2_target=target))
    assert result.verdict == verdict
    if verdict == "UNSAT":
        assert mpmath.mpf(result.epsilon_total) >= _joint(
            _scenario_bound(beta, n))
        _assert_documented_epsilon_total(result, target)
        _assert_joint_confidence(result, pset, beta)
        assert result.n_samples_used == n


@pytest.mark.parametrize("method", ["amls_bounded", "amls_bounded_union"])
@pytest.mark.parametrize("region, verdict", [
    pytest.param(_far, "UNSAT", id="region-beyond-the-disc"),
    pytest.param(_inside, "UNKNOWN", id="region-holds-half-the-disc"),
])
def test_bounded_amls_states_its_target_and_its_upper_bound(
        method, region, verdict):
    """Bounded AMLS (200 samples per level, 5 levels, quantile 0.2, beta =
    1e-6, target 0.05; the quantile is not the default 0.1, so a quantile
    that is not handed on changes the bound). Beyond the disc no chain
    reaches the region: the run uses all 5 levels, detects nothing and its
    upper bound is the closed form of ``_amls_upper`` (1.44e-3), so the
    answer is UNSAT with
    amls_bounded_eps_2_upper at least that bound and at most the target,
    epsilon_total the documented 1 - (1 - 0.01)(1 - 0.05), delta_total the
    set's confidence times 1 - beta and amls_levels_used 5. A region that
    holds half the disc is detected at once: UNKNOWN, detection flag set.
    Catches the upper bound read from a field the spec-level result does not
    have (always None), 'amls_bounded_union' refused by the configuration,
    beta or the quantile not handed on (the bound is then that of another
    run: at the default beta 0.001 it is 8.5e-4, at the default quantile 0.1
    it is 9.5e-5), and an epsilon_total that is not the product rule."""
    beta, target, quantile = 1e-6, 0.05, 0.2
    pset = _disc_set()
    result = verify_specification(pset, region(), config=ProbVerifyConfig(
        method=method, n_samples=200, amls_max_levels=5, beta=beta, seed=0,
        amls_quantile=quantile, amls_bounded_eps_2_target=target))
    assert result.verdict == verdict
    upper = result.amls_bounded_eps_2_upper
    assert upper is not None and 0.0 <= upper <= 1.0
    if verdict == "UNSAT":
        bound = _amls_upper(beta, quantile=quantile)
        assert bound * (1 - 1e-12) <= upper <= target, (
            f"stated upper bound {upper}, closed form {bound}")
        assert mpmath.mpf(result.epsilon_total) >= _joint(bound)
        _assert_documented_epsilon_total(result, target)
        _assert_joint_confidence(result, pset, beta)
        assert result.amls_levels_used == 5
        assert result.amls_bounded_detected_unsafe is False
    else:
        assert result.amls_bounded_detected_unsafe is True


@pytest.mark.parametrize("method", [
    pytest.param("amls_bounded", marks=known_bug(
        13, "with method 'amls_bounded' each member of an OR group is bounded "
            "on its own and the group is certified when each bound is at "
            "most the target, but epsilon_total is built from the target "
            "alone, below the sum of the members' bounds the union needs")),
    pytest.param("amls_bounded_union"),
])
def test_bounded_amls_states_the_bound_of_the_whole_certifying_group(method):
    """The unsafe region is the OR of y0 >= 5 and y1 >= 5, both beyond the
    disc, so every chain gives the closed-form bound u of ``_amls_upper``
    and the target is 1.5 u. 'amls_bounded' bounds each member on its own,
    so the union of the two is bounded only by 2 u (union bound); stating
    epsilon_2 = 1.5 u claims less than the run supports (one-sided, exact).
    A correct method states at least 2 u or answers UNKNOWN.
    'amls_bounded_union' runs one chain on the union, whose bound u is at
    most the target: its UNSAT stating 1.5 u is supported (the passing
    neighbour, same reference)."""
    beta = 1e-6
    u = _amls_upper(beta)
    result = verify_specification(
        _disc_set(), [_far(), _at_least([0.0, 1.0], 5.0)],
        config=ProbVerifyConfig(method=method, n_samples=200,
                                amls_max_levels=5, beta=beta, seed=0,
                                amls_bounded_eps_2_target=1.5 * u))
    if result.verdict == "UNKNOWN":
        return
    group_bound = 2 * u if method == "amls_bounded" else u
    assert mpmath.mpf(result.epsilon_total) >= _joint(group_bound), (
        f"epsilon_total {result.epsilon_total}; the group's members are "
        f"bounded by {group_bound} together")


@pytest.mark.parametrize("method", ["amls", "is_tilted"])
@pytest.mark.parametrize("region", [
    pytest.param(_far, id="region-beyond-the-disc"),
    pytest.param(_inside, id="region-holds-half-the-disc"),
])
def test_a_detector_states_no_epsilon_and_is_unknown_when_it_detects(
        method, region):
    """'amls' and 'is_tilted' are detectors: they produce no bound on the
    unsafe mass, so their answer states no epsilon_total (DESIGN section 14:
    estimate only), and a region they detect (half the disc) is UNKNOWN.
    'is_tilted' draws n_samples for a single half-space and says so. The
    answer UNSAT when nothing is detected is deferred bug 3 and 4 (the
    tests above) and is not asserted either way here. Catches an epsilon
    stated with no test behind it, a detection answered UNSAT, and the
    sample count of AMLS (per level times levels) reported for 'is_tilted'."""
    n = 200
    result = verify_specification(_disc_set(), region(), config=ProbVerifyConfig(
        method=method, n_samples=n, amls_max_levels=5, seed=0))
    assert result.epsilon_total is None
    if region is _inside:
        assert result.verdict == "UNKNOWN"
    if method == "is_tilted":
        assert result.n_samples_used == n


_LEVELS_ARE_THE_MAXIMUM = known_bug(
    13, "amls_levels_used reports amls_max_levels, the configured maximum, "
        "not the levels the run used: a region the first sample meets uses "
        "one level and is reported as 5")


@pytest.mark.parametrize("method, pset, region, levels, used", [
    pytest.param("amls", _standard_normal_set, lambda: _at_least([1.0], 5.0),
                 2, (2,), id="amls-nothing-detected"),
    pytest.param("amls", _disc_set, _inside, 5, (0, 1),
                 id="amls-detected-by-the-first-sample",
                 marks=_LEVELS_ARE_THE_MAXIMUM),
    pytest.param("amls_bounded", _disc_set, _inside, 5, (0, 1),
                 id="amls_bounded-detected-by-the-first-sample",
                 marks=_LEVELS_ARE_THE_MAXIMUM),
])
def test_amls_reports_the_levels_its_run_used(method, pset, region, levels,
                                              used):
    """``VerificationResult.amls_levels_used`` is the number of adaptive
    levels the run used. A run that detects nothing uses every level (2 of
    2 here: y >= 5 for a standard normal y, 100 samples per level, the case
    deferred bug 3 is about); a run whose first sample meets the region
    stops before its first level (one level, counted as 1 by the AMLS
    results themselves). Catches a count that is the configured maximum."""
    result = verify_specification(pset(), region(), config=ProbVerifyConfig(
        method=method, n_samples=100, amls_max_levels=levels, seed=0,
        amls_bounded_eps_2_target=0.05))
    assert result.amls_levels_used in used, (
        f"amls_levels_used {result.amls_levels_used}, the run used {used}")


_ESCALATED_SAMPLES = known_bug(
    13, "after the scenario check escalates to adaptive_n_samples, "
        "n_samples_used still reports the first-stage n_samples (200 of the "
        "1000 the certifying test drew)")


@pytest.mark.parametrize("field", [
    pytest.param("epsilon_total", id="epsilon-of-the-escalated-test"),
    pytest.param("n_samples_used", id="samples-of-the-escalated-test",
                 marks=_ESCALATED_SAMPLES),
])
def test_scenario_reports_the_escalated_test(field):
    """y0 >= 2.5 lies beyond the disc, but within 1 of its edge: the first
    stage (200 samples) certifies with a worst margin in (0, 1], so the
    check escalates to 1000 samples, and that second test is the one the
    answer rests on (``certify_halfspace_disjoint``). epsilon_total is
    the documented one for 1000 samples (at least the exact scenario bound
    for 1000, one-sided); n_samples_used is "the sample count actually drawn
    (after adaptive escalation)" (``VerificationResult``): 1000, or the 1200
    drawn in all, not 200. Catches the adaptive pair not handed on or
    swapped, and a count that ignores the escalation."""
    beta = 0.001
    result = verify_specification(
        _disc_set(), _at_least([1.0, 0.0], 2.5), config=ProbVerifyConfig(
            method="scenario", n_samples=200, beta=beta, seed=0,
            adaptive_threshold=1.0, adaptive_n_samples=1000))
    if result.verdict != "UNSAT":
        pytest.fail(f"the case needs an UNSAT answer, got {result.verdict}")
    if field == "epsilon_total":
        assert mpmath.mpf(result.epsilon_total) >= _joint(
            _scenario_bound(beta, 1000))
        _assert_documented_epsilon_total(result, math.log(1 / beta) / 1000)
    else:
        assert result.n_samples_used in (1000, 1200), (
            f"n_samples_used {result.n_samples_used}")


def test_a_set_whose_score_has_no_flow_model_is_refused():
    """The lane samples the flow behind the set's score; a set with a
    hyper-rectangle score has none, and the call is refused with the
    library's deliberate ``TypeError`` naming ``flow_model`` before any
    sampling, rather than answering or failing on a missing attribute."""
    pset = ProbabilisticSet(HyperrectScore(torch.zeros(2), torch.ones(2)),
                            threshold=1.0, m=1000, ell=999, epsilon=0.01,
                            dim=2)
    with pytest.raises(TypeError, match="flow_model"):
        verify_specification(pset, _far(), config=ProbVerifyConfig())


class _RecordingFlow(_IdentityFlow):
    """The identity flow, keeping the settings and the latent batch of every
    call of ``inverse``."""

    def __init__(self):
        self.calls = []

    def inverse(self, z, **kwargs):
        self.calls.append((kwargs, torch.as_tensor(z).detach().clone()))
        return z


# Settings that differ from every default of ProbVerifyConfig and of the
# certify functions, and the names the flow's ``inverse`` receives them under.
_ODE_SETTINGS = {"t": 0.75, "n_ode_steps": 7, "ode_method": "euler",
                 "ode_atol": 1e-3, "ode_rtol": 1e-4}
_FLOW_KWARGS = {"t": 0.75, "n_steps": 7, "method": "euler", "atol": 1e-3,
                "rtol": 1e-4}
# Methods whose first draw is the truncated Gaussian on the latent ball of
# radius threshold, the conformal level set (``sample_truncated_gaussian_ball``);
# 'qmc' draws the untruncated N(0, I) (``certify_halfspace_disjoint``).
_BALL_SAMPLERS = ("scenario", "amls_bounded", "amls_bounded_union",
                  "raw_mc_uniform")


@pytest.mark.parametrize("method, sampling", [
    pytest.param("scenario", "uniform", id="scenario"),
    pytest.param("scenario", "qmc", id="scenario-qmc"),
    pytest.param("amls", "uniform", id="amls"),
    pytest.param("amls_bounded", "uniform", id="amls_bounded"),
    pytest.param("amls_bounded_union", "uniform", id="amls_bounded_union"),
    pytest.param("is_tilted", "uniform", id="is_tilted"),
    pytest.param("raw_mc_uniform", "uniform", id="raw_mc_uniform"),
])
def test_every_setting_of_the_config_reaches_the_flow(method, sampling):
    """Each branch of the lane hands the configuration on to its certify
    method by its own call, so each is checked: a stand-in flow (the
    identity) records every call. Every call carries the configured ODE end
    time, step count, solver and tolerances (none is a default); the first
    draw has n_samples latent points; two seeds give two different draws;
    a method that samples the conformal set draws from the ball of the
    set's radius 2 (every point within it, some beyond radius 1.5: with 128
    points of the truncated Gaussian none beyond 1.5 has probability 2e-14),
    and 'qmc' draws the untruncated Gaussian (some point beyond radius 2).
    Catches a setting, the sample budget, the seed, the sampling strategy or
    the radius that one branch does not hand on (the certify method then
    runs at its own default)."""
    n = 128
    first = {}
    for seed in (1, 2):
        flow = _RecordingFlow()
        pset = ProbabilisticSet(SimpleNamespace(flow_model=flow),
                                threshold=_RADIUS, m=1000, ell=999,
                                epsilon=_SET_EPSILON, dim=2)
        verify_specification(pset, _far(), config=ProbVerifyConfig(
            method=method, n_samples=n, amls_max_levels=2, seed=seed,
            sampling_strategy=sampling, amls_bounded_eps_2_target=0.05,
            **_ODE_SETTINGS))
        if not flow.calls:
            pytest.fail("the certify method never called the flow")
        for kwargs, _ in flow.calls:
            assert kwargs == _FLOW_KWARGS, f"the flow was called with {kwargs}"
        first[seed] = flow.calls[0][1]
        assert tuple(first[seed].shape) == (n, 2), (
            f"first draw of shape {tuple(first[seed].shape)}, n_samples {n}")
    assert not torch.equal(first[1], first[2]), "seeds 1 and 2 draw the same"
    radius = first[1].double().norm(dim=1)
    if sampling == "qmc":
        assert radius.max() > _RADIUS, "'qmc' drew only points of the ball"
    elif method in _BALL_SAMPLERS:
        assert 1.5 < radius.max() <= _RADIUS * (1 + 1e-6), (
            f"largest latent radius {float(radius.max())}, set radius "
            f"{_RADIUS}")


_EMPTY_REGION_ANSWERED = known_bug(
    6, "an unsafe region given as an empty list of half-spaces is answered "
       "UNSAT on the probabilistic lane by this method instead of refused",
    raises=DID_NOT_RAISE)


@pytest.mark.parametrize("method", [
    pytest.param("scenario"),
    pytest.param("amls", marks=_EMPTY_REGION_ANSWERED),
    pytest.param("amls_bounded", marks=_EMPTY_REGION_ANSWERED),
    pytest.param("is_tilted", marks=_EMPTY_REGION_ANSWERED),
])
def test_an_empty_unsafe_region_is_refused_on_the_probabilistic_lane(method):
    """No half-space states nothing about what is unsafe, so the call is
    refused with ``ValueError`` (the rule of the sound branch, registry
    SPEC-12). Today the scenario check refuses (its group must not be
    empty), while 'amls', 'amls_bounded' and 'is_tilted' certify every
    member of the empty group and answer UNSAT ('amls_bounded' with an
    epsilon_total of 0.0595)."""
    with pytest.raises(ValueError):
        verify_specification(_disc_set(), [], config=ProbVerifyConfig(
            method=method, n_samples=100, amls_max_levels=3, seed=0,
            amls_bounded_eps_2_target=0.05))


@pytest.mark.parametrize("method", [
    pytest.param("scenario"),
    pytest.param("raw_mc_uniform", marks=known_bug(
        23, "a half-space with no rows (the whole output space as the unsafe "
            "region) crashes the torch margin of the AMLS family "
            "(_phi_halfspace_torch, used by amls, amls_bounded, "
            "amls_bounded_union and raw_mc_uniform) with an IndexError "
            "where a refusal or an UNKNOWN is due",
        raises=IndexError)),
])
def test_an_unsafe_region_with_no_rows_is_refused_or_not_unsat(method):
    """A half-space with no rows is the whole output space: every output is
    unsafe, so UNSAT is wrong, and refusing it with ``ValueError`` is also
    correct (the rule of the sound branch,
    ``test_verify_api.py::test_unsafe_region_with_no_rows_is_never_unsat``).
    Catches a crash of the sampling check on a legal half-space object. The
    crash is in the torch margin that 'amls', 'amls_bounded',
    'amls_bounded_union' and 'raw_mc_uniform' share, so one method marks
    it; 'scenario' refuses today (numpy refuses the empty reduction with
    ``ValueError``)."""
    whole_space = HalfSpace(np.zeros((0, 2)), np.zeros((0, 1)))
    try:
        result = verify_specification(_disc_set(), whole_space,
                                      config=ProbVerifyConfig(
                                          method=method, n_samples=100,
                                          seed=0,
                                          amls_bounded_eps_2_target=0.05))
    except ValueError:
        return
    assert result.verdict != "UNSAT"


@pytest.mark.parametrize("method", ["scenario", "amls"])
def test_verbose_prints_the_verdict_and_the_stated_epsilon(method, capsys):
    """``verbose=True`` prints the verdict, and the stated epsilon_total
    when there is one (the scenario check) and no such line when there is
    none (a detector), without changing the result. Catches the printing of
    an absent epsilon (a crash on the format of None) and a printed verdict
    or epsilon that is not the one returned."""
    result = verify_specification(_disc_set(), _far(), config=ProbVerifyConfig(
        method=method, n_samples=200, amls_max_levels=2, seed=0), verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert f"verdict: {result.verdict}" in lines
    printed = [line for line in lines if line.startswith("eps_total:")]
    if result.epsilon_total is None:
        assert printed == []
    else:
        assert printed == [f"eps_total: {result.epsilon_total:.6e}"]
