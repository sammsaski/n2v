"""Tests for the sound verify path of ``format='nonlinear'`` specs
(:mod:`n2v.utils.verify_nonlinear`).

The reach sets here are built by hand: an identity "network" maps a flat
input ``Star`` to an output star with the *same* predicates, so the joint
affine model ``[x; y]`` is exact (``y == x``) once the caller asserts the
sharing (``input_predicates_shared=True``). That lets each test pin a
hand-computed three-valued outcome without standing up a real model.
"""

import os
import tempfile

import numpy as np
import pytest

from n2v.sets import Star
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.verify_nonlinear import verify_nonlinear_reach, falsify_nonlinear


def _spec(body: str) -> dict:
    text = (
        "(vnnlib-version <2.0>)\n"
        "(declare-network f\n"
        "    (declare-input X real [1,1])\n"
        "    (declare-output Y real [1,1])\n"
        ")\n"
    ) + body
    fd, path = tempfile.mkstemp(suffix=".vnnlib")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        return load_vnnlib(path)
    finally:
        os.remove(path)


def _identity_reach(p):
    """Input star over [lb, ub] plus an identity output star (y == x)."""
    lb = np.asarray(p["lb"], dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(p["ub"], dtype=np.float64).reshape(-1, 1)
    s = Star.from_bounds(lb, ub)
    return s, [s]


# A trivially-true product term forces the parser onto the nonlinear path
# without otherwise constraining the region.
_DUMMY_NL = "(assert (>= (* X[0,0] X[0,0]) 0.0))\n"


def test_unsat_output_constraint_provably_unreachable():
    """y == x in [1, 2]; the unsafe region needs y >= 10 -> provably empty."""
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= Y[0,0] 10.0))\n" + _DUMMY_NL
    )
    assert p["format"] == "nonlinear"
    iset, reach = _identity_reach(p)
    assert verify_nonlinear_reach(reach, iset, p) == "UNSAT"


def test_unknown_when_output_constraint_straddles():
    """Unsafe region needs y >= 1.5 while y in [1, 2] straddles it -> UNKNOWN."""
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= Y[0,0] 1.5))\n" + _DUMMY_NL
    )
    iset, reach = _identity_reach(p)
    assert verify_nonlinear_reach(reach, iset, p) == "UNKNOWN"


def test_joint_affine_coupling_beats_decoupled_box():
    """`y - x >= 0.5` is unreachable because y == x (so y - x == 0), but the
    decoupled boxes y in [1,2], x in [1,2] give y - x in [-1,1] (MAYBE). The
    shared-predicate affine form proves it FALSE -> UNSAT. Sharing is only
    used when the caller asserts it; the default stays decoupled."""
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= (- Y[0,0] X[0,0]) 0.5))\n" + _DUMMY_NL
    )
    iset, reach = _identity_reach(p)
    assert verify_nonlinear_reach(reach, iset, p,
                                  input_predicates_shared=True) == "UNSAT"
    assert verify_nonlinear_reach(reach, iset, p) == "UNKNOWN"


def test_quadratic_input_term_enclosed_soundly():
    """x in [-2, 2], y == x. Unsafe needs x^2 >= 10; x^2 in [0,4] < 10 so the
    region is provably empty -> UNSAT (interval encloses the product)."""
    p = _spec(
        "(assert (and (>= X[0,0] -2.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= (* X[0,0] X[0,0]) 10.0))\n"
    )
    assert p["format"] == "nonlinear"
    iset, reach = _identity_reach(p)
    assert verify_nonlinear_reach(reach, iset, p) == "UNSAT"


def test_unknown_on_non_star_reach_set_is_sound():
    """A reach set we cannot read predicate-wise must not be claimed safe."""
    from n2v.sets import Box
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= Y[0,0] 10.0))\n" + _DUMMY_NL
    )
    iset, _ = _identity_reach(p)
    box = Box(np.array([[1.0]]), np.array([[2.0]]))
    assert verify_nonlinear_reach([box], iset, p) == "UNKNOWN"


def test_falsify_finds_concrete_violation():
    """y = 2x via a Linear layer; unsafe iff y >= 3.5 on x in [1, 2]
    (reached at x = 2 -> y = 4). The sampler must surface a witness."""
    import torch

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0]]))
        model.bias.copy_(torch.tensor([0.0]))
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= Y[0,0] 3.5))\n" + _DUMMY_NL
    )
    x = falsify_nonlinear(model, p["lb"], p["ub"], p, (1, 1),
                          n_samples=200, seed=0)
    assert x is not None
    assert 1.0 <= float(x[0]) <= 2.0
    assert 2.0 * float(x[0]) >= 3.5


def test_falsify_returns_none_when_safe():
    """No x in [1, 2] yields 2x >= 100, so the sampler returns None."""
    import torch

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0]]))
        model.bias.copy_(torch.tensor([0.0]))
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= Y[0,0] 100.0))\n" + _DUMMY_NL
    )
    assert falsify_nonlinear(model, p["lb"], p["ub"], p, (1, 1),
                             n_samples=200, seed=0) is None


def _write_spec(text: str) -> dict:
    fd, path = tempfile.mkstemp(suffix=".vnnlib")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        return load_vnnlib(path)
    finally:
        os.remove(path)


def test_appended_predicates_padding_and_coupling():
    """Output star carries an extra relaxation predicate beyond the input
    predicate, with a basis that differs from the input's (the real shape of a
    post-ReLU reach set). Proving ``y - x >= 0.5`` UNSAT requires the joint
    model to cancel the SHARED input predicate across the zero-padded input
    basis and the wider output basis -- the decoupled output box (y in [0,2],
    x in [1,2] -> y-x in [-2,1]) cannot."""
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= (- Y[0,0] X[0,0]) 0.5))\n" + _DUMMY_NL
    )
    iset = Star.from_bounds(np.array([[1.0]]), np.array([[2.0]]))  # x = 1.5 + 0.5 a0
    # y = 0.5 + 0.5 a0 + 1.0 a1, a0 in [-1,1], a1 in [0,1]  ->  y - x = -1 + a1
    out = Star(V=np.array([[0.5, 0.5, 1.0]]), C=None, d=None,
               pred_lb=np.array([-1.0, 0.0]), pred_ub=np.array([1.0, 1.0]))
    assert out.nVar == 2 and iset.nVar == 1  # genuinely an appended predicate
    assert verify_nonlinear_reach([out], iset, p,
                                  input_predicates_shared=True) == "UNSAT"


def test_pinned_input_dimension_still_decides():
    """A pinned input dim (lb==ub) drops its generator so n_pred < n_in; the
    verifier must still build the joint model (the pinned dim is constant) and
    decide, not bail to UNKNOWN as it did before the fix."""
    p = _write_spec(
        "(vnnlib-version <2.0>)\n"
        "(declare-network f (declare-input X real [1,2])"
        " (declare-output Y real [1,2]))\n"
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (and (>= X[0,1] 5.0) (<= X[0,1] 5.0)))\n"
        "(assert (>= Y[0,1] 10.0))\n"
        "(assert (>= (* X[0,0] X[0,0]) 0.0))\n"
    )
    iset = Star.from_bounds(np.array([[1.0], [5.0]]), np.array([[2.0], [5.0]]))
    assert iset.nVar == 1  # the pinned dim's generator was dropped
    # identity reach: Y == X, so Y[0,1] == 5 < 10 -> provably safe.
    assert verify_nonlinear_reach([iset], iset, p) == "UNSAT"


def test_multi_star_requires_every_star_safe():
    """UNSAT only when EVERY output star is provably violation-free; one star
    that might violate -> UNKNOWN (the exact/branching reach returns a union)."""
    p = _spec("(assert (>= Y[0,0] 5.0))\n" + _DUMMY_NL)  # unsafe: y >= 5
    iset = Star.from_bounds(np.array([[1.0]]), np.array([[2.0]]))
    safe = Star(V=np.array([[1.5, 0.5]]), C=None, d=None,   # y in [1, 2]
                pred_lb=np.array([-1.0]), pred_ub=np.array([1.0]))
    maybe = Star(V=np.array([[5.0, 1.0]]), C=None, d=None,  # y in [4, 6], straddles 5
                 pred_lb=np.array([-1.0]), pred_ub=np.array([1.0]))
    assert verify_nonlinear_reach([safe, safe], iset, p) == "UNSAT"
    assert verify_nonlinear_reach([safe, maybe], iset, p) == "UNKNOWN"


# --------------------------------------------------------------------------- #
# The falsifier's documented search, and conjuncts the                        #
# interval rules are documented to decide                                       #
# --------------------------------------------------------------------------- #
def _alternating_net(n):
    """y = x0 - x1 + x2 - ... (weights +1 and -1 by turns)."""
    import torch

    model = torch.nn.Linear(n, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[(-1.0) ** i for i in range(n)]]))
        model.bias.zero_()
    return model


def _box_text(n, out_assert):
    box = " ".join(f"(>= X[{i}] 0.0) (<= X[{i}] 1.0)" for i in range(n))
    return _write_spec(
        "(vnnlib-version <2.0>)\n"
        f"(declare-network f (declare-input X real [{n}]) (declare-output Y real [1]))\n"
        f"(assert (and {box}))\n"
        f"(assert {out_assert})\n")



@pytest.mark.parametrize("n", [1, 12])
def test_falsify_tries_every_corner_of_a_box_of_up_to_twelve_entries(n):
    """y = x0 - x1 + x2 - ... on [0, 1]^n with ``y * y >= m^2``, m = ceil(n/2):
    only the corners with x_i = 1 on the even entries and 0 on the odd ones
    (and, for even n, the mirror corner) reach it, a set of measure zero that
    interior draws never hit, and neither is the all-upper or all-lower
    corner. The docstring promises the corners first for a box of at most 12
    entries, so the search must return one of them. Catches the corner
    enumeration cut off below 12 entries and corners built from one bound
    only."""
    import numpy as np

    m = (n + 1) // 2
    p = _box_text(n, f"(>= (* Y[0] Y[0]) {float(m * m)})")
    x = falsify_nonlinear(_alternating_net(n), p["lb"], p["ub"], p, (n,),
                          n_samples=50, seed=0)
    even = np.array([1.0 - i % 2 for i in range(n)])
    witnesses = [even] + ([1.0 - even] if n % 2 == 0 else [])
    assert x is not None and any(np.array_equal(np.asarray(x), w) for w in witnesses)


@pytest.mark.parametrize("n, n_samples, corners", [(2, 37, 4), (2, 0, 4), (13, 5, 0)])
def test_falsify_draws_n_samples_points_of_the_box_from_its_seed(n, n_samples, corners):
    """The candidates handed to the forward are the 2^n corners (none above
    12 entries) and then ``n_samples`` draws inside [lb, ub] from ``seed``:
    the same seed gives the same draws, another seed other draws, and
    ``n_samples=0`` leaves the corners alone. The box [-3, -1] x [2, 7] (and
    its 13-entry extension) does not start at 0, so draws scaled by ub
    instead of ub - lb leave it. Catches a hard-wired draw count, a seed that
    is not used, and draws outside the box."""
    import numpy as np

    lb = np.array([-3.0, 2.0] + [0.5] * (n - 2))
    ub = np.array([-1.0, 7.0] + [1.5] * (n - 2))
    p = _box_text(1, "(>= (* Y[0] Y[0]) 1e9)")       # never satisfied
    seen = []

    def record(X):
        seen.append(np.array(X, dtype=np.float64))
        return np.zeros((len(X), 1))

    for seed in (1, 1, 2):
        assert falsify_nonlinear(None, lb, ub, p, (n,), n_samples=n_samples,
                                 seed=seed, forward=record) is None
    first, again, other = seen
    assert first.shape == (corners + n_samples, n)
    assert np.all(first >= lb) and np.all(first <= ub)
    interior = first[corners:]
    assert np.array_equal(interior, again[corners:])
    if n_samples:
        assert not np.array_equal(interior, other[corners:])


def _two_output_star():
    """Y0 = 0.5 + 0.5 a0, Y1 = 0.5 - 0.5 a0 (so Y0 + Y1 = 1), Y2 = a1, with
    a0, a1 in [-1, 1]."""
    return Star(V=np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 1.0]]),
                C=None, d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))


@pytest.mark.parametrize("assertions", [
    # affine sums keep the correlation through the star's own predicates
    ["(>= (+ Y[0] Y[1]) 1.5)"],
    # a constant factor keeps the affine form
    ["(>= (+ (* 2.0 Y[0]) (* Y[1] 2.0)) 2.5)"],
    # a constant 0 times an unbounded quotient is exactly 0
    ["(>= (* 0.0 (/ Y[0] Y[2])) 1.0)"],
    # a division by the constant 0 leaves the other conjunct to decide (inside
    # an 'or': as a top-level conjunct the parser's box pass divides by it)
    ["(or (and (>= (/ Y[0] 0.0) 1.0) (>= Y[0] 5.0)) (>= Y[0] 5.0))"],
    # each comparison, clear of its threshold
    ["(<= Y[0] -0.5)"], ["(< Y[0] -0.5)"], ["(>= Y[0] 1.5)"], ["(> Y[0] 1.5)"],
    ["(== Y[0] 2.0)"],
    # 'and' is false once a conjunct is, 'or' once every disjunct is
    ["(and (>= (* Y[2] Y[2]) 0.5) (>= Y[0] 1.5))"],
    ["(or (>= Y[0] 1.5) (<= (* Y[0] Y[1]) -0.5))"],
    # a provably false assertion decides although a later one is undecided
    ["(>= (* Y[0] Y[0]) 5.0)", "(>= (* Y[0] Y[1]) 0.1)"],
], ids=["affine-correlation", "constant-factor-correlation", "zero-times-unbounded",
        "division-by-constant-zero", "le", "lt", "ge", "gt", "eq", "and-false",
        "or-all-false", "false-then-undecided-assertion"])
def test_a_conjunct_the_interval_rules_decide_is_proved_false(assertions):
    """Y0 + Y1 = 1 with Y0, Y1 in [0, 1] and Y2 in [-1, 1]. Each spec is
    false at every point of the star, by a margin, and the rules the module
    documents decide it: the sum of affine forms is recomputed from the
    combined coefficients, a constant keeps the affine form, 0 times anything
    is the exact 0, a division by the constant 0 degrades only its own
    conjunct, a comparison clear of its threshold is surely false, ``and`` is
    false once one conjunct is, ``or`` once all are, and a false assertion
    ends the star although a later one is undecided. So the verdict must be
    UNSAT. Catches each of these rules lost (a wrong True, an interval sum
    for affine forms, NaN from 0 times inf, a raise on a zero divisor, an
    assertion loop that lets a later undecided assertion overwrite a
    false one)."""
    body = "".join(f"(assert {a})\n" for a in assertions)
    p = _write_spec(
        "(vnnlib-version <2.0>)\n"
        "(declare-network f (declare-input X real [1]) (declare-output Y real [3]))\n"
        "(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))\n"
        "(assert (>= (* X[0] X[0]) 0.0))\n" + body)
    assert p["format"] == "nonlinear"
    iset = Star.from_bounds(np.array([[0.0]]), np.array([[1.0]]))
    assert verify_nonlinear_reach([_two_output_star()], iset, p) == "UNSAT"
