"""Tests for the sound verify path of ``format='nonlinear'`` specs
(:mod:`n2v.utils.verify_nonlinear`).

The reach sets here are built by hand: an identity "network" maps a flat
input ``Star`` to an output star with the *same* predicates, so the joint
affine model ``[x; y]`` is exact (``y == x``). That lets each test pin a
hand-computed three-valued outcome without standing up a real model.
"""

import os
import tempfile

import numpy as np

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
    shared-predicate affine form proves it FALSE -> UNSAT."""
    p = _spec(
        "(assert (and (>= X[0,0] 1.0) (<= X[0,0] 2.0)))\n"
        "(assert (>= (- Y[0,0] X[0,0]) 0.5))\n" + _DUMMY_NL
    )
    iset, reach = _identity_reach(p)
    assert verify_nonlinear_reach(reach, iset, p) == "UNSAT"


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
