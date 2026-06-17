"""Tests for the VNNLIB 2.0 parser (n2v.utils.vnnlib2) and the
format-detecting dispatcher in n2v.utils.load_vnnlib."""

import os
import tempfile

import numpy as np
import pytest

from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import VNNLibParseError, load_vnnlib_v2
from n2v.sets.halfspace import HalfSpace


def _write(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False)
    f.write(text)
    f.close()
    return f.name


ACASXU_LIKE = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1, 1, 1, 5])
    (declare-output Y float32 [1, 5])
)
(assert (<= X[0,0,0,0] 0.68))
(assert (>= X[0,0,0,0] 0.60))
(assert (<= X[0,0,0,1] 0.50))
(assert (>= X[0,0,0,1] -0.50))
(assert (<= X[0,0,0,2] 0.50))
(assert (>= X[0,0,0,2] -0.50))
(assert (<= X[0,0,0,3] 0.50))
(assert (>= X[0,0,0,3] 0.45))
(assert (<= X[0,0,0,4] -0.45))
(assert (>= X[0,0,0,4] -0.50))
(assert (>= Y[0,0] 3.99))
"""


def test_acasxu_like_box_and_single_output():
    p = load_vnnlib(_write(ACASXU_LIKE))
    np.testing.assert_allclose(p["lb"], [0.60, -0.50, -0.50, 0.45, -0.50])
    np.testing.assert_allclose(p["ub"], [0.68, 0.50, 0.50, 0.50, -0.45])
    assert len(p["prop"]) == 1
    hg = p["prop"][0]["Hg"]
    assert isinstance(hg, HalfSpace)
    # (>= Y[0] 3.99) -> -Y[0] <= -3.99
    np.testing.assert_allclose(hg.G, [[-1, 0, 0, 0, 0]])
    np.testing.assert_allclose(hg.g, [[-3.99]])


ROBUSTNESS_OR = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1, 3])
    (declare-output Y float32 [1, 3])
)
(assert (and (<= X[0,0] 1.0) (>= X[0,0] 0.0)))
(assert (and (<= X[0,1] 1.0) (>= X[0,1] 0.0)))
(assert (and (<= X[0,2] 1.0) (>= X[0,2] 0.0)))
(assert (or (<= Y[0,0] Y[0,1]) (<= Y[0,0] Y[0,2])))
"""


def test_robustness_disjunction_and_y_to_y():
    p = load_vnnlib(_write(ROBUSTNESS_OR))
    np.testing.assert_allclose(p["lb"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(p["ub"], [1.0, 1.0, 1.0])
    assert len(p["prop"]) == 1
    hg = p["prop"][0]["Hg"]
    # OR -> list of two single-row halfspaces
    assert isinstance(hg, list) and len(hg) == 2
    # (<= Y[0] Y[1]) -> Y[0] - Y[1] <= 0
    np.testing.assert_allclose(hg[0].G, [[1, -1, 0]])
    np.testing.assert_allclose(hg[0].g, [[0]])
    np.testing.assert_allclose(hg[1].G, [[1, 0, -1]])


def test_disjunctive_input_regions():
    spec = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1, 2])
    (declare-output Y float32 [1, 1])
)
(assert (or (and (<= X[0,0] 1.0) (>= X[0,0] 0.0) (<= X[0,1] 1.0) (>= X[0,1] 0.0))
            (and (<= X[0,0] 3.0) (>= X[0,0] 2.0) (<= X[0,1] 3.0) (>= X[0,1] 2.0))))
(assert (<= Y[0,0] 0.0))
"""
    p = load_vnnlib(_write(spec))
    assert isinstance(p["lb"], list) and len(p["lb"]) == 2
    np.testing.assert_allclose(p["lb"][0], [0.0, 0.0])
    np.testing.assert_allclose(p["ub"][0], [1.0, 1.0])
    np.testing.assert_allclose(p["lb"][1], [2.0, 2.0])
    np.testing.assert_allclose(p["ub"][1], [3.0, 3.0])


def test_nonlinear_spec_loads_as_ast():
    """Specs outside the linear fragment load faithfully as resolved
    ASTs (format='nonlinear') instead of being rejected. Truth values
    hand-computed: violation iff x0 in [20,40], x1 in [-40,0], x1^2 >= y0."""
    spec = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X real [1,2])
    (declare-output Y real [1,1])
)
(assert (and (>= X[0,0] 20.0) (<= X[0,0] 40.0)))
(assert (and (>= X[0,1] -40.0) (<= X[0,1] 0.0)))
(assert (>= (* X[0,1] X[0,1]) Y[0,0]))
"""
    from n2v.utils.vnnlib2 import evaluate_nonlinear
    p = load_vnnlib(_write(spec))
    assert p["format"] == "nonlinear"
    assert len(p["assertions"]) == 3
    # best-effort box from the affine single-var atoms
    np.testing.assert_allclose(p["lb"], [20.0, -40.0])
    np.testing.assert_allclose(p["ub"], [40.0, 0.0])
    # hand-computed truth table for the quadratic constraint
    assert evaluate_nonlinear(p, np.array([30.0, -3.0]), np.array([9.0]))
    assert not evaluate_nonlinear(p, np.array([30.0, -3.0]), np.array([9.1]))
    assert evaluate_nonlinear(p, np.array([30.0, -3.0]), np.array([-50.0]))
    assert not evaluate_nonlinear(p, np.array([50.0, -3.0]), np.array([0.0]))


def test_multi_network_spec_parses_as_relational():
    """Multi-network specs now load as relational structures (see
    test_vnnlib_relational.py for the full fixtures); a spec leaving
    joint dims unconstrained still fails loudly."""
    spec = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [1])
    (declare-output Y_f real [1])
)
(declare-network g
    (equal-to f)
    (declare-input X_g real [1])
    (declare-output Y_g real [1])
)
(assert (and (<= X_f[0] 1.0) (>= X_f[0] 0.0)))
(assert (== X_f[0] X_g[0]))
(assert (< Y_f[0] Y_g[0]))
"""
    p = load_vnnlib(_write(spec))
    assert p["format"] == "relational"
    assert [n["name"] for n in p["networks"]] == ["f", "g"]


def test_dispatcher_routes_legacy_1_0():
    """A legacy 1.0 spec still parses via the dispatcher (declare-const)."""
    spec = """\
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= X_1 1.0))
(assert (>= X_1 0.0))
(assert (<= Y_0 5.0))
"""
    p = load_vnnlib(_write(spec))
    np.testing.assert_allclose(p["lb"], [0.0, 0.0])
    np.testing.assert_allclose(p["ub"], [1.0, 1.0])
    assert len(p["prop"]) >= 1
