"""Multi-network (relational) VNNLIB 2.0 parsing — monotonic/isomorphic.

Fixtures are the REAL deployed spec shapes with hand-computed expected
representations (the spec-line -> representation tables reviewed for this
task). Joint spaces concatenate per kind in network declaration order:
monotonic/isomorphic: f's inputs at joint dims 0-4, g's at 5-9; outputs
likewise.
"""

import tempfile

import numpy as np
import pytest

from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import VNNLibParseError


def _write(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False)
    f.write(text)
    f.close()
    return f.name


MONOTONIC = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [5])
    (declare-output Y_f real [5])
)
(declare-network g
    (equal-to f)
    (declare-input X_g real [5])
    (declare-output Y_g real [5])
)
(assert (and (<= X_f[0] 0.667245963) (>= X_f[0] -0.16247807)))
(assert (and (<= X_f[1] 0.0) (>= X_f[1] -0.25)))
(assert (and (<= X_f[2] 0.5) (>= X_f[2] 0.25)))
(assert (== X_f[3] 0.227272727))
(assert (== X_f[4] 0.25 ))
(assert (and (>= X_f[0] X_g[0]) (>= X_g[0] -0.16247807)))
(assert (== X_f[1] X_g[1]))
(assert (== X_f[2] X_g[2]))
(assert (== X_f[3] X_g[3]))
(assert (== X_f[4] X_g[4]))
(assert (Y_f[3] < Y_g[3]))
"""


class TestMonotonic:
    def test_networks_and_relation(self):
        p = load_vnnlib(_write(MONOTONIC))
        assert p["format"] == "relational"
        nets = p["networks"]
        assert [n["name"] for n in nets] == ["f", "g"]
        assert nets[0]["relation"] is None
        assert nets[1]["relation"] == ("equal-to", "f")
        assert nets[0]["input_offset"] == 0 and nets[1]["input_offset"] == 5
        assert nets[0]["output_offset"] == 0 and nets[1]["output_offset"] == 5

    def test_joint_box(self):
        p = load_vnnlib(_write(MONOTONIC))
        lb = np.asarray(p["lb"]).flatten()
        ub = np.asarray(p["ub"]).flatten()
        # f's box, per the spec lines
        np.testing.assert_allclose(lb[:5], [-0.16247807, -0.25, 0.25,
                                            0.227272727, 0.25])
        np.testing.assert_allclose(ub[:5], [0.667245963, 0.0, 0.5,
                                            0.227272727, 0.25])
        # g's dims: only X_g[0] has a direct bound; 6-9 bounded via coupling
        assert lb[5] == pytest.approx(-0.16247807)
        assert np.isinf(lb[6:]).all() and np.isinf(ub[5:]).all()

    def test_coupling_rows(self):
        """(>= X_f[0] X_g[0]) -> x5 - x0 <= 0; each == X_f[i]==X_g[i]
        -> two rows (xi - x(5+i) <= 0 and x(5+i) - xi <= 0)."""
        p = load_vnnlib(_write(MONOTONIC))
        C = p["input_coupling"]
        assert C is not None
        rows = {tuple(np.round(r, 9)) for r in
                np.hstack([C.G, C.g]).tolist()}
        expect = set()
        r = np.zeros(11)
        r[5], r[0] = 1.0, -1.0  # x_g0 - x_f0 <= 0
        expect.add(tuple(r))
        for i in range(1, 5):
            r1 = np.zeros(11)
            r1[i], r1[5 + i] = 1.0, -1.0
            r2 = np.zeros(11)
            r2[5 + i], r2[i] = 1.0, -1.0
            expect.add(tuple(r1))
            expect.add(tuple(r2))
        assert rows == expect

    def test_output_property(self):
        """(Y_f[3] < Y_g[3]) -> violation row y3 - y8 <= 0 (joint outputs)."""
        p = load_vnnlib(_write(MONOTONIC))
        assert len(p["prop"]) == 1
        hs = p["prop"][0]["Hg"]
        expect = np.zeros(10)
        expect[3], expect[8] = 1.0, -1.0
        np.testing.assert_allclose(hs.G, [expect])
        np.testing.assert_allclose(hs.g, [[0.0]])


ISOMORPHIC = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [2])
    (declare-output Y_f real [2])
)
(declare-network g
    (isomorphic-to f)
    (declare-input X_g real [2])
    (declare-output Y_g real [2])
)
(assert (and (<= X_f[0] 0.6) (>= X_f[0] 0.5)))
(assert (and (<= X_f[1] 0.2) (>= X_f[1] 0.1)))
(assert (== X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (and (> Y_g[0] (+ Y_f[0] 0.05)) (< Y_g[0] (- Y_f[0] 0.05))))
(assert (and (> Y_g[1] (+ Y_f[1] 0.05)) (< Y_g[1] (- Y_f[1] 0.05))))
"""


class TestIsomorphic:
    def test_relation_and_output_rows(self):
        """Each assert is an AND of two rows over joint outputs (parsed
        as written, vacuous or not — spec semantics are upstream's call):
        (> Y_g[i] (+ Y_f[i] 0.05)) -> y_fi - y_gi <= -0.05
        (< Y_g[i] (- Y_f[i] 0.05)) -> y_gi - y_fi <= -0.05
        Joint outputs: f at 0-1, g at 2-3."""
        p = load_vnnlib(_write(ISOMORPHIC))
        assert p["networks"][1]["relation"] == ("isomorphic-to", "f")
        assert len(p["prop"]) == 2
        for i, group in enumerate(p["prop"]):
            hs = group["Hg"]
            G = np.zeros((2, 4))
            G[0, i], G[0, 2 + i] = 1.0, -1.0   # y_fi - y_gi <= -0.05
            G[1, 2 + i], G[1, i] = 1.0, -1.0   # y_gi - y_fi <= -0.05
            rows_got = {tuple(np.round(r, 9)) for r in
                        np.hstack([hs.G, hs.g]).tolist()}
            rows_exp = {tuple(list(G[0]) + [-0.05]),
                        tuple(list(G[1]) + [-0.05])}
            assert rows_got == rows_exp


class TestRelationalRejections:
    def test_unbounded_uncoupled_dim_raises(self):
        spec = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [2])
    (declare-output Y_f real [1])
)
(declare-network g
    (equal-to f)
    (declare-input X_g real [2])
    (declare-output Y_g real [1])
)
(assert (and (<= X_f[0] 1.0) (>= X_f[0] 0.0)))
(assert (and (<= X_f[1] 1.0) (>= X_f[1] 0.0)))
(assert (== X_f[0] X_g[0]))
(assert (< Y_f[0] Y_g[0]))
"""
        # X_g[1] is neither bounded nor coupled -> loud failure
        with pytest.raises((VNNLibParseError, ValueError)):
            load_vnnlib(_write(spec))
