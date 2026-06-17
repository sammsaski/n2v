"""Tests for combined input/output spec parsing and (region, prop) pairs.

Covers the task-1 parser work:
  - v1 (legacy) combined-form rewrite: bound extraction, output-constraint
    retention, region multiplicity, float64, parse performance.
  - v2 combined-form lowering and rank-0 outputs.
  - The normalized ``pairs`` structure every parse must expose.

Fixture numbers are hand-computed from the real VNN-COMP files
(nn4sys lindex_1, test_tiny) that exposed the legacy bugs.
"""

import os
import tempfile
import time

import numpy as np
import pytest

from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import VNNLibParseError


def _write(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False)
    f.write(text)
    f.close()
    return f.name


def _pair_hs(pair):
    """Single-group, single-halfspace accessor for a pair's prop."""
    prop = pair["prop"]
    assert len(prop) == 1
    hg = prop[0]["Hg"]
    return hg if not isinstance(hg, list) else hg[0]


# ---------------------------------------------------------------------------
# v1 (legacy format) combined input/output
# ---------------------------------------------------------------------------

LINDEX_1 = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.0019679116085172) (<= X_0 0.0019679903052747) (<= Y_0 0.3195608170521065))
  (and (>= X_0 0.0019679116085172) (<= X_0 0.0019679903052747) (>= Y_0 0.3223194567668632))
))
"""


class TestV1Combined:
    def test_lindex_bounds_and_pairing(self):
        """The exact spec shape that the legacy parser corrupted (ub stuck
        at 0) — both regions must carry the true bounds, each paired with
        its OWN output constraint."""
        p = load_vnnlib(_write(LINDEX_1))
        pairs = p["pairs"]
        assert len(pairs) == 2

        for pair in pairs:
            np.testing.assert_allclose(
                np.asarray(pair["lb"]).flatten(), [0.0019679116085172])
            np.testing.assert_allclose(
                np.asarray(pair["ub"]).flatten(), [0.0019679903052747])

        hs0 = _pair_hs(pairs[0])  # Y_0 <= 0.31956... ->  y <= g
        np.testing.assert_allclose(hs0.G, [[1.0]])
        np.testing.assert_allclose(hs0.g, [[0.3195608170521065]])

        hs1 = _pair_hs(pairs[1])  # Y_0 >= 0.32231... -> -y <= -g
        np.testing.assert_allclose(hs1.G, [[-1.0]])
        np.testing.assert_allclose(hs1.g, [[-0.3223194567668632]])

    def test_output_constraint_not_dropped(self):
        """test_tiny (1.0): the legacy parser silently dropped the Y
        constraint (Hg=None)."""
        spec = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
    (and (>= X_0 -1) (<= X_0 1) (>= Y_0 100))
))
"""
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 1
        np.testing.assert_allclose(np.asarray(pairs[0]["lb"]).flatten(), [-1.0])
        np.testing.assert_allclose(np.asarray(pairs[0]["ub"]).flatten(), [1.0])
        hs = _pair_hs(pairs[0])
        np.testing.assert_allclose(hs.G, [[-1.0]])
        np.testing.assert_allclose(hs.g, [[-100.0]])

    def test_region_multiplicity(self):
        """The legacy parser collapsed N-region specs to 2 regions
        (lindex_500 -> 2). Every disjunct must become a pair."""
        blocks = "\n".join(
            f"  (and (>= X_0 {i}.0) (<= X_0 {i}.5) (<= Y_0 {i}.25))"
            for i in range(40)
        )
        spec = (
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            f"(assert (or\n{blocks}\n))\n"
        )
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 40
        for i, pair in enumerate(pairs):
            np.testing.assert_allclose(
                np.asarray(pair["lb"]).flatten(), [float(i)])
            np.testing.assert_allclose(
                np.asarray(pair["ub"]).flatten(), [i + 0.5])
            hs = _pair_hs(pair)
            np.testing.assert_allclose(hs.g, [[i + 0.25]])

    def test_input_only_disjunct_is_trivially_true(self):
        """A disjunct with no Y constraint means 'any output violates' —
        it must become an always-true prop (0 . y <= 0), never UNSAT-able."""
        spec = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
    (and (>= X_0 0.0) (<= X_0 1.0) (>= Y_0 5.0))
    (and (>= X_0 2.0) (<= X_0 3.0))
))
"""
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 2
        hs = _pair_hs(pairs[1])
        assert np.all(hs.G == 0.0)
        np.testing.assert_allclose(hs.g.flatten(), [0.0])
        # always satisfied: G @ y <= g for any y
        assert hs.contains(np.array([123.456]))

    def test_float64_precision(self):
        """Legacy stored constants as float32; -0.030001 must round-trip
        as float64 (the ml4acopf differential finding)."""
        spec = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 -0.030001))
"""
        p = load_vnnlib(_write(spec))
        hs = _pair_hs(p["pairs"][0])
        assert hs.g[0, 0] == -0.030001  # exact float64 of the literal
        assert hs.g[0, 0] != np.float64(np.float32(-0.030001))

    def test_parse_performance_large_or_block(self):
        """The legacy parser was quadratic (4-9 min on mscn files). A
        2000-disjunct spec must parse in seconds."""
        blocks = "\n".join(
            f"  (and (>= X_0 {i}.0) (<= X_0 {i}.5) (<= Y_0 1.0))"
            for i in range(2000)
        )
        spec = (
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            f"(assert (or\n{blocks}\n))\n"
        )
        path = _write(spec)
        t0 = time.time()
        p = load_vnnlib(path)
        dt = time.time() - t0
        assert len(p["pairs"]) == 2000
        assert dt < 5.0, f"parse took {dt:.1f}s (quadratic blowup?)"


# ---------------------------------------------------------------------------
# v2 (VNNLIB 2.0) combined input/output + rank-0
# ---------------------------------------------------------------------------

TEST_TINY_V2 = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1])
    (declare-output Y float32 [1])
)
(assert (or
    (and (>= X[0] -1.0) (<= X[0] 1.0) (>= Y[0] 100.0))
))
"""


class TestV2Combined:
    def test_test_tiny(self):
        """The 2.0 test-benchmark spec our v2 parser rejected."""
        p = load_vnnlib(_write(TEST_TINY_V2))
        pairs = p["pairs"]
        assert len(pairs) == 1
        np.testing.assert_allclose(np.asarray(pairs[0]["lb"]).flatten(), [-1.0])
        np.testing.assert_allclose(np.asarray(pairs[0]["ub"]).flatten(), [1.0])
        hs = _pair_hs(pairs[0])
        np.testing.assert_allclose(hs.G, [[-1.0]])
        np.testing.assert_allclose(hs.g, [[-100.0]])

    def test_combined_with_global_bounds(self):
        """Global X asserts must intersect into every combined pair."""
        spec = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [2])
    (declare-output Y float32 [1])
)
(assert (>= X[1] 0.0))
(assert (<= X[1] 9.0))
(assert (or
    (and (>= X[0] 0.0) (<= X[0] 1.0) (>= Y[0] 10.0))
    (and (>= X[0] 5.0) (<= X[0] 6.0) (<= Y[0] -10.0))
))
"""
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 2
        np.testing.assert_allclose(
            np.asarray(pairs[0]["lb"]).flatten(), [0.0, 0.0])
        np.testing.assert_allclose(
            np.asarray(pairs[0]["ub"]).flatten(), [1.0, 9.0])
        np.testing.assert_allclose(
            np.asarray(pairs[1]["lb"]).flatten(), [5.0, 0.0])
        np.testing.assert_allclose(
            np.asarray(pairs[1]["ub"]).flatten(), [6.0, 9.0])

    def test_rank0_output(self):
        """cgan transformer specs: rank-0 output declared `[]`, referenced
        as bare `Y`."""
        spec = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1, 2])
    (declare-output Y float32 [])
)
(assert (<= X[0,0] 1.0))
(assert (>= X[0,0] 0.0))
(assert (<= X[0,1] 1.0))
(assert (>= X[0,1] 0.0))
(assert (>= Y 0.5))
"""
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 1
        hs = _pair_hs(pairs[0])
        np.testing.assert_allclose(hs.G, [[-1.0]])
        np.testing.assert_allclose(hs.g, [[-0.5]])

    def test_disjunctive_inputs_with_separate_output(self):
        """prop_6 shape: input-or assert + separate output assert ->
        cross-product pairs (the case the official lib gets wrong)."""
        spec = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  X float32 [1])
    (declare-output Y float32 [1])
)
(assert (or
    (and (>= X[0] 0.0) (<= X[0] 1.0))
    (and (>= X[0] 2.0) (<= X[0] 3.0))
))
(assert (<= Y[0] 0.0))
"""
        p = load_vnnlib(_write(spec))
        pairs = p["pairs"]
        assert len(pairs) == 2
        for pair in pairs:
            hs = _pair_hs(pair)
            np.testing.assert_allclose(hs.G, [[1.0]])
            np.testing.assert_allclose(hs.g, [[0.0]])


class TestV2MultiInput:
    """Multiple declare-input blocks in ONE network (smart_turn shape):
    inputs concatenate into a joint index space in declaration order."""

    SPEC = """\
(vnnlib-version <2.0>)
(declare-network N
    (declare-input  A float32 [2])
    (declare-input  B float32 [1, 2])
    (declare-output Y float32 [1])
)
(assert (>= A[0] 0.0))
(assert (<= A[0] 1.0))
(assert (>= A[1] 2.0))
(assert (<= A[1] 3.0))
(assert (>= B[0,0] 4.0))
(assert (<= B[0,0] 5.0))
(assert (>= B[0,1] 6.0))
(assert (<= B[0,1] 7.0))
(assert (>= Y[0] 100.0))
"""

    def test_joint_input_space(self):
        """A occupies joint dims 0-1, B occupies 2-3 (declaration order)."""
        p = load_vnnlib(_write(self.SPEC))
        pairs = p["pairs"]
        assert len(pairs) == 1
        np.testing.assert_allclose(
            np.asarray(pairs[0]["lb"]).flatten(), [0.0, 2.0, 4.0, 6.0])
        np.testing.assert_allclose(
            np.asarray(pairs[0]["ub"]).flatten(), [1.0, 3.0, 5.0, 7.0])
        hs = _pair_hs(pairs[0])
        np.testing.assert_allclose(hs.G, [[-1.0]])
        np.testing.assert_allclose(hs.g, [[-100.0]])

    def test_input_tensor_metadata(self):
        """Consumers need per-tensor shapes/offsets to split the joint
        vector back into model inputs."""
        p = load_vnnlib(_write(self.SPEC))
        tensors = p["input_tensors"]
        assert [t["name"] for t in tensors] == ["A", "B"]
        assert tensors[0]["shape"] == (2,) and tensors[0]["offset"] == 0
        assert tensors[1]["shape"] == (1, 2) and tensors[1]["offset"] == 2


# ---------------------------------------------------------------------------
# pairs structure invariants
# ---------------------------------------------------------------------------

class TestPairsInvariants:
    def test_simple_spec_single_pair(self):
        """Plain box + output spec -> exactly one pair sharing the prop."""
        spec = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 5.0))
"""
        p = load_vnnlib(_write(spec))
        assert len(p["pairs"]) == 1
        assert p["pairs"][0]["prop"] is not None

    def test_lb_greater_than_ub_raises(self):
        """A parse producing an impossible region must fail loudly."""
        spec = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 2.0))
(assert (<= X_0 1.0))
(assert (<= Y_0 5.0))
"""
        with pytest.raises(ValueError):
            load_vnnlib(_write(spec))
