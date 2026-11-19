"""
Unit tests for falsification techniques.
"""

from fractions import Fraction

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.utils import falsify
from n2v.utils.falsify import _batch_total_margin, _build_group_arrays
from n2v.sets import HalfSpace
from tests.known_bugs import DID_NOT_RAISE, known_bug


class TestFalsifyRandom:
    """Tests for random sampling falsification (method='random')."""

    def test_finds_counterexample(self):
        """Test that random sampling finds a counterexample when one exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Property: output[0] >= 0.5 is unsafe
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='random', n_samples=100, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None, "Counterexample should be returned"
        inp, out = cex
        assert inp[0] >= 0.5, "Counterexample input should satisfy property"

    def test_no_counterexample(self):
        """Test that random sampling returns UNKNOWN when no counterexample exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])

        # Property: output[0] >= 0.5 is unsafe (but inputs never reach 0.5)
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='random', n_samples=100, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample found)"
        assert cex is None, "No counterexample should be returned"

    def test_with_dict_property(self):
        """Test that falsify handles dict property format (from vnnlib)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)
        prop = [{'Hg': hs}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=100, seed=42)

        assert result == 0, "Should find counterexample"

    def test_reproducible_with_seed(self):
        """Test that falsify is reproducible with the same seed."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result1, cex1 = falsify(model, lb, ub, hs, method='random', n_samples=10, seed=123)
        result2, cex2 = falsify(model, lb, ub, hs, method='random', n_samples=10, seed=123)

        assert result1 == result2
        if cex1 is not None and cex2 is not None:
            np.testing.assert_array_equal(cex1[0], cex2[0])
            np.testing.assert_array_equal(cex1[1], cex2[1])

    def test_default_method_is_random(self):
        """Test that the default method is random sampling."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # No method specified - should default to random
        result, cex = falsify(model, lb, ub, hs, n_samples=100, seed=42)

        assert result == 0, "Default method should find counterexample"


class TestFalsifyPGD:
    """Tests for PGD falsification (method='pgd')."""

    def test_finds_counterexample(self):
        """Test that PGD finds a counterexample when one exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None, "Counterexample should be returned"
        inp, out = cex
        # zero tolerance: a witness short of the boundary is not one
        assert 0.0 <= inp[0] <= 1.0 and inp[0] >= 0.5, \
            "Counterexample input should satisfy property"

    def test_no_counterexample(self):
        """Test that PGD returns UNKNOWN when no counterexample exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample found)"
        assert cex is None, "No counterexample should be returned"

    def test_with_relu_model(self):
        """Test PGD with a model containing ReLU (non-linear)."""
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.ReLU(),
            nn.Linear(2, 1, bias=False)
        )
        model[0].weight.data = torch.eye(2)
        model[2].weight.data = torch.ones(1, 2)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Property: output >= 1.5 is unsafe
        G = np.array([[-1.0]])
        g = np.array([-1.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=10, n_steps=50, seed=42)

        assert result == 0, "Should find counterexample"
        assert cex is not None
        inp, out = cex
        # zero tolerance: the returned output is the model's, in the region
        assert np.all((lb <= inp) & (inp <= ub))
        assert out[0] >= 1.5, "Output should be >= 1.5"

    def test_reproducible_with_seed(self):
        """Test that PGD is reproducible with the same seed."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result1, cex1 = falsify(model, lb, ub, hs, method='pgd',
                                n_restarts=3, n_steps=10, seed=123)
        result2, cex2 = falsify(model, lb, ub, hs, method='pgd',
                                n_restarts=3, n_steps=10, seed=123)

        assert result1 == result2
        if cex1 is not None and cex2 is not None:
            np.testing.assert_allclose(cex1[0], cex2[0], rtol=1e-5)
            np.testing.assert_allclose(cex1[1], cex2[1], rtol=1e-5)

    def test_custom_step_size(self):
        """Test PGD with custom step size."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=3, n_steps=30, step_size=0.05, seed=42)

        assert result == 0, "Should find counterexample with custom step size"


class TestFalsifyCombined:
    """Tests for combined falsification (method='random+pgd')."""

    def test_finds_counterexample_via_random(self):
        """Test that combined method finds counterexample via random (first)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # With enough random samples, should find via random first
        result, cex = falsify(model, lb, ub, hs, method='random+pgd',
                              n_samples=100, n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "Should find counterexample"
        assert cex is not None

    def test_finds_counterexample_via_pgd_when_random_fails(self):
        """Test that combined method falls back to PGD when random fails."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # With 0 random samples, PGD should find the counterexample
        result, cex = falsify(model, lb, ub, hs, method='random+pgd',
                              n_samples=0, n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "PGD should find counterexample when random has 0 samples"
        assert cex is not None


class TestFalsifyMultiGroupProperty:
    """Tests for multi-group property handling (AND logic across groups).

    VNN-LIB properties can have multiple top-level asserts that are ANDed.
    A counterexample must satisfy ALL property groups simultaneously.
    """

    def test_rejects_false_counterexample_when_second_group_not_satisfied(self):
        """Counterexample satisfying only first property group should not be reported as SAT.

        Reproduces lsnc_relu bug: falsify only checked property[0], ignoring property[1].
        """
        # Identity model: output = input
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3 (easy to satisfy in [0,1]^2)
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 2.0 (impossible in [0,1]^2)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        # No valid counterexample exists (group 1 is infeasible)
        assert result == 2, "Should return UNKNOWN — no input satisfies both groups"
        assert cex is None

    def test_finds_counterexample_when_all_groups_satisfied(self):
        """Counterexample satisfying all property groups should be reported as SAT."""
        # Identity model: output = input
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 0.3
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.3]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        assert result == 0, "Should find counterexample satisfying both groups"
        assert cex is not None
        inp, out = cex
        assert inp[0] >= 0.3 and inp[1] >= 0.3, "Counterexample must satisfy both groups"

    def test_pgd_rejects_false_counterexample_multi_group(self):
        """PGD should also respect multi-group AND logic."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3 (easy to satisfy)
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 2.0 (impossible)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 2, "PGD should return UNKNOWN — no input satisfies both groups"
        assert cex is None

    def test_multi_group_with_or_within_group(self):
        """Multi-group where one group has OR of halfspaces (list)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.8 OR x1 <= 0.2 (OR of two halfspaces)
        hs0a = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]]))
        hs0b = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]]))

        # Group 1: x2 >= 2.0 (impossible)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': [hs0a, hs0b]}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        assert result == 2, "Should not find counterexample — group 1 is infeasible"
        assert cex is None


class TestFalsifyValidation:
    """Tests for input validation."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        with pytest.raises(ValueError, match="Unknown method"):
            falsify(model, lb, ub, hs, method='invalid_method')

    def test_mismatched_bounds(self):
        """Test that mismatched bounds raise ValueError."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])  # Wrong shape

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        with pytest.raises(ValueError, match="same shape"):
            falsify(model, lb, ub, hs)


# --------------------------------------------------------------------------- #
# The random falsifier's sampling, the                                         #
# order of a cascade, the Square margin, and the refusals of unreadable input. #
# --------------------------------------------------------------------------- #
def _identity(n):
    model = nn.Sequential(nn.Linear(n, n, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(n))
    return model


class _Recorder(nn.Module):
    """y = x. Keeps every input batch it is called with; with
    ``batch_one_only`` it rejects batches larger than one with a
    RuntimeError, as an ONNX-converted graph with a fixed batch axis does."""

    def __init__(self, batch_one_only=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.batch_one_only = batch_one_only
        self.seen = []

    def forward(self, x):
        if self.batch_one_only and x.shape[0] > 1:
            raise RuntimeError("this graph takes batch size 1 only")
        self.seen.append(x.detach().reshape(x.shape[0], -1).numpy().copy())
        return self.scale * x.reshape(x.shape[0], -1)


class _SkewedBatches(nn.Module):
    """y = x; the rows of a batch larger than one read 1/2 lower."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        y = self.scale * x.reshape(x.shape[0], -1)
        return y - 0.5 if x.shape[0] > 1 else y


class TestFalsifyRandomSampling:
    """The random falsifier draws ``n_samples`` points of the box, checks
    every one, and returns one whose batch-1 output is unsafe."""

    def test_every_sample_of_every_chunk_is_drawn_from_the_box_and_checked(self):
        """y = x on [2, 3] (a box away from zero); the unsafe sliver
        x >= 3 - 1/2048 holds about two of the 4613 samples, which are
        evaluated in chunks of 2048. Over seeds 0 to 7 the samples the model
        sees must lie in the box; a run without a hit must have evaluated all
        4613; a run in which some evaluated sample is in the sliver must
        return a real counterexample, also when the first hit lies after the
        first chunk (seeds where it does are required). Catches samples drawn
        from [lb, ub - lb], a chunk slice that evaluates only the first 2048
        samples, and a hit of a later chunk re-checked at the sample with its
        index inside the chunk."""
        lb, ub = np.array([2.0]), np.array([3.0])
        top = 3.0 - 1.0 / 2048.0
        unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-top]]))
        n = 2 * 2048 + 517
        later = []
        for seed in range(8):
            model = _Recorder()
            result, cex = falsify(model, lb, ub, unsafe, method='random',
                                  seed=seed, n_samples=n)
            drawn = np.concatenate([b for b in model.seen if b.shape[0] > 1])
            assert np.all((drawn >= lb) & (drawn <= ub)), \
                f"seed {seed}: a sample outside [2, 3]"
            hits = np.nonzero(drawn[:, 0] >= top)[0]
            if hits.size == 0:
                assert drawn.shape[0] == n, \
                    f"seed {seed}: {drawn.shape[0]} of n_samples={n} evaluated"
                assert (result, cex) == (2, None)
                continue
            assert result == 0, \
                f"seed {seed}: sample {hits[0]} is a counterexample, " \
                f"falsify answered {(result, cex)}"
            x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
            assert lb[0] <= x[0] <= ub[0] and x[0] >= top, f"seed {seed}: {x}"
            assert np.array_equal(np.asarray(cex[1]).reshape(-1), x)
            if hits[0] >= 2048:
                later.append(seed)
        if not later:
            pytest.fail("setup: no seed put the first hit after the first "
                        "chunk of 2048 samples")

    def test_per_sample_fallback_checks_every_sample(self):
        """A model that rejects batches larger than one: every sample is run
        alone. The unsafe region is one half of [0, 1], then the other half;
        64 samples miss a half with probability 2^-64, and the first sample
        lies in exactly one of the two, so both runs must return a real
        counterexample. Catches a fallback that evaluates the first sample
        for every sample, and a fallback that catches fewer exceptions than
        the batched forward of a converted graph raises."""
        lb, ub = np.array([0.0]), np.array([1.0])
        halves = [HalfSpace(np.array([[-1.0]]), np.array([[-0.5]])),   # x >= 1/2
                  HalfSpace(np.array([[1.0]]), np.array([[0.5]]))]     # x <= 1/2
        for k, unsafe in enumerate(halves):
            model = _Recorder(batch_one_only=True)
            result, cex = falsify(model, lb, ub, unsafe, method='random',
                                  seed=0, n_samples=64)
            assert result == 0, f"half {k}: {(result, cex)}"
            x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
            assert lb[0] <= x[0] <= ub[0]
            assert (x[0] >= 0.5) if k == 0 else (x[0] <= 0.5)
            assert np.array_equal(np.asarray(cex[1]).reshape(-1), x)

    def test_a_batched_hit_that_batch_one_does_not_confirm_is_skipped(self):
        """Batched rows read 1/2 lower than the batch-1 output, so about half
        of the samples look unsafe for y <= 1/64 in the batch and only the
        samples with x <= 1/64 are. The falsifier must re-run each batched
        hit at batch size 1, skip the false ones and return a real one (2000
        samples miss x <= 1/64 with probability below 1e-13). Catches a
        falsifier that returns the first batched hit unconfirmed, which the
        final check then rejects with (2, None)."""
        lb, ub = np.array([0.0]), np.array([1.0])
        unsafe = HalfSpace(np.array([[1.0]]), np.array([[1.0 / 64.0]]))
        result, cex = falsify(_SkewedBatches(), lb, ub, unsafe,
                              method='random', seed=0, n_samples=2000)
        assert result == 0, (result, cex)
        x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
        assert 0.0 <= x[0] <= 1.0 / 64.0
        assert np.array_equal(np.asarray(cex[1]).reshape(-1), x)

    def test_seed_zero_makes_a_run_repeatable(self):
        """Seed 0 is a seed like any other: two runs with seed 0 return the
        same counterexample. Catches a generator built from ``seed or None``,
        which reads 0 as no seed."""
        lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        unsafe = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[0.0]]))
        first = falsify(_identity(2), lb, ub, unsafe, method='random', seed=0,
                        n_samples=50)
        second = falsify(_identity(2), lb, ub, unsafe, method='random',
                         seed=0, n_samples=50)
        assert first[0] == 0 and second[0] == 0
        np.testing.assert_array_equal(first[1][0], second[1][0])
        np.testing.assert_array_equal(first[1][1], second[1][1])


class TestFalsifyMethodOrder:
    """A cascade runs its legs in the order its name gives."""

    def test_random_plus_pgd_tries_random_first(self):
        """When random sampling finds a counterexample, 'random+pgd' returns
        exactly the point that method 'random' with the same seed and
        ``n_samples`` returns. The unsafe region is x0 >= 0, then x0 <= 0:
        the first sample lies in exactly one of the two, and for the other
        PGD (which starts from that same first draw and has the steps to
        cross x0 = 0) returns a point that random sampling does not draw.
        Catches a cascade that runs PGD first."""
        lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        for sign in (-1.0, 1.0):
            unsafe = HalfSpace(np.array([[sign, 0.0]]), np.array([[0.0]]))
            alone = falsify(_identity(2), lb, ub, unsafe, method='random',
                            seed=7, n_samples=50)
            cascade = falsify(_identity(2), lb, ub, unsafe,
                              method='random+pgd', seed=7, n_samples=50,
                              n_restarts=1, n_steps=100)
            assert alone[0] == 0 and cascade[0] == 0, (sign, alone, cascade)
            np.testing.assert_array_equal(cascade[1][0], alone[1][0])


# Plain specs over y: (rows, rhs) with sum_k rows[i][k] y[k] <= rhs[i] for
# every row i.
_A = ([[-1.0, 0.0], [0.0, 1.0]], [-0.5, 0.25])   # y0 >= 1/2 and y1 <= 1/4
_B = ([[1.0, 0.0]], [-0.5])                      # y0 <= -1/2
_C = ([[0.0, -1.0]], [0.0])                      # y1 >= 0


def _hs(spec):
    return HalfSpace(np.array(spec[0]), np.array(spec[1]).reshape(-1, 1))


class TestSquareMargin:
    """The Square attack's screening margin: <= 0 exactly in the unsafe
    region, as the AND of groups, the OR of half-spaces, the AND of rows."""

    def test_margin_is_the_and_of_or_of_and_reading(self):
        """For the groups [A or B] and [C], the margin at y is
        max(min(max_i(A_i y - a_i), max_i(B_i y - b_i)), max_i(C_i y - c_i)),
        computed here in rationals. Dyadic outputs make the float32 margin
        exact; at y = (3/4, 1/2) a min over rows, a max over half-spaces and
        a min over groups each change the value. Catches each of those
        reductions read the other way, and g kept as a column."""
        groups = [[_A, _B], [_C]]
        outs = [(0.75, 0.5), (0.75, 0.125), (-0.75, -0.25), (1.0, 0.0),
                (0.0, 0.75), (-2.0, 1.0), (0.5, 0.25)]

        def margin(y):
            def hs_margin(spec):
                return max(sum(Fraction(a) * Fraction(v)
                               for a, v in zip(row, y)) - Fraction(b)
                           for row, b in zip(*spec))
            return max(min(hs_margin(s) for s in g) for g in groups)

        arrays = _build_group_arrays([[_hs(s) for s in g] for g in groups])
        got = _batch_total_margin(np.array(outs, dtype=np.float32), arrays)
        assert got.shape == (len(outs),)
        for y, m in zip(outs, got):
            assert Fraction(float(m)) == margin(y), (y, float(m), margin(y))

    @pytest.mark.parametrize("name", ["two-groups", "two-rows",
                                      "unreachable-disjunct-first"])
    def test_square_finds_a_point_that_needs_every_group_row_and_disjunct(
            self, name):
        """y = x on [-2, 2] x [1, 3] x [-1, 1]; the unsafe region is
        y0 >= 3/2 and y1 <= 3/2 (one eighth of the box), written as two
        groups, as one half-space of two rows, and as the second disjunct
        after the unreachable y2 >= 2. Square is guided by the margin alone,
        so a margin that is <= 0 when one group or one row holds stops it
        short of the region, and one that reads the disjunction as AND never
        reaches 0. The witness must be real."""
        lb, ub = np.array([-2.0, 1.0, -1.0]), np.array([2.0, 3.0, 1.0])
        y0_high = ([[-1.0, 0.0, 0.0]], [-1.5])
        y1_low = ([[0.0, 1.0, 0.0]], [1.5])
        both = ([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [-1.5, 1.5])
        y2_high = ([[0.0, 0.0, -1.0]], [-2.0])
        prop = {
            "two-groups": [{'Hg': _hs(y0_high)}, {'Hg': _hs(y1_low)}],
            "two-rows": _hs(both),
            "unreachable-disjunct-first": [_hs(y2_high), _hs(both)],
        }[name]
        result, cex = falsify(_identity(3), lb, ub, prop, method='square',
                              seed=0, n_iters=1000, batch=32)
        assert result == 0, (name, result)
        x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
        assert np.all((lb <= x) & (x <= ub))
        assert x[0] >= 1.5 and x[1] <= 1.5
        assert np.array_equal(np.asarray(cex[1]).reshape(-1), x)


def _two_output_onnx(path):
    """A = X and B = -X, two output tensors of shape (1, 2) for one input of shape (1, 2)."""
    import onnx
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["A"]), helper.make_node("Neg", ["X"], ["B"])], "two_outputs",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


class TestFalsifyRefusals:
    """Input that falsify cannot read is refused, never answered and never a
    crash: ValueError for a malformed property or box (TypeError where the
    property reader names a wrong type, as it does today)."""

    _LB, _UB = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    _HS = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))

    @pytest.mark.parametrize("prop", [
        pytest.param(np.array([[-1.0, 0.0, -0.5]]), id="array"),
        pytest.param({'Hg': np.array([[-1.0, 0.0]])}, id="dict-Hg-array"),
        pytest.param([{'Hg': "y0 >= 1/2"}], id="group-Hg-text"),
    ])
    def test_property_of_an_unreadable_type_is_refused(self, prop):
        """A property that is no HalfSpace, list or dict with 'Hg', and an
        'Hg' that is no HalfSpace or list, in each of the three branches of
        the reader."""
        with pytest.raises(TypeError, match="must be HalfSpace"):
            falsify(_identity(2), self._LB, self._UB, prop, seed=0,
                    n_samples=10)

    @known_bug(23, "a property given as a list of groups (lists of "
                   "HalfSpaces), the form the readers return, is taken as "
                   "one OR group whose members are lists, and falsify "
                   "crashes with AttributeError instead of refusing it",
               raises=AttributeError)
    def test_property_given_as_a_list_of_groups_is_refused(self):
        """``[[hs_a, hs_b], [hs_c]]`` is not one of the accepted forms."""
        with pytest.raises((ValueError, TypeError)):
            falsify(_identity(2), self._LB, self._UB,
                    [[self._HS, self._HS], [self._HS]], seed=0, n_samples=10)

    @known_bug(23, "a dict property without the key 'Hg' (for example the "
                   "whole dict load_vnnlib returns) makes falsify crash with "
                   "KeyError instead of refusing it", raises=KeyError)
    def test_property_dict_without_hg_is_refused(self):
        """``{'lb': ..., 'ub': ..., 'prop': ...}`` passed where its 'prop' was
        meant."""
        spec = {'lb': self._LB, 'ub': self._UB, 'prop': [{'Hg': self._HS}]}
        with pytest.raises((ValueError, TypeError)):
            falsify(_identity(2), self._LB, self._UB, spec, seed=0,
                    n_samples=10)

    @pytest.mark.parametrize("prop", [
        pytest.param([], id="empty-list"),
        pytest.param({'Hg': []}, id="dict-with-empty-Hg"),
        pytest.param([{'Hg': []}], id="group-with-empty-Hg"),
    ])
    @known_bug(6, "a property with no half-space (an empty list, or an "
                  "empty 'Hg') is answered (2, None) instead of refused; "
                  "verify_specification must refuse the same input "
                  "(registry SPEC-12)", raises=DID_NOT_RAISE)
    def test_empty_property_is_refused(self, prop):
        """No half-space is no statement of what is unsafe (the decision of
        SPEC-12 for the verifier's reader of the same object)."""
        with pytest.raises(ValueError):
            falsify(_identity(2), self._LB, self._UB, prop, seed=0,
                    n_samples=10)

    def test_empty_box_is_never_answered_sat(self):
        """lb > ub holds no point, so there is no counterexample, even for a
        property that every output meets: refused with ValueError (today
        numpy's sampler raises it) or answered (2, None). Catches bounds
        that are sorted or swapped into a non-empty box."""
        everything = HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))
        try:
            result = falsify(_identity(2), np.array([0.5, 0.0]),
                             np.array([0.25, 1.0]), everything,
                             method='random', seed=0, n_samples=20)
        except ValueError:
            return
        assert result == (2, None)

    @known_bug(23, "a box with an infinite bound makes the random "
                   "falsifier's sampler raise OverflowError ('Range exceeds "
                   "valid bounds') instead of a refusal or an answer",
               raises=OverflowError)
    def test_unbounded_box_is_refused_or_answered(self):
        """x0 in [-inf, 1], x1 in [0, 1] (a VNN-LIB 2.0 input without bounds
        is read as infinite); the unsafe region y1 >= 1/2 does not need x0.
        Either a refusal (ValueError, NotImplementedError) or an answer whose
        witness lies in the box."""
        unsafe = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))
        lb, ub = np.array([-np.inf, 0.0]), np.array([1.0, 1.0])
        try:
            result, cex = falsify(_identity(2), lb, ub, unsafe,
                                  method='random', seed=0, n_samples=50)
        except (ValueError, NotImplementedError):
            return
        if result == 0:
            x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
            assert np.all((lb <= x) & (x <= ub)) and x[1] >= 0.5
        else:
            assert (result, cex) == (2, None)

    @known_bug(19, "falsify on a converted ONNX model with two output tensors crashes with AttributeError "
                   "('list' object has no attribute 'detach'): the sampler calls detach() on the list of "
                   "outputs, where a counterexample over the joint output or a refusal is due; the runner "
                   "then answers unknown on such an instance (registry row VERDICT-10)", raises=AttributeError)
    def test_model_with_two_output_tensors_is_refused_or_answered_on_the_joint_output(self, tmp_path):
        """A = X and B = -X on the box [0, 1]^2, with the unsafe row B0 <= -1/2 on the joint output (A0, A1, B0, B1), which every input with x0 >= 1/2 meets (half of the box). Either a refusal (ValueError, NotImplementedError), or the answer SAT with an input of the box that meets the row; 64 samples miss that half with probability 2^-64."""
        from n2v.utils.model_loader import load_onnx
        model = load_onnx(_two_output_onnx(tmp_path / "two.onnx"))
        unsafe = HalfSpace(np.array([[0.0, 0.0, 1.0, 0.0]]), np.array([[-0.5]]))
        lb, ub = np.zeros(2), np.ones(2)
        try:
            result, cex = falsify(model, lb, ub, [{'Hg': unsafe}], method='random', n_samples=64, seed=0)
        except (ValueError, NotImplementedError):
            return
        assert result == 0, f"no counterexample found where half of the box is unsafe (answer {result})"
        x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
        assert np.all((lb <= x) & (x <= ub)) and x[0] >= 0.5, f"the witness {x} is no counterexample"
