"""
Unit tests for ProbabilisticBox class.
"""

import math
import re
from fractions import Fraction
from itertools import product

import pytest
import numpy as np
from scipy.stats import beta

from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.sets.box import Box
from tests._oracles.membership import LP_TOL, distance
from tests.known_bugs import known_bug


class TestProbabilisticBoxConstruction:
    """Tests for ProbabilisticBox construction and validation."""

    def test_basic_construction(self):
        """Test basic construction with valid parameters."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        m = 1000
        ell = 999
        epsilon = 0.01

        pbox = ProbabilisticBox(lb, ub, m=m, ell=ell, epsilon=epsilon)

        assert pbox.dim == 2
        assert pbox.m == m
        assert pbox.ell == ell
        assert pbox.epsilon == epsilon
        assert pbox.coverage == 1 - epsilon
        np.testing.assert_array_equal(pbox.lb.flatten(), lb)
        np.testing.assert_array_equal(pbox.ub.flatten(), ub)

    def test_construction_with_column_vectors(self):
        """Test construction with column vectors."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])

        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.05)

        assert pbox.dim == 3
        np.testing.assert_array_equal(pbox.lb, lb)
        np.testing.assert_array_equal(pbox.ub, ub)

    def test_invalid_m_raises_error(self):
        """Test that m < 1 raises ValueError."""
        lb = np.array([0.0])
        ub = np.array([1.0])

        with pytest.raises(ValueError, match="m must be >= 1"):
            ProbabilisticBox(lb, ub, m=0, ell=1, epsilon=0.01)

        with pytest.raises(ValueError, match="m must be >= 1"):
            ProbabilisticBox(lb, ub, m=-5, ell=1, epsilon=0.01)

    def test_invalid_ell_raises_error(self):
        """Test that invalid ell raises ValueError."""
        lb = np.array([0.0])
        ub = np.array([1.0])

        # ell < 1
        with pytest.raises(ValueError, match="ell must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=0, epsilon=0.01)

        # ell > m
        with pytest.raises(ValueError, match="ell must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=101, epsilon=0.01)

    def test_invalid_epsilon_raises_error(self):
        """Test that invalid epsilon raises ValueError."""
        lb = np.array([0.0])
        ub = np.array([1.0])

        # epsilon <= 0
        with pytest.raises(ValueError, match="epsilon must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=-0.1)

        # epsilon >= 1
        with pytest.raises(ValueError, match="epsilon must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=1.0)

        with pytest.raises(ValueError, match="epsilon must be in"):
            ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=1.5)


class TestConfidenceComputation:
    """Tests for confidence computation matching scipy.stats.beta."""

    def test_confidence_computation_basic(self):
        """Test confidence computation matches expected formula."""
        lb = np.array([0.0])
        ub = np.array([1.0])
        m = 8000
        ell = 7999
        epsilon = 0.001

        pbox = ProbabilisticBox(lb, ub, m=m, ell=ell, epsilon=epsilon)

        # Compute expected confidence using scipy.stats.beta
        expected_confidence = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)

        assert abs(pbox.confidence - expected_confidence) < 1e-10

    def test_confidence_computation_various_params(self):
        """Test confidence computation for various parameter combinations."""
        lb = np.array([0.0])
        ub = np.array([1.0])

        test_cases = [
            (100, 99, 0.05),
            (1000, 999, 0.01),
            (8000, 7999, 0.001),
            (100000, 99999, 0.0001),
            (500, 500, 0.02),  # ell = m case
        ]

        for m, ell, epsilon in test_cases:
            pbox = ProbabilisticBox(lb, ub, m=m, ell=ell, epsilon=epsilon)
            expected_confidence = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)

            assert abs(pbox.confidence - expected_confidence) < 1e-10, \
                f"Failed for m={m}, ell={ell}, epsilon={epsilon}"

    def test_coverage_equals_one_minus_epsilon(self):
        """Test that coverage equals 1 - epsilon."""
        lb = np.array([0.0])
        ub = np.array([1.0])

        for epsilon in [0.001, 0.01, 0.05, 0.1]:
            pbox = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=epsilon)
            assert pbox.coverage == 1 - epsilon


class TestBoxInheritance:
    """Tests for Box inheritance - all Box methods should work."""

    def test_dim_attribute(self):
        """Test dim attribute from Box."""
        lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        assert pbox.dim == 5

    def test_center_attribute(self):
        """Test center attribute from Box."""
        lb = np.array([0.0, 2.0])
        ub = np.array([2.0, 4.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        expected_center = np.array([[1.0], [3.0]])
        np.testing.assert_array_equal(pbox.center, expected_center)

    def test_sample_method(self):
        """Test sample() method from Box."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        samples = pbox.sample(100)
        assert samples.shape == (2, 100)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_get_range_method(self):
        """Test get_range() method from Box."""
        lb = np.array([1.0, 2.0])
        ub = np.array([3.0, 4.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        lb_out, ub_out = pbox.get_range()
        np.testing.assert_array_equal(lb_out.flatten(), lb)
        np.testing.assert_array_equal(ub_out.flatten(), ub)

    def test_contains_method(self):
        """Test contains() method from Box."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        assert pbox.contains(np.array([0.5, 0.5]))
        assert pbox.contains(np.array([0.0, 0.0]))
        assert pbox.contains(np.array([1.0, 1.0]))
        assert not pbox.contains(np.array([1.5, 0.5]))
        assert not pbox.contains(np.array([-0.1, 0.5]))

    def test_to_zono_method(self):
        """Test to_zono() method from Box."""
        lb = np.array([0.0, 0.0])
        ub = np.array([2.0, 2.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        zono = pbox.to_zono()
        # Zono uses 'c' for center, not 'center'
        np.testing.assert_array_equal(zono.c.flatten(), np.array([1.0, 1.0]))


class TestMinkowskiSum:
    """Tests for Minkowski sum preserving guarantee."""

    def test_minkowski_sum_with_box(self):
        """Test Minkowski sum with a regular Box preserves guarantee."""
        lb1 = np.array([0.0, 0.0])
        ub1 = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb1, ub1, m=1000, ell=999, epsilon=0.01)

        lb2 = np.array([0.5, 0.5])
        ub2 = np.array([1.5, 1.5])
        box = Box(lb2, ub2)

        result = pbox.minkowski_sum(box)

        # Should be ProbabilisticBox
        assert isinstance(result, ProbabilisticBox)

        # Guarantee should be preserved
        assert result.m == pbox.m
        assert result.ell == pbox.ell
        assert result.epsilon == pbox.epsilon

        # Bounds should be sum
        np.testing.assert_array_almost_equal(result.lb.flatten(), np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(result.ub.flatten(), np.array([2.5, 2.5]))

    # The sum of two probabilistic boxes was covered here by a test that
    # asserted epsilon == max(epsilon1, epsilon2). The miss probabilities add
    # (union bound); the contract is tests/contracts/
    # test_probabilistic_contract.py
    # (``test_sum_of_two_probabilistic_boxes_adds_the_miss_probabilities`` and
    # the two tests after it).


class TestAffineMap:
    """Tests for affine map preserving guarantee."""

    def test_affine_map_preserves_guarantee(self):
        """Test affine map preserves probabilistic guarantee."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=0.01)

        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, 2.0])

        result = pbox.affine_map(W, b)

        # Should be ProbabilisticBox
        assert isinstance(result, ProbabilisticBox)

        # Guarantee should be preserved
        assert result.m == pbox.m
        assert result.ell == pbox.ell
        assert result.epsilon == pbox.epsilon
        assert result.coverage == pbox.coverage
        assert result.confidence == pbox.confidence

    def test_affine_map_transforms_bounds(self):
        """Test affine map correctly transforms bounds."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        # Scale by 2
        W = np.array([[2.0, 0.0], [0.0, 2.0]])

        result = pbox.affine_map(W)

        np.testing.assert_array_almost_equal(result.lb.flatten(), np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.ub.flatten(), np.array([2.0, 2.0]))


class TestToStar:
    """Tests for to_star() conversion warning."""

    def test_to_star_issues_warning(self):
        """Test that to_star() issues a warning about losing metadata."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        with pytest.warns(UserWarning, match="loses guarantee metadata"):
            star = pbox.to_star()

        # Star should still be created successfully
        assert star is not None


class TestGuaranteeUtilities:
    """Tests for guarantee utility methods."""

    def test_get_guarantee(self):
        """Test get_guarantee() returns tuple."""
        lb = np.array([0.0])
        ub = np.array([1.0])
        pbox = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=0.01)

        coverage, confidence = pbox.get_guarantee()

        assert coverage == pbox.coverage
        assert confidence == pbox.confidence

    def test_get_guarantee_string(self):
        """Test get_guarantee_string() returns readable string."""
        lb = np.array([0.0])
        ub = np.array([1.0])
        pbox = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=0.01)

        string = pbox.get_guarantee_string()

        assert "confidence" in string.lower()
        assert "%" in string  # Should have percentage formatting

    def test_compute_parameters(self):
        """Test compute_parameters() finds correct m, ell, epsilon."""
        target_coverage = 0.99
        target_confidence = 0.95

        m, ell, epsilon = ProbabilisticBox.compute_parameters(
            target_coverage, target_confidence
        )

        # Verify computed parameters achieve targets
        assert epsilon == 1 - target_coverage
        computed_confidence = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)
        assert computed_confidence >= target_confidence

    def test_compute_parameters_raises_if_impossible(self):
        """Test compute_parameters() raises if targets can't be achieved."""
        # Very high requirements with low max_m should fail
        with pytest.raises(ValueError, match="Cannot achieve"):
            ProbabilisticBox.compute_parameters(
                target_coverage=0.9999,
                target_confidence=0.9999,
                max_m=10
            )


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_key_info(self):
        """Test __repr__ contains key information."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=0.01)

        repr_str = repr(pbox)

        assert "ProbabilisticBox" in repr_str
        assert "dim=2" in repr_str
        assert "m=1000" in repr_str
        assert "999" in repr_str  # ell
        assert "0.01" in repr_str  # epsilon


# --------------------------------------------------------------------------- #
# The guarantee against the calibration, computed independently                #
# --------------------------------------------------------------------------- #
def _binomial_confidence(m, ell, epsilon):
    """The probability that the ell-th smallest of m calibration scores lies
    above the 1 - epsilon quantile of the score distribution: that happens
    exactly when fewer than ell of the m scores fall below the quantile, so
    it is P(Bin(m, 1 - epsilon) <= ell - 1). Exact, in fractions from the
    float epsilon; no Beta function and no scipy."""
    eps = Fraction(epsilon)
    p = 1 - eps

    def term(k):
        return math.comb(m, k) * p ** k * eps ** (m - k)

    if ell - 1 <= m // 2:
        return sum(term(k) for k in range(ell))
    return 1 - sum(term(k) for k in range(ell, m + 1))


class TestGuaranteeFromTheCalibration:
    """The stated (coverage, confidence) against the binomial value of the
    calibration, never through scipy's Beta function."""

    @pytest.mark.parametrize("m, ell, epsilon", [
        (20, 19, 0.1), (20, 20, 0.1), (50, 45, 0.05), (1000, 990, 0.02),
        (8000, 7999, 0.001),
    ])
    def test_confidence_is_the_probability_that_the_rank_clears_the_quantile(
            self, m, ell, epsilon):
        """Never above the binomial value (a higher one is a guarantee the
        calibration does not support), and equal to it as documented, with
        the 1e-12 slack of float64 rounding. Ranks m - 1, m and below m - 1.
        Catches a Beta second parameter m - ell instead of m + 1 - ell, a
        first parameter fixed at m - 1, a rank ell = m refused, and a
        ``get_guarantee`` tuple in the wrong order."""
        pbox = ProbabilisticBox(np.array([0.0, -1.0]), np.array([1.0, 2.0]),
                                m=m, ell=ell, epsilon=epsilon)
        exact = _binomial_confidence(m, ell, epsilon)
        slack = Fraction(1, 10 ** 12)
        assert Fraction(pbox.confidence) <= exact + slack, (
            f"states {pbox.confidence}, the calibration supports "
            f"{float(exact)}")
        assert Fraction(pbox.confidence) >= exact - slack
        coverage, confidence = pbox.get_guarantee()
        assert abs(Fraction(coverage) - (1 - Fraction(epsilon))) <= slack
        assert abs(Fraction(confidence) - exact) <= slack

    @pytest.mark.parametrize("coverage, confidence, max_m, smallest", [
        (0.99, 0.95, 1_000_000, 473),
        (0.9, 0.5, 1_000_000, 17),
        pytest.param(0.99, 0.95, 473, 473, id="max_m-is-the-answer"),
        pytest.param(0.8, 0.5, 1_000_000, 9, marks=known_bug(
            13, "ProbabilisticBox.compute_parameters starts its search at "
                "m = 10, so it returns 10 where a smaller m reaches the "
                "targets (9 for coverage 0.8 and confidence 0.5)")),
        pytest.param(0.5, 0.45, 5, 3, marks=known_bug(
            13, "ProbabilisticBox.compute_parameters starts its search at "
                "m = 10, so with max_m below 10 it raises although a smaller "
                "m reaches the targets (3 for coverage 0.5 and confidence "
                "0.45)", raises=ValueError)),
    ])
    def test_compute_parameters_returns_the_smallest_sufficient_m(
            self, coverage, confidence, max_m, smallest):
        """The docstring: the smallest m whose rank m - 1 reaches the target
        confidence at epsilon = 1 - coverage. The expected m is checked
        against the binomial value first (it reaches the target, m - 1 does
        not). Catches a rank m instead of m - 1 (299 instead of 473), a Beta
        CDF evaluated at epsilon instead of 1 - epsilon (10), and a search
        that stops before ``max_m``."""
        eps = 1 - coverage
        if not (_binomial_confidence(smallest, smallest - 1, eps)
                >= Fraction(confidence)
                > _binomial_confidence(smallest - 1, smallest - 2, eps)):
            pytest.fail(f"{smallest} is not the smallest m for the targets")
        m, ell, epsilon = ProbabilisticBox.compute_parameters(
            coverage, confidence, max_m=max_m)
        assert (m, ell) == (smallest, smallest - 1)
        assert epsilon == 1 - coverage

    def test_guarantee_string_states_confidence_then_coverage(self):
        """m = 100, rank 99, epsilon 0.05: confidence 96.29 %, coverage
        95.00 %. Catches the two numbers exchanged in the sentence."""
        pbox = ProbabilisticBox(np.array([0.0]), np.array([1.0]), m=100,
                                ell=99, epsilon=0.05)
        text = pbox.get_guarantee_string()
        stated_confidence = re.search(r"([0-9.]+)% confidence", text)
        stated_coverage = re.search(r"at least ([0-9.]+)% of outputs", text)
        assert stated_confidence and stated_coverage, text
        assert float(stated_confidence[1]) == pytest.approx(
            100 * float(_binomial_confidence(100, 99, 0.05)), abs=0.005)
        assert float(stated_coverage[1]) == pytest.approx(95.0, abs=0.005)

    def test_repr_states_the_coverage_and_confidence_of_the_calibration(self):
        """The repr is what ``conformal_reach`` logs: coverage 0.9500 and
        confidence 0.9629 for m = 100, rank 99, epsilon 0.05."""
        pbox = ProbabilisticBox(np.array([0.0]), np.array([1.0]), m=100,
                                ell=99, epsilon=0.05)
        text = repr(pbox)
        stated_coverage = re.search(r"coverage=([0-9.]+)", text)
        stated_confidence = re.search(r"confidence=([0-9.]+)", text)
        assert stated_coverage and stated_confidence, text
        assert float(stated_coverage[1]) == pytest.approx(0.95, abs=5e-5)
        assert float(stated_confidence[1]) == pytest.approx(
            float(_binomial_confidence(100, 99, 0.05)), abs=5e-5)

    def test_conformal_reach_states_the_confidence_of_its_calibration(self):
        """The public route: ``conformal_reach`` with m = 60 and rank 55
        (below m - 1) builds the box from its configuration; the box states
        the binomial confidence of that calibration. Catches a confidence
        that ignores the rank on the route a user takes."""
        import torch
        from n2v.probabilistic import ConformalReachConfig, conformal_reach

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(),
                                    torch.nn.Linear(3, 2)).eval()
        result = conformal_reach(
            model, Box(-np.ones(2), np.ones(2)),
            ConformalReachConfig(surrogate='naive', m=60, ell=55,
                                 epsilon=0.1, seed=0))
        assert isinstance(result, ProbabilisticBox)
        assert (result.m, result.ell, result.epsilon) == (60, 55, 0.1)
        exact = _binomial_confidence(60, 55, 0.1)
        assert abs(Fraction(result.confidence) - exact) <= Fraction(1, 10**12)
        assert result.coverage == pytest.approx(0.9, abs=1e-15)


class TestOperationsKeepTheImageAndTheGuarantee:
    """Set operations on a probabilistic box: the result holds the image of
    every point of the operand, and states no more than the operands
    support. Corners are mapped in fractions; the data are dyadic."""

    def test_affine_map_holds_the_image_of_every_corner(self):
        """[-3, 1] x [2, 5] under W (3 x 2, both signs) and an offset of
        magnitude up to 20: each corner's image lies in the result, whose
        (m, ell, epsilon) and stated guarantee are the operand's. Catches an
        offset that is not passed on (the image moved by up to 20)."""
        lo, hi = [-3.0, 2.0], [1.0, 5.0]
        pbox = ProbabilisticBox(np.array(lo), np.array(hi), m=1000, ell=990,
                                epsilon=0.02)
        W = [[2.0, -1.0], [0.5, 0.25], [-1.0, -3.0]]
        b = [10.0, -20.0, 4.0]
        result = pbox.affine_map(np.array(W), np.array(b))
        assert isinstance(result, ProbabilisticBox) and result.dim == 3
        assert (result.m, result.ell, result.epsilon) == (1000, 990, 0.02)
        assert result.get_guarantee() == pbox.get_guarantee()
        r_lo, r_hi = result.lb.ravel(), result.ub.ravel()
        for corner in product(*zip(lo, hi)):
            for i in range(3):
                y = Fraction(b[i]) + sum(Fraction(W[i][j]) * Fraction(corner[j])
                                         for j in range(2))
                assert Fraction(r_lo[i]) <= y <= Fraction(r_hi[i]), (
                    corner, i, float(y), r_lo[i], r_hi[i])

    def test_sum_of_two_probabilistic_boxes_states_no_more_coverage_than_either(self):
        """Coverages 0.99 and 0.98: the sum needs both statements, so it may
        state at most 0.98 (the full promise, the union bound, is
        ``tests/contracts/test_probabilistic_contract.py::test_sum_of_two_probabilistic_boxes_adds_the_miss_probabilities``,
        marked PROB-1). The bounds hold every sum of corners. Catches a sum
        that keeps the smaller epsilon."""
        a = ProbabilisticBox(np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                             m=1000, ell=999, epsilon=0.01)
        b = ProbabilisticBox(np.array([0.0, -1.0]), np.array([2.0, 0.0]),
                             m=1000, ell=999, epsilon=0.02)
        total = a.minkowski_sum(b)
        assert total.coverage <= min(a.coverage, b.coverage), (
            total.coverage, a.coverage, b.coverage)
        assert np.all(total.lb.ravel() <= [0.0, -1.0])
        assert np.all(total.ub.ravel() >= [3.0, 1.0])

    def test_sum_with_one_epsilon_states_no_more_confidence_than_either(self):
        """Same m and epsilon, ranks 1999 and 1990: confidences about
        0.99999996 and 0.98. The sum may state at most the smaller (the full
        promise, a.confidence + b.confidence - 1, is
        ``tests/contracts/test_probabilistic_contract.py::test_confidence_of_a_sum_is_at_most_the_union_bound_of_its_operands``,
        marked PROB-3). Catches a sum that keeps the larger rank."""
        a = ProbabilisticBox(np.array([0.0]), np.array([1.0]), m=2000,
                             ell=1999, epsilon=0.01)
        b = ProbabilisticBox(np.array([0.0]), np.array([1.0]), m=2000,
                             ell=1990, epsilon=0.01)
        if not a.confidence > b.confidence + 0.01:
            pytest.fail("the operands' confidences should differ clearly")
        total = a.minkowski_sum(b)
        assert total.confidence <= min(a.confidence, b.confidence), (
            total.confidence, a.confidence, b.confidence)

    def test_to_star_holds_the_box(self):
        """The Star keeps every corner of [-3, 1] x [2, 5] and not the point
        (1.5, 3), by LP membership on the Star's own rows; the conversion
        warns that the guarantee is lost."""
        pbox = ProbabilisticBox(np.array([-3.0, 2.0]), np.array([1.0, 5.0]),
                                m=100, ell=99, epsilon=0.01)
        with pytest.warns(UserWarning, match="loses guarantee metadata"):
            star = pbox.to_star()
        for corner in product([-3.0, 1.0], [2.0, 5.0]):
            assert distance(star, corner) <= LP_TOL, corner
        assert distance(star, [1.5, 3.0]) >= 0.5 - LP_TOL
