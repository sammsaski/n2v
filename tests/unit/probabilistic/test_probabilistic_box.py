"""
Unit tests for ProbabilisticBox class.
"""

import pytest
import numpy as np
from scipy.stats import beta

from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.sets.box import Box


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

    def test_inherits_from_box(self):
        """Test that ProbabilisticBox is a subclass of Box."""
        assert issubclass(ProbabilisticBox, Box)

    def test_isinstance_of_box(self):
        """Test that ProbabilisticBox instances are also Box instances."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        assert isinstance(pbox, Box)
        assert isinstance(pbox, ProbabilisticBox)

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

    def test_minkowski_sum_with_probabilistic_box(self):
        """Test Minkowski sum with another ProbabilisticBox uses conservative combination."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        pbox1 = ProbabilisticBox(lb, ub, m=1000, ell=999, epsilon=0.01)
        pbox2 = ProbabilisticBox(lb, ub, m=500, ell=490, epsilon=0.02)

        result = pbox1.minkowski_sum(pbox2)

        # Should use conservative (worse) parameters
        assert result.m == min(pbox1.m, pbox2.m)  # 500
        assert result.ell == min(pbox1.ell, pbox2.ell)  # 490
        assert result.epsilon == max(pbox1.epsilon, pbox2.epsilon)  # 0.02


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
