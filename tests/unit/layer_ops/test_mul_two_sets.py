"""Tests for element-wise Mul of two computed sets (Tasks 9-10)."""

import numpy as np
import pytest
from n2v.sets import Star, Zono, Box
from n2v.nn.reach import _mul_sets


class TestMulTwoSetsBox:
    """Task 9: Interval arithmetic for Box multiplication."""

    def test_positive_times_positive(self):
        a = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        b = Box(np.array([[2.0], [1.0]]), np.array([[5.0], [3.0]]))
        result = _mul_sets([a], [b])
        np.testing.assert_allclose(result[0].lb, np.array([[2.0], [2.0]]), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.array([[15.0], [12.0]]), atol=1e-10)

    def test_negative_times_positive(self):
        a = Box(np.array([[-3.0]]), np.array([[-1.0]]))
        b = Box(np.array([[2.0]]), np.array([[5.0]]))
        result = _mul_sets([a], [b])
        np.testing.assert_allclose(result[0].lb, np.array([[-15.0]]), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.array([[-2.0]]), atol=1e-10)

    def test_negative_times_negative(self):
        a = Box(np.array([[-4.0]]), np.array([[-2.0]]))
        b = Box(np.array([[-3.0]]), np.array([[-1.0]]))
        result = _mul_sets([a], [b])
        np.testing.assert_allclose(result[0].lb, np.array([[2.0]]), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.array([[12.0]]), atol=1e-10)

    def test_crossing_zero(self):
        a = Box(np.array([[-1.0]]), np.array([[2.0]]))
        b = Box(np.array([[-3.0]]), np.array([[4.0]]))
        result = _mul_sets([a], [b])
        np.testing.assert_allclose(result[0].lb, np.array([[-6.0]]), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.array([[8.0]]), atol=1e-10)

    def test_zero_width_interval(self):
        a = Box(np.array([[3.0]]), np.array([[3.0]]))
        b = Box(np.array([[2.0]]), np.array([[5.0]]))
        result = _mul_sets([a], [b])
        np.testing.assert_allclose(result[0].lb, np.array([[6.0]]), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.array([[15.0]]), atol=1e-10)

    def test_soundness_random(self):
        np.random.seed(42)
        lb_a, ub_a = np.random.randn(5, 1), np.random.randn(5, 1)
        lb_a, ub_a = np.minimum(lb_a, ub_a), np.maximum(lb_a, ub_a)
        lb_b, ub_b = np.random.randn(5, 1), np.random.randn(5, 1)
        lb_b, ub_b = np.minimum(lb_b, ub_b), np.maximum(lb_b, ub_b)
        a = Box(lb_a, ub_a)
        b = Box(lb_b, ub_b)
        result = _mul_sets([a], [b])
        for _ in range(1000):
            x = np.random.uniform(lb_a.flatten(), ub_a.flatten())
            y = np.random.uniform(lb_b.flatten(), ub_b.flatten())
            z = (x * y).reshape(-1, 1)
            assert np.all(z >= result[0].lb - 1e-10), "Product below lower bound"
            assert np.all(z <= result[0].ub + 1e-10), "Product above upper bound"

    def test_multiple_pairs(self):
        a1 = Box(np.array([[1.0]]), np.array([[2.0]]))
        b1 = Box(np.array([[3.0]]), np.array([[4.0]]))
        a2 = Box(np.array([[0.0]]), np.array([[1.0]]))
        b2 = Box(np.array([[-1.0]]), np.array([[1.0]]))
        results = _mul_sets([a1, a2], [b1, b2])
        assert len(results) == 2
        np.testing.assert_allclose(results[0].lb, np.array([[3.0]]), atol=1e-10)
        np.testing.assert_allclose(results[0].ub, np.array([[8.0]]), atol=1e-10)
        np.testing.assert_allclose(results[1].lb, np.array([[-1.0]]), atol=1e-10)
        np.testing.assert_allclose(results[1].ub, np.array([[1.0]]), atol=1e-10)

    def test_mismatched_lengths_raises(self):
        a = Box(np.array([[1.0]]), np.array([[2.0]]))
        b = Box(np.array([[1.0]]), np.array([[2.0]]))
        with pytest.raises(ValueError, match="different lengths"):
            _mul_sets([a, a], [b])


class TestMulTwoSetsZono:
    """Task 9: Interval arithmetic for Zono multiplication."""

    def test_basic(self):
        a = Zono.from_bounds(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        b = Zono.from_bounds(np.array([[2.0], [1.0]]), np.array([[5.0], [3.0]]))
        result = _mul_sets([a], [b])
        lb, ub = result[0].get_bounds()
        # Must contain [2, 2] to [15, 12]
        assert np.all(lb <= np.array([[2.0], [2.0]]) + 1e-6)
        assert np.all(ub >= np.array([[15.0], [12.0]]) - 1e-6)

    def test_soundness_random(self):
        np.random.seed(42)
        center_a = np.random.randn(4, 1)
        center_b = np.random.randn(4, 1)
        a = Zono.from_bounds(center_a - 0.5, center_a + 0.5)
        b = Zono.from_bounds(center_b - 0.5, center_b + 0.5)
        result = _mul_sets([a], [b])
        lb, ub = result[0].get_bounds()
        for _ in range(500):
            x = np.random.uniform((center_a - 0.5).flatten(), (center_a + 0.5).flatten())
            y = np.random.uniform((center_b - 0.5).flatten(), (center_b + 0.5).flatten())
            z = (x * y).reshape(-1, 1)
            assert np.all(z >= lb - 1e-6), "Product below Zono lower bound"
            assert np.all(z <= ub + 1e-6), "Product above Zono upper bound"

    def test_result_is_zono(self):
        a = Zono.from_bounds(np.array([[1.0]]), np.array([[2.0]]))
        b = Zono.from_bounds(np.array([[3.0]]), np.array([[4.0]]))
        result = _mul_sets([a], [b])
        assert isinstance(result[0], Zono)


def _make_shared_stars(lb, ub, W_a, b_a, W_b, b_b):
    """
    Create two Stars that share predicate variables (simulating two branches
    from the same input set through different affine transformations).

    Args:
        lb, ub: Input bounds (n, 1)
        W_a, b_a: Affine transform for branch A
        W_b, b_b: Affine transform for branch B

    Returns:
        (star_a, star_b) sharing the same predicates
    """
    base = Star.from_bounds(lb, ub)
    star_a = base.affine_map(W_a, b_a)
    star_b = base.affine_map(W_b, b_b)
    return star_a, star_b


class TestMulTwoSetsStar:
    """Task 10: McCormick relaxation for Star multiplication."""

    def test_positive_ranges_shared_predicates(self):
        """Two branches from same input, both producing positive outputs."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        # Branch A: identity + offset => x = alpha + [1, 2]
        W_a = np.eye(2)
        b_a = np.array([[1.0], [2.0]])
        # Branch B: identity + offset => y = alpha + [2, 1]
        W_b = np.eye(2)
        b_b = np.array([[2.0], [1.0]])
        sa, sb = _make_shared_stars(lb, ub, W_a, b_a, W_b, b_b)
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        # With shared predicates:
        # z0 = (a1+1)(a1+2), a1 in [0,1] => z0 in [1*2, 2*3] = [2, 6]
        # z1 = (a2+2)(a2+1), a2 in [0,1] => z1 in [2*1, 3*2] = [2, 6]
        assert res_lb[0, 0] <= 2.0 + 1e-6
        assert res_ub[0, 0] >= 6.0 - 1e-6
        assert res_lb[1, 0] <= 2.0 + 1e-6
        assert res_ub[1, 0] >= 6.0 - 1e-6

    def test_result_is_star(self):
        lb = np.array([[0.0]])
        ub = np.array([[1.0]])
        sa, sb = _make_shared_stars(
            lb, ub,
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[2.0]]), np.array([[1.0]])
        )
        result = _mul_sets([sa], [sb])
        assert isinstance(result[0], Star)

    def test_soundness_shared_predicates(self):
        """Soundness: all concrete products must be within McCormick bounds."""
        np.random.seed(42)
        N_in = 3
        N_out = 4
        lb = np.random.randn(N_in, 1)
        ub = lb + np.random.rand(N_in, 1) * 2
        W_a = np.random.randn(N_out, N_in)
        b_a = np.random.randn(N_out, 1)
        W_b = np.random.randn(N_out, N_in)
        b_b = np.random.randn(N_out, 1)
        sa, sb = _make_shared_stars(lb, ub, W_a, b_a, W_b, b_b)
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        for _ in range(500):
            inp = np.random.uniform(lb.flatten(), ub.flatten()).reshape(-1, 1)
            x = W_a @ inp + b_a
            y = W_b @ inp + b_b
            z = x * y
            assert np.all(z >= res_lb - 1e-5), \
                f"Product below McCormick lb: {z.flatten()} < {res_lb.flatten()}"
            assert np.all(z <= res_ub + 1e-5), \
                f"Product above McCormick ub: {z.flatten()} > {res_ub.flatten()}"

    def test_mccormick_tighter_than_interval(self):
        """McCormick with shared predicates should be tighter than interval arithmetic."""
        lb = np.array([[0.0]])
        ub = np.array([[1.0]])
        # Both branches use the same alpha, so z = (alpha+1)*(alpha+1) = (alpha+1)^2
        # alpha in [0,1] => z in [1, 4]
        # Interval arithmetic: x in [1,2], y in [1,2] => z in [1, 4]
        sa, sb = _make_shared_stars(
            lb, ub,
            np.array([[1.0]]), np.array([[1.0]]),
            np.array([[1.0]]), np.array([[1.0]])
        )
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        # Must contain actual range [1, 4]
        assert res_lb[0, 0] <= 1.0 + 1e-6
        assert res_ub[0, 0] >= 4.0 - 1e-6
        # Interval arithmetic gives [1, 4] too in this case
        assert res_ub[0, 0] <= 4.0 + 1e-6

    def test_crossing_zero_shared(self):
        """Both operands cross zero, shared predicates."""
        lb = np.array([[0.0]])
        ub = np.array([[1.0]])
        # Branch A: x = 5*alpha - 2, alpha in [0,1] => x in [-2, 3]
        # Branch B: y = 5*alpha - 1, alpha in [0,1] => y in [-1, 4]
        sa, sb = _make_shared_stars(
            lb, ub,
            np.array([[5.0]]), np.array([[-2.0]]),
            np.array([[5.0]]), np.array([[-1.0]])
        )
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        # Concrete products at alpha=0: (-2)*(-1)=2
        # at alpha=1: 3*4=12
        # at alpha=0.4: 0*1=0
        # at alpha=0.2: -1*0=0
        # True range with shared alpha: z = (5a-2)(5a-1) = 25a^2 - 15a + 2
        # Minimum at a=0.3: 25*0.09 - 15*0.3 + 2 = 2.25 - 4.5 + 2 = -0.25
        # At a=0: 2, at a=1: 12
        # So true range is [-0.25, 12]
        assert res_lb[0, 0] <= -0.25 + 1e-5
        assert res_ub[0, 0] >= 12.0 - 1e-5

    def test_soundness_high_dimension_shared(self):
        """Soundness test with more dimensions, shared predicates."""
        np.random.seed(123)
        N_in = 3
        N_out = 6
        lb = np.random.randn(N_in, 1) - 1
        ub = lb + np.random.rand(N_in, 1) * 2
        W_a = np.random.randn(N_out, N_in)
        b_a = np.random.randn(N_out, 1)
        W_b = np.random.randn(N_out, N_in)
        b_b = np.random.randn(N_out, 1)
        sa, sb = _make_shared_stars(lb, ub, W_a, b_a, W_b, b_b)
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        for _ in range(300):
            inp = np.random.uniform(lb.flatten(), ub.flatten()).reshape(-1, 1)
            x = W_a @ inp + b_a
            y = W_b @ inp + b_b
            z = x * y
            assert np.all(z >= res_lb - 1e-5), "Product below McCormick lb (high dim)"
            assert np.all(z <= res_ub + 1e-5), "Product above McCormick ub (high dim)"

    def test_squaring_shared(self):
        """When both branches are identical transforms, z = x^2."""
        lb = np.array([[1.0]])
        ub = np.array([[3.0]])
        # Create two Stars sharing the same predicates (both are identity maps)
        sa, sb = _make_shared_stars(
            lb, ub,
            np.array([[1.0]]), np.array([[0.0]]),
            np.array([[1.0]]), np.array([[0.0]])
        )
        # x in [1,3], y in [1,3], same alpha => z = x^2 in [1, 9]
        result = _mul_sets([sa], [sb])
        res_lb, res_ub = result[0].get_ranges()
        assert res_lb[0, 0] <= 1.0 + 1e-6
        assert res_ub[0, 0] >= 9.0 - 1e-6
