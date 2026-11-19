"""Phase 1.1: correctness of `sample_l_inf_ball` (uniform on L_inf ball).

The conformal guarantee is against the actual distribution the sampler
produces. Anything other than uniform breaks the paper's 'P_X = Uniform'
claim, so this audit is validity-relevant, not code hygiene.

The real signature is ``sample_l_inf_ball(x_center, radius, n_samples, seed, dim)``
with ``x_center`` a torch tensor, and returns a torch tensor. We convert to
numpy for statistical tests.
"""

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from n2v.probabilistic.flow.sampling import sample_l_inf_ball


def _sample(center_np, radius, n, seed):
    """Helper: convert numpy center to torch and return numpy output."""
    center_t = torch.as_tensor(center_np, dtype=torch.float32)
    x = sample_l_inf_ball(
        x_center=center_t, radius=radius, n_samples=n, seed=seed,
        dim=center_t.shape[0],
    )
    return x.numpy()


class TestSupportAndShape:
    def test_all_samples_inside_linf_ball(self):
        center = np.array([0.5, -0.3, 1.0])
        radius = 0.4
        x = _sample(center, radius, 10_000, seed=0)
        assert x.shape == (10_000, 3)
        delta = x - center
        assert np.abs(delta).max() <= radius + 1e-5

    def test_seed_determinism(self):
        x1 = _sample(np.zeros(2), 1.0, 500, seed=7)
        x2 = _sample(np.zeros(2), 1.0, 500, seed=7)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        x1 = _sample(np.zeros(2), 1.0, 500, seed=7)
        x2 = _sample(np.zeros(2), 1.0, 500, seed=8)
        assert not np.array_equal(x1, x2)


class TestUniformity:
    def test_per_dim_is_uniform(self):
        radius = 1.0
        center = np.zeros(3)
        x = _sample(center, radius, 50_000, seed=0)
        # Each dim should be Uniform([-radius, radius]).
        for k in range(3):
            u = (x[:, k] + radius) / (2 * radius)  # to U(0,1)
            ks_stat, p = kstest(u, 'uniform')
            assert p > 0.001, f"dim {k} KS p={p} (stat={ks_stat})"

    def test_empirical_moments(self):
        radius = 1.0
        center = np.array([0.2, -0.5])
        x = _sample(center, radius, 50_000, seed=0)
        emp_mean = x.mean(axis=0)
        emp_var = x.var(axis=0)
        expected_var = radius ** 2 / 3.0
        np.testing.assert_allclose(emp_mean, center, atol=0.02)
        np.testing.assert_allclose(emp_var, expected_var, rtol=0.05)

    def test_dimension_independence(self):
        x = _sample(np.zeros(3), 1.0, 50_000, seed=0)
        corr = np.corrcoef(x, rowvar=False)
        off_diag = corr[~np.eye(3, dtype=bool)]
        assert np.abs(off_diag).max() < 0.03


class TestDisjointSeedOffsets:
    def test_train_calib_test_disjoint(self):
        center = np.zeros(2)
        radius = 1.0
        base = 123
        x_train = _sample(center, radius, 1000, seed=base)
        x_calib = _sample(center, radius, 1000, seed=base + 1_000_000)
        x_test = _sample(center, radius, 1000, seed=base + 2_000_000)
        assert not np.array_equal(x_train, x_calib)
        assert not np.array_equal(x_train, x_test)
        assert not np.array_equal(x_calib, x_test)


# ---------------------------------------------------------------------------
# sample_box — uniform on an arbitrary axis-aligned box [lb, ub]
# ---------------------------------------------------------------------------


class TestSampleBox:
    def test_output_shape(self):
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([-1.0, 0.0, 2.0])
        ub = torch.tensor([1.0, 3.0, 5.0])
        x = sample_box(lb, ub, n_samples=100, seed=0)
        assert x.shape == (100, 3)
        assert x.dtype == lb.dtype

    def test_support(self):
        """Every sample must lie inside [lb, ub] componentwise."""
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([-2.0, 0.5, 10.0])
        ub = torch.tensor([-1.0, 0.7, 20.0])
        x = sample_box(lb, ub, n_samples=5000, seed=0).numpy()
        assert (x >= lb.numpy() - 1e-6).all()
        assert (x <= ub.numpy() + 1e-6).all()

    def test_determinism(self):
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        x1 = sample_box(lb, ub, n_samples=100, seed=42)
        x2 = sample_box(lb, ub, n_samples=100, seed=42)
        assert torch.allclose(x1, x2)

    def test_different_seeds_differ(self):
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([-1.0, -1.0])
        ub = torch.tensor([1.0, 1.0])
        x1 = sample_box(lb, ub, n_samples=100, seed=0)
        x2 = sample_box(lb, ub, n_samples=100, seed=1)
        assert not torch.allclose(x1, x2)

    def test_marginal_uniformity_ks(self):
        """Per-dim marginal should pass a loose KS test against U[lb, ub]."""
        from scipy.stats import kstest
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([-2.0, 5.0])
        ub = torch.tensor([3.0, 7.0])
        n = 10_000
        x = sample_box(lb, ub, n_samples=n, seed=0).numpy()
        for k in range(2):
            # KS vs U[lb[k], ub[k]]
            cdf = lambda v, l=lb[k].item(), u=ub[k].item(): (v - l) / (u - l)
            stat, p = kstest(x[:, k], cdf)
            # 10k samples with a true uniform should comfortably pass p>0.01
            assert p > 0.01, f'dim {k} KS p-value too small: {p}'

    def test_rejects_lb_above_ub(self):
        """Degenerate input where lb > ub should raise."""
        from n2v.probabilistic.flow.sampling import sample_box
        lb = torch.tensor([1.0, 0.0])
        ub = torch.tensor([0.0, 1.0])  # first dim violates lb <= ub
        with pytest.raises(ValueError):
            sample_box(lb, ub, n_samples=10, seed=0)

    def test_agrees_with_sample_l_inf_ball_on_symmetric_case(self):
        """sample_box with lb = center - r, ub = center + r should match
        sample_l_inf_ball to floating-point tolerance (same underlying RNG
        sequence)."""
        from n2v.probabilistic.flow.sampling import sample_box, sample_l_inf_ball
        center = torch.tensor([0.5, -0.3, 2.0])
        r = 0.25
        n = 500
        x_box = sample_box(center - r, center + r, n_samples=n, seed=7)
        x_ball = sample_l_inf_ball(center, r, n_samples=n, seed=7, dim=3)
        # Both use the same uniform-per-dim factorization; with the same
        # seed and the same generator construction they must agree.
        assert torch.allclose(x_box, x_ball, atol=1e-6)


# ---------------------------------------------------------------------------
# Boxes sample_box cannot sample
# ---------------------------------------------------------------------------
import math  # noqa: E402

from tests.known_bugs import DID_NOT_RAISE, known_bug  # noqa: E402


class TestSampleBoxRefusals:
    @pytest.mark.parametrize("lb, ub", [
        pytest.param([-1.0, 0.0, 0.5], [2.0], id="shapes-differ"),
        pytest.param([-math.inf, 0.0], [1.0, 1.0], id="infinite-bound",
                     marks=known_bug(13, "sample_box draws from a box with an "
                                         "infinite bound and returns NaN or "
                                         "infinite samples instead of refusing; "
                                         "no uniform law exists on an "
                                         "unbounded box", raises=DID_NOT_RAISE)),
    ])
    def test_refuses_a_box_it_cannot_sample(self, lb, ub):
        """Bounds of different shapes (documented) and an unbounded box
        raise ValueError. A shape check that compares only the number of
        axes broadcasts a one-entry ub silently (every lb entry is below it,
        so the order check cannot raise in its place); an unbounded box
        today gives NaN samples, and a coverage estimated on them is 0."""
        from n2v.probabilistic.flow.sampling import sample_box
        with pytest.raises(ValueError):
            sample_box(torch.tensor(lb), torch.tensor(ub), n_samples=10, seed=0)


class TestSampleLInfBallDimension:
    @pytest.mark.parametrize("centre", [
        pytest.param([0.0, 1.0, -2.0], id="centre-of-dim-entries"),
        pytest.param([0.0, 1.0], id="centre-of-fewer-entries", marks=known_bug(
            23, "sample_l_inf_ball with a centre of 2 entries and dim=3 raises "
                "a torch RuntimeError from broadcasting instead of a ValueError "
                "naming the mismatch", raises=RuntimeError)),
    ])
    def test_dim_must_match_the_centre(self, centre):
        """dim must equal the number of centre entries (documented). A
        matching centre gives n samples of dim coordinates inside the ball;
        a centre of 2 entries with dim = 3 must raise ValueError."""
        x_center = torch.tensor(centre)
        if x_center.numel() == 3:
            x = sample_l_inf_ball(x_center, 0.5, n_samples=50, seed=0, dim=3)
            assert x.shape == (50, 3)
            assert ((x - x_center).abs() <= 0.5).all()
        else:
            with pytest.raises(ValueError):
                sample_l_inf_ball(x_center, 0.5, n_samples=50, seed=0, dim=3)
