"""Tests for n2v.sets.volume."""

import numpy as np
import pytest

from n2v.sets.star import Star
from n2v.sets.volume import (
    VolumeEstimate,
    star_union_bbox,
    star_union_volume_mc,
    star_union_volume_sound_lower,
    star_volume,
)


def _axis_aligned_star(lo, hi):
    """Identity-basis Star whose image is the axis-aligned box [lo, hi]."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    dim = lo.shape[0]
    V = np.zeros((dim, dim + 1))
    V[:, 0] = 0.5 * (lo + hi)  # offset at center
    V[:, 1:] = np.diag(0.5 * (hi - lo))  # basis scales alpha in [-1, 1]
    plb = -np.ones(dim)
    pub = np.ones(dim)
    return Star(V=V, C=None, d=None, pred_lb=plb, pred_ub=pub)


class TestVolumeEstimateDataclass:
    def test_construct_with_required_fields(self):
        v = VolumeEstimate(
            mean=1.0, se=0.1, ci_low=0.8, ci_high=1.2,
            method='mc', n_samples=100,
        )
        assert v.mean == 1.0
        assert v.method == 'mc'
        assert v.meta == {}

    def test_construct_deterministic(self):
        v = VolumeEstimate(
            mean=1.0, se=None, ci_low=1.0, ci_high=1.0,
            method='exact', n_samples=None,
            meta={'notes': 'rank-deficient'},
        )
        assert v.se is None
        assert v.n_samples is None
        assert v.meta['notes'] == 'rank-deficient'


class TestStarUnionBbox:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            star_union_bbox([])

    def test_single_star_matches_star_box(self):
        s = _axis_aligned_star([-1.0, -2.0], [3.0, 4.0])
        lo, hi = star_union_bbox([s])
        np.testing.assert_allclose(lo, [-1.0, -2.0], atol=1e-8)
        np.testing.assert_allclose(hi, [3.0, 4.0], atol=1e-8)

    def test_two_disjoint_boxes(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([3.0, 3.0], [5.0, 5.0])
        lo, hi = star_union_bbox([a, b])
        np.testing.assert_allclose(lo, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(hi, [5.0, 5.0], atol=1e-8)

    def test_nested_boxes(self):
        outer = _axis_aligned_star([-2.0, -2.0], [2.0, 2.0])
        inner = _axis_aligned_star([-0.5, -0.5], [0.5, 0.5])
        lo, hi = star_union_bbox([outer, inner])
        np.testing.assert_allclose(lo, [-2.0, -2.0], atol=1e-8)
        np.testing.assert_allclose(hi, [2.0, 2.0], atol=1e-8)

    def test_inconsistent_dims_raises(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            star_union_bbox([a, b])


class TestStarVolumeMC:
    def test_identity_unit_box_2d(self):
        s = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_volume(s, method='mc', n_samples=50_000, seed=0)
        assert v.method == 'mc'
        assert v.n_samples == 50_000
        assert v.ci_low <= 1.0 <= v.ci_high

    def test_identity_unit_box_3d(self):
        s = _axis_aligned_star([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        v = star_volume(s, method='mc', n_samples=100_000, seed=0)
        assert v.ci_low <= 1.0 <= v.ci_high

    def test_determinant_scaling(self):
        # Star whose image is [0,2] x [0,3] x [0,5]; vol = 30.
        s = _axis_aligned_star([0.0, 0.0, 0.0], [2.0, 3.0, 5.0])
        v = star_volume(s, method='mc', n_samples=200_000, seed=0)
        assert v.ci_low <= 30.0 <= v.ci_high

    def test_half_box_via_cd(self):
        # [-1,1]^2 cut by alpha[0] >= 0 -> upper half, box_vol = 4, vol = 2.
        dim = 2
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        plb = -np.ones(dim)
        pub = np.ones(dim)
        # alpha[0] >= 0 <=> -alpha[0] <= 0 <=> C=[[-1, 0]], d=[0].
        C = np.array([[-1.0, 0.0]])
        d = np.array([0.0])
        s = Star(V=V, C=C, d=d, pred_lb=plb, pred_ub=pub)
        v = star_volume(s, method='mc', n_samples=100_000, seed=0)
        assert v.ci_low <= 2.0 <= v.ci_high

    def test_empty_predicate_box(self):
        dim = 2
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        plb = np.array([1.0, 1.0])
        pub = np.array([-1.0, -1.0])
        s = Star(V=V, C=None, d=None, pred_lb=plb, pred_ub=pub)
        v = star_volume(s, method='mc', n_samples=1_000, seed=0)
        assert v.mean == 0.0

    def test_rank_deficient_returns_zero(self):
        # 2D output with a 2x2 rank-1 basis. Image is a line, measure 0.
        V = np.zeros((2, 3))
        V[:, 1:] = np.array([[1.0, 0.0], [1.0, 0.0]])  # rank 1
        s = Star(V=V, C=None, d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
        v = star_volume(s, method='mc', n_samples=1_000, seed=0)
        assert v.mean == 0.0
        assert 'rank-deficient' in v.meta.get('notes', '')

    def test_invalid_method_raises(self):
        s = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        with pytest.raises(ValueError):
            star_volume(s, method='bogus')


class TestStarVolumeExact:
    def test_identity_unit_box_2d(self):
        s = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_volume(s, method='exact')
        assert v.method == 'exact'
        assert v.se is None
        np.testing.assert_allclose(v.mean, 1.0, atol=1e-6)

    def test_identity_unit_box_3d(self):
        s = _axis_aligned_star([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        v = star_volume(s, method='exact')
        np.testing.assert_allclose(v.mean, 1.0, atol=1e-6)

    def test_determinant_scaling_3d(self):
        s = _axis_aligned_star([0.0, 0.0, 0.0], [2.0, 3.0, 5.0])
        v = star_volume(s, method='exact')
        np.testing.assert_allclose(v.mean, 30.0, atol=1e-6)

    def test_half_box_via_cd_2d(self):
        dim = 2
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        C = np.array([[-1.0, 0.0]])
        d = np.array([0.0])
        s = Star(V=V, C=C, d=d, pred_lb=-np.ones(dim), pred_ub=np.ones(dim))
        v = star_volume(s, method='exact')
        np.testing.assert_allclose(v.mean, 2.0, atol=1e-6)

    def test_empty_predicate_returns_zero(self):
        dim = 2
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        s = Star(
            V=V, C=None, d=None,
            pred_lb=np.array([1.0, 1.0]),
            pred_ub=np.array([-1.0, -1.0]),
        )
        v = star_volume(s, method='exact')
        assert v.mean == 0.0

    def test_rank_deficient_returns_zero(self):
        V = np.zeros((2, 3))
        V[:, 1:] = np.array([[1.0, 0.0], [1.0, 0.0]])
        s = Star(V=V, C=None, d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
        v = star_volume(s, method='exact')
        assert v.mean == 0.0


class TestStarVolumeMCExactAgreement:
    def test_ci_contains_exact_across_random_stars(self):
        """20 random full-rank 3D Stars: MC 99% CI should contain exact
        for at least 18/20 (with some slack for 1% tail events)."""
        rng = np.random.default_rng(123)
        dim = 3
        covered = 0
        for _ in range(20):
            basis = rng.normal(size=(dim, dim))
            while abs(np.linalg.det(basis)) < 0.2:
                basis = rng.normal(size=(dim, dim))
            V = np.zeros((dim, dim + 1))
            V[:, 0] = rng.uniform(-1, 1, size=dim)
            V[:, 1:] = basis
            plb = rng.uniform(-1.0, 0.0, size=dim)
            pub = rng.uniform(0.0, 1.0, size=dim)
            # Optional C/d that is feasible at alpha=0.
            s = Star(V=V, C=None, d=None, pred_lb=plb, pred_ub=pub)
            ex = star_volume(s, method='exact').mean
            mc = star_volume(s, method='mc', n_samples=50_000, seed=0)
            if mc.ci_low <= ex <= mc.ci_high:
                covered += 1
        assert covered >= 18, f"CI coverage {covered}/20 too low"


class TestStarUnionVolumeMC:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            star_union_volume_mc([])

    def test_single_star_matches_star_volume(self):
        s = _axis_aligned_star([0.0, 0.0], [2.0, 3.0])
        v_single = star_volume(s, method='exact').mean
        v_union = star_union_volume_mc(
            [s], n_samples=30_000, batch_size=5_000, seed=0,
        )
        assert v_union.method == 'mc_union'
        assert v_union.ci_low <= v_single <= v_union.ci_high

    def test_two_disjoint_boxes_equals_sum(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])  # vol 1
        b = _axis_aligned_star([5.0, 5.0], [6.0, 6.0])  # vol 1
        v = star_union_volume_mc([a, b], n_samples=30_000, batch_size=5_000, seed=0)
        assert v.ci_low <= 2.0 <= v.ci_high

    def test_two_identical_equals_one(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_union_volume_mc([a, b], n_samples=30_000, batch_size=5_000, seed=0)
        assert v.ci_low <= 1.0 <= v.ci_high

    def test_two_half_overlap(self):
        # Squares [0,1]^2 and [0.5, 1.5]^2. Union = 2 - 0.25 = 1.75.
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.5, 0.5], [1.5, 1.5])
        v = star_union_volume_mc([a, b], n_samples=40_000, batch_size=5_000, seed=0)
        assert v.ci_low <= 1.75 <= v.ci_high

    def test_honors_provided_bbox(self):
        # Provide a bbox that's bigger than the union; volume should still
        # come back as ~1 (frac shrinks, bbox_vol grows -> product same).
        s = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        lo = np.array([-5.0, -5.0])
        hi = np.array([6.0, 6.0])
        v = star_union_volume_mc(
            [s], bbox=(lo, hi), n_samples=40_000, batch_size=5_000, seed=0,
        )
        assert v.ci_low <= 1.0 <= v.ci_high

    def test_algebraic_path_matches_lp_path(self):
        """Both contains methods should give statistically-equivalent volumes."""
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.5, 0.5], [1.5, 1.5])
        v_lp = star_union_volume_mc(
            [a, b], n_samples=20_000, batch_size=5_000, seed=0,
            contains_method='lp',
        )
        v_alg = star_union_volume_mc(
            [a, b], n_samples=20_000, batch_size=5_000, seed=0,
            contains_method='algebraic',
        )
        # Same seed + same bbox + same sample points -> identical inside counts.
        assert v_lp.mean == pytest.approx(v_alg.mean, abs=1e-10)


class TestStarUnionSoundLower:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            star_union_volume_sound_lower([])

    def test_unknown_method_raises(self):
        s = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        with pytest.raises(ValueError):
            star_union_volume_sound_lower([s], method='bogus')

    def test_single_star_matches_star_exact_volume(self):
        s = _axis_aligned_star([0.0, 0.0], [2.0, 3.0])
        exact = star_volume(s, method='exact').mean
        lower = star_union_volume_sound_lower([s]).mean
        np.testing.assert_allclose(lower, exact, atol=1e-6)

    def test_two_identical_equals_one(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_union_volume_sound_lower([a, b]).mean
        np.testing.assert_allclose(v, 1.0, atol=1e-6)

    def test_max_star_picks_larger(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])  # vol 1
        b = _axis_aligned_star([0.0, 0.0], [2.0, 3.0])  # vol 6
        v = star_union_volume_sound_lower([a, b])
        np.testing.assert_allclose(v.mean, 6.0, atol=1e-6)
        assert v.meta['argmax_star'] == 1

    def test_lower_bound_below_mc_for_disjoint_union(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])  # vol 1
        b = _axis_aligned_star([5.0, 5.0], [6.0, 6.0])  # vol 1
        lower = star_union_volume_sound_lower([a, b]).mean
        mc = star_union_volume_mc(
            [a, b], n_samples=20_000, batch_size=5_000, seed=0,
        )
        assert lower <= mc.ci_high
