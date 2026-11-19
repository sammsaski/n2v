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
from tests.known_bugs import known_bug


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


class TestStarVolumeUnusedPredicates:
    """A predicate variable that no basis vector and no constraint row uses
    leaves the set unchanged, so its box must not scale the volume.
    Star.fresh_from_bounds leads with such a variable (box [0, t], t in
    (1, 2)), which made the volume t times too large."""

    @pytest.mark.parametrize('method', ['mc', 'exact'])
    def test_fresh_from_bounds_matches_from_bounds(self, method):
        lb, ub = np.zeros(2), np.array([1.0, 2.0])
        s = Star.fresh_from_bounds(lb, ub)
        assert float(s.predicate_ub[0, 0]) > 1.0
        v = star_volume(s, method=method, n_samples=2_000, seed=0)
        np.testing.assert_allclose(v.mean, 2.0, rtol=1e-9)

    @pytest.mark.parametrize('method', ['mc', 'exact'])
    def test_zero_width_unused_variable(self, method):
        s = _axis_aligned_star([0.0, 0.0], [1.0, 2.0])
        V = np.hstack([s.V, np.zeros((2, 1))])
        s = Star(V=V, C=None, d=None, pred_lb=np.array([-1.0, -1.0, 3.0]),
                 pred_ub=np.array([1.0, 1.0, 3.0]))
        v = star_volume(s, method=method, n_samples=2_000, seed=0)
        np.testing.assert_allclose(v.mean, 2.0, rtol=1e-9)


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

    # 3000 samples: the Hoeffding 99 % half-width is 0.030 in fraction, 0.067
    # in volume for the half-overlap case, so 1.5, 1.75 and 2.0 stay separated
    # at a tenth of the LP count of the original 30k-40k samples.
    def test_single_star_matches_star_volume(self):
        s = _axis_aligned_star([0.0, 0.0], [2.0, 3.0])
        v_single = star_volume(s, method='exact').mean
        v_union = star_union_volume_mc(
            [s], n_samples=3_000, batch_size=1_000, seed=0,
        )
        assert v_union.method == 'mc_union'
        assert v_union.ci_low <= v_single <= v_union.ci_high

    def test_two_disjoint_boxes_equals_sum(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])  # vol 1
        b = _axis_aligned_star([5.0, 5.0], [6.0, 6.0])  # vol 1
        v = star_union_volume_mc([a, b], n_samples=3_000, batch_size=1_000, seed=0)
        assert v.ci_low <= 2.0 <= v.ci_high

    def test_two_identical_equals_one(self):
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_union_volume_mc([a, b], n_samples=3_000, batch_size=1_000, seed=0)
        assert v.ci_low <= 1.0 <= v.ci_high

    def test_two_half_overlap(self):
        # Squares [0,1]^2 and [0.5, 1.5]^2. Union = 2 - 0.25 = 1.75.
        a = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        b = _axis_aligned_star([0.5, 0.5], [1.5, 1.5])
        v = star_union_volume_mc([a, b], n_samples=3_000, batch_size=1_000, seed=0)
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
        # 3000 draws (was 20000, 14 s): the comparison is exact (same draws),
        # so the instance size does not change what it catches.
        v_lp = star_union_volume_mc(
            [a, b], n_samples=3_000, batch_size=1_000, seed=0,
            contains_method='lp',
        )
        v_alg = star_union_volume_mc(
            [a, b], n_samples=3_000, batch_size=1_000, seed=0,
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


def _cut_parallelogram():
    """x = (5, -7) + [[2, 3], [1, -1]] a, a in [0.5, 2] x [1, 3], cut by
    a0 + a1 <= 3.5 and a0 - a1 <= 0.5: about 58 % of the predicate box, and
    one Monte Carlo draw is 0 or (box volume 3) x (|det| 5) = 15."""
    return Star(V=np.array([[5.0, 2.0, 3.0], [-7.0, 1.0, -1.0]]),
                C=np.array([[1.0, 1.0], [1.0, -1.0]]), d=np.array([3.5, 0.5]),
                pred_lb=np.array([0.5, 1.0]), pred_ub=np.array([2.0, 3.0]))


def _diamond_and_bar():
    """|x| + |y| <= 1 and the box [0, 2] x [-0.5, 0.5]."""
    diamond = Star(V=np.array([[0.0, 0.5, 0.5], [0.0, 0.5, -0.5]]), C=None,
                   d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
    return [diamond, _axis_aligned_star([0.0, -0.5], [2.0, 0.5])]


class TestMonteCarloUnitValues:
    """Values of the Monte Carlo estimates that do not decide whether the
    interval holds the volume, computed by hand: the standard error, the
    number of draws, the seed."""

    def test_star_volume_standard_error_is_the_binomial_error_of_the_volume(self):
        """se = sqrt(f (1 - f) / n) times the range of one draw (15), with f
        the inside fraction. Catches a standard error scaled by the box
        volume without the Jacobian |det| = 5."""
        v = star_volume(_cut_parallelogram(), method='mc', n_samples=4_000,
                        seed=0)
        frac = v.mean / 15.0
        if not 0.0 < frac < 1.0:
            pytest.fail(f"inside fraction {frac}: the case needs one in (0, 1)")
        assert v.se == pytest.approx(
            np.sqrt(frac * (1 - frac) / 4_000) * 15.0, rel=1e-9)

    def test_union_standard_error_is_the_binomial_error_of_the_volume(self):
        """With the sampling box [-1, 2] x [-1, 1] (area 6) given: se =
        sqrt(f (1 - f) / n) times 6. Catches the standard error of the
        inside fraction reported as that of the volume."""
        v = star_union_volume_mc(
            _diamond_and_bar(), bbox=(np.array([-1.0, -1.0]),
                                      np.array([2.0, 1.0])),
            n_samples=2_000, batch_size=2_000, seed=0,
            contains_method='algebraic')
        frac = v.mean / 6.0
        if not 0.0 < frac < 1.0:
            pytest.fail(f"inside fraction {frac}: the case needs one in (0, 1)")
        assert v.se == pytest.approx(
            np.sqrt(frac * (1 - frac) / 2_000) * 6.0, rel=1e-9)

    def test_last_partial_batch_draws_only_the_remaining_samples(self):
        """1100 draws in batches of 500: two full batches and one of 100.
        Catches a last batch that draws a full batch (1500 reported, and an
        interval computed for draws that were not asked for)."""
        v = star_union_volume_mc(_diamond_and_bar(), n_samples=1_100,
                                 batch_size=500, seed=0,
                                 contains_method='algebraic')
        assert v.n_samples == 1_100

    def test_seed_selects_the_draws(self):
        """Ten seeds give ten independent estimates of an inside fraction
        near 0.58 from 1000 draws: the chance that all ten agree is below
        0.026^9 < 1e-14. Eight seeds of the union estimate (fraction near
        0.54 from 300 draws) all agree with chance below 0.047^7 < 1e-9. The
        same seed gives the same estimate. Catches a sampler of either
        function seeded with a constant instead of ``seed``."""
        star = _cut_parallelogram()
        means = [star_volume(star, method='mc', n_samples=1_000, seed=k).mean
                 for k in range(10)]
        assert len(set(means)) > 1, means
        assert star_volume(star, method='mc', n_samples=1_000,
                           seed=4).mean == means[4]
        union = [star_union_volume_mc(_diamond_and_bar(), n_samples=300,
                                      batch_size=300, seed=k,
                                      contains_method='algebraic').mean
                 for k in range(8)]
        assert len(set(union)) > 1, union

    @pytest.mark.parametrize('function, n_samples, confidence', [
        pytest.param('star', 1, 0.99, id='star-one-draw'),
        pytest.param('star', 0, 0.99, id='star-no-draws', marks=known_bug(
            8, "star_volume(method='mc', n_samples=0) divides the inside "
               "count by zero (ZeroDivisionError) where a ValueError or an "
               "unbounded interval is due; _hoeffding_half_width's guard for "
               "no draws is never reached", raises=ZeroDivisionError)),
        pytest.param('star', 100, 1.0, id='star-confidence-one',
                     marks=known_bug(
            8, "the Monte Carlo volume functions take confidence=1 into "
               "log(2 / (1 - confidence)) and raise ZeroDivisionError where a "
               "ValueError or an unbounded interval is due",
            raises=ZeroDivisionError)),
        pytest.param('union', 1, 0.99, id='union-one-draw'),
        pytest.param('union', 0, 0.99, id='union-no-draws', marks=known_bug(
            8, "star_union_volume_mc(n_samples=0) divides the inside count "
               "by zero (ZeroDivisionError) where a ValueError or an "
               "unbounded interval is due", raises=ZeroDivisionError)),
    ])
    def test_a_request_without_a_finite_interval_is_refused_or_unbounded(
            self, function, n_samples, confidence):
        """No draws, or confidence 1, admit no finite Hoeffding interval
        (``_hoeffding_half_width`` returns an infinite half-width for no
        draws). The call must refuse with ``ValueError`` or return an
        interval that holds the area 6 of [0, 2] x [0, 3]. One draw at 99 %
        is the passing neighbour: its half-width 6 sqrt(ln(200) / 2) = 9.8
        exceeds the range of a draw, so its interval holds 6 whatever the
        draw. Catches a zero option value read as a division by zero."""
        box = Star.from_bounds(np.zeros((2, 1)), np.array([[2.0], [3.0]]))
        try:
            if function == 'star':
                v = star_volume(box, method='mc', n_samples=n_samples,
                                confidence=confidence)
            else:
                v = star_union_volume_mc([box], n_samples=n_samples,
                                         batch_size=10, confidence=confidence,
                                         contains_method='algebraic')
        except ValueError:
            return
        assert v.ci_low <= 6.0 <= v.ci_high, v

    @pytest.mark.parametrize('method', ['exact', 'mc'])
    def test_empty_star_has_volume_zero(self, method):
        """``Star()`` holds no point. Catches an empty-star test that looks
        only for ``V is None`` (the empty constructor stores a 0 x 0 array,
        on which the Monte Carlo branch counts every draw inside a box of
        volume 1)."""
        v = star_volume(Star(), method=method, n_samples=100)
        assert v.mean == 0.0 and v.ci_high == 0.0, v


class TestStarUnionSoundLowerArgmax:
    def test_max_star_names_the_largest_member_when_it_comes_first(self):
        """[[0, 2] x [0, 3], [0, 1]^2]: the bound is 6 and names member 0.
        Catches an argmax that is updated for every member, not only for a
        larger one, and so names the last member."""
        big = _axis_aligned_star([0.0, 0.0], [2.0, 3.0])
        small = _axis_aligned_star([0.0, 0.0], [1.0, 1.0])
        v = star_union_volume_sound_lower([big, small])
        assert v.meta['argmax_star'] == 0, v.meta
        assert v.mean == pytest.approx(6.0, rel=1e-9)


class TestVolumeZeroRegimes:
    def test_sound_lower_of_members_of_volume_zero_is_zero(self):
        """``Star()`` and the diagonal segment of a rank-deficient square
        basis: no member has volume, so the bound is 0 and names no member
        (-1). Catches an argmax that names a member anyway."""
        flat = Star(V=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), C=None,
                    d=None, pred_lb=-np.ones(2), pred_ub=np.ones(2))
        v = star_union_volume_sound_lower([Star(), flat])
        assert v.mean == 0.0
        assert v.meta['argmax_star'] == -1, v.meta

    def test_union_of_a_star_with_a_zero_width_coordinate_has_volume_zero(self):
        """[0, 1] x {2} x [0, 3]: the union box has volume 0, so the union
        has volume 0 with an interval [0, 0]. Catches a box volume that is
        not the product of the widths (a sum gives 4, and every draw of the
        flat box lies in the flat star)."""
        flat = Star.from_bounds(np.array([[0.0], [2.0], [0.0]]),
                                np.array([[1.0], [2.0], [3.0]]))
        v = star_union_volume_mc([flat], n_samples=200, batch_size=200)
        assert v.mean == 0.0 and v.ci_high == 0.0, v
