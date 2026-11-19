"""Unit tests for ProbabilisticSet."""

import pytest
import torch
import numpy as np


class TestProbabilisticSetContains:
    """Tests for membership queries."""

    def test_center_is_inside(self):
        """Center point should be inside the set."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[0.0, 0.0]])
        assert pset.contains(y).item() is True

    def test_far_point_is_outside(self):
        """Point far from center should be outside."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[10.0, 10.0]])
        assert pset.contains(y).item() is False

    def test_batch_contains(self):
        """Should handle batches, returning (batch,) boolean tensor."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5]])
        result = pset.contains(y)
        assert result.shape == (3,)
        assert result[0].item() is True
        assert result[1].item() is False
        assert result[2].item() is True


class TestProbabilisticSetGuarantee:
    """Tests for guarantee metadata."""

    def test_get_guarantee(self):
        """Should return (coverage, confidence) tuple."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=8000, ell=7999, epsilon=0.001, dim=2
        )

        coverage, confidence = pset.get_guarantee()
        assert coverage == pytest.approx(0.999)
        assert 0.99 < confidence < 1.0


class TestProbabilisticSetVolume:
    """Tests for MC volume estimation."""

    def test_ball_volume_2d(self):
        """Volume of L2 ball with radius r in 2D = pi*r^2."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        radius = 2.0
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=radius,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bbox = (torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        volume, std_err = pset.estimate_volume(n_samples=500_000,
                                               bounding_box=bbox)
        expected = np.pi * radius**2
        # MC estimate should be within ~5% of true volume
        assert volume == pytest.approx(expected, rel=0.05)

    def test_hyperrect_volume_2d(self):
        """Volume of hyperrect should match closed-form."""
        from n2v.probabilistic.flow.scores import HyperrectScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)
        q = 2.0
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=q,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bbox = (torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0]))
        volume, std_err = pset.estimate_volume(n_samples=500_000,
                                               bounding_box=bbox)
        expected = score_fn.sublevel_set_volume(torch.tensor(q))
        assert volume == pytest.approx(expected, rel=0.05)


class TestProbabilisticSetBoundary2D:
    """Tests for 2D boundary extraction."""

    def test_returns_arrays(self):
        """boundary_2d should return list of (N, 2) numpy arrays."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bounds = (torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0]))
        contours = pset.boundary_2d(resolution=50, bounds=bounds)
        assert isinstance(contours, list)
        assert len(contours) > 0
        for path in contours:
            assert isinstance(path, np.ndarray)
            assert path.ndim == 2
            assert path.shape[1] == 2

    def test_raises_for_non_2d(self):
        """boundary_2d should raise ValueError for dim != 2."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=3
        )

        with pytest.raises(ValueError, match="dim=2"):
            pset.boundary_2d()


# ---------------------------------------------------------------------------
# AffineTransform, __repr__ and estimate_coverage
# ---------------------------------------------------------------------------
from fractions import Fraction  # noqa: E402
from math import comb  # noqa: E402

import torch.nn as nn  # noqa: E402
from scipy.stats import binom  # noqa: E402

from tests.known_bugs import known_bug  # noqa: E402

_MEAN3 = np.array([8.0, -4.0, 0.75])
_STD3 = np.array([4.0, 0.25, 1.5])


def _exact(values):
    return [Fraction(v) for v in values]


class TestAffineTransform:
    """The documented whitening y_w = (y - mean) / std, its inverse, the
    dimension and the refusals of AffineTransform."""

    def test_whiten_and_unwhiten_are_the_documented_maps(self):
        """whiten is (y - mean) / std and unwhiten is y_w * std + mean, on a
        batch and on one point, compared exactly with fractions: the data
        are dyadic, so the difference is exact and the one division is
        correctly rounded (the case with std 1.5 has the quotient 1/6).
        Catches whiten written y / std - mean and unwhiten written
        (y_w + mean) * std; both agree with the true maps only when mean = 0
        or std = 1, which is why the axes here have both signs of mean and
        scales above and below 1."""
        from n2v.probabilistic.flow.sets import AffineTransform

        at = AffineTransform(mean=_MEAN3, std=_STD3)
        y = np.array([[10.0, -4.5, 3.0], [-2.0, 0.125, -0.75], [9.0, -3.0, 1.0]])
        mean, std = _exact(_MEAN3), _exact(_STD3)
        white = np.array([[float((v - mu) / s) for v, mu, s in
                           zip(_exact(row), mean, std)] for row in y])
        np.testing.assert_array_equal(at.whiten(y), white)
        np.testing.assert_array_equal(at.whiten(y[2]), white[2])

        y_w = np.array([[0.5, -2.0, 1.5], [-2.5, 17.0, -1.0]])
        raw = np.array([[float(v * s + mu) for v, mu, s in
                         zip(_exact(row), mean, std)] for row in y_w])
        np.testing.assert_array_equal(at.unwhiten(y_w), raw)
        np.testing.assert_array_equal(at.unwhiten(y_w[1]), raw[1])

    def test_normalises_its_arrays_and_states_its_dimension(self):
        """Column vectors (the layout Box keeps its bounds in) and float32
        arrays are stored flat in float64, dim is the number of outputs, and
        whiten then works on a flat point. Catches dim returning the number
        of array axes (always 1)."""
        from n2v.probabilistic.flow.sets import AffineTransform

        at = AffineTransform(
            mean=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            std=np.array([[1.0], [2.0], [4.0]], dtype=np.float32))
        assert at.mean.shape == (3,) and at.std.shape == (3,)
        assert at.mean.dtype == np.float64 and at.std.dtype == np.float64
        assert at.dim == 3
        np.testing.assert_array_equal(at.whiten(np.array([5.0, 6.0, 7.0])),
                                      [4.0, 2.0, 1.0])

    @pytest.mark.parametrize("mean, std", [
        pytest.param([0.0, 1.0, 2.0], [1.0, 1.0], id="lengths-differ"),
        pytest.param([0.0, 1.0], [1.0, 0.0], id="zero-std"),
        pytest.param([0.0, 1.0], [-2.0, 1.0], id="negative-std"),
        pytest.param([0.0, 1.0], [float("nan"), 1.0], id="nan-std"),
    ])
    def test_refuses_arrays_it_cannot_use(self, mean, std):
        """A mean and std of different lengths, and a std entry that is
        zero, negative or NaN, raise ValueError as documented. A zero std
        divides by zero in whiten and zeroes a column of every transformed
        half-space. Catches the positivity check written std >= 0."""
        from n2v.probabilistic.flow.sets import AffineTransform

        with pytest.raises(ValueError):
            AffineTransform(mean=np.array(mean), std=np.array(std))


class TestProbabilisticSetRepr:
    def test_repr_states_coverage_and_confidence(self):
        """The repr names the dimension, the threshold, the coverage
        1 - epsilon and the confidence P(Bin(m, 1 - epsilon) <= ell - 1)
        (computed in fractions: 0.6083 for m = 20, ell = 19, epsilon = 0.1),
        each in its own field. Catches coverage and confidence printed in
        each other's place."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        m, ell, eps = 20, 19, Fraction(0.1)
        confidence = sum(comb(m, k) * (1 - eps)**k * eps**(m - k)
                         for k in range(ell))
        pset = ProbabilisticSet(BallScore(torch.zeros(2)), threshold=1.25,
                                m=m, ell=ell, epsilon=0.1, dim=2)
        assert repr(pset) == (
            f"ProbabilisticSet(dim=2, threshold=1.2500, coverage=0.9000, "
            f"confidence={float(confidence):.4f})")


# The model y = (4 x0 + 1, 2 x1 - 3) on the box x0 in [-3, 1], x1 in
# [0.5, 1.5]: with the whitening mean (-3, -1) and std (8, 1) the whitened
# output is uniform on [-1, 1]^2, so the unit disc covers pi / 4 of it.
_COVERAGE_LB = np.array([-3.0, 0.5])
_COVERAGE_UB = np.array([1.0, 1.5])
_SCALE = [4.0, 2.0]
_SHIFT = [1.0, -3.0]


class _ParameterFree(nn.Module):
    def forward(self, x):
        return x * torch.tensor(_SCALE, dtype=x.dtype) + torch.tensor(_SHIFT, dtype=x.dtype)


class _BuffersOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(_SCALE))
        self.register_buffer("shift", torch.tensor(_SHIFT))

    def forward(self, x):
        return x * self.scale + self.shift


def _linear(dtype):
    lin = nn.Linear(2, 2).to(dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.diag(torch.tensor(_SCALE, dtype=dtype)))
        lin.bias.copy_(torch.tensor(_SHIFT, dtype=dtype))
    return lin


class TestEstimateCoverage:
    @pytest.mark.parametrize("kind", [
        "module-with-parameters",
        "module-without-parameters-or-buffers",
        "module-with-buffers-only",
        "numpy-callable",
        pytest.param("float64-module", marks=known_bug(
            13, "estimate_coverage feeds float32 samples to an nn.Module without "
               "casting them to the module's dtype, so a float64 model raises a "
               "torch RuntimeError (mat1 and mat2 must have the same dtype)",
            raises=RuntimeError)),
    ])
    def test_coverage_is_the_fraction_of_the_box_mapped_into_the_set(self, kind):
        """estimate_coverage samples the input box, runs the model, whitens
        with the set's transform and returns the share inside the set: here
        pi / 4 in closed form, and the estimate over 20000 samples must lie
        in its two-sided 1e-6 binomial interval. Every way of passing the
        model is used: an nn.Module with parameters, one with neither
        parameters nor buffers, one with buffers only, a numpy callable and
        a float64 module. Catches an upper bound read from the box's lower
        bound (every sample the same point: coverage 0 or 1), whitening that
        multiplies by std (coverage near pi / 256) and a device lookup
        without the CPU fallback (StopIteration on a parameter-free
        module)."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import AffineTransform, ProbabilisticSet
        from n2v.sets import Box

        model = {
            "module-with-parameters": lambda: _linear(torch.float32),
            "module-without-parameters-or-buffers": _ParameterFree,
            "module-with-buffers-only": _BuffersOnly,
            "numpy-callable": lambda: (lambda x: x * np.array(_SCALE) + np.array(_SHIFT)),
            "float64-module": lambda: _linear(torch.float64),
        }[kind]()
        pset = ProbabilisticSet(
            BallScore(torch.zeros(2)), threshold=1.0, m=100, ell=99,
            epsilon=0.01, dim=2,
            affine_transform=AffineTransform(mean=np.array([-3.0, -1.0]),
                                             std=np.array([8.0, 1.0])))
        n = 20000
        coverage = pset.estimate_coverage(model, Box(_COVERAGE_LB, _COVERAGE_UB),
                                          n_test=n, seed=3)
        lo = binom.ppf(5e-7, n, np.pi / 4) / n
        hi = binom.isf(5e-7, n, np.pi / 4) / n
        assert lo <= coverage <= hi, f"coverage {coverage}, true pi/4 = {np.pi / 4:.4f}"

    def test_seed_selects_the_sample(self):
        """estimate_coverage draws its test inputs with the seed it is
        given, and with seed 0 when the seed is None (documented); every
        input lies in the box. Catches a seed that is ignored (every call
        evaluates the same sample)."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet
        from n2v.sets import Box

        seen = []

        def model(x):
            seen.append(np.array(x, copy=True))
            return x

        pset = ProbabilisticSet(BallScore(torch.zeros(2)), threshold=1.0,
                                m=100, ell=99, epsilon=0.01, dim=2)
        box = Box(_COVERAGE_LB, _COVERAGE_UB)
        for seed in (None, 0, 7):
            pset.estimate_coverage(model, box, n_test=64, seed=seed)
        np.testing.assert_array_equal(seen[0], seen[1])
        assert not np.array_equal(seen[1], seen[2]), "seed 7 drew the sample of seed 0"
        for x in seen:
            assert np.all(x >= _COVERAGE_LB) and np.all(x <= _COVERAGE_UB)
