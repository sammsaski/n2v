"""
Soundness tests for SpikingNeuralNetwork.reach().

Verifies that the LP relaxation is a true over-approximation: for any input
sampled from within the input set, the true model output must lie within the
reported bounds. Tests both 'approx' and 'exact' methods, and both Box and
Star inputs.
"""

import pytest
pytest.importorskip("snntorch", reason="snntorch not installed; pip install n2v[snn]")

import numpy as np
import torch

from n2v.sets import Box, Star
from n2v.snn.model import F2FMLP
from n2v.nn.spiking_neural_network import SpikingNeuralNetwork

# Number of random samples used to check soundness.
N_SAMPLES = 300
ATOL = 1e-4  # numerical tolerance for LP solve imprecision


@pytest.fixture(scope="module")
def snn():
    torch.manual_seed(0)
    model = F2FMLP(input_size=4, hidden_sizes=[8], num_classes=3, num_steps=8)
    model.eval()
    return SpikingNeuralNetwork(model)


def _sample_scores(snn: SpikingNeuralNetwork, lb: np.ndarray, ub: np.ndarray,
                   n: int, seed: int = 42) -> np.ndarray:
    """Sample n random points from [lb, ub] and return their class scores."""
    rng = np.random.default_rng(seed)
    # lb and ub are (D, 1); flatten for sampling
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()
    scores = []
    for _ in range(n):
        x = rng.uniform(lb_flat, ub_flat).astype(np.float32)
        x_t = torch.from_numpy(x)
        with torch.no_grad():
            s = snn.forward(x_t).numpy().flatten()
        scores.append(s)
    return np.array(scores)  # (N, num_classes)


def _check_soundness(out_box: Box, scores: np.ndarray, label: str):
    """Assert that every sampled score vector is within out_box bounds."""
    lb = out_box.lb.flatten()
    ub = out_box.ub.flatten()
    violations_lb = np.any(scores < lb - ATOL, axis=1)
    violations_ub = np.any(scores > ub + ATOL, axis=1)
    n_viol = np.sum(violations_lb | violations_ub)
    assert n_viol == 0, (
        f"[{label}] {n_viol}/{len(scores)} samples fell outside LP bounds.\n"
        f"  lb={lb}\n  ub={ub}\n"
        f"  worst lb violation: {(lb - scores).max():.6f}\n"
        f"  worst ub violation: {(scores - ub).max():.6f}"
    )


# ---------------------------------------------------------------------------
# Box input
# ---------------------------------------------------------------------------

class TestSoundnessBoxInput:

    def test_approx_box_all_symbolic(self, snn):
        lb = np.array([[0.2], [0.3], [0.1], [0.4]])
        ub = np.array([[0.5], [0.7], [0.6], [0.9]])
        input_box = Box(lb, ub)

        out_box = snn.reach(input_box, method='approx')[0]
        scores = _sample_scores(snn, lb, ub, N_SAMPLES)
        _check_soundness(out_box, scores, "approx/box/all-symbolic")

    def test_approx_box_partial_symbolic(self, snn):
        # Two dimensions fixed, two symbolic
        lb = np.array([[0.2], [0.6], [0.6], [0.4]])
        ub = np.array([[0.8], [0.6], [0.6], [0.9]])
        input_box = Box(lb, ub)

        out_box = snn.reach(input_box, method='approx')[0]
        scores = _sample_scores(snn, lb, ub, N_SAMPLES)
        _check_soundness(out_box, scores, "approx/box/partial-symbolic")

    def test_exact_box_two_symbolic(self, snn):
        # Only 2 symbolic dims → 'exact' is tractable
        lb = np.array([[0.2], [0.6], [0.6], [0.4]])
        ub = np.array([[0.8], [0.6], [0.6], [0.9]])
        input_box = Box(lb, ub)

        out_box = snn.reach(input_box, method='exact')[0]
        scores = _sample_scores(snn, lb, ub, N_SAMPLES)
        _check_soundness(out_box, scores, "exact/box/two-symbolic")

    def test_approx_box_tight_bounds(self, snn):
        lb = np.array([[0.3], [0.5], [0.2], [0.5]])
        ub = np.array([[0.6], [0.8], [0.5], [0.8]])
        input_box = Box(lb, ub)

        out_box = snn.reach(input_box, method='approx', tight_bounds=True)[0]
        scores = _sample_scores(snn, lb, ub, N_SAMPLES)
        _check_soundness(out_box, scores, "approx/box/tight-bounds")


# ---------------------------------------------------------------------------
# Star input
# ---------------------------------------------------------------------------

class TestSoundnessStarInput:

    def test_approx_star_from_bounds(self, snn):
        lb = np.array([[0.2], [0.3], [0.1], [0.4]])
        ub = np.array([[0.5], [0.7], [0.6], [0.9]])
        input_star = Star.from_bounds(lb, ub)

        out_box = snn.reach(input_star, method='approx')[0]
        # Sample from the STAR's bounding box (which is [lb, ub] for this Star)
        scores = _sample_scores(snn, lb, ub, N_SAMPLES)
        _check_soundness(out_box, scores, "approx/star/from-bounds")


# ---------------------------------------------------------------------------
# Exact is at least as tight as approx
# ---------------------------------------------------------------------------

class TestExactTighterThanApprox:

    def test_exact_bounds_inside_approx_bounds(self, snn):
        lb = np.array([[0.2], [0.6], [0.6], [0.4]])
        ub = np.array([[0.8], [0.6], [0.6], [0.9]])
        input_box = Box(lb, ub)

        approx_box = snn.reach(input_box, method='approx')[0]
        exact_box  = snn.reach(input_box, method='exact')[0]

        # Exact lb >= approx lb (exact is at least as tight from below)
        assert np.all(exact_box.lb >= approx_box.lb - ATOL), (
            f"Exact lb {exact_box.lb.flatten()} < approx lb {approx_box.lb.flatten()}"
        )
        # Exact ub <= approx ub (exact is at least as tight from above)
        assert np.all(exact_box.ub <= approx_box.ub + ATOL), (
            f"Exact ub {exact_box.ub.flatten()} > approx ub {approx_box.ub.flatten()}"
        )


# ---------------------------------------------------------------------------
# Point input (degenerate case — lb == ub)
# ---------------------------------------------------------------------------

class TestSoundnessPointInput:

    def test_point_input_box(self, snn):
        # All dimensions fixed → single-point input
        x = np.array([0.4, 0.6, 0.3, 0.7], dtype=np.float32)
        lb = x.reshape(-1, 1)
        ub = x.reshape(-1, 1)
        input_box = Box(lb, ub)

        out_box = snn.reach(input_box, method='approx')[0]

        # The only possible output is the exact forward-pass score
        x_t = torch.from_numpy(x)
        true_scores = snn.forward(x_t).numpy().flatten()

        lb_scores = out_box.lb.flatten()
        ub_scores = out_box.ub.flatten()

        assert np.all(true_scores >= lb_scores - ATOL), (
            f"True score {true_scores} below lb {lb_scores}"
        )
        assert np.all(true_scores <= ub_scores + ATOL), (
            f"True score {true_scores} above ub {ub_scores}"
        )
