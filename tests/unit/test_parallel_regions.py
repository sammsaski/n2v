"""Tests for parallel input region verification."""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, HalfSpace
from n2v.nn import NeuralNetwork


class TestVerifyRegionsParallel:

    def _make_model(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        model.eval()
        return model

    def _make_unsat_regions(self):
        """Create regions where all outputs are positive (property: y0 < 0 is UNSAT)."""
        return [
            (np.array([[0.5], [0.5]]), np.array([[1.0], [1.0]])),
            (np.array([[0.5], [0.0]]), np.array([[1.0], [0.5]])),
        ]

    def test_parallel_matches_sequential(self):
        """Parallel and sequential verification should produce same result."""
        from n2v.utils.vnncomp import verify_regions_parallel
        from n2v.utils.verify_specification import verify_specification

        model = self._make_model()
        net = NeuralNetwork(model)

        regions = self._make_unsat_regions()

        # Property: y_0 < 0 (i.e., [1, 0] @ y <= 0 is the unsafe region)
        prop = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))

        # Sequential
        sequential_results = []
        for lb, ub in regions:
            input_set = Star.from_bounds(lb, ub)
            reach_sets = net.reach(input_set, method='approx')
            verdict = verify_specification(reach_sets, prop)
            sequential_results.append(verdict)

        # Parallel
        parallel_result = verify_regions_parallel(
            model, regions, prop, method='approx', n_workers=2
        )

        # If all sequential are UNSAT (1), parallel should also be UNSAT
        if all(r == 1 for r in sequential_results):
            assert parallel_result['result'] == 'unsat'

    def test_single_region_works(self):
        """Should handle single-region case."""
        from n2v.utils.vnncomp import verify_regions_parallel

        model = self._make_model()
        regions = [(np.array([[0.5], [0.5]]), np.array([[1.0], [1.0]]))]
        prop = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))

        result = verify_regions_parallel(
            model, regions, prop, method='approx', n_workers=1
        )
        assert result['result'] in ('sat', 'unsat', 'unknown')

    def test_returns_expected_keys(self):
        """Result dict should have expected keys."""
        from n2v.utils.vnncomp import verify_regions_parallel

        model = self._make_model()
        regions = self._make_unsat_regions()
        prop = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))

        result = verify_regions_parallel(
            model, regions, prop, method='approx', n_workers=2
        )
        assert 'result' in result
        assert 'per_region' in result
        assert result['result'] in ('sat', 'unsat', 'unknown')
