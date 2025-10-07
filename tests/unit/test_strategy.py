"""Tests for VNN-COMP strategy module."""

import numpy as np
import pytest


class TestAnalyzeDifficulty:

    def test_counts_stable_neurons(self):
        """Should correctly count stable active/inactive/uncertain neurons."""
        from n2v.utils.vnncomp import analyze_difficulty

        layer_bounds = {
            1: (
                np.array([[0.5], [-1.0], [-0.5], [0.3]]),  # lb
                np.array([[1.5], [-0.2], [0.5], [1.0]]),   # ub
            ),
        }
        result = analyze_difficulty(layer_bounds)

        assert len(result['per_layer']) == 1
        layer_info = result['per_layer'][0]
        assert layer_info['total'] == 4
        assert layer_info['stable_active'] == 2   # neurons 0, 3 (lb >= 0)
        assert layer_info['stable_inactive'] == 1  # neuron 1 (ub <= 0)
        assert layer_info['uncertain'] == 1         # neuron 2 (crosses zero)

    def test_total_uncertain(self):
        """Should sum uncertain neurons across layers."""
        from n2v.utils.vnncomp import analyze_difficulty

        layer_bounds = {
            1: (np.array([[-1.0], [0.5]]), np.array([[1.0], [1.5]])),
            3: (np.array([[-0.5], [-0.5], [0.5]]), np.array([[0.5], [0.5], [1.5]])),
        }
        result = analyze_difficulty(layer_bounds)
        # Layer 1: 1 uncertain; Layer 3: 2 uncertain
        assert result['total_uncertain'] == 3
        assert result['max_layer_uncertain'] == 2

    def test_empty_bounds(self):
        """Should handle empty layer_bounds."""
        from n2v.utils.vnncomp import analyze_difficulty

        result = analyze_difficulty({})
        assert result['total_uncertain'] == 0
        assert result['per_layer'] == []


class TestGetBenchmarkConfig:

    def test_returns_list_of_reach_options(self):
        """Should return a list of ReachOptions for known benchmarks."""
        from n2v.utils.vnncomp import get_benchmark_config, ReachOptions

        config = get_benchmark_config('acasxu')
        assert isinstance(config, list)
        assert len(config) >= 1
        assert all(isinstance(opt, ReachOptions) for opt in config)

    def test_unknown_benchmark_returns_default(self):
        """Unknown benchmark should return default 3-stage pipeline."""
        from n2v.utils.vnncomp import get_benchmark_config

        config = get_benchmark_config('unknown_benchmark_xyz')
        assert len(config) >= 1  # At least one strategy

    def test_default_config_has_approx(self):
        """Default config should include approx method."""
        from n2v.utils.vnncomp import get_benchmark_config

        config = get_benchmark_config('unknown_benchmark_xyz')
        methods = [opt.method for opt in config]
        assert 'approx' in methods

    def test_acasxu_config(self):
        """ACAS Xu should use exact for most props."""
        from n2v.utils.vnncomp import get_benchmark_config

        config = get_benchmark_config('acasxu')
        methods = [opt.method for opt in config]
        assert 'exact' in methods
