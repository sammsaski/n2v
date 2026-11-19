"""Tests for VNN-COMP strategy module.

``n2v/utils/vnncomp.py`` is slated for deletion (PLAN step 1); until then the
cases at the end check what its strategy table and difficulty count
promise, with expected values worked out by hand: the counts partition the
neurons with a bound at zero counted as stable, the cost class follows the
largest layer, the table is looked up case-insensitively, names only sound
methods with the bound pre-pass, and its options reach ``to_kwargs``.
"""

import numpy as np
import pytest

from tests.known_bugs import known_bug


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
        """Unknown benchmark should return the default pipeline (approx, then
        exact; checked in full by
        ``test_the_default_strategy_and_every_named_one_are_sound_with_the_prepass``)."""
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


# --------------------------------------------------------------------------- #
# What the difficulty count and the strategy table promise                    #
# --------------------------------------------------------------------------- #

def _layer(n_active, n_inactive, n_uncertain):
    """Bounds of one layer with the given counts, in a shuffled order and as
    a (k, 1) column; a bound of exactly 0 on one stable neuron of each kind."""
    lb = [0.0] + [0.25 * (i + 1) for i in range(n_active - 1)]
    ub = [1.0] + [2.0] * (n_active - 1)
    lb += [-1.0] + [-3.0] * (n_inactive - 1)
    ub += [0.0] + [-0.5] * (n_inactive - 1)
    lb += [-0.5] * n_uncertain
    ub += [0.75] * n_uncertain
    order = np.random.default_rng(0).permutation(len(lb))
    return (np.array(lb)[order].reshape(-1, 1),
            np.array(ub)[order].reshape(-1, 1))


def test_a_bound_at_zero_counts_as_stable():
    """A neuron with lower bound 0 is always active and one with upper bound
    0 always inactive (ReLU is the identity or zero on it): three active,
    two inactive and four uncertain neurons, with one of each stable kind
    exactly at 0, are counted as such. Catches a strict comparison with 0
    on either side."""
    from n2v.utils.vnncomp import analyze_difficulty
    result = analyze_difficulty({'relu_1': _layer(3, 2, 4)})
    assert result['per_layer'] == [{'layer_id': 'relu_1', 'total': 9,
                                    'stable_active': 3, 'stable_inactive': 2,
                                    'uncertain': 4}]
    assert result['total_uncertain'] == 4


@known_bug(1, "analyze_difficulty counts a neuron with lower and upper bound "
              "0 as both stable active and stable inactive, so the layer's "
              "uncertain count is one too low for each such neuron (below "
              "zero for a layer of only such neurons); the module is deleted "
              "in PLAN step 1")
def test_a_neuron_fixed_at_zero_is_counted_once():
    """A layer of one neuron fixed at 0 (lower and upper bound 0, stable)
    and one uncertain neuron: the stable neuron is counted once, the other
    as uncertain. Its passing neighbour is
    ``test_a_bound_at_zero_counts_as_stable``."""
    from n2v.utils.vnncomp import analyze_difficulty
    info = analyze_difficulty({1: (np.array([[0.0], [-1.0]]),
                                   np.array([[0.0], [1.0]]))})['per_layer'][0]
    assert info['total'] == 2
    assert info['stable_active'] + info['stable_inactive'] == 1
    assert info['uncertain'] == 1


@pytest.mark.parametrize("largest,cost", [(0, 'low'), (5, 'low'),
                                          (6, 'medium'), (20, 'medium'),
                                          (21, 'high')])
def test_the_cost_class_follows_the_largest_layer(largest, cost):
    """Two layers, one with ``largest`` uncertain neurons (its bounds given
    as a row) and one with fewer: the largest per-layer count decides the
    class, at most 5 'low', at most 20 'medium', else 'high', and the total
    adds both. Catches a threshold moved by one and a class taken from the
    total."""
    from n2v.utils.vnncomp import analyze_difficulty
    small = max(largest - 1, 0)
    big_lb, big_ub = _layer(2, 1, largest)
    result = analyze_difficulty({
        0: _layer(1, 1, small),
        'conv': (big_lb.reshape(1, -1), big_ub.reshape(1, -1))})
    assert result['max_layer_uncertain'] == largest
    assert result['total_uncertain'] == largest + small
    assert result['estimated_exact_cost'] == cost
    assert [p['layer_id'] for p in result['per_layer']] == [0, 'conv']


def test_the_category_name_is_looked_up_case_insensitively():
    """'ACASXU', 'AcasXu' and 'acasxu' get the same strategy: approx, then
    exact, both with the bound pre-pass. Catches the lower-casing dropped
    (the upper-case name then gets the default)."""
    from n2v.utils.vnncomp import get_benchmark_config
    for name in ('ACASXU', 'AcasXu', 'acasxu', 'CORA'):
        got = [opt.to_dict() for opt in get_benchmark_config(name)]
        assert got == [opt.to_dict()
                       for opt in get_benchmark_config(name.lower())]
    cora = get_benchmark_config('CORA')
    assert [(o.method, o.relax_factor, o.relax_method) for o in cora] == [
        ('approx', 0.5, 'area'), ('approx', None, 'standard')]


def test_the_default_strategy_and_every_named_one_are_sound_with_the_prepass():
    """An unknown name gets approx then exact; every named strategy and the
    default use only sound methods (approx, exact), each with the bound
    pre-pass on. Catches a default that drops exact, a probabilistic method
    in the table and a pre-pass left off."""
    from n2v.utils.vnncomp import get_benchmark_config
    default = get_benchmark_config('unknown_benchmark_xyz')
    assert [o.method for o in default] == ['approx', 'exact']
    names = ['acasxu', 'safenlp', 'tllverify', 'sat_relu', 'cersyve',
             'malbeware', 'metaroom', 'cora', 'cifar100', 'unknown']
    for name in names:
        for opt in get_benchmark_config(name):
            assert opt.method in ('approx', 'exact'), name
            assert opt.precompute_bounds is True, name


@pytest.mark.parametrize("kwargs,expected", [
    ({}, {'lp_solver': 'linprog', 'precompute_bounds': False}),
    ({'relax_factor': 0.5, 'relax_method': 'area', 'precompute_bounds': True},
     {'lp_solver': 'linprog', 'precompute_bounds': True, 'relax_factor': 0.5,
      'relax_method': 'area'}),
    ({'relax_factor': 0.0, 'lp_solver': 'highs',
      'extra_kwargs': {'parallel': False}},
     {'lp_solver': 'highs', 'precompute_bounds': False, 'relax_factor': 0.0,
      'parallel': False}),
], ids=["defaults", "relax-area", "relax-factor-zero-and-extras"])
def test_reach_options_become_the_reach_keywords(kwargs, expected):
    """``ReachOptions.to_kwargs``: the solver by its name, the pre-pass flag,
    a relax factor whenever one is set (0.0 included), a relax method other
    than 'standard', and the extra keywords. Catches a relax factor of 0.0
    dropped because it reads as false, and the extras left out."""
    from n2v.utils.vnncomp import ReachOptions
    assert ReachOptions(method='approx', **kwargs).to_kwargs() == expected
