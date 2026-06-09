"""
Unit tests for SNNReachConfig validation and behaviour.
"""

import pytest
pytest.importorskip("snntorch", reason="snntorch not installed; pip install n2v[snn]")

from n2v.nn.spiking_neural_network import SNNReachConfig


class TestSNNReachConfigDefaults:

    def test_default_method(self):
        cfg = SNNReachConfig()
        assert cfg.method == 'approx'

    def test_default_parallel_workers(self):
        cfg = SNNReachConfig()
        assert cfg.parallel_workers == 0

    def test_default_tight_bounds(self):
        cfg = SNNReachConfig()
        assert cfg.tight_bounds is False

    def test_default_singleton_bounds(self):
        cfg = SNNReachConfig()
        assert cfg.singleton_bounds is False

    def test_default_split_strategy(self):
        cfg = SNNReachConfig()
        assert cfg.split_strategy == 'choice-influence'

    def test_default_label(self):
        cfg = SNNReachConfig()
        assert cfg.label is None


class TestSNNReachConfigValidation:

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            SNNReachConfig(method='gibberish')

    def test_negative_parallel_workers_raises(self):
        with pytest.raises(ValueError, match="parallel_workers"):
            SNNReachConfig(parallel_workers=-1)

    def test_invalid_split_strategy_raises(self):
        with pytest.raises(ValueError, match="split_strategy"):
            SNNReachConfig(split_strategy='not-a-strategy')

    def test_zero_parallel_workers_ok(self):
        cfg = SNNReachConfig(parallel_workers=0)
        assert cfg.parallel_workers == 0

    def test_positive_parallel_workers_ok(self):
        cfg = SNNReachConfig(parallel_workers=4)
        assert cfg.parallel_workers == 4

    def test_exact_method_ok(self):
        cfg = SNNReachConfig(method='exact')
        assert cfg.method == 'exact'

    def test_approx_method_ok(self):
        cfg = SNNReachConfig(method='approx')
        assert cfg.method == 'approx'


class TestSNNReachConfigValidStrategies:

    @pytest.mark.parametrize("strategy", [
        'selected', 'influence', 'choice', 'choice-influence', 'random'
    ])
    def test_valid_strategies_accepted(self, strategy):
        cfg = SNNReachConfig(split_strategy=strategy)
        assert cfg.split_strategy == strategy


class TestSNNReachConfigImmutability:

    def test_is_frozen(self):
        cfg = SNNReachConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.method = 'exact'  # type: ignore

    def test_equality(self):
        cfg1 = SNNReachConfig(method='exact', parallel_workers=2)
        cfg2 = SNNReachConfig(method='exact', parallel_workers=2)
        assert cfg1 == cfg2

    def test_inequality(self):
        cfg1 = SNNReachConfig(method='approx')
        cfg2 = SNNReachConfig(method='exact')
        assert cfg1 != cfg2


class TestSNNReachConfigFieldCombinations:

    def test_all_fields_set(self):
        cfg = SNNReachConfig(
            method='exact',
            parallel_workers=8,
            tight_bounds=True,
            singleton_bounds=True,
            split_strategy='influence',
            label=2,
        )
        assert cfg.method == 'exact'
        assert cfg.parallel_workers == 8
        assert cfg.tight_bounds is True
        assert cfg.singleton_bounds is True
        assert cfg.split_strategy == 'influence'
        assert cfg.label == 2
