"""Tests for VNN-COMP per-benchmark configuration."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'vnncomp'))
from benchmark_configs import get_config, BENCHMARK_CONFIGS, DEFAULT_CONFIG


class TestGetConfig:
    """Test config resolution logic."""

    def test_acasxu_default_is_exact(self):
        cfg = get_config('acasxu_2023', vnnlib_path='prop_1.vnnlib')
        assert cfg['reach_methods'] == [('exact', {})]
        assert cfg['n_rand'] == 500

    def test_acasxu_prop3_is_approx_then_exact(self):
        cfg = get_config('acasxu_2023', vnnlib_path='prop_3.vnnlib')
        assert cfg['reach_methods'] == [('approx', {}), ('exact', {})]

    def test_acasxu_prop4_is_approx_then_exact(self):
        cfg = get_config('acasxu_2023', vnnlib_path='prop_4.vnnlib')
        assert cfg['reach_methods'] == [('approx', {}), ('exact', {})]

    def test_cora_set_model(self):
        cfg = get_config('cora_2024', onnx_path='mnist-set.onnx')
        methods = cfg['reach_methods']
        assert len(methods) == 2
        assert methods[0] == ('approx', {'relax_factor': 0.5, 'relax_method': 'area'})
        assert methods[1] == ('approx', {})
        assert cfg['n_rand'] == 100

    def test_cora_other_model(self):
        cfg = get_config('cora_2024', onnx_path='mnist-point.onnx')
        methods = cfg['reach_methods']
        assert len(methods) == 1
        assert methods[0] == ('approx', {'relax_factor': 0.7, 'relax_method': 'area'})

    def test_cgan_non_transformer(self):
        cfg = get_config('cgan_2023', onnx_path='cGAN_imgSz32_nCh_1.onnx')
        methods = cfg['reach_methods']
        assert methods[0] == ('approx', {'relax_factor': 0.8, 'relax_method': 'area'})

    def test_cgan_transformer(self):
        cfg = get_config('cgan_2023', onnx_path='cGAN_transformer_model.onnx')
        assert cfg['reach_methods'] == [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})]

    def test_nn4sys_lindex(self):
        cfg = get_config('nn4sys', onnx_path='lindex_deep.onnx')
        assert cfg['reach_methods'] == [('approx', {})]

    def test_nn4sys_pensieve(self):
        cfg = get_config('nn4sys', onnx_path='pensieve_big_parallel.onnx')
        assert cfg['reach_methods'] == [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})]

    def test_dist_shift_exact_only(self):
        """NNV overwrite bug: only exact-star runs."""
        cfg = get_config('dist_shift_2023')
        assert cfg['reach_methods'] == [('exact', {})]

    def test_malbeware_exact_only(self):
        """NNV overwrite bug: only exact-star runs."""
        cfg = get_config('malbeware')
        assert cfg['reach_methods'] == [('exact', {})]

    def test_tllverify_relax_then_approx(self):
        cfg = get_config('tllverifybench_2023')
        methods = cfg['reach_methods']
        assert methods[0] == ('approx', {'relax_factor': 0.9, 'relax_method': 'area'})
        assert methods[1] == ('approx', {})

    def test_relusplitter_relax_area_1(self):
        cfg = get_config('relusplitter')
        assert cfg['reach_methods'] == [('approx', {'relax_factor': 1.0, 'relax_method': 'area'})]

    def test_unknown_benchmark_gets_default(self):
        cfg = get_config('nonexistent_benchmark_2099')
        assert cfg['reach_methods'] == DEFAULT_CONFIG['reach_methods']
        assert cfg['n_rand'] == DEFAULT_CONFIG['n_rand']

    def test_none_paths_dont_crash(self):
        """get_config should handle None paths gracefully."""
        cfg = get_config('acasxu_2023')
        assert cfg['n_rand'] == 500

    def test_cora_uses_random_falsify(self):
        """cora uses random-only falsification (PGD too slow for OnnxMatMul)."""
        cfg = get_config('cora_2024', onnx_path='mnist-set.onnx')
        assert cfg['falsify_method'] == 'random'

    def test_default_falsify_method_is_random_pgd(self):
        """Benchmarks without explicit falsify_method default to random+pgd."""
        cfg = get_config('acasxu_2023', vnnlib_path='prop_1.vnnlib')
        assert cfg['falsify_method'] == 'random+pgd'

    def test_probabilistic_configs_have_kwargs(self):
        """All probabilistic methods must specify m, epsilon, and surrogate."""
        PROBABILISTIC_BENCHMARKS = [
            'cersyve', 'cifar100_2024', 'soundnessbench', 'tinyimagenet_2024',
            'collins_aerospace_benchmark', 'ml4acopf_2024', 'vggnet16_2022',
            'vit_2023', 'yolo_2023',
        ]
        for category in PROBABILISTIC_BENCHMARKS:
            cfg = get_config(category)
            for method, kwargs in cfg['reach_methods']:
                if method == 'probabilistic':
                    assert 'm' in kwargs, f"{category} probabilistic missing 'm'"
                    assert 'epsilon' in kwargs, f"{category} probabilistic missing 'epsilon'"
                    assert 'surrogate' in kwargs, f"{category} probabilistic missing 'surrogate'"
                    assert kwargs['m'] > 0
                    assert 0 < kwargs['epsilon'] < 1

    def test_nn4sys_probabilistic_has_kwargs(self):
        cfg = get_config('nn4sys', onnx_path='pensieve_big_parallel.onnx')
        kwargs = cfg['reach_methods'][0][1]
        assert kwargs == {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'}

    def test_cgan_transformer_probabilistic_has_kwargs(self):
        cfg = get_config('cgan_2023', onnx_path='cGAN_transformer_model.onnx')
        kwargs = cfg['reach_methods'][0][1]
        assert kwargs == {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'}

    def test_all_configs_have_required_keys(self):
        """Every config must resolve to reach_methods, n_rand, and falsify_method."""
        for category in BENCHMARK_CONFIGS:
            cfg = get_config(category)
            assert 'reach_methods' in cfg, f"{category} missing reach_methods"
            assert 'n_rand' in cfg, f"{category} missing n_rand"
            assert 'falsify_method' in cfg, f"{category} missing falsify_method"
            assert cfg['falsify_method'] in ('random', 'pgd', 'random+pgd'), \
                f"{category} has invalid falsify_method: {cfg['falsify_method']}"
            assert isinstance(cfg['reach_methods'], list)
            assert len(cfg['reach_methods']) > 0
            for method, kwargs in cfg['reach_methods']:
                assert method in ('exact', 'approx', 'probabilistic'), \
                    f"{category} has invalid method: {method}"
                assert isinstance(kwargs, dict)
