"""
Per-benchmark verification strategies for VNN-COMP 2025.

Mirrors NNV's load_vnncomp_network() from run_vnncomp_instance.m exactly,
including NNV's index-overwrite bugs (dist_shift, linearize, malbeware
only run exact-star despite seemingly intending approx->exact).

Each config has:
    reach_methods: Ordered list of (method, kwargs) tuples.
        method is one of: 'exact', 'approx', 'probabilistic'
        kwargs are passed to net.reach() — e.g. relax_factor, relax_method
    n_rand: Number of random falsification samples.
    falsify_method: Falsification method ('random', 'pgd', 'random+pgd').
        Defaults to 'random+pgd' if not specified.

For benchmarks with model/property-specific strategies, use get_config()
which resolves the correct method list based on file names.
"""


# Default config for unknown benchmarks
DEFAULT_CONFIG = {
    'reach_methods': [('approx', {}), ('exact', {})],
    'n_rand': 100,
}


BENCHMARK_CONFIGS = {

    # =========================================================================
    # Main Track
    # =========================================================================

    'acasxu_2023': {
        # prop_3/4: [approx, exact]; all others: exact only
        # Resolved by get_config() using vnnlib_path
        'reach_methods_by_prop': {
            'prop_3': [('approx', {}), ('exact', {})],
            'prop_4': [('approx', {}), ('exact', {})],
            '_default': [('exact', {})],
        },
        'n_rand': 500,
    },

    'cersyve': {
        # NNV uses cp-star; n2v matched via falsification already
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'cgan_2023': {
        # Non-transformer: relax-star-area(0.8) -> approx-star
        # Transformer: probabilistic
        # Resolved by get_config() using onnx_path
        'reach_methods': [
            ('approx', {'relax_factor': 0.8, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        'reach_methods_transformer': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'cifar100_2024': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
        'falsify_method': 'random',  # PGD too slow for ResNet medium (~220s for 500 gradient steps)
    },

    'collins_rul_cnn_2022': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'cora_2024': {
        # '-set' models: relax-star-area(0.5) -> approx-star
        # Other models: relax-star-area(0.7) (NNV overwrite bug: only this runs)
        # Resolved by get_config() using onnx_path
        'reach_methods_by_model': {
            '-set': [
                ('approx', {'relax_factor': 0.5, 'relax_method': 'area'}),
                ('approx', {}),
            ],
            '_default': [
                ('approx', {'relax_factor': 0.7, 'relax_method': 'area'}),
            ],
        },
        'n_rand': 100,
        'falsify_method': 'random',  # PGD too slow for 784-dim OnnxMatMul models
    },

    'dist_shift_2023': {
        # NNV overwrite bug: reachOptionsList{1} overwritten, only exact-star runs
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'linearizenn_2024': {
        # NNV overwrite bug: only exact-star runs
        # NNV falls back to cp-star if matlab2nnv fails, but we don't need that
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'malbeware': {
        # NNV overwrite bug: only exact-star runs
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'metaroom_2023': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'nn4sys': {
        # lindex models: approx-star; others: probabilistic
        # Resolved by get_config() using onnx_path
        'reach_methods_by_model': {
            'lindex': [('approx', {})],
            '_default': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        },
        'n_rand': 100,
    },

    'safenlp_2024': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 500,
    },

    'sat_relu': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },

    'soundnessbench': {
        # SAT instance — counterexample exists. All layers (Gemm, ReLU, Conv, Flatten)
        # are fully supported, so use deterministic reachability.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
    },

    'tinyimagenet_2024': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 500,
    },

    'tllverifybench_2023': {
        'reach_methods': [
            ('approx', {'relax_factor': 0.9, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        'n_rand': 100,
    },

    # =========================================================================
    # Extended Track
    # =========================================================================

    'collins_aerospace_benchmark': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'lsnc_relu': {
        # NNV can't handle this (MATLAB opset issue); n2v uses approx/exact with
        # probabilistic fallback for models with unsupported ONNX ops
        'reach_methods': [
            ('approx', {}),
            ('exact', {}),
            ('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'clipping_block'}),
        ],
        'n_rand': 100,
    },

    'ml4acopf_2024': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'relusplitter': {
        'reach_methods': [
            ('approx', {'relax_factor': 1.0, 'relax_method': 'area'}),
        ],
        'n_rand': 100,
    },

    'traffic_signs_recognition_2023': {
        # Binarized NN with Sign activations — now supported by dispatcher.
        # Try approx first (Sign over-approximation), fall back to probabilistic.
        # PGD still useless (Sign has zero gradients almost everywhere).
        'reach_methods': [
            ('approx', {}),
            ('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'}),
        ],
        'n_rand': 10000,
        'falsify_method': 'random',  # PGD useless for Sign activations (zero gradients)
    },

    'vggnet16_2022': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'vit_2023': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
    },

    'yolo_2023': {
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 100,
        'falsify_method': 'random',  # PGD too slow for TinyYOLO (~233s for 500 gradient steps)
    },

    # =========================================================================
    # Test Track
    # =========================================================================

    'test': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },

    # cctsdb_yolo_2023 intentionally omitted — NNV also can't handle it
}


def get_config(category, onnx_path=None, vnnlib_path=None):
    """
    Resolve per-benchmark config, handling model/property variants.

    Args:
        category: Benchmark category name (e.g. 'acasxu_2023')
        onnx_path: Path to ONNX model (for model-specific configs)
        vnnlib_path: Path to VNNLIB spec (for property-specific configs)

    Returns:
        Dict with:
        - 'reach_methods': list of (method, kwargs) tuples
        - 'n_rand': int
        - 'falsify_method': str ('random', 'pgd', or 'random+pgd')
    """
    config = BENCHMARK_CONFIGS.get(category, DEFAULT_CONFIG)
    onnx_path = onnx_path or ''
    vnnlib_path = vnnlib_path or ''

    falsify_method = config.get('falsify_method', 'random+pgd')

    # Resolve property-specific methods (acasxu)
    if 'reach_methods_by_prop' in config:
        for key, methods in config['reach_methods_by_prop'].items():
            if key != '_default' and key in vnnlib_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'], 'falsify_method': falsify_method}
        return {
            'reach_methods': config['reach_methods_by_prop']['_default'],
            'n_rand': config['n_rand'],
            'falsify_method': falsify_method,
        }

    # Resolve model-specific methods (cora, nn4sys)
    if 'reach_methods_by_model' in config:
        for key, methods in config['reach_methods_by_model'].items():
            if key != '_default' and key in onnx_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'], 'falsify_method': falsify_method}
        return {
            'reach_methods': config['reach_methods_by_model']['_default'],
            'n_rand': config['n_rand'],
            'falsify_method': falsify_method,
        }

    # Resolve transformer variant (cgan)
    if 'reach_methods_transformer' in config:
        if 'transformer' in onnx_path.lower():
            return {
                'reach_methods': config['reach_methods_transformer'],
                'n_rand': config['n_rand'],
                'falsify_method': falsify_method,
            }

    return {
        'reach_methods': config.get('reach_methods', DEFAULT_CONFIG['reach_methods']),
        'n_rand': config.get('n_rand', DEFAULT_CONFIG['n_rand']),
        'falsify_method': falsify_method,
    }
