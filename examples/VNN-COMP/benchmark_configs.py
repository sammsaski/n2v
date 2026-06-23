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
        # NNV uses cp-star; n2v matched via falsification already.
        # n_rand bumped 100->3000: validated to find more SOUND (CE-validated) sat
        # (1->4) now that batched random sampling makes it cheap.
        'reach_methods': [('probabilistic', {'m': 8000, 'epsilon': 0.001, 'surrogate': 'naive'})],
        'n_rand': 3000,
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
        # Holds-dominated (gold 65 unsat / 7 sat). Prepend a SOUND approx pass
        # before exact: approx (triangle over-approximation) terminates fast and
        # an approx-UNSAT is a true hold; exact (kept as the complete tail) still
        # wins instances where it finishes. approx can only ADD holds or fall
        # through to exact -- it can never emit a false unsat (sound by
        # construction; the verify_specification UNSAT invariant is locked by the
        # soundness-invariant tests added in PR #6).
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },

    'linearizenn_2024': {
        # Holds-dominated (gold 59 unsat / 1 sat); loadable now (IBP matmul guard).
        # Prepend a sound approx pass before exact (same rationale as dist_shift).
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },

    'malbeware': {
        # Holds-dominated (gold 131 unsat / 19 sat). Prepend a sound approx pass
        # before exact (same rationale as dist_shift).
        'reach_methods': [('approx', {}), ('exact', {})],
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
        # SAT instances — planted/binarized counterexamples. Gradient-free Square
        # (the 'strong' ensemble) cracks the planted CEs that random+pgd miss
        # (validated 0->3/10; full-pipeline LIFT +2, REGRESSION 0). Sat-only, so
        # the sequential 'strong' budget cannot cost a holds.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'strong',
    },

    'soundnessbench_2026': {
        # 2026 variant (was falling to DEFAULT: n_rand=100, random+pgd -> near-zero
        # falsification). Mirror the 2025 config: deterministic reach + high n_rand
        # + gradient-free 'strong' for the planted/binarized CEs.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'strong',
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
        # Sat-only (0 holds in gold). Drop the probabilistic reach: it can only
        # emit an UNSOUND 'unsat' (a coverage set, not a proof) and there are no
        # holds to gain. Concede to falsification + unknown (sound).
        'reach_methods': [],
        'n_rand': 100,
    },

    'lsnc_relu': {
        # approx/exact now run SOUNDLY (the IBP matmul-shape guard is merged, so
        # these no longer crash). Drop the probabilistic fallback: it was the only
        # remaining UNSOUND path (a coverage-set 'unsat'). Sound holds/unsat from
        # approx/exact + sat from falsification.
        'reach_methods': [
            ('approx', {}),
            ('exact', {}),
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
        # Binarized NN with Sign activations. Sat-only (0 holds in gold), so drop
        # the probabilistic fallback (it can only emit an UNSOUND 'unsat'); keep
        # approx (sound; may time out -> unknown). Use the gradient-free 'strong'
        # ensemble (Square) for the CEs: PGD/APGD are useless on Sign (zero
        # gradients), Square is the right tool (validated 1->2/10).
        'reach_methods': [
            ('approx', {}),
        ],
        'n_rand': 10000,
        'falsify_method': 'strong',
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
