"""PROPOSED VNN-COMP 2026 per-benchmark strategy (zero-risk + falsification wins).

Sandboxed copy of examples/VNN-COMP/benchmark_configs.py with two changes,
validated in this directory's STRATEGY_LOG.md. NOT yet promoted to the shipped
config — drop-in compatible (same get_config signature).

CHANGE 1 — Zero-risk: every ('probabilistic', ...) reach entry is removed.
  Probabilistic reach can emit an UNSOUND 'unsat' = a -150 landmine under the
  +10/0/-150 model. We confirmed sound 'approx' reach does NOT recover the
  ml4acopf holds (returns 'unknown'), so the conceded holds are a genuine cost,
  but removing the probabilistic path removes ALL false-UNSAT risk.
    - Small nets where sound holds are still plausible -> ('approx',) [+exact]:
      cersyve, ml4acopf_2024, nn4sys(_default), yolo_2023.
    - Frontier/large nets where sound reach can't finish in budget -> [] (falsify
      only; conceding holds we couldn't prove anyway, saving wall-clock):
      cifar100_2024, tinyimagenet_2024, vggnet16_2022, vit_2023, cgan transformer.

CHANGE 2 — Falsification (zero soundness risk; runs before reach):
    - sat_relu -> 'random+square'. Boolean-SAT-encoded ReLU nets have flat
      gradients; random/PGD find 35/100, Square finds 50/100 (validated full
      corpus, ~0.1s each, strict superset -> no regression). +15 sat.
    - cersyve n_rand 3000 -> 5000: random sampling then cracks pendulum_pretrain_con
      (+1, validated).

Everything else is byte-for-byte the current config.
"""


DEFAULT_CONFIG = {
    'reach_methods': [('approx', {}), ('exact', {})],
    'n_rand': 100,
}


BENCHMARK_CONFIGS = {

    # ===================== Main Track =====================

    'acasxu_2023': {
        'reach_methods_by_prop': {
            'prop_3': [('approx', {}), ('exact', {})],
            'prop_4': [('approx', {}), ('exact', {})],
            '_default': [('exact', {})],
        },
        'n_rand': 500,
    },

    'cersyve': {
        # WAS probabilistic. Zero-risk: sound approx/exact (small FC control nets,
        # sound reach may prove holds; returns unknown otherwise -- never -150).
        # n_rand 3000 -> 5000 cracks pendulum_pretrain_con via random sampling (+1).
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
    },

    'cgan_2023': {
        'reach_methods': [
            ('approx', {'relax_factor': 0.8, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        # WAS probabilistic for the transformer sub-model. Zero-risk: falsify-only
        # (random finds the small_transformer CE instantly; reach is the attention
        # frontier -> unknown anyway).
        'reach_methods_transformer': [],
        'n_rand': 100,
    },

    'cgan2026': {
        'reach_methods': [
            ('approx', {'relax_factor': 0.8, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        'n_rand': 100,
    },

    'challenging_certified_training_2026': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
    },

    'nn4sys_2023': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
    },

    'relusplitter_2026': {
        'reach_methods': [('approx', {'relax_factor': 1.0, 'relax_method': 'area'})],
        'n_rand': 100,
    },

    'cifar100_2024': {
        # WAS probabilistic. Zero-risk: ResNet approx reach can't finish at scale
        # (-> timeout), so falsify-only (random+apgd) -- concedes holds we couldn't
        # soundly prove anyway, saves budget. SAT path unchanged.
        'reach_methods': [],
        'n_rand': 100,
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
    },

    'collins_rul_cnn_2022': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'cora_2024': {
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
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
    },

    'dist_shift_2023': {
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'linearizenn_2024': {
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'malbeware': {
        'reach_methods': [('exact', {})],
        'n_rand': 100,
    },

    'metaroom_2023': {
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'nn4sys': {
        # WAS _default probabilistic. Zero-risk: sound approx (nn4sys has provable
        # holds; returns unknown otherwise). lindex already approx.
        'reach_methods_by_model': {
            'lindex': [('approx', {})],
            '_default': [('approx', {})],
        },
        'n_rand': 100,
    },

    'safenlp_2024': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 500,
    },

    'sat_relu': {
        # CHANGE 2: Boolean-SAT-encoded ReLU nets -> flat gradients. random+pgd
        # finds 35/100; gradient-free Square finds 50/100 (validated full corpus,
        # ~0.1s/instance, strict superset). +15 sat. Sound reach kept for holds.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
        'falsify_method': 'random+square',
        'falsify_kwargs': {'n_iters': 20000},
    },

    'soundnessbench': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'strong',
    },

    'soundnessbench_2026': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'strong',
    },

    'tinyimagenet_2024': {
        # WAS probabilistic. Zero-risk: ResNet -> falsify-only (random+apgd).
        'reach_methods': [],
        'n_rand': 500,
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
    },

    'tllverifybench_2023': {
        'reach_methods': [
            ('approx', {'relax_factor': 0.9, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        'n_rand': 100,
    },

    # ===================== Extended Track =====================

    'collins_aerospace_benchmark': {
        'reach_methods': [],
        'n_rand': 100,
    },

    'lsnc_relu': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },

    'ml4acopf_2024': {
        # WAS probabilistic. Zero-risk: sound approx (validated -> 'unknown' on the
        # 14_ieee holds, so the 6 conceded holds are a real cost, but no -150 risk).
        # Keep random+apgd nr3/ns50 (cracks the gold-sat 300_ieee_prop2 CE).
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 3, 'n_steps': 50},
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'relusplitter': {
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
        'reach_methods': [
            ('approx', {'relax_factor': 1.0, 'relax_method': 'area'}),
        ],
        'n_rand': 100,
    },

    'traffic_signs_recognition_2023': {
        'reach_methods': [
            ('approx', {}),
        ],
        'n_rand': 10000,
        'falsify_method': 'strong',
    },

    'vggnet16_2022': {
        # WAS probabilistic. Zero-risk: 150k-dim ImageNet scale -> falsify-only
        # (caught the sat in the baseline sweep). Sound reach can't finish.
        'reach_methods': [],
        'n_rand': 100,
    },

    'vit_2023': {
        # WAS probabilistic. Zero-risk: bilinear attention frontier -> falsify-only.
        'reach_methods': [],
        'n_rand': 100,
    },

    'yolo_2023': {
        # WAS probabilistic. Zero-risk: sound approx (TinyYOLO is small; may prove
        # holds, else unknown -- never -150). random falsify (PGD too slow).
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
    },

    # ===================== Test Track =====================

    'test': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
    },
}


def get_config(category, onnx_path=None, vnnlib_path=None):
    config = BENCHMARK_CONFIGS.get(category, DEFAULT_CONFIG)
    onnx_path = onnx_path or ''
    vnnlib_path = vnnlib_path or ''
    falsify_method = config.get('falsify_method', 'random+pgd')
    falsify_kwargs = config.get('falsify_kwargs', {})

    if 'reach_methods_by_prop' in config:
        for key, methods in config['reach_methods_by_prop'].items():
            if key != '_default' and key in vnnlib_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'],
                        'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs}
        return {'reach_methods': config['reach_methods_by_prop']['_default'],
                'n_rand': config['n_rand'], 'falsify_method': falsify_method,
                'falsify_kwargs': falsify_kwargs}

    if 'reach_methods_by_model' in config:
        for key, methods in config['reach_methods_by_model'].items():
            if key != '_default' and key in onnx_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'],
                        'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs}
        return {'reach_methods': config['reach_methods_by_model']['_default'],
                'n_rand': config['n_rand'], 'falsify_method': falsify_method,
                'falsify_kwargs': falsify_kwargs}

    if 'reach_methods_transformer' in config:
        if 'transformer' in onnx_path.lower():
            return {'reach_methods': config['reach_methods_transformer'],
                    'n_rand': config['n_rand'], 'falsify_method': falsify_method,
                    'falsify_kwargs': falsify_kwargs}

    return {'reach_methods': config.get('reach_methods', DEFAULT_CONFIG['reach_methods']),
            'n_rand': config.get('n_rand', DEFAULT_CONFIG['n_rand']),
            'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs}
