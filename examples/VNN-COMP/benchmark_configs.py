"""
Per-benchmark verification strategies for VNN-COMP (2025 base, 2026 tuned).

ZERO-RISK POLICY (competition): NO probabilistic reach. Probabilistic verification
returns a coverage set, not a proof, so its 'unsat' is UNSOUND -- a false unsat
scores -150 under the +10/0/-150 model. Every reach method here is sound (approx /
exact star); benchmarks where sound reach can't finish concede to `unknown` (0)
rather than gamble. SAT comes only from falsification, ORT-revalidated at zero
output tolerance before emission.

Mirrors NNV's load_vnncomp_network() from run_vnncomp_instance.m,
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
        # ZERO-RISK: dropped probabilistic (a coverage-set 'unsat' is UNSOUND =
        # -150 under the +10/0/-150 model). Sound approx/exact (small FC control
        # nets; may prove holds, else unknown). n_rand 3000->5000 cracks
        # pendulum_pretrain_con via random sampling (validated +1 sat).
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
    },

    'cgan_2023': {
        # Non-transformer: relax-star-area(0.8) -> approx-star
        # Transformer: probabilistic
        # Resolved by get_config() using onnx_path
        'reach_methods': [
            ('approx', {'relax_factor': 0.8, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        # ZERO-RISK: dropped probabilistic transformer branch. Falsify-only
        # (random finds the small_transformer CE; reach = attention frontier).
        'reach_methods_transformer': [],
        'n_rand': 100,
    },

    # --- 2026 variants that previously fell to DEFAULT (untuned). All SOUND;
    #     none introduce a probabilistic (unsound-unsat) path. ---
    'cgan2026': {
        # Mirror cgan_2023 non-transformer relax-area(0.8). NO probabilistic
        # transformer branch: small_transformer sub-models fall to approx -> unknown
        # (sound); falsify harvests sat.
        'reach_methods': [
            ('approx', {'relax_factor': 0.8, 'relax_method': 'area'}),
            ('approx', {}),
        ],
        'n_rand': 100,
        # The default random+pgd's 500-step PGD leg burns the budget on the
        # UNSAT instances (the SAT ones are found instantly by the random leg,
        # A/B-verified: same 3/3 SATs, 0 regressions), starving the slow cGAN-
        # generator reach. Bounded random+apgd (1x30) frees that time for reach.
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
    },

    'challenging_certified_training_2026': {
        # CNN7: exact never finishes (burns the whole budget) -> approx-only (sound);
        # random falsify (PGD too slow on CNN7, like cifar100/tinyimagenet).
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
    },

    'nn4sys_2023': {
        # mscn-only 2026 dir (no local instances; insurance). SOUND approx -- must
        # NOT be probabilistic (unlike the 'nn4sys' _default). random falsify.
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
    },

    'relusplitter_2026': {
        # Mirror relusplitter (relax-area 1.0); avoids the exact-star timeout.
        'reach_methods': [('approx', {'relax_factor': 1.0, 'relax_method': 'area'})],
        'n_rand': 100,
        # Default 'random+pgd' (500 grad steps on 784-dim) burned ~94s finding
        # no CE on these safe MNIST specs, starving the fast (~30s) sound reach
        # -> timeout. Mirror the extended relusplitter config's bounded
        # 'random+apgd' (1 restart / 30 steps): stronger-per-step AND cheaper,
        # so the reach gets to run and the reach-solvable instances certify.
        # random+apgd was gold-validated to be a superset of random+pgd's CEs
        # on relusplitter (+2, no loss), so this cannot drop a SAT.
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
    },

    'cifar100_2024': {
        # Tiny-eps adversarial CEs on a differentiable ResNet: random alone can't
        # reach them (curse of dimensionality). Use a BOUNDED gradient attack —
        # 'random+apgd' with 1 restart / 30 steps (~31 fwd + 30 bwd) — affordable
        # where the full default PGD (10x50=500 steps, ~220s) was not. Reach stays
        # ZERO-RISK: dropped probabilistic (ResNet sound reach can't finish in
        # budget -> would only ever emit an UNSOUND 'unsat'). Falsify-only
        # (random+apgd) for the tiny-eps adversarial CEs; holds concede to
        # unknown (sound). Reach=[] saves the wasted reach time.
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
        # Bounded gradient attack: full PGD was too slow on 784-dim OnnxMatMul, but
        # random+apgd at 1 restart / 30 steps is affordable and (like cifar100) may
        # crack thin-sliver CEs random misses. cora is the largest sat pool (131).
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
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
            # ZERO-RISK: was probabilistic. Sound approx (mscn/pensieve supported).
            '_default': [('approx', {})],
        },
        'n_rand': 100,
    },

    'safenlp_2024': {
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 500,
    },

    'sat_relu': {
        # Boolean-SAT-encoded ReLU nets have flat gradients: random/PGD find
        # 35/100, gradient-free Square finds 50/100 (validated full corpus,
        # ~0.1s each, strict superset -> no regression). +15 sat. Sound reach
        # kept for the unsat instances.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 100,
        'falsify_method': 'random+square',
        'falsify_kwargs': {'n_iters': 20000},
    },

    'soundnessbench': {
        # SAT instances — planted/binarized counterexamples. Square (gradient-free)
        # is what actually cracks the planted CEs (gradients are flat on binarized
        # nets, so APGD is useless here — same reason sat_relu uses random+square).
        # The old 'strong' cascade ran random->APGD->Square, and APGD's default
        # budget consumed the clock on these heavy (~137ms/forward) nets BEFORE
        # Square ran, so the whole pipeline timed out. Drop APGD (random+square):
        # A/B-verified this keeps every SAT 'strong' found (0 regressions) at
        # ~2.5x the speed, so Square runs within budget. n_rand held at 5000 (no
        # coverage cut); n_iters=20000 mirrors sat_relu.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'random+square',
        'falsify_kwargs': {'n_iters': 20000},
    },

    'soundnessbench_2026': {
        # 2026 variant. Same fix as soundnessbench: random+square (drop the
        # budget-eating, gradient-useless APGD) so Square runs on the planted CEs.
        # A/B-verified 0 regressions vs 'strong', ~2.5x faster.
        'reach_methods': [('approx', {}), ('exact', {})],
        'n_rand': 5000,
        'falsify_method': 'random+square',
        'falsify_kwargs': {'n_iters': 20000},
    },

    'tinyimagenet_2024': {
        # Had no falsify_method -> inherited DEFAULT 'random+pgd' at DEFAULT budget
        # (10x50=500 grad steps, ~200s on a ResNet) -> the falsifier was likely
        # timing out before reach even ran. Switch to a BOUNDED 'random+apgd'
        # (1 restart / 30 steps): fixes the latent timeout AND upgrades to APGD's
        # ZERO-RISK: dropped probabilistic (ResNet sound reach times out -> would
        # only emit an unsound unsat). Falsify-only (random+apgd); holds -> unknown.
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
        # Falsify with random+apgd (was the random+pgd default). Validated (gold-aware,
        # 2026-06-23): at nr=3/ns=50 APGD cracks the gold-sat 300_ieee_prop2 (linear-
        # residual) CE that random/pgd miss and it survives the onnxruntime re-check ->
        # one more sound +10 sat AND one fewer instance that falls through to the
        # (unsound) probabilistic reach and emits a false `unsat` (-150). The cheaper
        # nr=1/ns=30 budget does NOT crack it, so the stronger budget is required here.
        # Independent of the reach decision (#36). (300_ieee base onnx stays uncrackable.)
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 3, 'n_steps': 50},
        # ZERO-RISK: dropped probabilistic. Sound approx (14/118_ieee reach
        # end-to-end via Pow/trig; returns unknown if it can't prove). random+apgd
        # nr3/ns50 cracks the gold-sat 300_ieee_prop2 CE.
        'reach_methods': [('approx', {})],
        'n_rand': 100,
    },

    'relusplitter': {
        # Falsify with random+apgd (was the random+pgd default). Validated gold-aware
        # (2026-06-23): random+apgd cracks gold-sat CEs (mnist_fc, oval21-cifar) that
        # random+pgd misses (+2 on the K=8 sample, ORT-confirmed, no regression).
        'falsify_method': 'random+apgd',
        'falsify_kwargs': {'n_restarts': 1, 'n_steps': 30},
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
        # ZERO-RISK: was probabilistic. 150k-dim ImageNet scale -> falsify-only.
        'reach_methods': [],
        'n_rand': 100,
    },

    'vit_2023': {
        # The runner routes vit -> the LP-free CROWN verifier (verify_vit_instance),
        # so this reach list is unused; cleaned to [] (was probabilistic). n_rand
        # feeds the CROWN path's falsification sample count.
        'reach_methods': [],
        'n_rand': 200,
    },

    'yolo_2023': {
        # ZERO-RISK: was probabilistic. Sound approx (TinyYOLO is small; may prove
        # holds, else unknown). random falsify (PGD too slow for TinyYOLO).
        'reach_methods': [('approx', {})],
        'n_rand': 100,
        'falsify_method': 'random',
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
        - 'falsify_method': str — any method in n2v.utils.falsify.METHODS
          (e.g. 'random', 'random+pgd', 'apgd', 'square', 'strong',
          'random+square', 'random+apgd'); defaults to 'random+pgd'
        - 'falsify_kwargs': dict of per-method budget knobs (e.g.
          {'n_iters': ...} for square, {'n_restarts','n_steps'} for apgd);
          defaults to {}. The runner whitelists these before calling falsify().
    """
    config = BENCHMARK_CONFIGS.get(category, DEFAULT_CONFIG)
    onnx_path = onnx_path or ''
    vnnlib_path = vnnlib_path or ''

    falsify_method = config.get('falsify_method', 'random+pgd')
    # Per-method falsify budgets (e.g. {'n_iters': ...} for square, {'n_restarts','n_steps'}
    # for apgd). The runner whitelists these before passing to falsify(). Default {}.
    falsify_kwargs = config.get('falsify_kwargs', {})

    # Resolve property-specific methods (acasxu)
    if 'reach_methods_by_prop' in config:
        for key, methods in config['reach_methods_by_prop'].items():
            if key != '_default' and key in vnnlib_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'], 'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs}
        return {
            'reach_methods': config['reach_methods_by_prop']['_default'],
            'n_rand': config['n_rand'],
            'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs,
        }

    # Resolve model-specific methods (cora, nn4sys)
    if 'reach_methods_by_model' in config:
        for key, methods in config['reach_methods_by_model'].items():
            if key != '_default' and key in onnx_path:
                return {'reach_methods': methods, 'n_rand': config['n_rand'], 'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs}
        return {
            'reach_methods': config['reach_methods_by_model']['_default'],
            'n_rand': config['n_rand'],
            'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs,
        }

    # Resolve transformer variant (cgan)
    if 'reach_methods_transformer' in config:
        if 'transformer' in onnx_path.lower():
            return {
                'reach_methods': config['reach_methods_transformer'],
                'n_rand': config['n_rand'],
                'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs,
            }

    return {
        'reach_methods': config.get('reach_methods', DEFAULT_CONFIG['reach_methods']),
        'n_rand': config.get('n_rand', DEFAULT_CONFIG['n_rand']),
        'falsify_method': falsify_method, 'falsify_kwargs': falsify_kwargs,
    }
