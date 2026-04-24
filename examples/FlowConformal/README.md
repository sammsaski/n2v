# FlowConformal — flow-matching-based conformal probabilistic reachability

Runnable entry points for the flow-matching probabilistic-reachability project. Library code lives in `n2v/probabilistic/flow/` (models, score functions, training, calibration, scenario verification) and `n2v/sets/volume.py` (volume estimators). This directory holds the demos, benchmarks, and ablations.

For the research context and the consolidated results-so-far, see [`.claude/research/flow-matching-probabilistic-reach/flow-conformal-reference.md`](../../.claude/research/flow-matching-probabilistic-reach/flow-conformal-reference.md).

## Directory map

```
FlowConformal/
├── networks.py          — toy networks (RotatedBananaNet, ThreeBlob{2D,3D})
├── benchmarks/          — production benchmarks, referenced by the paper
│   ├── _common.py                   (star-union-ground-truth runner)
│   ├── _common_analytical.py        (analytical-ground-truth runner)
│   ├── test_rotated_linear.py       (2D + 3D rotated cube, analytical)
│   ├── test_rotated_linear_production.py
│   ├── test_identity_network.py     (cube, sanity)
│   ├── test_banana.py               (2D curved strip)
│   └── test_three_blob_3d.py        (3D multimodal classifier output)
├── ablations/           — quality sweeps that inform config defaults
│   ├── multi_seed_three_blob.py     (6-config × 3-seed training sweep)
│   ├── sweep_logdensity_vs_naive.py (naive vs LogDet comparison)
│   ├── sweep_three_blob_training.py (sinusoidal-time / bigger-net / ReFlow)
│   └── sweep_three_blob_enhanced.py (tight config + dopri5 inference)
├── smokes/              — fast correctness checks
│   ├── smoke_logdensity_gaussian.py (sign-error canary)
│   ├── verify_exact_caches.py       (cross-check cached Star-union volumes)
│   └── fm_validation/               (two_moons, 8_gaussians, checkerboard)
├── viz/                 — visualization demos + figure output dirs
│   ├── star_viz_demo.py
│   ├── flow_viz_demo_3d.py          (flow reachset vs Star-union overlay)
│   ├── figures_star_union/          (Plotly HTMLs for Star union)
│   └── figures_flow_reachset/       (Plotly HTMLs for flow reachset)
├── figures/             — paper-figure generators + their outputs
│   └── flow_matching_training/
└── utils/               — shared benchmark-side helpers (not library)
    └── reach.py         — compute_exact_reach wrapper around n2v.nn.NeuralNetwork
```

The `_archive/` subtree under `.claude/research/flow-matching-probabilistic-reach/` holds the superseded designs, old audits, and 12+ exploratory experiment attempts. Worth a look for design rationale; not worth running.

## How to run

All commands use the project's conda env:

```bash
CONDA=/home/sasakis/miniconda3/envs/n2v/bin/python
```

**A single benchmark (~3 min):**

```bash
$CONDA -m examples.FlowConformal.benchmarks.test_banana
# or: test_three_blob_3d, test_rotated_linear, test_rotated_linear_production
```

**An ablation sweep (~1–2 h, best run overnight with nohup):**

```bash
cd /home/sasakis/v/tools/n2v
nohup $CONDA -u -m examples.FlowConformal.ablations.multi_seed_three_blob \
    > /tmp/sweep.log 2>&1 &
disown

# monitor
tail -f /tmp/sweep.log
pgrep -af multi_seed_three_blob
```

**A smoke (~30 s):**

```bash
$CONDA -m examples.FlowConformal.smokes.smoke_logdensity_gaussian
```

**A viz demo (~3–5 min, writes HTMLs you open in a browser):**

```bash
$CONDA -m examples.FlowConformal.viz.flow_viz_demo_3d
```

## Test suite

Fast (~20 s) per-module subset while iterating:

```bash
$CONDA -m pytest \
    tests/unit/probabilistic/flow/test_logdet_scores.py \
    tests/unit/probabilistic/flow/test_scores.py \
    tests/unit/probabilistic/flow/test_sets.py \
    tests/unit/probabilistic/flow/test_calibrate.py \
    tests/unit/probabilistic/flow/test_scenario_verify.py \
    -q --tb=short
```

Full fast suite (~1 min, skips training-heavy slow-marked tests):

```bash
$CONDA -m pytest tests/unit/probabilistic/flow/ -m "not slow" -q
```

Full suite including slow tests (~5 min; trains flows on Gaussian targets):

```bash
$CONDA -m pytest tests/unit/probabilistic/flow/ -q
```

## Current results headline

From [`benchmarks/`](benchmarks/), production config, 3 seeds:

| Benchmark | flow-naive | hyperrect | ball | coverage |
|---|---:|---:|---:|---:|
| rotated-linear-2D | **1.00×** | 1.65× | 1.38× | 0.990 |
| rotated-linear-3D | **1.02×** | 2.60× | 1.81× | 0.989 |
| banana-2D | **1.42×** | 4.69× | 4.99× | 0.989 |
| three-blob-3D | **5.52×** | 352× | 349× | 0.989 |

Ratio = `volume / ((1-α) × analytical_or_star_union_volume)`, α = 0.01.

On `three-blob-3D`, flow-naive is also **~23× tighter than n2v's approximate deterministic Star reach**, which is the real-world scalable baseline. See the reference doc for full context.

## The production config

If you need to reproduce the default settings anywhere:

```python
# Flow training (tight config used in all PoC benchmarks)
train_flow(
    velocity_field=VelocityField(
        dim=output_dim, hidden=256, n_layers=6, activation='silu',
        time_embed='sinusoidal',
    ),
    training_outputs=y_train,            # 10_000 samples from f(P_X)
    n_epochs=5000, batch_size=2048, lr=1e-3,
    coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=10,
    time_sampling='uniform',
    use_ema=True, ema_decay=0.999,
    standardize_outputs=True,
)

# Inference
FlowScore(flow, t=1.0, n_steps=30, method='rk4', batch_size=65536)

# Calibration
m = 8000; ell = 7999; alpha = 0.001   # Hashemi double-step; β ≈ 0.003
```

See [the reference doc](../../.claude/research/flow-matching-probabilistic-reach/flow-conformal-reference.md) §3 for which knobs are load-bearing and why.
