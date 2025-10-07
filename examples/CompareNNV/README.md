# Compare n2v with NNV

This directory contains a comprehensive framework for comparing verification results between **n2v** (Python) and **NNV** (MATLAB).

**Goal**: Verify that both implementations produce equivalent results across multiple model architectures, reachability methods, and set representations.

---

## Quick Start

### 1. Setup

```bash
# Ensure n2v conda environment is active
conda activate n2v

# Navigate to CompareNNV directory
cd examples/CompareNNV

# Install dependencies (if not already installed)
pip install -e ../..
pip install -e ../../third_party/onnx2torch
```

### 2. Train Models

```bash
# Train all models and export to ONNX
python models/train_all.py
```

This creates:
- FC models (fc_mnist, fc_mnist_small)
- CNN models (cnn_conv_relu, cnn_avgpool, cnn_maxpool)
- Toy models (toy_fc_4_3_2, toy_fc_8_4_2)

### 3. Run n2v Verification

```bash
# Run all experiments (recommended: skip CNN exact to avoid memory issues)
python python/run_all.py --skip-cnn-exact

# Run all experiments including CNN exact (may run out of memory)
python python/run_all.py

# Or run specific model/method
python python/verify.py --model fc_mnist --method exact
python python/verify.py --exp-id 1
```

**Note:** The `--skip-cnn-exact` flag skips exact reachability for CNN models. Exact reachability on CNNs causes exponential set splitting at ReLU layers, which can exhaust memory. Use `approx` or `relax-star-*` methods for CNNs instead.

### 4. Run NNV Verification (Docker)

NNV runs in a Docker container. The verification scripts are pre-generated in `matlab/scripts/`.

```bash
# Build and run NNV experiments in Docker
./matlab/start_container.sh

# Or run in background with logging
nohup ./matlab/start_container.sh > outputs/logs/nnv.log 2>&1 &

# Monitor progress
tail -f outputs/logs/nnv.log

# Build only (no experiments)
./matlab/start_container.sh --build-only

# Interactive MATLAB shell
./matlab/start_container.sh --shell
```

**Note:** CNN exact reachability experiments are automatically skipped (intractable due to thousands of ReLU neurons). Results are saved to `outputs/nnv/`.

### 5. Compare Results

```bash
# Compare all experiments
python compare/compare_results.py

# Compare specific model
python compare/compare_results.py --model fc_mnist
```

---

## Directory Structure

```
CompareNNV/
├── README.md                     # This file
├── config.py                     # Experiment configurations
├── models/
│   ├── architectures.py          # Model definitions
│   ├── train_all.py              # Training script
│   ├── fc_mnist/                 # FC model files (.onnx, .pth)
│   ├── cnn_avgpool/              # CNN with AvgPool files
│   └── ...
├── python/
│   ├── verify.py                 # n2v verification (single experiment)
│   ├── run_all.py                # Run all experiments
│   └── utils.py                  # Helper functions
├── matlab/
│   ├── Dockerfile                # Docker image for NNV
│   ├── start_container.sh        # Docker container launcher
│   ├── run_all_nnv.m             # Master script to run all experiments
│   └── scripts/                  # Individual verification scripts
│       ├── verify_fc_mnist_exact.m
│       ├── verify_fc_mnist_approx.m
│       └── ...
├── compare/
│   └── compare_results.py        # Comparison tool
├── samples/                      # Test samples (.mat)
├── outputs/
│   ├── n2v/                      # n2v results (.mat)
│   ├── nnv/                      # NNV results (.mat)
│   ├── logs/                     # Experiment logs
│   └── comparisons/              # Comparison reports
```

---

## Model Architectures

### FC Models (MNIST Classification)

| Model | Architecture | Description |
|-------|-------------|-------------|
| fc_mnist | 784→50→20→10 | Standard FC network |
| fc_mnist_small | 784→32→16→10 | Smaller, faster testing |

### CNN Models (MNIST Classification)

| Model | Architecture | Description |
|-------|-------------|-------------|
| cnn_conv_relu | Conv2D→ReLU→FC | Basic CNN |
| cnn_avgpool | Conv2D→ReLU→AvgPool→FC | CNN with average pooling |
| cnn_maxpool | Conv2D→ReLU→MaxPool→FC | CNN with max pooling |

### Toy Models (Zono/Box Testing)

| Model | Architecture | Description |
|-------|-------------|-------------|
| toy_fc_4_3_2 | 4→3→2 | Very small FC |
| toy_fc_8_4_2 | 8→4→2 | Small FC |

---

## Experiment Matrix

### Reachability Methods

| Method | n2v API | NNV Method | Description |
|--------|---------|------------|-------------|
| exact | `method='exact'` | `exact-star` | Exact reachability with ReLU splitting |
| approx | `method='approx'` | `approx-star` | Over-approximate reachability |
| relax-star-area | `method='approx', relax_method='area'` | `relax-star-area` | Relaxed, prioritize by area |
| relax-star-range | `method='approx', relax_method='range'` | `relax-star-range` | Relaxed, prioritize by range |

### Set Types

| Set Type | Use Case | n2v Class |
|----------|----------|-----------|
| star | FC models | `Star` |
| imagestar | CNN models | `Star` (flattened) |
| zono | Toy models | `Zono` |
| box | Toy models | `Box` |

### Full Experiment List

Run `python config.py` to see all experiments:

```
ID   Model                Method               Set Type     Relax Factor
----------------------------------------------------------------------
1    fc_mnist             exact                star         -
2    fc_mnist             approx               star         -
3    fc_mnist             relax-star-area      star         0.25
4    fc_mnist             relax-star-area      star         0.50
5    fc_mnist             relax-star-area      star         0.75
...
```

**Total: ~50 experiments**

---

## Comparison Metrics

The comparison tool checks:

1. **Bound Equivalence**: `max(|n2v_lb - nnv_lb|) < tolerance`
2. **Robustness Agreement**: Both tools agree on ROBUST/NOT ROBUST
3. **Output Set Count**: Number of output stars matches (for exact methods)
4. **Nominal Output**: Forward pass outputs match

### Tolerance Levels

- **Bounds (exact)**: < 1e-6
- **Bounds (approx)**: May differ due to different approximation strategies
- **Nominal output**: < 1e-5

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│  python models/train_all.py                                         │
│     │                                                               │
│     ├── Train FC models on MNIST                                    │
│     ├── Train CNN models on MNIST                                   │
│     ├── Create toy models                                           │
│     ├── Export all to ONNX                                          │
│     └── Save test samples (.mat)                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│            n2v               │   │         NNV (Docker)        │
├─────────────────────────────┤   ├─────────────────────────────┤
│  python python/run_all.py    │   │  ./matlab/start_container.sh│
│    --skip-cnn-exact          │   │     │                       │
│     │                        │   │     ├── Build Docker image  │
│     ├── Load ONNX model      │   │     ├── Mount volumes       │
│     ├── Load test sample     │   │     ├── Run run_all_nnv.m   │
│     ├── Create input set     │   │     └── Save results (.mat) │
│     ├── Run reachability     │   │                             │
│     └── Save results (.mat)  │   │  CNN exact: auto-skipped    │
└─────────────────────────────┘   └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        COMPARISON PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│  python compare/compare_results.py                                  │
│     │                                                               │
│     ├── Load n2v results                                            │
│     ├── Load NNV results                                            │
│     ├── Compare bounds, robustness, timing                          │
│     └── Generate summary report (outputs/comparisons/summary.csv)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Result Format

Both n2v and NNV save results in `.mat` format with these fields:

```
{
    'experiment_id': int,
    'model': str,
    'method': str,
    'set_type': str,
    'epsilon': float,
    'relax_factor': float,
    'test_label': int,
    'nominal_output': array(num_classes,),
    'nominal_pred': int,
    'output_lb': array(num_classes,),
    'output_ub': array(num_classes,),
    'num_output_sets': int,
    'computation_time': float,
    'robust': int (1 or -1),
    'success': int (1 or 0)
}
```

---

## Skipped Experiments

Some experiments are automatically skipped because they are computationally intractable:

| Model | Method | ReLU Count | Reason |
|-------|--------|------------|--------|
| cnn_conv_relu | exact | 608 | Exponential splitting |
| cnn_avgpool | exact | 3,168 | Exponential splitting |
| cnn_maxpool | exact | 3,168 | Exponential splitting |

Exact reachability has worst-case O(2^n) complexity where n is the number of ReLU neurons. With thousands of ReLUs, this is infeasible.

**To include these experiments anyway:**
- Python: Remove `--skip-cnn-exact` flag (not recommended)
- MATLAB: Edit `skipExperiments` list in `run_all_nnv.m`

---

## Docker Requirements (for NNV)

The NNV experiments run in a Docker container with MATLAB R2024b.

### Prerequisites
- Docker installed and running
- `sudo` access for Docker commands (or add user to docker group)
- MATLAB license server accessible (configured in Dockerfile)

### First-time Setup
```bash
# Build the Docker image (takes ~10-15 minutes)
./matlab/start_container.sh --build-only
```

### Troubleshooting Docker
```bash
# Permission denied
# Solution: Use sudo or add user to docker group
sudo usermod -aG docker $USER

# Check container logs
docker logs nnv-verification-run

# Interactive debugging
./matlab/start_container.sh --shell
```

---

## Troubleshooting

### Model not found

```
FileNotFoundError: Model not found: models/fc_mnist/fc_mnist.onnx
```

**Solution**: Run `python models/train_all.py` first.

### MATLAB import fails

```matlab
Error using importONNXNetwork
```

**Solution**: Ensure ONNX opset version is 11 (already set in train_all.py).

### Bounds differ significantly

**Possible causes**:
1. Different LP solvers (numerical precision)
2. Different approximation strategies
3. Implementation bug

**Check**: Compare nominal outputs first - they should match within 1e-5.

### n2v vs NNV Architecture Differences

n2v and NNV use different approaches for MNIST verification that lead to different (but both valid) results:

**n2v approach:**
- Creates a flat `Star` directly from row-major flattened image bounds
- Matches PyTorch's Flatten layer behavior
- Applies Linear layers to the already-flattened Star

**NNV approach:**
- Creates an `ImageStar` from 2D image bounds
- Uses MATLAB's column-major representation internally
- Network's internal Flatten layer converts to column-major flat representation

**Key insight**: The ONNX model was trained with PyTorch's row-major convention. n2v processes it correctly with row-major flattening. NNV uses column-major ImageStar which effectively reorders the input.

Both tools correctly implement their respective approaches, but they process the same network differently. The trace scripts (`trace_n2v.py` and `trace_nnv.m`) demonstrate that when using identical flattening order and bypassing the high-level APIs, both tools produce matching bounds (within ~1e-6).

**Note**: This architectural difference is inherent to how MATLAB and Python handle array indexing. It does not indicate a bug in either tool.

### Different number of output sets

For exact methods, this may indicate:
1. Different ReLU splitting strategies
2. Different constraint handling
3. Numerical precision issues

---

## Extending the Framework

### Adding a New Model

1. Add model class to `models/architectures.py`
2. Add to `MODEL_REGISTRY`
3. Run `python models/train_all.py --model your_model`

### Adding a New Method

1. Add n2v mapping to `N2V_METHOD_CONFIG` in `config.py`
2. Add NNV mapping to `NNV_METHOD_CONFIG` in `config.py`
3. Add experiments to `generate_experiments()` in `config.py`

### Adding a New Set Type

1. Implement in `python/verify.py` `create_input_set()`
2. Add NNV template if needed
3. Add experiments to `config.py`

---

## References

- **n2v**: [../../README.md](../../README.md)
- **NNV**: https://github.com/verivital/nnv
- **ONNX**: https://onnx.ai/

