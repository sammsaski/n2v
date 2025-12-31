# n2v Benchmarks

Performance benchmarks for n2v neural network verification.

## Quick Start

```bash
# Run benchmarks and compare to baseline (does NOT overwrite latest.json)
python run_benchmarks.py

# Run specific category
python run_benchmarks.py --category cnn

# Save results after confirming improvements
python run_benchmarks.py --save
```

## Development Workflow

1. **Before changes**: `results/latest.json` is the committed baseline
2. **Make changes**: Implement your optimization
3. **Compare**: `python run_benchmarks.py` (automatically compares to latest.json)
4. **If acceptable**: `python run_benchmarks.py --save` to update baseline
5. **Commit**: Include updated `latest.json` with your changes

---

## Network Architectures

### Fully-Connected Networks

| Model | Architecture | Input | Output | Parameters |
|-------|--------------|-------|--------|------------|
| fc_mnist | 784→50→20→10 | 784 | 10 | 40,480 |
| fc_mnist_small | 784→32→16→10 | 784 | 10 | 25,818 |
| toy_fc_4_3_2 | 4→3→2 | 4 | 2 | 23 |
| toy_fc_8_4_2 | 8→4→2 | 8 | 2 | 46 |

### Convolutional Networks

| Model | Architecture | Input | Output | Parameters |
|-------|--------------|-------|--------|------------|
| cnn_conv_relu | Conv(1→4,k5,s2)→ReLU→576→32→10 | 1×28×28 | 10 | 18,898 |
| cnn_avgpool | Conv(1→4,k3)→ReLU→AvgPool(k4)→196→32→10 | 1×28×28 | 10 | 6,674 |
| cnn_maxpool | Conv(1→4,k3)→ReLU→MaxPool(k4)→196→32→10 | 1×28×28 | 10 | 6,674 |

All networks use ReLU activations. MNIST models trained on MNIST dataset.

**Architecture notation:** `k` = kernel size, `s` = stride (default s=1 if omitted)

---

## Input Set Construction

Benchmarks verify **local robustness** using L∞ perturbations around test samples.

### L∞ Perturbation Model

Given an input sample `x` and perturbation radius `ε`, the input set is:
```
S = { x' : ||x' - x||_∞ ≤ ε }
```

This is equivalent to element-wise bounds:
```
lb[i] = x[i] - ε
ub[i] = x[i] + ε
```

### Epsilon Values

| Model Type | Epsilon (ε) | Notes |
|------------|-------------|-------|
| MNIST (fc_mnist, cnn_*) | 1/255 ≈ 0.0039 | Standard adversarial robustness |
| Toy models | 0.1 | Larger for testing |

### Clamping

For image inputs (MNIST), bounds are clamped to valid pixel range [0, 1]:
```python
lb = max(image - epsilon, 0)
ub = min(image + epsilon, 1)
```

---

## Current Results

**Last Updated:** 2024-12-31 (after vectorized `estimate_ranges()` optimization)

### Summary by Category

| Category | Benchmarks | Total Time | Avg Time |
|----------|------------|------------|----------|
| FC (Star) | 10 | 1.23s | 0.12s |
| CNN (ImageStar) | 21 | 35.1s | 1.67s |
| Toy (Zono/Box) | 4 | <0.01s | <0.01s |
| **Total** | **35** | **36.3s** | **1.04s** |

*3 CNN exact benchmarks skipped (exponential complexity)*

### Results by Method

| Method | FC Time | CNN Time | Description |
|--------|---------|----------|-------------|
| exact | 0.63s | skipped | Exact reachability (splits on ReLU) |
| approx | 0.03s | 0.35s | Triangle relaxation (no splitting) |
| relax-area-0.25 | 0.18s | 8.50s | Area-based relaxation, 25% relaxed |
| relax-area-0.50 | 0.08s | 5.84s | Area-based relaxation, 50% relaxed |
| relax-area-0.75 | 0.02s | 3.23s | Area-based relaxation, 75% relaxed |
| relax-range-0.25 | 0.19s | 8.43s | Range-based relaxation, 25% relaxed |
| relax-range-0.50 | 0.09s | 5.82s | Range-based relaxation, 50% relaxed |
| relax-range-0.75 | 0.03s | 3.07s | Range-based relaxation, 75% relaxed |

### Results by Network

#### Fully-Connected Networks (Star)

| Benchmark | exact | approx | area-0.25 | area-0.50 | area-0.75 | range-0.25 | range-0.50 | range-0.75 |
|-----------|-------|--------|-----------|-----------|-----------|------------|------------|------------|
| fc_mnist | 0.55s | 0.02s | 0.18s | 0.08s | 0.02s | 0.19s | 0.09s | 0.03s |
| fc_mnist_small | 0.08s | 0.01s | - | - | - | - | - | - |

#### CNN: cnn_conv_relu (ImageStar)

| Method | Time | NNV Time | Speedup | Notes |
|--------|------|----------|---------|-------|
| exact | skipped | - | - | Exponential complexity |
| approx | 0.04s | 4.28s | **100x** | Fast triangle relaxation |
| relax-area-0.25 | 6.67s | 3.26s | 0.5x | LP-based bound refinement |
| relax-area-0.50 | 4.45s | 2.12s | 0.5x | |
| relax-area-0.75 | 2.25s | 1.07s | 0.5x | |
| relax-range-0.25 | 6.64s | 3.17s | 0.5x | |
| relax-range-0.50 | 4.46s | 2.13s | 0.5x | |
| relax-range-0.75 | 2.25s | 1.07s | 0.5x | |

#### CNN: cnn_avgpool (ImageStar)

| Method | Time | NNV Time | Speedup | Notes |
|--------|------|----------|---------|-------|
| exact | skipped | - | - | Exponential complexity |
| approx | 0.16s | 0.14s | 0.9x | AvgPool is linear (no splitting) |
| relax-area-0.25 | 0.19s | 0.11s | 0.6x | |
| relax-area-0.50 | 0.15s | 0.11s | 0.7x | |
| relax-area-0.75 | 0.14s | 0.08s | 0.6x | |
| relax-range-0.25 | 0.18s | 0.11s | 0.6x | |
| relax-range-0.50 | 0.16s | 0.10s | 0.6x | |
| relax-range-0.75 | 0.14s | 0.08s | 0.5x | |

#### CNN: cnn_maxpool (ImageStar)

| Method | Time | NNV Time | Speedup | Notes |
|--------|------|----------|---------|-------|
| exact | skipped | - | - | Exponential complexity |
| approx | 0.15s | 17.64s | **118x** | MaxPool uses over-approximation |
| relax-area-0.25 | 1.64s | 17.57s | **11x** | |
| relax-area-0.50 | 1.24s | 17.14s | **14x** | |
| relax-area-0.75 | 0.84s | 16.61s | **20x** | |
| relax-range-0.25 | 1.61s | 17.34s | **11x** | |
| relax-range-0.50 | 1.20s | 17.00s | **14x** | |
| relax-range-0.75 | 0.68s | 16.57s | **24x** | |

#### Toy Models

| Benchmark | Set Type | Time |
|-----------|----------|------|
| toy_fc_4_3_2 | Zono | <0.001s |
| toy_fc_4_3_2 | Box | <0.001s |
| toy_fc_8_4_2 | Zono | <0.001s |
| toy_fc_8_4_2 | Box | <0.001s |

---

## Methods

| Method | Splitting | Speed | Precision | Use Case |
|--------|-----------|-------|-----------|----------|
| exact | Yes | Slow | Exact | Small networks, precise bounds needed |
| approx | No | Fast | Over-approx | Quick analysis, large networks |
| relax-star-area | Partial | Medium | Tunable | Balance speed/precision via relax factor |
| relax-star-range | Partial | Medium | Tunable | Alternative heuristic for neuron selection |

**Relax factor**: 0 = exact (all neurons optimized), 1 = fully relaxed (no LP optimization)

---

## Command Reference

```bash
# Basic usage (compares to latest.json, does NOT save)
python run_benchmarks.py                        # Run all benchmarks
python run_benchmarks.py --category cnn         # Run CNN benchmarks only
python run_benchmarks.py --model fc_mnist       # Run specific model
python run_benchmarks.py --method approx        # Run specific method

# Save results (updates latest.json baseline)
python run_benchmarks.py --save                 # Run all and save
python run_benchmarks.py --category cnn --save  # Run CNN and save

# Other options
python run_benchmarks.py --include-slow         # Include CNN exact (very slow)
python run_benchmarks.py --no-warmup            # Skip warmup runs
python run_benchmarks.py --quiet                # Minimal output
```

---

## Technical Notes

1. **Timing**: Measures only `reach()` call, not model loading or result extraction
2. **Warmup**: Each benchmark runs once for warmup before timed run
3. **Set types**: FC uses Star, CNN uses ImageStar, Toy uses Zono/Box
4. **Models**: Same as `examples/CompareNNV/` for NNV comparison
5. **NNV Timings**: Static NNV timing data is stored in `run_benchmarks.py` (see `NNV_TIMINGS` dictionary). Update this after re-running CompareNNV experiments.

---

## Test Environment

Results were obtained on the following hardware:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Gold 6238R @ 2.20GHz |
| Cores | 112 (logical) |
| RAM | 504 GB |
| OS | Linux 5.15.0 |
| Python | 3.10+ with NumPy, CVXPY |

Note: Timing results may vary on different hardware. The relative speedups between methods and vs NNV should remain consistent.
