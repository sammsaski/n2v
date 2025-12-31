# n2v Benchmarks

Performance benchmarks for n2v neural network verification.

## Quick Start

```bash
# Run all benchmarks
python run_benchmarks.py

# Run by category
python run_benchmarks.py --category cnn

# Compare against baseline after making changes
python run_benchmarks.py --compare latest.json
```

## Development Workflow

1. **Before changes**: `results/latest.json` is the committed baseline
2. **Make changes**: Implement your optimization
3. **Compare**: `python run_benchmarks.py --compare latest.json`
4. **If acceptable**: Commit updated `latest.json` with your changes

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

**Last Updated:** 2024-12-31

### Summary by Category

| Category | Benchmarks | Total Time | Avg Time |
|----------|------------|------------|----------|
| FC (Star) | 10 | 1.67s | 0.17s |
| CNN (ImageStar) | 21 | 61.9s | 2.95s |
| Toy (Zono/Box) | 4 | <0.01s | <0.01s |
| **Total** | **35** | **63.6s** | **1.82s** |

*3 CNN exact benchmarks skipped (exponential complexity)*

### Results by Method

| Method | FC Time | CNN Time | Description |
|--------|---------|----------|-------------|
| exact | 0.83s | skipped | Exact reachability (splits on ReLU) |
| approx | 0.12s | 3.71s | Triangle relaxation (no splitting) |
| relax-area-0.25 | 0.21s | 12.8s | Area-based relaxation, 25% relaxed |
| relax-area-0.50 | 0.10s | 9.71s | Area-based relaxation, 50% relaxed |
| relax-area-0.75 | 0.05s | 6.80s | Area-based relaxation, 75% relaxed |
| relax-range-0.25 | 0.22s | 12.6s | Range-based relaxation, 25% relaxed |
| relax-range-0.50 | 0.10s | 9.71s | Range-based relaxation, 50% relaxed |
| relax-range-0.75 | 0.05s | 6.62s | Range-based relaxation, 75% relaxed |

### Results by Network

#### Fully-Connected Networks (Star)

| Benchmark | exact | approx | area-0.25 | area-0.50 | area-0.75 | range-0.25 | range-0.50 | range-0.75 |
|-----------|-------|--------|-----------|-----------|-----------|------------|------------|------------|
| fc_mnist | 0.73s | 0.09s | 0.21s | 0.10s | 0.05s | 0.22s | 0.10s | 0.05s |
| fc_mnist_small | 0.09s | 0.03s | - | - | - | - | - | - |

#### CNN: cnn_conv_relu (ImageStar)

| Method | Time | Notes |
|--------|------|-------|
| exact | skipped | Exponential complexity |
| approx | 0.41s | Fast triangle relaxation |
| relax-area-0.25 | 7.86s | LP-based bound refinement |
| relax-area-0.50 | 5.29s | |
| relax-area-0.75 | 2.84s | |
| relax-range-0.25 | 7.78s | |
| relax-range-0.50 | 5.36s | |
| relax-range-0.75 | 2.82s | |

#### CNN: cnn_avgpool (ImageStar)

| Method | Time | Notes |
|--------|------|-------|
| exact | skipped | Exponential complexity |
| approx | 1.60s | AvgPool is linear (no splitting) |
| relax-area-0.25 | 1.64s | |
| relax-area-0.50 | 1.57s | |
| relax-area-0.75 | 1.57s | |
| relax-range-0.25 | 1.63s | |
| relax-range-0.50 | 1.57s | |
| relax-range-0.75 | 1.57s | |

#### CNN: cnn_maxpool (ImageStar)

| Method | Time | Notes |
|--------|------|-------|
| exact | skipped | Exponential complexity |
| approx | 1.70s | MaxPool uses over-approximation |
| relax-area-0.25 | 3.25s | |
| relax-area-0.50 | 2.85s | |
| relax-area-0.75 | 2.40s | |
| relax-range-0.25 | 3.21s | |
| relax-range-0.50 | 2.79s | |
| relax-range-0.75 | 2.23s | |

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
# Filter options
python run_benchmarks.py --model fc_mnist       # Specific model
python run_benchmarks.py --method approx        # Specific method
python run_benchmarks.py --category cnn         # Category: fc, cnn, toy

# Execution options
python run_benchmarks.py --include-slow         # Include CNN exact (very slow)
python run_benchmarks.py --no-warmup            # Skip warmup runs
python run_benchmarks.py --quiet                # Minimal output

# Comparison
python run_benchmarks.py --compare latest.json  # Compare to baseline
```

---

## Technical Notes

1. **Timing**: Measures only `reach()` call, not model loading or result extraction
2. **Warmup**: Each benchmark runs once for warmup before timed run
3. **Set types**: FC uses Star, CNN uses ImageStar, Toy uses Zono/Box
4. **Models**: Same as `examples/CompareNNV/` for NNV comparison
