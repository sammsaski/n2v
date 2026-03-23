# n2v Benchmarks

**Status: Future Work**

This directory is a placeholder for a standardized benchmark suite that
measures both performance and correctness of n2v verification methods.

## Why This Doesn't Exist Yet

A previous benchmark runner measured only runtime. This was removed because
speed-only benchmarks are misleading for verification tools: a method that
returns trivially loose bounds will always be "faster" than one that computes
tight, sound bounds. Without correctness and tightness metrics, regressions
in verification quality go undetected.

## What a Good Benchmark Suite Needs

A useful regression suite for n2v should capture the **speed-tightness
trade-off** across methods. Each benchmark run should record:

1. **Correctness** -- Are the output bounds sound? Do they contain the true
   reachable set? Compare against ground truth from exact Star analysis.
2. **Tightness** -- How close are approximate bounds to exact bounds?
   Measure as a ratio or absolute gap per output dimension.
3. **Runtime** -- Wall-clock time for the `reach()` call only (not model
   loading or result extraction).
4. **Set count** -- Number of output sets produced (relevant for exact
   methods that split).

### Proposed Metrics Per Benchmark

```
benchmark_name:
  method: exact | approx | relax-area-0.5 | ...
  set_type: star | imagestar | zono | box | hexatope | octatope
  runtime_s: 1.234
  num_output_sets: 42
  bounds_sound: true | false
  tightness_ratio: 0.95      # 1.0 = exact, lower = looser
  lb_max_gap: 0.003           # worst-case gap vs exact lower bound
  ub_max_gap: 0.005           # worst-case gap vs exact upper bound
```

### Proposed Workflow

1. **Establish ground truth**: Run exact Star analysis on each benchmark
   network + input and store the output bounds as reference.
2. **Before changes**: Run the suite, record all metrics.
3. **After changes**: Run the suite again, compare.
4. **Regression criteria**: Flag if any benchmark becomes unsound, loses
   more than 5% tightness, or slows down more than 20% without a
   corresponding tightness improvement.

## Existing Validation

Until this suite is built, use:

- `pytest tests/soundness/` -- Mathematical soundness tests (~190 tests)
- `pytest tests/unit/` -- Unit tests (~1060 tests)
- `examples/ACASXu/run_benchmark.sh` -- Real-world ACAS Xu verification
  with actual SAT/UNSAT outcomes
- `examples/VNN-COMP/` -- VNN-COMP benchmark infrastructure

## Network Architectures (for future use)

### Fully-Connected Networks

| Model | Architecture | Input | Output | Parameters |
|-------|--------------|-------|--------|------------|
| fc_mnist | 784->50->20->10 | 784 | 10 | 40,480 |
| fc_mnist_small | 784->32->16->10 | 784 | 10 | 25,818 |
| toy_fc_4_3_2 | 4->3->2 | 4 | 2 | 23 |
| toy_fc_8_4_2 | 8->4->2 | 8 | 2 | 46 |

### Convolutional Networks

| Model | Architecture | Input | Output | Parameters |
|-------|--------------|-------|--------|------------|
| cnn_conv_relu | Conv(1->4,k5,s2)->ReLU->576->32->10 | 1x28x28 | 10 | 18,898 |
| cnn_avgpool | Conv(1->4,k3)->ReLU->AvgPool(k4)->196->32->10 | 1x28x28 | 10 | 6,674 |
| cnn_maxpool | Conv(1->4,k3)->ReLU->MaxPool(k4)->196->32->10 | 1x28x28 | 10 | 6,674 |

Models and samples are in `examples/CompareNNV/models/` and
`examples/CompareNNV/samples/`.
