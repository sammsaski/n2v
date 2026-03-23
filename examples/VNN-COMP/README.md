# VNN-COMP Benchmark Infrastructure

This directory contains the infrastructure for running [VNN-COMP](https://sites.google.com/view/vnn-comp) (Verification of Neural Networks Competition) benchmarks with n2v.

## Overview

The VNN-COMP runner implements a **3-stage verification strategy**:

1. **Falsification**: Quick counterexample search using random sampling and/or PGD
2. **Approximate reachability**: Sound over-approximation (polynomial time, no splitting)
3. **Exact reachability**: Sound and complete verification (may split, exponential worst-case)

Each stage can prove the result and short-circuit the remaining stages. Per-benchmark configurations in `benchmark_configs.py` tune which stages run and with what parameters.

## Files

| File | Description |
|------|-------------|
| `run_instance.py` | Verify a single ONNX model against a VNNLIB specification |
| `prepare_instance.py` | Load ONNX models, parse VNNLIB specs, create input sets |
| `benchmark_configs.py` | Per-benchmark verification strategies (28 benchmarks configured) |
| `smoke_test.sh` | Run 1 instance from each benchmark (quick compatibility check) |
| `run_benchmark.sh` | Run all instances from a single benchmark directory |

## Prerequisites

```bash
conda activate n2v
pip install -e /path/to/n2v
pip install -e /path/to/n2v/third_party/onnx2torch
```

## Verifying a Single Instance

```bash
python run_instance.py <onnx_model> <vnnlib_spec> [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `onnx` | positional | — | Path to ONNX model file |
| `vnnlib` | positional | — | Path to VNNLIB specification file |
| `--timeout` | int | 120 | Timeout in seconds |
| `--workers` | int | CPU count | Number of parallel LP workers |
| `--no-falsify` | flag | — | Skip falsification stage |
| `--no-approx` | flag | — | Skip approximate reachability stage |
| `--no-exact` | flag | — | Skip exact reachability stage |
| `--falsify-method` | str | `random+pgd` | Falsification method: `random`, `pgd`, or `random+pgd` |
| `--falsify-samples` | int | 500 | Number of random falsification samples |
| `--pgd-restarts` | int | 10 | Number of PGD restarts |
| `--pgd-steps` | int | 50 | Steps per PGD restart |
| `--parallel-regions` | flag | — | Verify disjunctive input regions in parallel |
| `--category` | str | — | Benchmark category (loads per-benchmark config) |
| `--precompute-bounds` | flag | — | Enable Zono pre-pass for dead neuron elimination |

### Examples

```bash
# Basic verification
python run_instance.py model.onnx spec.vnnlib

# Skip falsification, only run approximate
python run_instance.py model.onnx spec.vnnlib --no-falsify --no-exact

# Use a benchmark-specific configuration
python run_instance.py model.onnx spec.vnnlib --category acasxu_2023

# With Zono pre-pass for tighter bounds
python run_instance.py model.onnx spec.vnnlib --precompute-bounds
```

### Output

VNN-COMP compliant output on stdout:
- `sat` — property is violated (counterexample found)
- `unsat` — property is satisfied (verified)
- `unknown` — could not determine
- `timeout` — exceeded time limit
- `error` — an error occurred

If the result is `sat`, the counterexample is printed on the next line:
```
sat
((X_0  0.123456)
 (X_1  0.789012)
 (Y_0  0.345678)
 (Y_1  0.901234))
```

## Running the Smoke Test

The smoke test runs **1 instance from each benchmark** for a quick compatibility check.

```bash
./smoke_test.sh <benchmarks_root> [options]
```

### Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--timeout N` | 120 | Fallback timeout if instance has none |
| `--python PATH` | `python` | Python interpreter to use |
| `--output FILE` | `smoke_test_results.csv` | Output CSV file path |
| `--nnv-csv FILE` | — | NNV results CSV for instance selection (picks instances NNV solved) |

### Example

```bash
# Basic smoke test
./smoke_test.sh /path/to/vnncomp_benchmarks

# Use a specific Python environment and custom output
./smoke_test.sh /path/to/vnncomp_benchmarks \
    --python /path/to/conda/envs/n2v/bin/python \
    --output my_results.csv

# Pick instances that NNV solved (for comparison)
./smoke_test.sh /path/to/vnncomp_benchmarks \
    --nnv-csv /path/to/nnv_results.csv
```

### Instance Selection

- If `--nnv-csv` is provided: picks the first instance where NNV got a definitive result (`sat` or `unsat`)
- Otherwise: picks the first instance from `instances.csv`
- Compressed `.gz` files are automatically decompressed

### Output

The smoke test writes a CSV file and prints a summary:

```csv
benchmark,onnx,vnnlib,result,time,error
acasxu_2023,model.onnx,spec.vnnlib,sat,7.0,
safenlp_2024,model.onnx,spec.vnnlib,unsat,11.1,
cifar100_2024,model.onnx,spec.vnnlib,timeout,100,exceeded 100s
```

```
  sat:     7
  unsat:   5
  unknown: 6
  timeout: 4
  error:   5
  skip:    1
  ----------------------------------------
  solved:  12 / 28
```

## Running a Full Benchmark

The benchmark runner executes **all instances** from a single benchmark directory.

```bash
./run_benchmark.sh <benchmark_dir> [options]
```

### Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--timeout N` | 120 | Timeout per instance |
| `--output FILE` | `results.csv` | Output CSV file |
| `--workers N` | CPU count | Parallel LP workers |
| `--no-falsify` | — | Skip falsification stage |
| `--no-approx` | — | Skip approximate reachability |
| `--no-exact` | — | Skip exact reachability |
| `--subset N` | — | Run only the first N instances |

### Expected Directory Structure

```
benchmark_dir/
├── instances.csv       # List of (onnx_file, vnnlib_file, timeout) tuples
├── model1.onnx
├── model2.onnx
├── spec1.vnnlib
└── spec2.vnnlib
```

The `instances.csv` file has no header and each line is:
```
onnx_file,vnnlib_file,timeout
```

### Examples

```bash
# Run all instances with 2-minute timeout
./run_benchmark.sh /path/to/benchmarks/acasxu_2023 --timeout 120

# Run first 10 instances only
./run_benchmark.sh /path/to/benchmarks/acasxu_2023 --subset 10

# Run without falsification
./run_benchmark.sh /path/to/benchmarks/acasxu_2023 --no-falsify
```

### Output

```csv
onnx_file,vnnlib_file,result,time
model1.onnx,spec1.vnnlib,sat,7.3
model2.onnx,spec2.vnnlib,unsat,45.2
```

Summary:
```
Total instances: 100
sat:     30 (30.0%)
unsat:   45 (45.0%)
unknown: 15 (15.0%)
timeout:  5 (5.0%)
error:    5 (5.0%)
----------------------------------------
Solved:  75 (75.0%)
Total time: 2847.3s (47.5m)
```

## Benchmark Configurations

The `benchmark_configs.py` file defines per-benchmark verification strategies. Each config specifies:

- **`reach_methods`**: Ordered list of `(method, kwargs)` tuples to try
- **`n_rand`**: Number of random falsification samples
- **`falsify_method`**: Falsification strategy (`random`, `pgd`, `random+pgd`)

### Configured Benchmarks (28 total)

**Main Track**: acasxu_2023, cersyve, cgan_2023, cifar100_2024, collins_rul_cnn_2022, cora_2024, dist_shift_2023, linearizenn_2024, malbeware, metaroom_2023, nn4sys, safenlp_2024, sat_relu, soundnessbench, tinyimagenet_2024, tllverifybench_2023

**Extended Track**: collins_aerospace_benchmark, lsnc_relu, ml4acopf_2024, relusplitter, traffic_signs_recognition_2023, vggnet16_2022, vit_2023, yolo_2023

**Test Track**: test

### Strategy Examples

| Benchmark | Strategy | Rationale |
|-----------|----------|-----------|
| acasxu_2023 | Property-dependent: prop_3/4 use [approx, exact]; others use [exact] | Small networks; some properties need approx first |
| cifar100_2024 | [probabilistic] | Large ResNet; deterministic methods too slow |
| cora_2024 | [relax-star, approx] with model-specific relaxation factors | Complex models need tuned relaxation |
| traffic_signs_recognition_2023 | [approx, probabilistic] | Sign activations have zero gradients (PGD ineffective) |
| dist_shift_2023 | [exact] | Only exact produces correct results |

### Adding a New Benchmark

Add a new entry to the `CONFIGS` dict in `benchmark_configs.py`:

```python
'my_benchmark': {
    'reach_methods': [
        ('approx', {}),
        ('exact', {}),
    ],
    'n_rand': 100,
    'falsify_method': 'random+pgd',
}
```

For benchmarks that need different configs per model or property, use a function:

```python
'my_benchmark': lambda onnx_path, vnnlib_path: {
    'reach_methods': [('exact', {})] if 'small' in onnx_path else [('approx', {})],
    'n_rand': 500,
}
```

## Smoke Test Results: n2v vs NNV

Comparison of n2v smoke test results against NNV's official VNN-COMP 2025 results on the **exact same instances** (1 instance per benchmark). NNV results are from `vnncomp2025_results/nnv/results.csv`.

### Instances Where Both Tools Produced a Result

| Benchmark | Instance | n2v | Time (s) | NNV | Time (s) | Match |
|-----------|----------|-----|----------|-----|----------|-------|
| acasxu_2023 | ACASXU_run2a_2_1 / prop_2 | sat | 3.9 | sat | 17.0 | yes |
| cersyve | lane_keep_pretrain_inv / prop_lane_keep | sat | 7.2 | sat | 15.6 | yes |
| cgan_2023 | cGAN_imgSz32_nCh_1 / eps_0.010 | sat | 4.3 | sat | 18.9 | yes |
| cifar100_2024 | CIFAR100_resnet_medium / idx_7641 | unsat | 7.9 | unsat | 75.3 | yes |
| collins_rul_cnn_2022 | NN_rul_small_window_20 / delta5_eps10 | sat | 3.8 | sat | 15.0 | yes |
| cora_2024 | mnist-set / mnist-img0 | unsat | 16.1 | unsat | 23.4 | yes |
| dist_shift_2023 | mnist_concat / index6323_delta0.13 | unsat | 129.3 | unsat | 59.3 | yes |
| linearizenn_2024 | AllInOne_10_10 / prop_10_10 | sat | 11.1 | sat | 16.5 | yes |
| malbeware | malimg_family_linear-25 / Obfuscator.AD | unsat | 21.4 | unsat | 49.1 | yes |
| metaroom_2023 | 6cnn_tz_35_5 / spec_idx_176 | unsat | 22.8 | unsat | 17.4 | yes |
| ml4acopf_2024 | 118_ieee_ml4acopf / prop2 | unsat | 13.4 | unsat | 158.6 | yes |
| nn4sys | pensieve_big_parallel / parallel_71 | unsat | 4.5 | unsat | 108.2 | yes |
| relusplitter | mnist-net_256x4 / prop_1_0.05 | sat | 7.3 | sat | 15.6 | yes |
| safenlp_2024 | perturbations_0 / hyperrectangle_2839 | unsat | 7.6 | unsat | 17.0 | yes |
| sat_relu | sat_v65_c187 | sat | 7.0 | sat | 14.6 | yes |
| tinyimagenet_2024 | TinyImageNet_resnet_medium / idx_9262 | sat | 4.2 | sat | 23.4 | yes |
| tllverifybench_2023 | tllBench_n=2_N=M=8 / property_N=8_3 | sat | 4.0 | sat | 16.2 | yes |
| yolo_2023 | TinyYOLO / prop_000005_eps_1_255 | unsat | 19.1 | unsat | 229.6 | yes |

**18/18 agreement** on all instances where both tools returned sat/unsat.

### Instances Where Only One Tool Solved

| Benchmark | Instance | n2v | NNV | Notes |
|-----------|----------|-----|-----|-------|
| test | test_nano | unsat (7.1s) | error | NNV errors on test benchmark |
| vggnet16_2022 | vgg16-7 / spec0_suit | error | sat (62.1s) | n2v missing ONNX file |

### Instances Where Neither Tool Solved

| Benchmark | Instance | n2v | NNV |
|-----------|----------|-----|-----|
| soundnessbench | model / model_0 | timeout (150s) | unknown (107.5s) |
| vit_2023 | pgd_2_3_16 / pgd_2_3_16_2446 | error | timeout (107.1s) |

### Benchmarks Not in NNV

These benchmarks were not run by NNV in VNN-COMP 2025:

| Benchmark | Instance | n2v |
|-----------|----------|-----|
| cctsdb_yolo_2023 | patch-1 / spec_idx_00559 | error |
| collins_aerospace_benchmark | yolov5nano_LRelu_640 / img_14421 | sat (14.3s) |
| lsnc_relu | relu_quadrotor2d_state / state_0 | timeout (25s) |
| traffic_signs_recognition_2023 | 3_30_30_QConv / model_30_idx_7573 | unsat (17.3s) |

### Summary

- **Result agreement**: 18/18 (100%) on instances where both tools returned a definitive result
- **n2v solved**: 21/26 benchmarks (10 sat, 11 unsat, 3 error, 2 timeout)
- **n2v faster on 16/18** shared solved instances (NNV faster on dist_shift_2023 and metaroom_2023)
- n2v results from `smoke_test_results.csv` (112 parallel LP workers)

## Integration with ACAS Xu

The `examples/ACASXu/` directory provides a standalone ACAS Xu runner with additional features. The VNN-COMP infrastructure here is more general and supports all 28+ benchmarks.

For ACAS Xu specifically, see `examples/ACASXu/README.md`.
