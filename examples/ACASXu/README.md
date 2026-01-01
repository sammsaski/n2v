# ACAS Xu Verification Examples

This directory contains examples for verifying ACAS Xu neural network properties using n2v.

## Overview

ACAS Xu (Airborne Collision Avoidance System X for Unmanned Aircraft) is a safety-critical system that uses neural networks to provide collision avoidance advisories. The verification of these networks is a challenging benchmark problem in the neural network verification community.

## Files

### Data
- `onnx/` - ACAS Xu neural networks in ONNX format (45 networks)
- `vnnlib/` - VNN-LIB property files (10 properties)
- `outputs/` - Output directory for benchmark results and logs

### Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `verify_acasxu.py` | Interactive verification with detailed output | Exploring set types, debugging, learning |
| `run_instance.py` | VNN-COMP verification strategy | Single instance verification for benchmarking |
| `run_benchmark.sh` | Full benchmark runner | Running all 186 VNN-COMP instances |

**`verify_acasxu.py`** - Generalized verification script supporting all set types (Box, Zono, Star, Hexatope, Octatope). Provides detailed output including input/output bounds, timing breakdowns, and step-by-step progress. Best for interactive exploration and comparing different verification approaches.

**`run_instance.py`** - Implements NNV's VNN-COMP 2025 strategy:
1. Falsification (random sampling, PGD, or both)
2. Two-stage verification for prop_3/4 (approx first, then exact if needed)
3. Exact verification for other properties

Outputs machine-parseable results (`RESULT:`, `TIME:`, `METHOD:`) for use by `run_benchmark.sh`.

**`run_benchmark.sh`** - Bash script that runs `run_instance.py` on all 186 VNN-COMP instances with proper timeout handling using bash `timeout`.

## Usage

### Interactive Verification

Use `verify_acasxu.py` for exploring different set types and methods:

```bash
# Default: Star sets with exact method
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib

# Fast verification with Box sets
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set box --method approx

# Star sets with parallel processing
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set star --method exact --parallel --workers 8
```

#### Supported Set Types

| Set Type | Methods | Speed | Precision |
|----------|---------|-------|-----------|
| box | approx | Fastest | Basic |
| zono | approx | Fast | Good |
| star | exact, approx | Moderate | High |
| hexatope | exact, exact-differentiable, approx | Slow | Very High |
| octatope | exact, exact-differentiable, approx | Slow | Very High |

### VNN-COMP Benchmarking

Use `run_instance.py` for single instances with the VNN-COMP strategy:

```bash
# Single instance
python run_instance.py onnx/ACASXU_run2a_2_1_batch_2000.onnx vnnlib/prop_2.vnnlib

# With custom workers and falsification samples
python run_instance.py onnx/ACASXU_run2a_3_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --workers 8 --falsify-samples 1000

# Using PGD falsification
python run_instance.py onnx/ACASXU_run2a_2_1_batch_2000.onnx vnnlib/prop_2.vnnlib \
  --falsify-method pgd --pgd-restarts 20 --pgd-steps 100

# Combined: random sampling first, then PGD
python run_instance.py onnx/ACASXU_run2a_2_1_batch_2000.onnx vnnlib/prop_2.vnnlib \
  --falsify-method random+pgd
```

### Full Benchmark

Use `run_benchmark.sh` to run the complete VNN-COMP benchmark suite:

```bash
# Activate your conda environment first
conda activate n2v

# Run all 186 instances (default 120s timeout)
./run_benchmark.sh

# Run with custom timeout
./run_benchmark.sh --timeout 60

# Run only prop_3 instances
./run_benchmark.sh --property 3

# Run 10 random instances
./run_benchmark.sh --subset 10

# Custom output file
./run_benchmark.sh --csv outputs/my_results.csv

# Use PGD falsification
./run_benchmark.sh --falsify-method pgd

# Combined falsification with custom PGD settings
./run_benchmark.sh --falsify-method random+pgd --pgd-restarts 20 --pgd-steps 100
```

#### Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout N` | Timeout per instance in seconds | 120 |
| `--workers N` | Number of parallel workers | CPU count |
| `--falsify-method M` | Falsification method: random, pgd, random+pgd | random |
| `--falsify-samples N` | Random falsification samples | 500 |
| `--pgd-restarts N` | PGD restarts | 10 |
| `--pgd-steps N` | PGD steps per restart | 50 |
| `--property N` | Only run property N (1-10) | All |
| `--subset N` | Run N randomly selected instances | All |
| `--csv FILE` | Output CSV file | outputs/benchmark_results.csv |

## Verification Results

Results are one of:
- **SAT**: Property violated (counterexample found)
- **UNSAT**: Property holds (verified safe)
- **UNKNOWN**: Cannot determine
- **TIMEOUT**: Time limit exceeded

## VNN-COMP 2025 Benchmark

The full benchmark consists of 186 instances:
- prop_1 through prop_4: 45 instances each (all 45 networks)
- prop_5 through prop_10: 1 instance each (specific networks)

### NNV's VNN-COMP 2025 Results (for comparison)

| Property | Total | SAT | UNSAT | Timeout |
|----------|-------|-----|-------|---------|
| prop_1 | 45 | 0 | 0 | 45 |
| prop_2 | 45 | 31 | 0 | 14 |
| prop_3 | 45 | 3 | 32 | 10 |
| prop_4 | 45 | 3 | 37 | 5 |
| prop_5-10 | 6 | 1 | 0 | 4 |
| **Total** | **186** | **38** | **69** | **78** |

NNV success rate: 57% (107/186 solved) with 116s timeout.

## References

1. Katz, G., et al. "Reluplex: An efficient SMT solver for verifying deep neural networks." *CAV 2017*.
2. VNN-COMP: International Verification of Neural Networks Competition
3. ACAS Xu: Airborne Collision Avoidance System for unmanned aircraft
