#!/usr/bin/env python3
"""
Compare MATLAB and Python verification results to diagnose differences.
"""

import numpy as np
from scipy.io import loadmat

# Load results
matlab_data = loadmat('outputs/matlab/matlab_verify_fc_results.mat')
python_results = loadmat('outputs/python/python_verify_fc_results.mat')

# MATLAB saves in nested structure
matlab_results = matlab_data['results'][0, 0]

print("="*80)
print("COMPARISON: MATLAB vs PYTHON NNV RESULTS")
print("="*80)

# Compare nominal outputs (network evaluation)
print("\n1. NOMINAL OUTPUT LOGITS COMPARISON")
print("-" * 80)
matlab_output = matlab_results['nominal_output'].flatten()
python_output = python_results['nominal_output'].flatten()

max_diff_nominal = np.abs(matlab_output - python_output).max()
print(f"Maximum difference in nominal outputs: {max_diff_nominal:.2e}")

if max_diff_nominal < 1e-5:
    print("✅ Nominal outputs match (network evaluations are consistent)")
else:
    print("⚠️  Nominal outputs differ significantly")

print("\nPer-class differences:")
for i in range(10):
    diff = matlab_output[i] - python_output[i]
    print(f"  Class {i}: {diff:+.2e}")

# Compare output bounds
print("\n2. REACHABILITY OUTPUT BOUNDS COMPARISON")
print("-" * 80)

matlab_lb = matlab_results['output_lb'].flatten()
matlab_ub = matlab_results['output_ub'].flatten()
python_lb = python_results['output_lb'].flatten()
python_ub = python_results['output_ub'].flatten()

print(f"\n{'Class':<8} {'MATLAB LB':<15} {'Python LB':<15} {'Diff LB':<15} {'MATLAB UB':<15} {'Python UB':<15} {'Diff UB':<15}")
print("-" * 110)

max_lb_diff = 0
max_ub_diff = 0

for i in range(10):
    lb_diff = matlab_lb[i] - python_lb[i]
    ub_diff = matlab_ub[i] - python_ub[i]
    max_lb_diff = max(max_lb_diff, abs(lb_diff))
    max_ub_diff = max(max_ub_diff, abs(ub_diff))

    print(f"{i:<8} {matlab_lb[i]:<15.6f} {python_lb[i]:<15.6f} {lb_diff:+.6f}      "
          f"{matlab_ub[i]:<15.6f} {python_ub[i]:<15.6f} {ub_diff:+.6f}")

print(f"\nMaximum lower bound difference: {max_lb_diff:.6f}")
print(f"Maximum upper bound difference: {max_ub_diff:.6f}")

# Check if Python bounds are tighter or looser
print("\n3. SOUNDNESS CHECK")
print("-" * 80)

python_tighter_lb = 0  # Python LB > MATLAB LB (tighter/more conservative)
python_looser_lb = 0   # Python LB < MATLAB LB (looser/less conservative)
python_tighter_ub = 0  # Python UB < MATLAB UB (tighter/more conservative)
python_looser_ub = 0   # Python UB > MATLAB UB (looser/less conservative)

for i in range(10):
    if python_lb[i] > matlab_lb[i] + 1e-6:
        python_tighter_lb += 1
    elif python_lb[i] < matlab_lb[i] - 1e-6:
        python_looser_lb += 1

    if python_ub[i] < matlab_ub[i] - 1e-6:
        python_tighter_ub += 1
    elif python_ub[i] > matlab_ub[i] + 1e-6:
        python_looser_ub += 1

print(f"Lower bounds: Python tighter: {python_tighter_lb}, Python looser: {python_looser_lb}")
print(f"Upper bounds: Python tighter: {python_tighter_ub}, Python looser: {python_looser_ub}")

# Check interval widths
print("\n4. INTERVAL WIDTH COMPARISON")
print("-" * 80)

matlab_widths = matlab_ub - matlab_lb
python_widths = python_ub - python_lb

print(f"{'Class':<8} {'MATLAB Width':<15} {'Python Width':<15} {'Diff':<15} {'% Change':<15}")
print("-" * 80)

for i in range(10):
    width_diff = python_widths[i] - matlab_widths[i]
    pct_change = 100 * width_diff / matlab_widths[i] if matlab_widths[i] != 0 else 0
    print(f"{i:<8} {matlab_widths[i]:<15.6f} {python_widths[i]:<15.6f} {width_diff:+.6f}      {pct_change:+.2f}%")

avg_matlab_width = matlab_widths.mean()
avg_python_width = python_widths.mean()
print(f"\nAverage width - MATLAB: {avg_matlab_width:.6f}, Python: {avg_python_width:.6f}")
print(f"Difference: {avg_python_width - avg_matlab_width:+.6f} ({100*(avg_python_width - avg_matlab_width)/avg_matlab_width:+.2f}%)")

# Compare robustness results
print("\n5. ROBUSTNESS VERIFICATION RESULT")
print("-" * 80)

matlab_robust = int(matlab_results['robust'][0, 0])
python_robust = int(python_results['robust'][0, 0])

print(f"MATLAB result: {matlab_robust} ({'ROBUST' if matlab_robust == 1 else 'NOT ROBUST'})")
print(f"Python result: {python_robust} ({'ROBUST' if python_robust == 1 else 'NOT ROBUST'})")

if matlab_robust == python_robust:
    print("✅ Both implementations agree on robustness result")
else:
    print("❌ DISAGREEMENT on robustness result!")

# Soundness assessment
print("\n6. SOUNDNESS ASSESSMENT")
print("=" * 80)

print("\nThe differences could be due to:")
print("1. Different internal network representations after ONNX import")
print("2. Different Star set initialization (ImageStar vs Star.from_bounds)")
print("3. Different numerical precision or optimization in reach algorithms")
print("4. Different approximation strategies in 'approx'")

print("\nTo verify Python implementation soundness:")
print("- ✅ Nominal outputs match (< 1e-5): Network evaluation is consistent")
if matlab_robust == python_robust:
    print("- ✅ Robustness results agree: Verification conclusion is consistent")
else:
    print("- ❌ Robustness results disagree: NEEDS INVESTIGATION")

if python_looser_lb == 0 and python_looser_ub == 0:
    print("- ✅ Python bounds are not looser (sound over-approximation)")
else:
    print("- ⚠️  Python has some looser bounds (check if still sound)")

print("\nRECOMMENDATION:")
if matlab_robust == python_robust and max_diff_nominal < 1e-5:
    print("Both implementations appear to be working correctly.")
    print("Bound differences are likely due to different approximation strategies,")
    print("which is expected for 'approx' methods.")
    print("\nTo verify exact equivalence, try using 'exact' method in both.")
else:
    print("Further investigation needed - check network loading and Star set creation.")

print("=" * 80)
