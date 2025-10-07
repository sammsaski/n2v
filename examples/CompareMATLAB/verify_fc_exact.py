#!/usr/bin/env python3
"""
Robustness verification of a NN (L infinity adversarial attack)
If f(x) = y, then forall x' in X s.t. ||x - x'||_inf <= eps, then f(x') = y = f(x)

This script follows the structure of the MATLAB verify_fc.m for direct comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
import time
from pathlib import Path
from onnx2torch import convert
from scipy.io import savemat, loadmat

# Import NNV-Python
from nnv_py.sets import Star
from nnv_py.nn.reach.reach_star import reach_star_exact, reach_star_approx
from nnv_py.utils.model_loader import load_onnx

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Load ONNX Model and Convert to PyTorch
# =============================================================================

print("Loading ONNX model...")
model_name = 'outputs/fc_mnist.onnx'
net = load_onnx(model_name)

num_classes = 10
print(f"✅ Model loaded. Output size: {num_classes}\n")

# =============================================================================
# Load Test Sample
# =============================================================================

print("Loading test sample...")
data = loadmat('test_sample.mat')
img = data['image']
l = 8  # since 7 + 1 (MATLAB indexing)
test_label = int(data['label'][0, 0])

print(f"Test sample loaded. True label: {test_label} (MATLAB index: {l})")

# =============================================================================
# Verification Settings
# =============================================================================

# In MATLAB: reachOptions.reachMethod = "relax-star-area"; reachOptions.relaxFactor = 0.5;
# For comparison, we'll use exact-star method
reach_method = "exact-star"
print(f"Reach method: {reach_method}\n")

# =============================================================================
# Define Perturbation (L-infinity attack)
# =============================================================================

eps = 1/255

print(f"Starting verification with epsilon {eps:.6f}\n")

# Perform L_inf attack (MATLAB lines 33-37)
lb_min = np.zeros_like(img)  # minimum allowed values for lower bound is 0
ub_max = np.ones_like(img)   # maximum allowed values for upper bound is 1
lb_clip = np.maximum((img - eps), lb_min)
ub_clip = np.minimum((img + eps), ub_max)

# Flatten for Star set
img_flat = img.flatten().reshape(-1, 1)
lb_flat = lb_clip.flatten().reshape(-1, 1)
ub_flat = ub_clip.flatten().reshape(-1, 1)

# Create input Star set
IS = Star.from_bounds(lb_flat, ub_flat)

print(f"Input set created:")
print(f"  Dimension: {IS.dim}")
print(f"  L∞ epsilon: {eps}")

# =============================================================================
# Evaluate Nominal, LB, and UB Images (MATLAB lines 39-46)
# =============================================================================

print("\nEvaluating nominal, LB, and UB images...")

# Prepare inputs for PyTorch (add batch and channel dimensions)
img_torch = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
lb_torch = torch.from_numpy(lb_clip).unsqueeze(0).unsqueeze(0).float()
ub_torch = torch.from_numpy(ub_clip).unsqueeze(0).unsqueeze(0).float()

# Evaluate
with torch.no_grad():
    output = net(img_torch).squeeze().numpy()
    P = output.argmax()

    LB_output = net(lb_torch).squeeze().numpy()
    LBPred = LB_output.argmax()

    UB_output = net(ub_torch).squeeze().numpy()
    UBPred = UB_output.argmax()

print(f"Nominal image prediction: {P} (MATLAB index: {P+1})")
print(f"Lower bound prediction:   {LBPred} (MATLAB index: {LBPred+1})")
print(f"Upper bound prediction:   {UBPred} (MATLAB index: {UBPred+1})")

print(f"\nNominal output logits:")
for i, val in enumerate(output):
    marker = " <-- PREDICTED" if i == P else ""
    print(f"  Class {i}: {val:20.10f}{marker}")

# =============================================================================
# Run Verification Algorithm
# =============================================================================

print(f"\nVerification algorithm starting...")
t_start = time.time()

try:
    # Run reachability analysis (MATLAB verify_robustness)
    # In Python NNV, we compute reach set directly
    R_list = reach_star_exact(net, [IS])

    # Check robustness: true class should be highest for all reachable outputs
    res = -1  # -1 means not robust, 1 means robust

    # Get combined bounds from all output stars
    lb_out = np.ones(num_classes) * 1000
    ub_out = np.ones(num_classes) * -1000

    for star in R_list:
        lb_temp, ub_temp = star.estimate_ranges()
        lb_temp = lb_temp.flatten()
        ub_temp = ub_temp.flatten()
        lb_out = np.minimum(lb_temp, lb_out)
        ub_out = np.maximum(ub_temp, ub_out)

    # Check if true class (test_label) is always highest
    true_class_lb = lb_out[test_label]
    robust = True
    for i in range(num_classes):
        if i != test_label:
            if ub_out[i] >= true_class_lb:
                robust = False
                break

    res = 1 if robust else -1

    print("Verification algorithm finished.")

except Exception as e:
    print(f"Error during verification: {e}")
    res = -1
    R_list = []

time_elapsed = time.time() - t_start

print(f"\nResult: {res}")
print(f"Time: {time_elapsed:.6f} seconds")

# =============================================================================
# Get Reachable Sets and Ranges
# =============================================================================

print("\nComputing reach sets...")

# R_list already contains the output stars from reach_star_approx
R = R_list  # List of output Star objects

print("Done computing reach sets!")

print("\nGet the ranges for ub/lb")

# Already computed above, but let's be explicit
lb_out = np.ones(num_classes) * 1000
ub_out = np.ones(num_classes) * -1000

for star in R:
    lb_temp, ub_temp = star.estimate_ranges()
    lb_temp = lb_temp.flatten()
    ub_temp = ub_temp.flatten()
    lb_out = np.minimum(lb_temp, lb_out)
    ub_out = np.maximum(ub_temp, ub_out)

# =============================================================================
# Plotting (MATLAB lines 86-105)
# =============================================================================

print("\nNow to plotting!")

# Get middle point for each output and range sizes
mid_range = (lb_out + ub_out) / 2
range_size = ub_out - mid_range

# Label for x-axis
x = np.arange(10)

# Visualize set ranges and evaluation points
fig = plt.figure(figsize=(10, 6))
plt.errorbar(x, mid_range, yerr=range_size, fmt='.', color='b', linewidth=2,
             capsize=5, capthick=2, markersize=10, label='Reachable Range')
plt.xlim([-0.5, 9.5])
plt.scatter(x, output, s=100, marker='x', color='r', linewidths=2,
            label='Nominal Output', zorder=5)
plt.title('Reachable Outputs')
plt.xlabel('Label')
plt.ylabel('Reachable Output Range on the Input Set')
plt.xticks(x)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

# Save the figure
Path('outputs/exact').mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/exact/python_reach_stmnist_plot.png", dpi=150, bbox_inches='tight')
print("Figure saved to: outputs/exact/python_reach_stmnist_plot.png")
plt.close()

# =============================================================================
# Print Results
# =============================================================================

print("\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)
print(f"Test sample label: {test_label}")
print(f"L∞ epsilon: {eps}")
print(f"Reach method: {reach_method}")
print(f"Number of output stars: {len(R)}")
print(f"Computation time: {time_elapsed:.6f} seconds")
print(f"\nRobustness result: {'VERIFIED ROBUST' if res == 1 else 'NOT ROBUST'}")

print(f"\nOutput bounds:")
print(f"{'Class':<10} {'Lower':<15} {'Upper':<15} {'Width':<15}")
print("-" * 60)
for i in range(num_classes):
    width = ub_out[i] - lb_out[i]
    marker = " <- TRUE" if i == test_label else ""
    print(f"{i:<10} {lb_out[i]:<15.6f} {ub_out[i]:<15.6f} {width:<15.6f}{marker}")

print("="*70)

# =============================================================================
# Save Results for Comparison with MATLAB
# =============================================================================

results = {
    'test_label': test_label,
    'epsilon': eps,
    'reach_method': reach_method,
    'nominal_output': output,
    'lb_image_output': LB_output,
    'ub_image_output': UB_output,
    'nominal_pred': int(P),
    'lb_pred': int(LBPred),
    'ub_pred': int(UBPred),
    'output_lb': lb_out,
    'output_ub': ub_out,
    'mid_range': mid_range,
    'range_size': range_size,
    'num_output_stars': len(R),
    'computation_time': time_elapsed,
    'robust': res,
}

savemat('outputs/exact/python_verify_fc_results.mat', results)
print("\n✅ Results saved to: outputs/exact/python_verify_fc_results.mat")

# Save detailed text report
with open('outputs/exact/python_verify_fc_results.txt', 'w') as f:
    f.write("NNV-PYTHON VERIFICATION RESULTS\n")
    f.write("Matching MATLAB verify_fc.m structure\n")
    f.write("="*70 + "\n\n")

    f.write("TEST CONFIGURATION:\n")
    f.write(f"  Test sample label: {test_label}\n")
    f.write(f"  L∞ epsilon: {eps}\n")
    f.write(f"  Reach method: {reach_method}\n\n")

    f.write("EVALUATION RESULTS:\n")
    f.write(f"  Nominal prediction: {P}\n")
    f.write(f"  Lower bound prediction: {LBPred}\n")
    f.write(f"  Upper bound prediction: {UBPred}\n\n")

    f.write("NOMINAL OUTPUT LOGITS:\n")
    for i, val in enumerate(output):
        marker = " <-- PREDICTED" if i == P else ""
        f.write(f"  Class {i}: {val:20.10f}{marker}\n")
    f.write("\n")

    f.write("REACHABILITY ANALYSIS:\n")
    f.write(f"  Number of output stars: {len(R)}\n")
    f.write(f"  Computation time: {time_elapsed:.6f} seconds\n")
    f.write(f"  Robustness result: {'VERIFIED ROBUST (1)' if res == 1 else 'NOT ROBUST (-1)'}\n\n")

    f.write("OUTPUT BOUNDS:\n")
    f.write(f"{'Class':<10} {'Lower Bound':<20} {'Upper Bound':<20} {'Width':<15}\n")
    f.write("-" * 70 + "\n")
    for i in range(num_classes):
        width = ub_out[i] - lb_out[i]
        marker = " <-- TRUE CLASS" if i == test_label else ""
        f.write(f"{i:<10} {lb_out[i]:<20.10f} {ub_out[i]:<20.10f} {width:<15.10f}{marker}\n")

print("✅ Results saved to: outputs/exact/python_verify_fc_results.txt")

print("\n" + "="*70)
print("COMPARISON INSTRUCTIONS:")
print("="*70)
print("1. Run the MATLAB script: verify_fc.m")
print("2. Compare the following values:")
print("   - Output bounds (lb_out, ub_out)")
print("   - Robustness result (res)")
print("   - Number of output stars")
print("   - Nominal output logits")
print("\nExpected: Bounds should match within numerical precision (< 1e-6)")
print("="*70)
