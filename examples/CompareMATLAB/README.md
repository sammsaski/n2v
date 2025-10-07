# Compare NNV-Python with MATLAB NNV

This directory contains examples for comparing verification results between **NNV-Python** and **MATLAB NNV**.

**Goal**: Verify that both implementations produce equivalent results by using a common ONNX model format.

---

## Overview

These notebooks demonstrate:
1. Training a simple FC network on MNIST
2. Exporting to **ONNX** format (usable in both Python and MATLAB)
3. Converting ONNX → PyTorch for NNV-Python
4. Running exact reachability analysis
5. Exporting results for comparison with MATLAB NNV

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   PYTHON WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Train FC model in PyTorch                                │
│ 2. Export to ONNX (fc_mnist.onnx)                           │
│ 3. Save test sample (test_sample.mat)                       │
│ 4. Convert ONNX → PyTorch                                   │
│ 5. Run NNV-Python verification                              │
│ 6. Export results (python_verification_results.mat)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MATLAB WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Load ONNX model (fc_mnist.onnx)                          │
│ 2. Load test sample (test_sample.mat)                       │
│ 3. Create equivalent input set                              │
│ 4. Run MATLAB NNV verification                              │
│ 5. Compare with Python results                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

### Notebooks

1. **`01_train_and_export_onnx.ipynb`** - Training and ONNX export
   - Trains simple FC network (784→50→20→10)
   - Exports to ONNX format
   - Saves test sample in multiple formats
   - Verifies ONNX export

2. **`02_verify_and_compare.ipynb`** - Verification and comparison
   - Loads ONNX model
   - Converts ONNX → PyTorch using `onnx2torch` library
   - Runs NNV-Python reachability analysis
   - Exports results for MATLAB comparison

---

## Quick Start

### Python Setup

```bash
# Install dependencies
pip install torch torchvision numpy scipy matplotlib
pip install onnx onnxruntime onnx2torch
pip install pandas  # For nice tables

# Install NNV-Python
cd ../../
pip install -e .
```

### Run Python Workflow

```bash
jupyter notebook
```

Run in order:
1. `01_train_and_export_onnx.ipynb`
2. `02_verify_and_compare.ipynb`

### Output Files (in `outputs/` directory)

After running Python notebooks:

**Model Files**:
- `fc_mnist.onnx` - ONNX model (use in MATLAB!)
- `fc_mnist_pytorch.pth` - PyTorch checkpoint (reference)

**Test Sample Files**:
- `test_sample.npy` - NumPy format
- `test_sample.csv` - CSV format
- `test_sample.mat` - **MATLAB format** (use in MATLAB!)
- `test_sample_info.txt` - Metadata

**Verification Results**:
- `python_verification_results.mat` - **MATLAB format** (compare!)
- `python_verification_results.npz` - NumPy format
- `python_results_detailed.txt` - Detailed text output
- `results_table.csv` - CSV table

**Visualizations**:
- `test_sample_visualization.png`
- `perturbation_bounds.png`
- `reachable_set_visualization.png`

---

## MATLAB NNV Comparison

### Step 1: Load Model in MATLAB

```matlab
% Load ONNX model
net = importONNXNetwork('outputs/fc_mnist.onnx', 'OutputLayerType', 'regression');
```

### Step 2: Load Test Sample

```matlab
% Load test sample
data = load('outputs/test_sample.mat');
test_image = data.image;
test_label = data.label;
epsilon = 0.01;  % Same as Python
```

### Step 3: Create Input Set

```matlab
% Flatten image
img_flat = reshape(test_image, [784, 1]);

% Create bounds
lb = max(img_flat - epsilon, 0);
ub = min(img_flat + epsilon, 1);

% Create Star set
input_star = Star(lb, ub);
```

### Step 4: Run Verification

```matlab
% Exact reachability
tic;
output_stars = net.reach(input_star, 'exact-star');
elapsed_time = toc;

fprintf('Computation time: %.2f seconds\\n', elapsed_time);
fprintf('Number of output stars: %d\\n', length(output_stars));
```

### Step 5: Compute Output Bounds

```matlab
% Get overall bounds
output_lb = zeros(10, 1);
output_ub = zeros(10, 1);

for i = 1:10
    output_lb(i) = inf;
    output_ub(i) = -inf;

    for j = 1:length(output_stars)
        [lb, ub] = output_stars(j).getRange(i);
        output_lb(i) = min(output_lb(i), lb);
        output_ub(i) = max(output_ub(i), ub);
    end
end

% Display
fprintf('\\nOutput Bounds:\\n');
for i = 1:10
    fprintf('Class %d: [%.10f, %.10f]\\n', i-1, output_lb(i), output_ub(i));
end
```

### Step 6: Compare with Python Results

```matlab
% Load Python results
python_results = load('outputs/python_verification_results.mat');

% Compare bounds
diff_lb = abs(output_lb - python_results.output_lb');
diff_ub = abs(output_ub - python_results.output_ub');

fprintf('\\nComparison with Python:\\n');
fprintf('Max LB difference: %.2e\\n', max(diff_lb));
fprintf('Max UB difference: %.2e\\n', max(diff_ub));

if max([diff_lb; diff_ub]) < 1e-6
    fprintf('✅ Results match! Implementations are equivalent.\\n');
else
    fprintf('⚠️  Differences detected.\\n');
end
```

---

## Expected Results

### Model Architecture

```
Input: 28×28 MNIST image (flattened to 784)
│
├─ Flatten
├─ Linear(784, 50)
├─ ReLU
├─ Linear(50, 20)
├─ ReLU
├─ Linear(20, 10)
│
Output: 10 class logits
```

### Network Output (Nominal Input)

Example output for test sample (values will vary):
```
Class 0: -5.234567
Class 1:  2.456789
Class 2:  0.123456
Class 3: -1.234567
Class 4:  8.901234  <-- PREDICTED
Class 5: -2.345678
Class 6:  1.234567
Class 7: -0.123456
Class 8:  3.456789
Class 9: -4.567890
```

### Output Reachable Set (ε=0.01)

Example bounds (values will vary):
```
Class    Lower Bound         Upper Bound         Range Width
---------------------------------------------------------------
0        -5.245678           -5.223456           0.022222
1         2.445678            2.467890           0.022212
2         0.112345            0.134567           0.022222
3        -1.245678           -1.223456           0.022222
4         8.890123            8.912345           0.022222  <- TRUE
5        -2.356789           -2.334567           0.022222
6         1.223456            1.245678           0.022222
7        -0.134567           -0.112345           0.022222
8         3.445678            3.467890           0.022222
9        -4.578901           -4.556789           0.022222
```

### Comparison Metrics

**Expected tolerances**:
- Network output difference (ONNX vs PyTorch): < 1e-5
- Bound differences (Python vs MATLAB): < 1e-6
- Number of output stars: Should match exactly

---

## Key Comparison Points

### 1. Nominal Network Output

**Python** (from notebook):
```python
onnx_output  # 10 values
```

**MATLAB**:
```matlab
net.evaluate(test_image)  # Should match within 1e-6
```

### 2. Output Reachable Set Bounds

**Python**:
```python
overall_lb  # 10 values (lower bounds)
overall_ub  # 10 values (upper bounds)
```

**MATLAB**:
```matlab
output_lb  % Should match Python overall_lb within 1e-6
output_ub  % Should match Python overall_ub within 1e-6
```

### 3. Number of Output Stars

**Python**:
```python
len(output_stars)  # Number of stars
```

**MATLAB**:
```matlab
length(output_stars)  % Should match exactly
```

### 4. Computation Time

- May differ slightly due to implementation details
- Both should be on same order of magnitude

### 5. Robustness Verdict

- Should agree on robust/not robust
- Based on: `true_class_lb > all_other_classes_ub`

---

## Troubleshooting

### Issue: ONNX import fails in MATLAB

**Solution**: Ensure ONNX opset version compatibility
```python
# In Python, export with specific opset
torch.onnx.export(..., opset_version=11)
```

### Issue: Output bounds don't match

**Possible causes**:
1. Different epsilon values (check!)
2. Different LP solvers → slight numerical differences
3. Implementation bug (investigate)

**Check**:
- Are you using same test sample?
- Same epsilon?
- Same input bounds?

### Issue: Different number of output stars

**Possible causes**:
1. Different splitting strategies
2. Different LP solver tolerances
3. Implementation difference in ReLU handling

**Note**: This could indicate a real difference in implementations!

### Issue: PyTorch conversion doesn't match ONNX

**Solution**: Check layer types and parameters
```python
# Verify conversion
verify_conversion(onnx_path, pytorch_model, test_input)
```

---

## File Formats

### ONNX Model (`fc_mnist.onnx`)

- **Format**: ONNX (Open Neural Network Exchange)
- **Opset Version**: 11
- **Input**: `input` [batch, 1, 28, 28]
- **Output**: `output` [batch, 10]
- **Use in Python**: `onnxruntime` or convert to PyTorch
- **Use in MATLAB**: `importONNXNetwork()`

### Test Sample (`test_sample.mat`)

MATLAB `.mat` file containing:
- `image`: 28×28 image matrix
- `label`: true class label
- `sample_idx`: index in MNIST test set
- `predicted`: predicted class
- `logits`: network output (10 values)

### Results (`python_verification_results.mat`)

MATLAB `.mat` file containing:
- `test_sample_idx`: sample index
- `test_label`: true label
- `epsilon`: perturbation magnitude
- `input_image`: 28×28 image
- `input_lb`: lower bound image
- `input_ub`: upper bound image
- `onnx_output`: network output (10 values)
- `pytorch_output`: converted model output (10 values)
- `output_lb`: reachable set lower bounds (10 values)
- `output_ub`: reachable set upper bounds (10 values)
- `num_output_stars`: number of output stars
- `computation_time`: verification time (seconds)
- `is_robust`: robustness verdict (0 or 1)

---

## Understanding the Results

### What is Being Compared?

**Forward Pass** (Nominal Input):
- Single test image → Single output
- Should match exactly between ONNX and PyTorch
- Use to verify model conversion

**Reachability Analysis** (Perturbed Input):
- Input set (infinitely many inputs) → Output set
- Lower/upper bounds represent ALL possible outputs
- Should match between Python and MATLAB implementations

### Why Use ONNX?

1. **Common Format**: Both Python and MATLAB can load ONNX
2. **Weight Sharing**: Exact same weights in both tools
3. **Fair Comparison**: Eliminates training differences
4. **Reproducibility**: Same model, same input, compare outputs

### Interpretation

**If results match (< 1e-6 difference)**:
✅ Implementations are equivalent
✅ NNV-Python correctly translated from MATLAB
✅ Can trust NNV-Python for verification

**If results differ significantly**:
❌ Potential implementation bug
❌ Different assumptions or algorithms
❌ Needs investigation

---

## Next Steps

After confirming equivalence:

1. **Test on larger models** (more layers, more neurons)
2. **Test different architectures** (CNNs with AvgPool)
3. **Test different perturbation magnitudes**
4. **Test different input samples**
5. **Performance comparison** (speed, memory)

---

## References

- **ONNX**: https://onnx.ai/
- **ONNX Runtime**: https://onnxruntime.ai/
- **MATLAB NNV**: https://github.com/verivital/nnv
- **NNV-Python**: Main README in `../../README.md`

---

## Contact

For issues or questions about these comparison examples:
- Check main [README](../../README.md)
- Report issues on GitHub

---

**Status**: ✅ Ready for MATLAB NNV comparison

These notebooks provide a complete framework for verifying equivalence between NNV-Python and MATLAB NNV implementations.
