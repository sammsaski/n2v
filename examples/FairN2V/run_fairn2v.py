"""
FairN2V - Main Runner Script
Runs the complete FairN2V verification pipeline on the Adult-Income
classifier (counterfactual + individual fairness).

USAGE:
  python run_fairn2v.py

OUTPUTS (under FairN2V/results/<timestamp>/):
  - CSV files with verification results
  - PNG/PDF figures
  - LaTeX tables (counterfactual, timing)

REQUIREMENTS:
  - n2v toolbox installed (import n2v)
  - ONNX models in FairN2V/models/ (AC-1.onnx, AC-3.onnx)
  - adult_data.npz in FairN2V/data/
"""

import datetime
import warnings
from pathlib import Path

import n2v
import adult_verify
import plot_results

## ================== CONFIGURATION ==================
# Defaults below are applied only for fields the caller has not already
# set in a pre-populated `config` struct (e.g., from run_all.sh --smoke)
if 'config' not in locals():
    config = {}

script_dir = Path(__file__).resolve().parent

# Bundled ONNX models and data live next to this script
config.setdefault('models_dir', script_dir / 'models')
config.setdefault('data_dir', script_dir / 'data')

# Timestamped output directory: results/<timestamp>/
if 'output_dir' not in config:
    ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    config['output_dir'] = script_dir / 'results' / ts

# Data file name
config.setdefault('data_file', 'adult_data.npz')

# Models to verify (must match filenames without .onnx extension)
# AC-1: 13→16→8→2 (2 hidden, narrow)
# AC-3: 13→50→2 (1 hidden, shallow)
config.setdefault('model_list', ['AC-1', 'AC-3'])

# Number of samples to test (default: 100)
config.setdefault('num_obs', 100)

# Random seed for reproducibility
config.setdefault('random_seed', 500)

# Timeout per epsilon value in seconds (default: 600 = 10 minutes)
config.setdefault('timeout', 600)

# Epsilon values for verification
# 0.0 = counterfactual fairness (flip sensitive attribute only)
# >0.0 = individual fairness (flip SA + perturb numerical features)
config.setdefault('epsilon_counterfactual', [0.0])
config.setdefault('epsilon_individual', [0.01, 0.02, 0.03, 0.05, 0.07, 0.1])

# Figure export formats (set to false to skip)
config.setdefault('save_png', True)
config.setdefault('save_pdf', True)

## ================== END CONFIGURATION ==================

## Initialize N2V
print("======= FairN2V Pipeline ==========")
print(" ")
print("Initializing N2V toolbox...")

## Validate Configuration
print("Validating configuration...")

# Check if models directory exists
if not config['models_dir'].is_dir():
    raise FileNotFoundError(f"Models directory not found: {config['models_dir']}")

# Check if data directory exists
if not config['data_dir'].is_dir():
    raise FileNotFoundError(f"Data directory not found: {config['data_dir']}")

# Check if data file exists
data_file_path = config['data_dir'] / config['data_file']
if not data_file_path.is_file():
    raise FileNotFoundError(f"Data file not found: {data_file_path}")

# Check if at least one model exists
model_found = False
for model_name in config['model_list']:
    model_path = config['models_dir'] / f"{model_name}.onnx"
    if model_path.is_file():
        model_found = True
        print(f"  Found model: {model_name}")
    else:
        warnings.warn(f"Model not found: {model_path}")
if not model_found:
    raise FileNotFoundError(f"No models found in: {config['models_dir']}")

# Create output directory if it doesn't exist
if not config['output_dir'].is_dir():
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    print(f"  Created output directory: {config['output_dir']}")
else:
    print(f"  Output directory: {config['output_dir']}")

print(" ")
print("Configuration validated successfully.")
print(" ")

## Run Verification
print("======= STEP 1: Running Verification ==========")
print(" ")

adult_verify.main(config)

print(" ")
print("Verification complete.")
print(" ")

## Run Plotting
print("======= STEP 2: Generating Figures ==========")
print(" ")

plot_results.main(config)

print(" ")
print("======= FairN2V Pipeline Complete ==========")
print(" ")
print(f"All results saved to: {config['output_dir']}")
