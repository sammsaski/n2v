#!/usr/bin/env python3
"""
n2v verification script for NNV vs n2v comparison.

This script performs reachability analysis using n2v and saves results
in a format compatible with NNV for comparison.

Usage:
    python verify.py --exp-id 1                    # Run single experiment
    python verify.py --model fc_mnist --method exact  # Run by model/method
    python verify.py --all                         # Run all experiments
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ALL_EXPERIMENTS,
    ExperimentConfig,
    N2V_METHOD_CONFIG,
    get_experiment,
    get_experiments_by_model,
)
from python.utils import (
    save_results,
    load_test_sample,
    compute_robustness,
    aggregate_bounds,
    format_bounds_table,
)

# Import n2v components
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn import NeuralNetwork
from n2v.utils.model_loader import load_onnx


def create_input_set(sample_data: dict, epsilon: float, set_type: str, model_type: str = 'mnist'):
    """
    Create input set based on set type.

    Args:
        sample_data: Sample data dictionary
        epsilon: Perturbation magnitude
        set_type: 'star', 'imagestar', 'zono', or 'box'
        model_type: 'mnist' or 'toy'

    Returns:
        Input set (Star, ImageStar, Zono, or Box)
    """
    if model_type == 'mnist':
        image = sample_data['image']
        # Compute bounds with L-infinity perturbation
        lb = np.maximum(image - epsilon, 0)
        ub = np.minimum(image + epsilon, 1)

        # Flatten for Star/Zono/Box
        # Use C order (row-major) to match PyTorch's flattening convention
        # Note: NNV uses column-major (Fortran order) due to MATLAB convention,
        # so FC bounds will differ but both are sound for their respective input orderings
        lb_flat = lb.flatten().reshape(-1, 1)
        ub_flat = ub.flatten().reshape(-1, 1)

        if set_type == 'star':
            return Star.from_bounds(lb_flat, ub_flat)
        elif set_type == 'imagestar':
            # For CNNs, create ImageStar with proper spatial dimensions
            # MNIST images are 28x28x1
            height, width, num_channels = 28, 28, 1
            return ImageStar.from_bounds(lb_flat, ub_flat, height, width, num_channels)
        elif set_type == 'zono':
            return Zono.from_bounds(lb_flat, ub_flat)
        elif set_type == 'box':
            return Box(lb_flat, ub_flat)
        else:
            raise ValueError(f"Unknown set type: {set_type}")

    else:  # toy model
        input_vec = sample_data['input']
        lb = input_vec - epsilon
        ub = input_vec + epsilon
        lb_flat = lb.reshape(-1, 1)
        ub_flat = ub.reshape(-1, 1)

        if set_type in ['star', 'imagestar']:
            return Star.from_bounds(lb_flat, ub_flat)
        elif set_type == 'zono':
            return Zono.from_bounds(lb_flat, ub_flat)
        elif set_type == 'box':
            return Box(lb_flat, ub_flat)
        else:
            raise ValueError(f"Unknown set type: {set_type}")


def run_verification(exp: ExperimentConfig, base_dir: Path, verbose: bool = True) -> dict:
    """
    Run a single verification experiment.

    Args:
        exp: Experiment configuration
        base_dir: Base directory for CompareNNV
        verbose: Print progress messages

    Returns:
        Dictionary with verification results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Experiment {exp.id}: {exp.model} / {exp.method}")
        print(f"{'='*70}")

    # Determine model type
    model_type = 'toy' if exp.model.startswith('toy') else 'mnist'

    # Load model
    model_dir = base_dir / 'models' / exp.model
    onnx_path = model_dir / f"{exp.model}.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}. Run train_all.py first.")

    if verbose:
        print(f"Loading model: {onnx_path}")
    net = load_onnx(str(onnx_path))

    # Load test sample
    sample_path = base_dir / 'samples' / f"{exp.model}_sample.mat"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}. Run train_all.py first.")

    if verbose:
        print(f"Loading sample: {sample_path}")
    sample_data = load_test_sample(sample_path, model_type)

    # Get label for MNIST, or use 0 for toy
    if model_type == 'mnist':
        true_label = sample_data['label']
        num_classes = 10
    else:
        true_label = 0  # Not applicable for toy
        num_classes = sample_data['output_dim']

    # Create input set
    if verbose:
        print(f"Creating {exp.set_type} input set with epsilon={exp.epsilon:.6f}")
    input_set = create_input_set(sample_data, exp.epsilon, exp.set_type, model_type)

    # Evaluate nominal input
    if model_type == 'mnist':
        image = sample_data['image']
        # Reshape for network: add batch and channel dims if needed
        if image.ndim == 2:
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        with torch.no_grad():
            nominal_output = net(image_tensor).squeeze().numpy()
            nominal_pred = int(nominal_output.argmax())
    else:
        input_vec = sample_data['input']
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).float()
        with torch.no_grad():
            nominal_output = net(input_tensor).squeeze().numpy()
            nominal_pred = int(nominal_output.argmax())

    if verbose:
        print(f"Nominal prediction: {nominal_pred}")

    # Configure reachability method
    method_config = N2V_METHOD_CONFIG[exp.method]
    reach_method = method_config['method']
    reach_kwargs = method_config['kwargs'].copy()

    if exp.relax_factor is not None:
        reach_kwargs['relax_factor'] = exp.relax_factor

    if verbose:
        print(f"Running reachability: method={reach_method}, kwargs={reach_kwargs}")

    # Run reachability analysis
    net_wrapper = NeuralNetwork(net)
    t_start = time.time()

    try:
        output_sets = net_wrapper.reach(input_set, method=reach_method, **reach_kwargs)
        elapsed_time = time.time() - t_start

        # Aggregate bounds
        lb_out, ub_out = aggregate_bounds(output_sets)

        # Check robustness (only for MNIST)
        if model_type == 'mnist':
            is_robust, reason = compute_robustness(lb_out, ub_out, true_label)
            robust = 1 if is_robust else -1
        else:
            is_robust = True  # Not applicable for toy
            robust = 1
            reason = "N/A for toy models"

        success = True
        error_msg = None

    except Exception as e:
        elapsed_time = time.time() - t_start
        lb_out = np.zeros(num_classes)
        ub_out = np.zeros(num_classes)
        is_robust = False
        robust = -1
        reason = str(e)
        success = False
        error_msg = str(e)
        output_sets = []

    # Print results
    if verbose:
        print(f"\nResults:")
        print(f"  Computation time: {elapsed_time:.4f} seconds")
        print(f"  Number of output sets: {len(output_sets)}")
        print(f"  Robustness: {'ROBUST' if is_robust else 'NOT ROBUST'}")
        print(f"  Reason: {reason}")
        print(f"\nOutput bounds:")
        print(format_bounds_table(lb_out, ub_out, true_label if model_type == 'mnist' else None))

    # Prepare results
    results = {
        'experiment_id': exp.id,
        'model': exp.model,
        'method': exp.method,
        'set_type': exp.set_type,
        'epsilon': exp.epsilon,
        'relax_factor': exp.relax_factor if exp.relax_factor else 0,
        'test_label': true_label,
        'nominal_output': nominal_output,
        'nominal_pred': nominal_pred,
        'output_lb': lb_out,
        'output_ub': ub_out,
        'num_output_sets': len(output_sets),
        'computation_time': elapsed_time,
        'robust': robust,
        'success': 1 if success else 0,
    }

    if error_msg:
        results['error'] = error_msg

    # Save results
    output_dir = exp.get_output_dir(base_dir, tool='n2v')
    output_path = output_dir / exp.get_result_filename()
    save_results(results, output_path)

    if verbose:
        print(f"\nResults saved to: {output_path}")

    return results


def run_all_experiments(base_dir: Path, verbose: bool = True):
    """Run all experiments."""
    print(f"\n{'='*70}")
    print(f"Running all {len(ALL_EXPERIMENTS)} experiments")
    print(f"{'='*70}")

    results_summary = []
    for exp in ALL_EXPERIMENTS:
        try:
            result = run_verification(exp, base_dir, verbose=verbose)
            results_summary.append({
                'id': exp.id,
                'model': exp.model,
                'method': exp.method,
                'success': result['success'],
                'robust': result['robust'],
                'time': result['computation_time'],
            })
        except Exception as e:
            print(f"ERROR in experiment {exp.id}: {e}")
            results_summary.append({
                'id': exp.id,
                'model': exp.model,
                'method': exp.method,
                'success': 0,
                'robust': -1,
                'time': 0,
                'error': str(e),
            })

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'ID':<4} {'Model':<20} {'Method':<20} {'Success':<8} {'Robust':<8} {'Time':<10}")
    print("-" * 70)
    for r in results_summary:
        success_str = "OK" if r['success'] else "FAIL"
        robust_str = "YES" if r['robust'] == 1 else "NO"
        print(f"{r['id']:<4} {r['model']:<20} {r['method']:<20} {success_str:<8} {robust_str:<8} {r['time']:<10.4f}")

    succeeded = sum(1 for r in results_summary if r['success'])
    print(f"\nTotal: {succeeded}/{len(results_summary)} succeeded")


def main():
    parser = argparse.ArgumentParser(description='Run n2v verification experiments')
    parser.add_argument('--exp-id', type=int, help='Run specific experiment by ID')
    parser.add_argument('--model', type=str, help='Run all experiments for a model')
    parser.add_argument('--method', type=str, help='Run experiments with specific method')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    verbose = not args.quiet

    if args.all:
        run_all_experiments(base_dir, verbose=verbose)

    elif args.exp_id:
        exp = get_experiment(args.exp_id)
        run_verification(exp, base_dir, verbose=verbose)

    elif args.model:
        experiments = get_experiments_by_model(args.model)
        if args.method:
            experiments = [e for e in experiments if e.method == args.method]
        for exp in experiments:
            try:
                run_verification(exp, base_dir, verbose=verbose)
            except Exception as e:
                print(f"ERROR in experiment {exp.id}: {e}")

    else:
        parser.print_help()
        print("\nAvailable experiments:")
        from config import list_experiments
        list_experiments()


if __name__ == '__main__':
    main()
