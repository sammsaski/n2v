"""
Utility functions for Python verification scripts.
"""

import numpy as np
from scipy.io import savemat, loadmat
from pathlib import Path
from typing import Dict, Any, Tuple, List


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save verification results in MATLAB-compatible format.

    Args:
        results: Dictionary with verification results
        output_path: Path to save .mat file
    """
    # Ensure all values are MATLAB-compatible
    mat_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            mat_results[key] = value
        elif isinstance(value, (int, float)):
            mat_results[key] = np.array([[value]])
        elif isinstance(value, str):
            mat_results[key] = value
        elif isinstance(value, bool):
            mat_results[key] = np.array([[1 if value else 0]])
        elif isinstance(value, list):
            mat_results[key] = np.array(value)
        else:
            mat_results[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), mat_results)


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load verification results from .mat file.

    Args:
        results_path: Path to .mat file

    Returns:
        Dictionary with verification results
    """
    data = loadmat(str(results_path))

    # Extract scalar values
    results = {}
    for key, value in data.items():
        if key.startswith('_'):
            continue
        if isinstance(value, np.ndarray):
            if value.size == 1:
                results[key] = value.item()
            else:
                results[key] = value.flatten() if value.ndim > 1 and 1 in value.shape else value
        else:
            results[key] = value

    return results


def load_test_sample(sample_path: Path, model_type: str = 'mnist') -> Dict[str, Any]:
    """
    Load a test sample from .mat file.

    Args:
        sample_path: Path to sample .mat file
        model_type: 'mnist' or 'toy'

    Returns:
        Dictionary with sample data
    """
    data = loadmat(str(sample_path))

    if model_type == 'mnist':
        return {
            'image': data['image'],
            'image_full': data.get('image_full', data['image']),
            'label': int(data['label'].item()) if 'label' in data else 0,
            'predicted': int(data['predicted'].item()) if 'predicted' in data else 0,
            'logits': data['logits'].flatten() if 'logits' in data else None,
        }
    else:  # toy
        return {
            'input': data['input'].flatten(),
            'output': data['output'].flatten() if 'output' in data else None,
            'input_dim': int(data['input_dim'].item()) if 'input_dim' in data else len(data['input'].flatten()),
            'output_dim': int(data['output_dim'].item()) if 'output_dim' in data else 2,
        }


def compute_robustness(
    lb_out: np.ndarray,
    ub_out: np.ndarray,
    true_label: int
) -> Tuple[bool, str]:
    """
    Check if the network is robust.

    The network is robust if the lower bound of the true class is greater than
    the upper bounds of all other classes.

    Args:
        lb_out: Lower bounds of output (num_classes,)
        ub_out: Upper bounds of output (num_classes,)
        true_label: True class label

    Returns:
        Tuple of (is_robust, reason)
    """
    num_classes = len(lb_out)
    true_class_lb = lb_out[true_label]

    for i in range(num_classes):
        if i != true_label:
            if ub_out[i] >= true_class_lb:
                return False, f"Class {i} upper bound ({ub_out[i]:.6f}) >= true class lower bound ({true_class_lb:.6f})"

    return True, "True class lower bound exceeds all other upper bounds"


def aggregate_bounds(output_sets: List, use_lp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate bounds from multiple output sets.

    Args:
        output_sets: List of output Star/Zono/Box sets
        use_lp: If True, use LP-based get_ranges() for tight bounds.
                If False, use estimate_ranges() for fast over-approximation.

    Returns:
        Tuple of (lb_out, ub_out) arrays
    """
    if not output_sets:
        raise ValueError("No output sets provided")

    # Choose bound computation method
    # get_ranges() uses LP solving for exact bounds (matches NNV's getRanges)
    # estimate_ranges() uses interval arithmetic (faster but looser)
    get_bounds = (lambda s: s.get_ranges()) if use_lp else (lambda s: s.estimate_ranges())

    # Get dimension from first set
    first_lb, first_ub = get_bounds(output_sets[0])
    dim = first_lb.size

    lb_out = np.ones(dim) * np.inf
    ub_out = np.ones(dim) * -np.inf

    for s in output_sets:
        lb_temp, ub_temp = get_bounds(s)
        lb_out = np.minimum(lb_temp.flatten(), lb_out)
        ub_out = np.maximum(ub_temp.flatten(), ub_out)

    return lb_out, ub_out


def format_bounds_table(lb: np.ndarray, ub: np.ndarray, true_label: int = None) -> str:
    """
    Format output bounds as a table string.

    Args:
        lb: Lower bounds
        ub: Upper bounds
        true_label: Optional true class label to highlight

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"{'Class':<8} {'Lower Bound':<20} {'Upper Bound':<20} {'Width':<15}")
    lines.append("-" * 65)

    for i in range(len(lb)):
        width = ub[i] - lb[i]
        marker = " <-- TRUE" if i == true_label else ""
        lines.append(f"{i:<8} {lb[i]:<20.10f} {ub[i]:<20.10f} {width:<15.10f}{marker}")

    return "\n".join(lines)
