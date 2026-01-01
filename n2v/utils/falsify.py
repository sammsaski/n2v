"""
Falsification via random sampling.

This module provides functionality to find counterexamples by randomly sampling
inputs and checking if any output violates the property specification.
"""

import numpy as np
import torch
from typing import Union, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from n2v.sets import HalfSpace


def falsify(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Attempt to find a counterexample by random sampling.

    Samples random inputs uniformly from [lb, ub], runs them through the model,
    and checks if any output violates the property (i.e., falls inside the
    unsafe halfspace region).

    Args:
        model: PyTorch neural network model
        lb: Lower bounds of input region (n,) or (n, 1)
        ub: Upper bounds of input region (n,) or (n, 1)
        property: Property specification (unsafe region), can be:
                  - dict with 'Hg' field containing HalfSpace(s)
                  - list of dicts with 'Hg' field
                  - HalfSpace object
                  - list of HalfSpace objects
        n_samples: Number of random samples to try (default: 500)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Tuple of (result, counterexample) where:
        - result: 0 if counterexample found (SAT), 2 if no counterexample (unknown)
        - counterexample: Tuple of (input, output) if found, None otherwise

    Example:
        >>> import torch
        >>> import numpy as np
        >>> from n2v.utils import falsify, load_vnnlib
        >>>
        >>> model = torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.ReLU())
        >>> prop = load_vnnlib('property.vnnlib')
        >>> result, cex = falsify(model, prop['lb'], prop['ub'], prop['prop'], n_samples=500)
        >>> if result == 0:
        ...     print(f"Counterexample found: input={cex[0]}, output={cex[1]}")
    """
    from n2v.sets import HalfSpace

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Ensure lb, ub are 1D
    lb = np.asarray(lb, dtype=np.float32).flatten()
    ub = np.asarray(ub, dtype=np.float32).flatten()

    if lb.shape != ub.shape:
        raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

    input_dim = lb.shape[0]

    # Process property to get list of HalfSpaces
    halfspaces = _extract_halfspaces(property)

    # Generate random samples uniformly in [lb, ub]
    # Shape: (n_samples, input_dim)
    samples = np.random.uniform(lb, ub, size=(n_samples, input_dim)).astype(np.float32)

    # Run model in eval mode without gradients
    model.eval()
    with torch.no_grad():
        # Process samples one at a time (some ONNX models don't support batching)
        for i in range(n_samples):
            sample = samples[i:i+1]  # Keep batch dimension (1, input_dim)
            sample_tensor = torch.from_numpy(sample)

            output = model(sample_tensor)
            output_np = output.numpy().flatten()

            # Check if output violates property (is inside unsafe region)
            for hs in halfspaces:
                if hs.contains(output_np):
                    # Property violated - counterexample found
                    counterexample = (samples[i], output_np)
                    return 0, counterexample

    # No counterexample found
    return 2, None


def _extract_halfspaces(property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']]) -> List['HalfSpace']:
    """
    Extract list of HalfSpace objects from various property formats.

    Args:
        property: Property specification in various formats

    Returns:
        List of HalfSpace objects
    """
    from n2v.sets import HalfSpace

    # Handle list of dicts (from vnnlib)
    if isinstance(property, list) and len(property) > 0 and isinstance(property[0], dict):
        property = property[0]
        property = property['Hg']
    elif isinstance(property, dict):
        property = property['Hg']

    # Ensure we have a list
    if isinstance(property, HalfSpace):
        return [property]
    elif isinstance(property, list):
        return property
    else:
        raise TypeError(f"Property must be HalfSpace, list of HalfSpace, or dict with 'Hg' field, got {type(property)}")
