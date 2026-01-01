"""
Falsification techniques for neural network verification.

This module provides functionality to find counterexamples using various methods:
- random: Fast, broad exploration via uniform sampling
- pgd: Targeted search using Projected Gradient Descent

NOTE: The current implementation assumes the input set is a hyperbox
(axis-aligned bounds). It samples uniformly from [lb, ub] and projects PGD
steps onto these bounds. For more complex input regions (e.g., polytopes defined
by general linear constraints), this approach may:
- Miss valid counterexamples outside the hyperbox but inside the true input set
- Find false counterexamples inside the hyperbox but outside the true input set

For ACAS Xu and similar benchmarks where inputs are axis-aligned bounds, this
is not an issue. Future work could extend this to support polytope input sets
using hit-and-run sampling (random) or LP-based projection (PGD).

Usage:
    from n2v.utils import falsify

    # Random sampling (default)
    result, cex = falsify(model, lb, ub, property)

    # PGD
    result, cex = falsify(model, lb, ub, property, method='pgd')

    # Combined: try random first, then PGD
    result, cex = falsify(model, lb, ub, property, method='random+pgd')
"""

import numpy as np
import torch
from typing import Union, List, Optional, Tuple, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from n2v.sets import HalfSpace


# Type alias for falsification results
FalsifyResult = Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]

# Available falsification methods
METHODS = ['random', 'pgd', 'random+pgd']


def falsify(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    method: str = 'random',
    seed: Optional[int] = None,
    **kwargs
) -> FalsifyResult:
    """
    Attempt to find a counterexample using the specified falsification method.

    Note:
        This function assumes the input set is a hyperbox [lb, ub]. For input
        sets defined by general linear constraints (polytopes), the sampling
        and projection may not cover the true input region correctly.

    Args:
        model: PyTorch neural network model
        lb: Lower bounds of input region (n,) or (n, 1). Defines hyperbox input set.
        ub: Upper bounds of input region (n,) or (n, 1). Defines hyperbox input set.
        property: Property specification (unsafe region), can be:
                  - dict with 'Hg' field containing HalfSpace(s)
                  - list of dicts with 'Hg' field
                  - HalfSpace object
                  - list of HalfSpace objects
        method: Falsification method to use:
                - 'random': Uniform random sampling (default)
                - 'pgd': Projected Gradient Descent
                - 'random+pgd': Try random first, then PGD if no counterexample found
        seed: Random seed for reproducibility (default: None)
        **kwargs: Method-specific arguments:
            For 'random':
                - n_samples (int): Number of random samples (default: 500)
            For 'pgd':
                - n_restarts (int): Number of random restarts (default: 10)
                - n_steps (int): Steps per restart (default: 50)
                - step_size (float): Step size (default: auto)
            For 'random+pgd':
                - All of the above

    Returns:
        Tuple of (result, counterexample) where:
        - result: 0 if counterexample found (SAT), 2 if no counterexample (unknown)
        - counterexample: Tuple of (input, output) if found, None otherwise

    Example:
        >>> import torch
        >>> from n2v.utils import falsify, load_vnnlib
        >>>
        >>> model = torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.ReLU())
        >>> prop = load_vnnlib('property.vnnlib')
        >>>
        >>> # Random sampling
        >>> result, cex = falsify(model, prop['lb'], prop['ub'], prop['prop'])
        >>>
        >>> # PGD
        >>> result, cex = falsify(model, prop['lb'], prop['ub'], prop['prop'],
        ...                       method='pgd', n_restarts=20)
        >>>
        >>> # Combined approach
        >>> result, cex = falsify(model, prop['lb'], prop['ub'], prop['prop'],
        ...                       method='random+pgd', n_samples=1000, n_restarts=10)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {METHODS}")

    if method == 'random':
        return _falsify_random(model, lb, ub, property, seed=seed, **kwargs)
    elif method == 'pgd':
        return _falsify_pgd(model, lb, ub, property, seed=seed, **kwargs)
    elif method == 'random+pgd':
        # Try random first
        result, cex = _falsify_random(model, lb, ub, property, seed=seed, **kwargs)
        if result == 0:
            return result, cex
        # Then try PGD
        return _falsify_pgd(model, lb, ub, property, seed=seed, **kwargs)

    # Should not reach here
    raise ValueError(f"Unknown method '{method}'")


def _falsify_random(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    n_samples: int = 500,
    seed: Optional[int] = None,
    **kwargs  # Ignore extra kwargs for compatibility with combined methods
) -> FalsifyResult:
    """
    Attempt to find a counterexample by random sampling.

    Samples random inputs uniformly from [lb, ub], runs them through the model,
    and checks if any output violates the property.

    Args:
        model: PyTorch neural network model
        lb: Lower bounds of input region
        ub: Upper bounds of input region
        property: Property specification (unsafe region)
        n_samples: Number of random samples to try (default: 500)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (result, counterexample)
    """
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
    samples = np.random.uniform(lb, ub, size=(n_samples, input_dim)).astype(np.float32)

    # Run model in eval mode without gradients
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            sample = samples[i:i+1]
            sample_tensor = torch.from_numpy(sample)

            output = model(sample_tensor)
            output_np = output.numpy().flatten()

            # Check if output violates property (is inside unsafe region)
            for hs in halfspaces:
                if hs.contains(output_np):
                    counterexample = (samples[i], output_np)
                    return 0, counterexample

    return 2, None


def _falsify_pgd(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    n_restarts: int = 10,
    n_steps: int = 50,
    step_size: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs  # Ignore extra kwargs for compatibility with combined methods
) -> FalsifyResult:
    """
    Attempt to find a counterexample using Projected Gradient Descent (PGD).

    PGD iteratively optimizes inputs to find outputs that violate the property.
    For each halfspace constraint G @ y <= g, PGD minimizes the maximum constraint
    margin to push the output into the unsafe region.

    Args:
        model: PyTorch neural network model
        lb: Lower bounds of input region
        ub: Upper bounds of input region
        property: Property specification (unsafe region)
        n_restarts: Number of random restarts (default: 10)
        n_steps: Number of PGD steps per restart (default: 50)
        step_size: Step size for gradient descent (default: auto)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (result, counterexample)
    """
    # Set random seeds if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Ensure lb, ub are 1D numpy arrays
    lb = np.asarray(lb, dtype=np.float32).flatten()
    ub = np.asarray(ub, dtype=np.float32).flatten()

    if lb.shape != ub.shape:
        raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

    input_dim = lb.shape[0]

    # Convert bounds to tensors
    lb_tensor = torch.from_numpy(lb)
    ub_tensor = torch.from_numpy(ub)

    # Auto-compute step size if not provided (1% of input range)
    if step_size is None:
        input_range = (ub - lb).max()
        step_size = input_range * 0.01

    # Process property to get list of HalfSpaces
    halfspaces = _extract_halfspaces(property)

    # Convert HalfSpace constraints to tensors for gradient computation
    hs_tensors = []
    for hs in halfspaces:
        G = torch.from_numpy(hs.G.astype(np.float32))
        g = torch.from_numpy(hs.g.astype(np.float32).flatten())
        hs_tensors.append((G, g))

    # Put model in eval mode but we need gradients
    model.eval()

    for _ in range(n_restarts):
        # Initialize with random input in [lb, ub]
        x = torch.from_numpy(
            np.random.uniform(lb, ub, size=(1, input_dim)).astype(np.float32)
        )
        x.requires_grad = True

        for _ in range(n_steps):
            # Forward pass
            output = model(x)

            # Compute loss: minimize max(G @ y - g) to push into unsafe region
            total_loss = torch.tensor(float('inf'))

            for G, g in hs_tensors:
                margins = G @ output.flatten() - g
                max_margin = margins.max()

                if max_margin < total_loss:
                    total_loss = max_margin

            # Check if we found a counterexample
            if total_loss.item() <= 0:
                output_np = output.detach().numpy().flatten()
                input_np = x.detach().numpy().flatten()
                return 0, (input_np, output_np)

            # Backward pass
            if x.grad is not None:
                x.grad.zero_()
            total_loss.backward()

            # PGD step: move in negative gradient direction
            with torch.no_grad():
                x = x - step_size * x.grad.sign()
                x = torch.clamp(x, lb_tensor, ub_tensor)

            x.requires_grad = True

        # Final check after all steps
        with torch.no_grad():
            output = model(x)
            output_np = output.numpy().flatten()

            for hs in halfspaces:
                if hs.contains(output_np):
                    input_np = x.numpy().flatten()
                    return 0, (input_np, output_np)

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
