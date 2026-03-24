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
from typing import Union, List, Optional, Tuple, Literal
from n2v.sets.halfspace import HalfSpace


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

    Bounds can be any shape matching the model's expected input (excluding the
    batch dimension). For example, pass lb/ub with shape (1, 28, 28) for a CNN
    model that expects (batch, C, H, W) input. Samples are generated uniformly
    in the flattened space, then reshaped to match the bounds' shape before
    passing to the model.

    Args:
        model: PyTorch neural network model
        lb: Lower bounds of input region. Shape should match model input
            (excluding batch dim), e.g. (n,) for FC or (C, H, W) for CNN.
        ub: Upper bounds of input region, same shape as lb.
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
        lb: Lower bounds of input region (any shape matching model input)
        ub: Upper bounds of input region (same shape as lb)
        property: Property specification (unsafe region)
        n_samples: Number of random samples to try (default: 500)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (result, counterexample)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)

    # Remember original shape, flatten for sampling
    orig_shape = lb.shape
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()

    if lb_flat.shape != ub_flat.shape:
        raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

    input_dim = lb_flat.shape[0]

    # Process property to get list of groups (AND of OR)
    groups = _extract_halfspace_groups(property)

    # Generate random samples uniformly in [lb, ub]
    samples = np.random.uniform(lb_flat, ub_flat, size=(n_samples, input_dim)).astype(np.float32)

    # Run model in eval mode without gradients
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            sample_tensor = torch.from_numpy(samples[i]).reshape(1, *orig_shape)

            output = model(sample_tensor)
            output_np = output.numpy().flatten()

            # Check if output satisfies all property groups (AND of OR)
            if _output_satisfies_property(output_np, groups):
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
        lb: Lower bounds of input region (any shape matching model input)
        ub: Upper bounds of input region (same shape as lb)
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

    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)

    # Remember original shape, flatten for sampling/clamping
    orig_shape = lb.shape
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()

    if lb_flat.shape != ub_flat.shape:
        raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

    input_dim = lb_flat.shape[0]

    # Convert bounds to tensors (flat for clamping)
    lb_tensor = torch.from_numpy(lb_flat)
    ub_tensor = torch.from_numpy(ub_flat)

    # Auto-compute step size if not provided (1% of input range)
    if step_size is None:
        input_range = (ub_flat - lb_flat).max()
        step_size = input_range * 0.01

    # Process property to get list of groups (AND of OR)
    groups = _extract_halfspace_groups(property)

    # Convert all HalfSpace constraints to tensors for gradient computation
    group_tensors = []
    for group in groups:
        tensors = []
        for hs in group:
            G = torch.from_numpy(hs.G.astype(np.float32))
            g = torch.from_numpy(hs.g.astype(np.float32).flatten())
            tensors.append((G, g))
        group_tensors.append(tensors)

    # Put model in eval mode but we need gradients
    model.eval()

    for _ in range(n_restarts):
        # Initialize with random input in [lb, ub] (flat for gradient/clamping)
        x = torch.from_numpy(
            np.random.uniform(lb_flat, ub_flat, size=(1, input_dim)).astype(np.float32)
        )
        x.requires_grad = True

        for _ in range(n_steps):
            # Reshape for model forward pass
            output = model(x.reshape(1, *orig_shape))

            # Compute loss: for AND of OR, we need all groups satisfied.
            # For each group (OR): min over halfspaces of max_margin → want <= 0
            # For all groups (AND): max over groups of that min → want <= 0
            # Loss = max over groups of (min over hs in group of max(G @ y - g))
            group_losses = []
            for group_t in group_tensors:
                best_in_group = torch.tensor(float('inf'))
                for G, g in group_t:
                    margins = G @ output.flatten() - g
                    max_margin = margins.max()
                    if max_margin < best_in_group:
                        best_in_group = max_margin
                group_losses.append(best_in_group)

            total_loss = torch.stack(group_losses).max()

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
            output = model(x.reshape(1, *orig_shape))
            output_np = output.numpy().flatten()

            if _output_satisfies_property(output_np, groups):
                input_np = x.numpy().flatten()
                return 0, (input_np, output_np)

    return 2, None


def _extract_halfspace_groups(property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']]) -> List[List['HalfSpace']]:
    """
    Extract property groups from various property formats.

    VNN-LIB properties can have multiple groups (from separate top-level asserts)
    that are ANDed together. Within each group, halfspaces are ORed.

    A counterexample must satisfy ALL groups (AND), where satisfying a group
    means satisfying ANY halfspace within it (OR).

    Args:
        property: Property specification in various formats

    Returns:
        List of groups, where each group is a list of HalfSpace objects (OR within group).
    """
    from n2v.sets.halfspace import HalfSpace

    # Handle list of dicts (from vnnlib) — each dict is a property group
    if isinstance(property, list) and len(property) > 0 and isinstance(property[0], dict):
        groups = []
        for p in property:
            hg = p['Hg']
            if isinstance(hg, HalfSpace):
                groups.append([hg])
            elif isinstance(hg, list):
                groups.append(hg)
            else:
                raise TypeError(f"Property group 'Hg' must be HalfSpace or list, got {type(hg)}")
        return groups
    elif isinstance(property, dict):
        hg = property['Hg']
        if isinstance(hg, HalfSpace):
            return [[hg]]
        elif isinstance(hg, list):
            return [hg]
        else:
            raise TypeError(f"Property 'Hg' must be HalfSpace or list, got {type(hg)}")

    # Single HalfSpace or list of HalfSpace (OR)
    if isinstance(property, HalfSpace):
        return [[property]]
    elif isinstance(property, list):
        return [property]
    else:
        raise TypeError(f"Property must be HalfSpace, list of HalfSpace, or dict with 'Hg' field, got {type(property)}")


def _output_satisfies_property(output_np: np.ndarray, groups: List[List['HalfSpace']]) -> bool:
    """Check if an output satisfies all property groups (AND of OR)."""
    for group in groups:
        # Within each group, at least one halfspace must be satisfied (OR)
        if not any(hs.contains(output_np) for hs in group):
            return False
    return True
