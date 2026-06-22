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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from n2v.sets.halfspace import HalfSpace


# Soft dependency: AutoAttack (Croce & Hein 2020). Enables
# method='autoattack' as an opt-in adversarial-robustness ensemble
# backend. Not installed by default; `pip install git+https://github.com/
# fra31/auto-attack.git` to enable.
try:
    import autoattack as _autoattack_pkg  # noqa: F401
    _HAS_AUTOATTACK = True
except ImportError:
    _HAS_AUTOATTACK = False


# Type alias for falsification results
FalsifyResult = Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]

# Available falsification methods
METHODS = ['random', 'pgd', 'apgd', 'square', 'strong', 'random+pgd',
           'random+pgd+apgd', 'autoattack']


def _detect_model_device(model) -> torch.device:
    """Return the device the model's parameters/buffers live on.

    Falls back to CPU when the model has neither parameters nor
    buffers (e.g. some ONNX-converted graphs hold their constants as
    Python attributes rather than as registered buffers).
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device('cpu')


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
    elif method == 'apgd':
        return _falsify_apgd(model, lb, ub, property, seed=seed, **kwargs)
    elif method == 'random+pgd+apgd':
        # Cascade: random -> pgd -> apgd. Return on first SAT.
        result, cex = _falsify_random(model, lb, ub, property, seed=seed, **kwargs)
        if result == 0:
            return result, cex
        result, cex = _falsify_pgd(model, lb, ub, property, seed=seed, **kwargs)
        if result == 0:
            return result, cex
        return _falsify_apgd(model, lb, ub, property, seed=seed, **kwargs)
    elif method == 'square':
        return _falsify_square(model, lb, ub, property, seed=seed, **kwargs)
    elif method == 'strong':
        # Self-contained ensemble (no external deps): random sampling ->
        # gradient APGD -> gradient-free Square. Returns on the first SAT.
        for _fn in (_falsify_random, _falsify_apgd, _falsify_square):
            result, cex = _fn(model, lb, ub, property, seed=seed, **kwargs)
            if result == 0:
                return result, cex
        return 2, None
    elif method == 'autoattack':
        return _falsify_autoattack(model, lb, ub, property, seed=seed, **kwargs)

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
    # Use a dedicated numpy Generator seeded from `seed` so the falsifier's
    # randomness depends ONLY on `seed` and not on global numpy state. This
    # makes the verdict order-independent: running this falsifier after any
    # other code that touched np.random gives the same result.
    rng = np.random.default_rng(seed)

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
    samples = rng.uniform(lb_flat, ub_flat, size=(n_samples, input_dim)).astype(np.float32)

    # Run the model BATCHED for speed: a single (chunked) forward over all samples
    # instead of n_samples Python-level forwards. In eval mode the forward is
    # per-sample independent (BatchNorm uses running stats), so batched outputs are
    # identical to one-at-a-time -- this is a pure speedup (e.g. ~28.9s -> <1s for
    # n=5000), and any hit is still CE-validated downstream. Push samples to the
    # model's device for CUDA-resident networks. Falls back to per-sample if a
    # converted graph rejects batch>1.
    device = _detect_model_device(model)
    model.eval()
    CHUNK = 2048
    with torch.no_grad():
        for start in range(0, n_samples, CHUNK):
            chunk = samples[start:start + CHUNK]
            try:
                bt = torch.from_numpy(chunk).reshape(chunk.shape[0], *orig_shape).to(device)
                out_np = model(bt).detach().cpu().numpy().reshape(chunk.shape[0], -1)
            except Exception:  # noqa: BLE001 - converted graph may reject batch>1; fall back to per-sample
                out_np = np.stack([
                    model(torch.from_numpy(chunk[k]).reshape(1, *orig_shape).to(device))
                    .detach().cpu().numpy().flatten()
                    for k in range(chunk.shape[0])
                ])
            for j in range(out_np.shape[0]):
                # Check if output satisfies all property groups (AND of OR)
                if _output_satisfies_property(out_np[j], groups):
                    # The batched forward's row j can differ from the true
                    # single-sample forward (float reordering, or a converted
                    # graph that silently mishandles batch>1). Re-run this one
                    # sample at batch=1 and re-check before emitting `sat`, so the
                    # returned CE is sound by construction and matches what the
                    # external onnxruntime checker will see. A mismatch here can
                    # only cost breadth (miss a CE), never yield a false `sat`.
                    x_ce = samples[start + j]
                    out1 = (model(torch.from_numpy(x_ce).reshape(1, *orig_shape).to(device))
                            .detach().cpu().numpy().flatten())
                    if _output_satisfies_property(out1, groups):
                        return 0, (x_ce, out1)

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
    # Dedicated RNG instances seeded from `seed`. PGD inits use numpy
    # uniform draws; torch operations downstream are deterministic given
    # those inits, but we still seed a torch Generator defensively for any
    # future stochastic torch op. Using local generators (not np.random.seed
    # / torch.manual_seed) makes this falsifier order-independent.
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    if seed is not None:
        torch_gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)

    # Remember original shape, flatten for sampling/clamping
    orig_shape = lb.shape
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()

    if lb_flat.shape != ub_flat.shape:
        raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

    input_dim = lb_flat.shape[0]

    # Detect the model's device so all tensors fed to it (and constraint
    # tensors used in gradient computation) live on the same device.
    device = _detect_model_device(model)

    # Convert bounds to tensors (flat for clamping)
    lb_tensor = torch.from_numpy(lb_flat).to(device)
    ub_tensor = torch.from_numpy(ub_flat).to(device)

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
            G = torch.from_numpy(hs.G.astype(np.float32)).to(device)
            g = torch.from_numpy(hs.g.astype(np.float32).flatten()).to(device)
            tensors.append((G, g))
        group_tensors.append(tensors)

    # Put model in eval mode but we need gradients
    model.eval()

    for _ in range(n_restarts):
        # Initialize with random input in [lb, ub] (flat for gradient/clamping)
        x = torch.from_numpy(
            rng.uniform(lb_flat, ub_flat, size=(1, input_dim)).astype(np.float32)
        ).to(device)
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
                best_in_group = torch.tensor(float('inf'), device=device)
                for G, g in group_t:
                    margins = G @ output.flatten() - g
                    max_margin = margins.max()
                    if max_margin < best_in_group:
                        best_in_group = max_margin
                group_losses.append(best_in_group)

            total_loss = torch.stack(group_losses).max()

            # Check if we found a counterexample
            if total_loss.item() <= 0:
                output_np = output.detach().cpu().numpy().flatten()
                input_np = x.detach().cpu().numpy().flatten()
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
            output_np = output.detach().cpu().numpy().flatten()

            if _output_satisfies_property(output_np, groups):
                input_np = x.detach().cpu().numpy().flatten()
                return 0, (input_np, output_np)

    return 2, None


def _falsify_apgd(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    n_restarts: int = 10,
    n_steps: int = 50,
    step_size: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> FalsifyResult:
    """Auto-PGD (Croce & Hein 2020). PGD with a step-size schedule that
    halves on plateau windows and restarts from best-so-far.

    Args:
        model: PyTorch model accepting the flat-shaped input.
        lb, ub: bounds of the input box (same shape as model input).
        property: VNN-LIB-shaped property.
        n_restarts: independent random inits.
        n_steps: steps per restart.
        step_size: initial step size (default: 5% of max input range).
        seed: RNG seed (numpy + torch).

    Returns:
        (result, counterexample) where result=0 means SAT found,
        result=2 means unknown, and counterexample is (x, y) or None.
    """
    # Dedicated RNG instances; see _falsify_pgd for rationale.
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    if seed is not None:
        torch_gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)
    orig_shape = lb.shape
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()
    input_dim = lb_flat.shape[0]

    if step_size is None:
        step_size = float((ub_flat - lb_flat).max() * 0.05)
    initial_step = step_size

    # Detect the model's device so all tensors live on the same device.
    device = _detect_model_device(model)

    groups = _extract_halfspace_groups(property)
    group_tensors = []
    for group in groups:
        group_tensors.append([
            (torch.from_numpy(hs.G.astype(np.float32)).to(device),
             torch.from_numpy(hs.g.astype(np.float32).flatten()).to(device))
            for hs in group
        ])

    model.eval()

    plateau_window = max(5, n_steps // 5)
    lb_tensor = torch.from_numpy(lb_flat).to(device)
    ub_tensor = torch.from_numpy(ub_flat).to(device)

    for _ in range(n_restarts):
        x = torch.from_numpy(
            rng.uniform(lb_flat, ub_flat, size=(1, input_dim)).astype(np.float32)
        ).to(device)
        x.requires_grad_(True)
        best_loss = float('inf')
        best_x = x.detach().clone()
        local_step = initial_step
        steps_since_improvement = 0

        for _ in range(n_steps):
            output = model(x.reshape(1, *orig_shape))
            group_losses = []
            for group_t in group_tensors:
                best_in_group = torch.tensor(float('inf'), device=device)
                for G, g in group_t:
                    margins = G @ output.flatten() - g
                    max_margin = margins.max()
                    if max_margin < best_in_group:
                        best_in_group = max_margin
                group_losses.append(best_in_group)
            total_loss = torch.stack(group_losses).max()

            if total_loss.item() <= 0:
                output_np = output.detach().cpu().numpy().flatten()
                input_np = x.detach().cpu().numpy().flatten()
                return 0, (input_np, output_np)

            if total_loss.item() < best_loss - 1e-9:
                best_loss = total_loss.item()
                best_x = x.detach().clone()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement >= plateau_window:
                    local_step *= 0.5
                    x = best_x.clone().requires_grad_(True)
                    steps_since_improvement = 0
                    continue

            if x.grad is not None:
                x.grad.zero_()
            total_loss.backward()
            with torch.no_grad():
                grad = x.grad
                x_new = x - local_step * grad.sign()
                x_new = torch.clamp(x_new, lb_tensor, ub_tensor)
            x = x_new.detach().requires_grad_(True)

    return 2, None


def _build_group_arrays(groups: List[List['HalfSpace']]):
    """Precompute (G, g) numpy arrays per halfspace for batched margin evaluation."""
    return [[(np.asarray(hs.G, dtype=np.float32),
              np.asarray(hs.g, dtype=np.float32).flatten()) for hs in group]
            for group in groups]


def _batch_total_margin(outs: np.ndarray, group_arrays) -> np.ndarray:
    """Vectorized AND-of-OR unsafe-region margin for a batch of outputs.

    ``outs``: (N, out_dim). Returns (N,) where a value <= 0 means the point lies in
    the unsafe region (a counterexample). Mirrors ``_output_satisfies_property``:
    AND across groups (max), OR within a group (min over halfspaces), all rows of a
    halfspace must hold (max over rows of ``G @ y - g``).
    """
    group_margins = []
    for group in group_arrays:
        hs_margins = [(outs @ G.T - g).max(axis=1) for G, g in group]  # each (N,)
        group_margins.append(np.min(np.stack(hs_margins, axis=1), axis=1))  # OR -> min
    return np.max(np.stack(group_margins, axis=1), axis=1)  # AND -> max


def _falsify_square(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property: Union[dict, List[dict], 'HalfSpace', List['HalfSpace']],
    n_iters: int = 8000,
    batch: int = 256,
    p_init: float = 0.3,
    seed: Optional[int] = None,
    **kwargs,  # Ignore extra kwargs for compatibility with combined methods
) -> FalsifyResult:
    """Gradient-free Square-style attack (Andriushchenko et al. 2020, adapted to
    VNN-LIB AND-of-OR losses). Self-contained — NO external deps.

    Random search that perturbs random coordinate blocks to the input-box extremes
    (where L-inf adversarial points concentrate) and greedily keeps improvements,
    evaluated in batches. Uses NO gradients, so it attacks models where PGD/APGD
    fail because gradients vanish (e.g. Sign/binarized activations like
    traffic_signs). Every returned counterexample is verified against the canonical
    ``_output_satisfies_property`` check, so a SAT result is always sound.

    Args:
        n_iters: max model evaluations (across the batched search).
        batch: candidates evaluated per iteration (one batched forward).
        p_init: initial fraction of coordinates perturbed per candidate (decays).
        seed: RNG seed (order-independent via a dedicated Generator).
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=np.float32)
    ub = np.asarray(ub, dtype=np.float32)
    orig_shape = lb.shape
    lbf, ubf = lb.flatten(), ub.flatten()
    d = lbf.shape[0]
    groups = _extract_halfspace_groups(property)
    garr = _build_group_arrays(groups)
    device = _detect_model_device(model)
    model.eval()

    def forward(X):  # X: (N, d) -> (N, out_dim)
        with torch.no_grad():
            t = torch.from_numpy(X.reshape(X.shape[0], *orig_shape)).to(device)
            return model(t).detach().cpu().numpy().reshape(X.shape[0], -1)

    # Initialize at a random point; track the best (lowest) margin so far.
    best = rng.uniform(lbf, ubf).astype(np.float32)
    by = forward(best[None])
    bm = float(_batch_total_margin(by, garr)[0])
    if bm <= 0 and _output_satisfies_property(by[0], groups):
        return 0, (best, by[0])

    evals = 0
    while evals < n_iters:
        frac = max(p_init * (1.0 - evals / max(n_iters, 1)), 0.02)
        blk = max(1, int(frac * d))
        cand = np.tile(best, (batch, 1))
        for i in range(batch):
            idx = rng.choice(d, size=blk, replace=False)
            if rng.random() < 0.5:
                # box extremes (optimal for L-inf perturbation-ball CEs)
                cand[i, idx] = np.where(rng.random(blk) < 0.5, ubf[idx], lbf[idx])
            else:
                # interior values (VNN-LIB CEs are not always at a box corner)
                cand[i, idx] = rng.uniform(lbf[idx], ubf[idx]).astype(np.float32)
        outs = forward(cand)
        evals += batch
        m = _batch_total_margin(outs, garr)
        j = int(np.argmin(m))
        if m[j] < bm:
            bm = float(m[j])
            best = cand[j].copy()
            # Verify against the canonical check before claiming SAT (soundness).
            if bm <= 0 and _output_satisfies_property(outs[j], groups):
                return 0, (best, outs[j])
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


def _falsify_autoattack(
    model: torch.nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    property,
    seed: Optional[int] = None,
    **kwargs,
) -> FalsifyResult:
    """Tier-2 scaffold wrapping the external ``autoattack`` package.

    AutoAttack (Croce & Hein 2020, ICML) is the robustness community's
    standard adversarial-attack ensemble (APGD-CE + APGD-DLR + FAB +
    Square). It's designed for classification-robustness losses on
    input perturbation balls, not the general AND-of-OR-of-AND VNN-LIB
    unsafe-region losses our pipeline uses.

    This function is a scaffold. If ``autoattack`` is not installed, it
    raises ImportError with install instructions. If it IS installed, it
    currently raises NotImplementedError — wiring the VNN-LIB loss into
    AutoAttack's API is deferred as Phase 4 tier-2 work. Invoke via
    ``method='autoattack'``.

    Raises:
        ImportError: ``autoattack`` pip package is not installed.
        NotImplementedError: package is installed but full integration
            with the VNN-LIB loss is not yet wired up. See
            ``docs/plans/2026-04-24-phase4-full-spec-support-design.md``
            Tier 2 notes for the remaining work.
    """
    if not _HAS_AUTOATTACK:
        raise ImportError(
            "AutoAttack backend requires the 'autoattack' pip package. "
            "Install: pip install git+https://github.com/fra31/auto-attack.git"
        )
    raise NotImplementedError(
        "AutoAttack wrapper is a Phase 4 scaffold. Full integration "
        "with the AND-of-OR-of-AND loss is deferred to tier-2 work. "
        "Use method='random+pgd+apgd' for the production ensemble."
    )
