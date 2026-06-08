"""
SNN certified robustness module.

Trains a First-to-Fire (F2F) Leaky Integrate-and-Fire MLP and verifies it
using a triangle LP relaxation over latency variables.

Intended to be used as a library via SNNVerifier. Can also be run standalone:
    uv run python experiments/snn_comparison.py --dataset pathmnist --hidden-sizes 128 64
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm as _tqdm

import snntorch as snn
from snntorch import surrogate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class F2FMLP(nn.Module):
    """Multi-hidden-layer F2F SNN for flattened image classification.

    Encoding: bright pixels fire at timestep 0 (earliest), dark pixels fire
    at timestep T-1 (latest), zero-valued pixels never fire (latency = T).

    Each neuron fires at most once across all T timesteps (the at-most-once
    property). The predicted class is the output neuron with the highest
    accumulated score, where earlier firing gives higher score.
    """

    def __init__(self, input_size: int = 784,
                 hidden_sizes: list[int] | tuple[int, ...] = (64,),
                 num_classes: int = 10, beta: float = 0.9,
                 threshold: float = 1.0, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.hidden_sizes = list(hidden_sizes)

        # Build one Linear + one LIF per layer (hidden layers + output layer).
        layer_sizes = [input_size] + list(hidden_sizes) + [num_classes]
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        # One LIF per layer; all layers share the same beta and threshold.
        # fast_sigmoid is the surrogate gradient for training.
        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
            for _ in range(len(layer_sizes) - 1)
        ])

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Return F2F class scores from spike train shaped (B, input_size, T).

        At each timestep t, the current layer's input (x) is the spike output
        of the previous layer. The at-most-once constraint is enforced by the
        'fired' mask: once a neuron has spiked, its future spikes are zeroed.

        Score for class c = sum_t  (T - t) * output_spike[c, t]
        Earlier spikes contribute larger weights, so the first-to-fire class
        accumulates the highest score.
        """
        batch_size = spike_train.shape[0]
        n_layers = len(self.fcs)

        # Initialize membrane potentials for all layers.
        mems = [self.lifs[l].init_leaky() for l in range(n_layers)]

        # fired[l][i] tracks whether neuron i in layer l has already fired.
        # Shape: (batch_size, layer_width). Initialized to all-zeros (no neuron has fired).
        fired = [
            torch.zeros(batch_size, self.fcs[l].out_features, device=spike_train.device)
            for l in range(n_layers)
        ]

        # Accumulator for the F2F class scores.
        scores = torch.zeros(batch_size, self.num_classes, device=spike_train.device)

        for t in range(self.num_steps):
            # Input at timestep t: which input pixels fire at this timestep.
            x = spike_train[:, :, t]

            for l in range(n_layers):
                cur = self.fcs[l](x)                         # linear pre-activation
                spk_raw, mems[l] = self.lifs[l](cur, mems[l])  # LIF step: membrane update + spike

                # Enforce at-most-once: zero out spikes from neurons that already fired.
                spk = spk_raw * (1.0 - fired[l])

                # Record newly fired neurons so they cannot fire again.
                fired[l] = torch.clamp(fired[l] + spk, 0.0, 1.0)

                x = spk  # Spikes from this layer become input to the next layer.

            # F2F scoring: output spikes at timestep t contribute (T - t) to their class score.
            # t=0 contributes T (maximum), t=T-1 contributes 1 (minimum).
            scores = scores + float(self.num_steps - t) * x

        return scores

    @torch.no_grad()
    def simulate_with_patterns(self, spike_train: torch.Tensor):
        """Return (scores, hidden_spikes, output_spikes) for one sample.

        Used by the exhaustive fallback verifier, which needs the exact scores
        for every spike-timing combination, and by Monte Carlo sampling.

        hidden_spikes: uint8 array of shape (T, total_hidden_neurons)
        output_spikes: uint8 array of shape (T, num_classes)
        """
        if spike_train.ndim == 2:
            # Add batch dimension if a single sample was passed.
            spike_train = spike_train.unsqueeze(0)
        batch_size = spike_train.shape[0]
        n_layers = len(self.fcs)
        mems = [self.lifs[l].init_leaky() for l in range(n_layers)]
        fired = [
            torch.zeros(batch_size, self.fcs[l].out_features, device=spike_train.device)
            for l in range(n_layers)
        ]
        scores = torch.zeros(batch_size, self.num_classes, device=spike_train.device)
        layer_spikes = [[] for _ in range(n_layers)]  # collect per-timestep spike arrays

        for t in range(self.num_steps):
            x = spike_train[:, :, t]
            for l in range(n_layers):
                cur = self.fcs[l](x)
                spk_raw, mems[l] = self.lifs[l](cur, mems[l])
                spk = spk_raw * (1.0 - fired[l])
                fired[l] = torch.clamp(fired[l] + spk, 0.0, 1.0)
                x = spk
                # Save the spike pattern at this timestep for this layer.
                layer_spikes[l].append(spk.detach().cpu().numpy()[0])
            scores = scores + float(self.num_steps - t) * x

        # Stack to (T, neurons_in_layer) for each layer, cast to uint8 (0 or 1).
        stacked = [np.stack(layer_spikes[l], axis=0).astype(np.uint8) for l in range(n_layers)]

        # Concatenate all hidden layer spikes along the neuron axis.
        if len(stacked) > 1:
            hidden = np.concatenate(stacked[:-1], axis=1)  # all but last (output) layer
        else:
            hidden = np.zeros((self.num_steps, 0), dtype=np.uint8)
        output = stacked[-1]  # output layer spikes only
        return scores.detach().cpu().numpy()[0], hidden, output


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def latency_from_values(values: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Encode normalized pixel values in [0, 1] to 0-based latency integers.

    Encoding formula: latency = round((T-1) * (1 - x))
      - x = 1.0  → latency = 0  (fires at timestep 0, the earliest)
      - x = 0.5  → latency ≈ (T-1)/2 (fires in the middle)
      - x = 0.0  → treated as background: latency = T (silent, never fires)
      - x ≤ 0    → silent sentinel value T

    Zero-valued pixels are background; assigning latency T means the spike
    loop (which runs t = 0..T-1) never emits a spike for them.
    """
    # Map brightness linearly to latency, round to nearest integer.
    z = torch.floor((num_steps - 1) * (1.0 - values) + 0.5).long()
    z = torch.clamp(z, 0, num_steps - 1)

    # Background pixels (value == 0) are assigned the silent sentinel T.
    silent = values <= 0
    z[silent] = num_steps
    return z


def encode_batch(images: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Convert a batch of images to a spike train tensor of shape (B, pixels, T).

    For each pixel, exactly one timestep is set to 1.0 (the latency timestep),
    or no timestep is set if the pixel is silent (background).
    """
    flat = images.view(images.shape[0], -1)       # flatten spatial dims: (B, pixels)
    lat = latency_from_values(flat, num_steps)     # (B, pixels) integer latencies

    # Build the spike train: one-hot along the time axis.
    spikes = torch.zeros(flat.shape[0], flat.shape[1], num_steps, device=flat.device)
    for t in range(num_steps):
        spikes[:, :, t] = (lat == t).float()      # 1 where this pixel fires at time t
    return spikes


def spike_train_from_latencies(latencies: np.ndarray, num_steps: int) -> torch.Tensor:
    """Build a single-sample spike train from a latency array.

    latencies[i] = t  → pixel i fires at timestep t
    latencies[i] = T  → pixel i is silent (no spike)

    Returns shape (num_pixels, num_steps).
    Used by the exhaustive fallback verifier and Monte Carlo sampling, where
    we need to simulate the model for one specific latency assignment.
    """
    spikes = np.zeros((latencies.shape[0], num_steps), dtype=np.float32)
    finite = latencies < num_steps          # True for non-silent pixels
    spikes[np.where(finite)[0], latencies[finite]] = 1.0
    return torch.from_numpy(spikes)


# ---------------------------------------------------------------------------
# LP helpers
# ---------------------------------------------------------------------------

def feasible_latencies(lb: float, ub: float, num_steps: int) -> list[int]:
    """Return the set of timesteps t that are consistent with pixel value in [lb, ub].

    A timestep t is feasible if the pixel-value interval [x_low(t), x_high(t)]
    corresponding to latency t overlaps with the perturbation interval [lb, ub].

    The pixel value that maps to latency t satisfies:
        x_low(t)  = 1 - (t + 0.5) / (T-1)   (lower edge of the latency-t bin)
        x_high(t) = 1 - (t - 0.5) / (T-1)   (upper edge)

    The silent latency T is feasible whenever lb ≤ 0 (the pixel might become
    zero or below-zero after perturbation).
    """
    vals = []

    # Silent (T) is feasible if the lower bound of the pixel value is ≤ 0,
    # meaning the pixel could become background after perturbation.
    if lb <= 0.0:
        vals.append(num_steps)  # sentinel T = silent

    lo = max(lb, np.nextafter(0.0, 1.0))  # only non-zero values can fire
    hi = ub
    if hi <= 0.0:
        return vals  # all perturbations force silence, no finite latency is feasible

    for t in range(num_steps):
        # Compute the pixel-value bin edges for latency t.
        x_low = 1.0 - (t + 0.5) / (num_steps - 1)
        x_high = 1.0 - (t - 0.5) / (num_steps - 1)
        x_low = max(0.0, x_low)
        x_high = min(1.0, x_high)
        # Feasible if the bins overlap: max(lo, x_low) <= min(hi, x_high).
        if max(lo, x_low) <= min(hi, x_high):
            vals.append(t)
    return sorted(set(vals))


def make_bounds(image_flat: np.ndarray, indices: np.ndarray, epsilon: float):
    """Build per-pixel lower and upper bound arrays for the perturbation set.

    For selected pixels: lb[i] = clip(x[i] - ε, 0, 1), ub[i] = clip(x[i] + ε, 0, 1).
    For unselected pixels: lb[i] = ub[i] = x[i] (no perturbation).
    """
    lb = image_flat.copy()
    ub = image_flat.copy()
    for idx in indices:
        lb[idx] = np.clip(image_flat[idx] - epsilon, 0.0, 1.0)
        ub[idx] = np.clip(image_flat[idx] + epsilon, 0.0, 1.0)
    return lb, ub


def effective_pixel_bounds(image_flat: np.ndarray, indices: np.ndarray,
                            epsilon: float, num_steps: int):
    """Compute per-pixel effective lower/upper bounds based on SNN latency quantization.

    The SNN LP certifies robustness over the union of all feasible latency bins
    that overlap [x_i - ε, x_i + ε], not just the continuous interval itself.
    Each latency bin has width 1/(T-1) in pixel-value space, so the effective
    input region is slightly wider than the stated ε on both sides.

    For each perturbed pixel i with nominal value x_i:
      - Compute the set of feasible latencies (bins overlapping [x_i-ε, x_i+ε])
      - t_min = lowest feasible latency (brightest reachable, highest pixel value)
      - t_max = highest feasible latency (darkest reachable, lowest pixel value)
      - eff_ub[i] = right edge of t_min bin = 1 - (t_min - 0.5)/(T-1)  (≥ x_i + ε)
      - eff_lb[i] = left  edge of t_max bin = 1 - (t_max + 0.5)/(T-1)  (≤ x_i - ε)

    Both effective bounds are always at least as wide as the original ε bounds.
    Non-perturbed pixels are unchanged: eff_lb[i] = eff_ub[i] = x_i.

    These bounds can be passed directly to the ANN verifiers for a fair comparison:
    the ANN is certified over the same input region the SNN LP actually covers.
    """
    lb, ub = make_bounds(image_flat, indices, epsilon)
    eff_lb = lb.copy()
    eff_ub = ub.copy()
    T = num_steps

    for i in indices:
        feas = feasible_latencies(float(lb[i]), float(ub[i]), T)
        feas_finite = [t for t in feas if t < T]   # exclude silent sentinel
        if not feas_finite:
            continue   # pixel forced silent by perturbation — keep original bounds

        t_min = min(feas_finite)   # earliest fire → highest pixel value
        t_max = max(feas_finite)   # latest fire  → lowest pixel value

        # Right edge of t_min bin: pixel values above this map to latency < t_min.
        eff_ub[i] = min(1.0, 1.0 - (t_min - 0.5) / (T - 1))
        # Left edge of t_max bin: pixel values below this map to latency > t_max.
        eff_lb[i] = max(0.0, 1.0 - (t_max + 0.5) / (T - 1))

    return eff_lb, eff_ub


ALL_SPLIT_STRATEGIES = ["random", "selected", "influence", "choice", "choice-influence"]

# Mutable context dict populated before each LP solve so that failure messages
# can report which image/pixels triggered the infeasibility.  Updated by
# SNNVerifier.verify (image_idx) and build_symbolic_relaxation_lp (pixel info).
_LP_CONTEXT: dict = {}

# When False, equality constraints (a@x = b) are converted to pairs of
# inequalities (a@x ≤ b, -a@x ≤ -b) before passing to the solver.
# Set by SNNVerifier.verify; controlled by --equality-constraints CLI flag.
_USE_EQ_CONSTRAINTS: bool = True

# When True, print LP context (image, pixels, dimensions) for every LP build,
# regardless of success or failure.  Controlled by --debug-lp CLI flag.
_DEBUG_LP: bool = False

# When True, add explicit at-most-once LP constraints (Σ_t spk[i,t] ≤ 1) for
# every hidden and output neuron.  Off by default because these constraints can
# become contradictory with the lower-triangle bounds.  Controlled by --amo.
_USE_AMO_CONSTRAINTS: bool = False


def order_split_indices(model, indices: np.ndarray, lb_x: np.ndarray, ub_x: np.ndarray,
                        num_steps: int, strategy: str = "selected") -> np.ndarray:
    """Return the pixel indices in the order they should be split on first.

    When the symbolic split descends to depth d, it branches on the first d
    pixels in this ordering. Splitting on the 'most important' pixels first
    tends to prune the LP branch tree more aggressively.

    Strategies:
      selected:         keep original index order (no reordering)
      influence:        sum of |W1[:, pix]| — pixels with large first-layer
                        weight impact are split first
      choice:           number of feasible latency values — ambiguous pixels
                        (widest perturbation interval) are split first
      choice-influence: influence * max(choices - 1, 0) — splits pixels that
                        are both high-influence AND have multiple feasible
                        latencies; pixels with only one feasible latency need
                        not be split at all (they contribute 0 to this score)
      random:           shuffled with a fixed seed (for ablations)
    """
    if strategy == "selected":
        return indices  # no reordering

    # First-layer weight matrix: shape (hidden_size, input_size).
    W1 = model.fcs[0].weight.detach().cpu().numpy()

    # Sum of absolute weights for each selected pixel across all hidden neurons.
    influence = np.sum(np.abs(W1[:, indices]), axis=0)  # shape (k,)

    # Number of feasible latency values for each selected pixel.
    choice_counts = np.array([
        len(feasible_latencies(float(lb_x[p]), float(ub_x[p]), num_steps))
        for p in indices
    ], dtype=np.float64)  # shape (k,)

    if strategy == "influence":
        scores = influence
    elif strategy == "choice":
        scores = choice_counts
    elif strategy == "choice-influence":
        # Pixels with only one feasible latency are already determined; no
        # benefit to splitting on them, so subtract 1 before multiplying.
        scores = influence * np.maximum(choice_counts - 1.0, 0.0)
    elif strategy == "random":
        shuffled = indices.copy()
        np.random.default_rng(42).shuffle(shuffled)
        return shuffled
    elif strategy.startswith("split_"):
        # Manual override: "split_0" puts index 0 first, "split_1_2" puts
        # indices 1 and 2 first. Used for debugging specific pixel orderings.
        parts = strategy.split("_")[1:]
        if len(parts) == 1:
            pos = int(parts[0])
            if pos >= len(indices):
                raise ValueError(f"{strategy}: position {pos} out of range for k={len(indices)}")
            return np.concatenate([[indices[pos]], np.delete(indices, pos)])
        elif len(parts) == 2:
            pos0, pos1 = int(parts[0]), int(parts[1])
            if pos0 >= len(indices) or pos1 >= len(indices):
                raise ValueError(f"{strategy}: positions out of range for k={len(indices)}")
            rest = [i for i in range(len(indices)) if i != pos0 and i != pos1]
            return np.array([indices[pos0], indices[pos1]] + [indices[i] for i in rest])
        else:
            raise ValueError(f"Unsupported split strategy: {strategy}")
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    # Sort descending: highest-scoring pixels are split first.
    return indices[np.argsort(scores)[::-1]]


def bounds_cover_outputs(row: dict, outputs: np.ndarray, tol: float = 1e-6) -> float:
    """Check what fraction of Monte Carlo output samples fall within the LP bounds.

    Used as a sanity check: if the LP bounds are sound, every MC sample should
    be covered. A coverage below 100% indicates an LP soundness violation.

    Returns the coverage percentage (should be ~100% for correct bounds).
    """
    lb = np.array(row["lb"], dtype=float)
    ub = np.array(row["ub"], dtype=float)
    # covered[i] = True iff all class bounds contain the i-th MC output.
    covered = np.logical_and(outputs >= lb[None, :] - tol, outputs <= ub[None, :] + tol)
    return 100.0 * float(np.all(covered, axis=1).mean())


def bounds_cover_bounds(row: dict, reference: dict, tol: float = 1e-6) -> bool:
    """Return True if row's bounds contain the reference bounds (subset check).

    Used to verify that a tighter result is contained within a looser one.
    """
    lb = np.array(row["lb"], dtype=float)
    ub = np.array(row["ub"], dtype=float)
    ref_lb = np.array(reference["lb"], dtype=float)
    ref_ub = np.array(reference["ub"], dtype=float)
    return bool(np.all(lb <= ref_lb + tol) and np.all(ub >= ref_ub - tol))


def monte_carlo_outputs(model, image_flat: np.ndarray, indices: np.ndarray,
                        epsilon: float, num_steps: int, num_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample random perturbations within the ε-ball and collect model outputs.

    For each sample, a random pixel value is drawn uniformly from [lb[i], ub[i]]
    for each perturbed pixel i, converted to a spike train, and simulated.

    Returns shape (num_samples, num_classes) of output scores.
    """
    lb, ub = make_bounds(image_flat, indices, epsilon)
    outputs = []
    for _ in range(num_samples):
        point = image_flat.copy()
        if len(indices) > 0:
            # Sample a random perturbation for each selected pixel.
            point[indices] = rng.uniform(lb[indices], ub[indices])
        # Convert pixel values to latency, then to a spike train.
        lat = latency_from_values(torch.from_numpy(point).float(), num_steps).numpy()
        spike_train = spike_train_from_latencies(lat, num_steps)
        score, _, _ = model.simulate_with_patterns(spike_train)
        outputs.append(score)
    return np.stack(outputs, axis=0)


# ---------------------------------------------------------------------------
# LP solver internals
# ---------------------------------------------------------------------------

def _linear_interval(center: float, coeffs: np.ndarray,
                     bounds: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute a fast interval bound for a linear expression: center + coeffs @ vars.

    Uses the current variable bounds (not an LP) to get a quick lower/upper
    estimate. This is used for the at-most-one membrane potential bounds
    (u_lb_amo, u_ub_amo) when tight_bounds=False.
    """
    lb = center
    ub = center
    nz = np.nonzero(coeffs)[0]
    for j in nz:
        lo, hi = bounds[j]
        val = coeffs[j]
        if val >= 0.0:
            lb += val * lo   # positive coeff: lb uses variable's lb
            ub += val * hi   # positive coeff: ub uses variable's ub
        else:
            lb += val * hi   # negative coeff: lb uses variable's ub
            ub += val * lo   # negative coeff: ub uses variable's lb
    return float(lb), float(ub)


def _solve_bound(obj: np.ndarray, A_ub: list[np.ndarray], b_ub: list[float],
                 A_eq: list[np.ndarray], b_eq: list[float],
                 bounds: list[tuple[float, float]], maximize: bool = False) -> float:
    """Solve a single LP to get the tightest bound on the linear objective 'obj'.

    Used only when tight_bounds=True to recompute u_lb and u_ub via LP rather
    than the cheaper interval arithmetic. This is much slower but produces
    tighter neuron-firing bounds.
    """
    lp_matrices = _prepare_lp_matrices(A_ub, b_ub, A_eq, b_eq)
    return _solve_bound_prepared(obj, lp_matrices, bounds, maximize=maximize)


def _prepare_lp_matrices(A_ub: list[np.ndarray], b_ub: list[float],
                         A_eq: list[np.ndarray], b_eq: list[float]):
    """Convert lists of constraint rows into sparse matrices for the LP solver.

    If _USE_EQ_CONSTRAINTS is False, each equality a@x = b is replaced by the
    pair a@x ≤ b and -a@x ≤ -b, so A_eq_mat/b_eq_vec are passed as None.
    This can help when equality constraints cause spurious infeasibility.

    If SNNV_LP_BACKEND=highspy is set, returns a _HighsPreparedLP object that
    caches the HiGHS model and only swaps out the objective vector between
    solves — faster than re-building the LP from scratch each time.

    Otherwise returns (A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec) for scipy.linprog.
    """
    from scipy import sparse

    if not _USE_EQ_CONSTRAINTS:
        # Drop equality constraints entirely — the LP feasible set is strictly
        # larger, so certification is less tight but the solver cannot become
        # infeasible due to an over-constrained equality.
        A_eq = []
        b_eq = []

    A_ub_mat = sparse.csr_matrix(np.vstack(A_ub)) if A_ub else None
    b_ub_vec = np.asarray(b_ub, dtype=np.float64) if A_ub else None
    A_eq_mat = sparse.csr_matrix(np.vstack(A_eq)) if A_eq else None
    b_eq_vec = np.asarray(b_eq, dtype=np.float64) if A_eq else None
    lp_matrices = A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec
    if os.environ.get("SNNV_LP_BACKEND", "").lower() == "highspy":
        return _HighsPreparedLP(lp_matrices, bounds=None)
    return lp_matrices


class _HighsPreparedLP:
    """Optional HiGHS-backed LP wrapper that reuses the model across objective changes.

    Building a HiGHS model is expensive. This class builds it once on the first
    call to solve(), then reuses it for subsequent solves by changing only the
    cost vector. This is important when solving many LPs with the same constraint
    matrix but different objectives (as happens in build_symbolic_relaxation_lp).
    """
    def __init__(self, lp_matrices, bounds):
        self.lp_matrices = lp_matrices
        self.bounds = bounds
        self.highs = None       # lazily initialized on first solve
        self.highspy = None
        self.indices = None

    def _init_model(self, bounds: list[tuple[float, float]]):
        """Build the HiGHS model from the constraint matrices and variable bounds."""
        from scipy import sparse
        import highspy

        A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = self.lp_matrices

        # Combine inequality and equality constraints into a single row-ranged
        # system: row_lower[i] <= A[i,:] @ x <= row_upper[i].
        mats = []
        lower_parts = []
        upper_parts = []
        if A_ub_mat is not None:
            mats.append(A_ub_mat)
            lower_parts.append(np.full(A_ub_mat.shape[0], -highspy.kHighsInf))
            upper_parts.append(b_ub_vec)
        if A_eq_mat is not None:
            mats.append(A_eq_mat)
            lower_parts.append(b_eq_vec)
            upper_parts.append(b_eq_vec)
        if mats:
            A = sparse.vstack(mats, format="csr")
            row_lower = np.concatenate(lower_parts).astype(np.float64)
            row_upper = np.concatenate(upper_parts).astype(np.float64)
        else:
            A = sparse.csr_matrix((0, len(bounds)), dtype=np.float64)
            row_lower = np.empty(0, dtype=np.float64)
            row_upper = np.empty(0, dtype=np.float64)

        lower = np.asarray([lo for lo, _ in bounds], dtype=np.float64)
        upper = np.asarray([hi for _, hi in bounds], dtype=np.float64)
        lower[np.isneginf(lower)] = -highspy.kHighsInf
        upper[np.isposinf(upper)] = highspy.kHighsInf

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)    # suppress HiGHS console output
        highs.setOptionValue("threads", int(os.environ.get("SNNV_HIGHS_THREADS", "1")))
        highs.setOptionValue("solver", "simplex")
        highs.addVars(len(bounds), lower, upper)
        highs.addRows(
            A.shape[0], row_lower, row_upper,
            A.nnz, A.indptr.astype(np.int32),
            A.indices.astype(np.int32), A.data.astype(np.float64),
        )
        self.highs = highs
        self.highspy = highspy
        self.indices = np.arange(len(bounds), dtype=np.int32)
        self.bounds = bounds

    def solve(self, obj: np.ndarray, bounds: list[tuple[float, float]],
              maximize: bool = False) -> float:
        """Solve the LP by swapping only the cost vector and re-running HiGHS."""
        from scipy.optimize import linprog

        if self.highs is None:
            self._init_model(bounds)
        c = np.asarray(-obj if maximize else obj, dtype=np.float64)
        self.highs.changeColsCost(len(c), self.indices, c)
        self.highs.run()
        status = self.highs.getModelStatus()
        if status != self.highspy.HighsModelStatus.kOptimal:
            # Fall back to scipy if HiGHS fails (e.g., infeasible sub-problem).
            A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = self.lp_matrices
            res = linprog(c, A_ub=A_ub_mat, b_ub=b_ub_vec,
                          A_eq=A_eq_mat, b_eq=b_eq_vec, bounds=bounds, method="highs")
            if not res.success:
                A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = self.lp_matrices
                ctx = _LP_CONTEXT
                print(f"  [LP-fail/highs] status={res.status} msg={res.message!r} "
                      f"maximize={maximize}")
                print(f"    image_idx={ctx.get('image_idx', '?')}  "
                      f"label={ctx.get('label', '?')}  "
                      f"epsilon={ctx.get('epsilon', '?')}")
                pidx = ctx.get("pixel_indices", [])
                pval = ctx.get("pixel_values", [])
                print(f"    perturbed pixels (in selection order): {pidx}")
                print(f"    pixel values at those positions:       "
                      f"{[round(v, 4) for v in pval]}")
                fixed = ctx.get("fixed_latencies") or {}
                if fixed:
                    print(f"    fixed latencies (split branch):        {fixed}")
                _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec, bounds)
                return float("inf") if maximize else float("-inf")
            return float(-res.fun if maximize else res.fun)
        value = float(self.highs.getInfo().objective_function_value)
        return -value if maximize else value


def _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec,
                         bounds: list[tuple[float, float]],
                         max_report: int = 10) -> None:
    """Print which individual constraints are infeasible given the variable bounds.

    Uses bound propagation: for each constraint row a, the achievable range of
    a @ x given variable bounds [lb, ub] is:
        min = sum(a[j]*lb[j] for a[j]>0) + sum(a[j]*ub[j] for a[j]<0)
        max = sum(a[j]*ub[j] for a[j]>0) + sum(a[j]*lb[j] for a[j]<0)

    A constraint is individually infeasible if the rhs falls outside [min, max]
    (equality) or if min > rhs (inequality a@x <= rhs).

    This catches constraints that are impossible in isolation; joint infeasibility
    from combinations of constraints is not diagnosed here.
    """
    lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)
    ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)

    # Report variables whose own bounds are contradictory.
    bad = np.where(lb_arr > ub_arr + 1e-9)[0]
    if len(bad):
        print(f"    Bad variable bounds (lb>ub): {bad[:max_report].tolist()}"
              + (" ..." if len(bad) > max_report else ""))

    def _check(mat, rhs_vec, kind: str):
        if mat is None:
            return
        # Convert to CSR so we can iterate rows efficiently.
        from scipy import sparse
        A = mat.tocsr()
        n_flagged = 0
        for i in range(A.shape[0]):
            row = A.getrow(i).toarray().ravel()
            pos = row > 0
            neg = row < 0
            row_min = float(row[pos] @ lb_arr[pos]) + float(row[neg] @ ub_arr[neg])
            row_max = float(row[pos] @ ub_arr[pos]) + float(row[neg] @ lb_arr[neg])
            rhs = float(rhs_vec[i])
            infeasible = False
            if kind == "eq":
                infeasible = rhs < row_min - 1e-9 or rhs > row_max + 1e-9
            else:  # ub: a@x <= rhs
                infeasible = row_min > rhs + 1e-9
            if infeasible:
                nnz = int((row != 0).sum())
                print(f"    Infeasible {kind} constraint {i}: "
                      f"achievable=[{row_min:.6g}, {row_max:.6g}]  rhs={rhs:.6g}  nnz={nnz}")
                n_flagged += 1
                if n_flagged >= max_report:
                    remaining = A.shape[0] - i - 1
                    if remaining > 0:
                        print(f"    ... ({remaining} more {kind} constraints not shown)")
                    break
        if n_flagged == 0:
            print(f"    No individually infeasible {kind} constraints found "
                  f"(infeasibility may be joint).")

    _check(A_eq_mat, b_eq_vec, "eq")
    _check(A_ub_mat, b_ub_vec, "ub")


def _solve_bound_prepared(obj: np.ndarray, lp_matrices,
                          bounds: list[tuple[float, float]],
                          maximize: bool = False) -> float:
    """Solve a single LP bound using the prepared (possibly cached) matrices.

    Dispatches to HiGHS or scipy depending on whether lp_matrices is a
    _HighsPreparedLP or a plain (A_ub, b_ub, A_eq, b_eq) tuple.
    """
    from scipy.optimize import linprog

    if isinstance(lp_matrices, _HighsPreparedLP):
        return lp_matrices.solve(obj, bounds, maximize=maximize)

    A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = lp_matrices
    c = -obj if maximize else obj
    
    # HELP EASE BOUNDS
    b_ub_vec = b_ub_vec + 1e-5 

    res = linprog(c, 
                A_ub=A_ub_mat, 
                b_ub=b_ub_vec,
                A_eq=A_eq_mat, 
                b_eq=b_eq_vec, 
                bounds=bounds, 
                method="highs-ipm")

    if not res.success:
        n_ub = A_ub_mat.shape[0] if A_ub_mat is not None else 0
        n_eq = A_eq_mat.shape[0] if A_eq_mat is not None else 0
        n_v = len(bounds)
        ctx = _LP_CONTEXT
        print(f"  [LP-fail] status={res.status} msg={res.message!r} "
              f"maximize={maximize} n_vars={n_v} n_ub={n_ub} n_eq={n_eq}")
        print(f"    image_idx={ctx.get('image_idx', '?')}  "
              f"label={ctx.get('label', '?')}  "
              f"epsilon={ctx.get('epsilon', '?')}")
        pidx = ctx.get("pixel_indices", [])
        pval = ctx.get("pixel_values", [])
        print(f"    perturbed pixels (in selection order): {pidx}")
        print(f"    pixel values at those positions:       "
              f"{[round(v, 4) for v in pval]}")
        # print(f"    multiple choices: {multi_choices}")
        fixed = ctx.get("fixed_latencies") or {}
        if fixed:
            print(f"    fixed latencies (split branch):        {fixed}")
        _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec, bounds)
        return float("inf") if maximize else float("-inf")
    return float(-res.fun if maximize else res.fun)


# ---------------------------------------------------------------------------
# Process-pool worker (used by build_symbolic_relaxation_lp_split)
# ---------------------------------------------------------------------------

# Module-level dict used to share state with worker processes/threads.
# Populated by _init_symbolic_split_worker before any _solve_symbolic_split_worker
# calls; process-safe because each worker process has its own copy.
_SPLIT_WORKER = {}


def _init_symbolic_split_worker(config, state_dict, image_flat, epsilon, k,
                                num_steps, keys, indices, base_lat,
                                tight_bounds, label, cert_only,
                                singleton_bounds=False):
    """Initializer for each worker process in the ProcessPoolExecutor.

    Builds and stores the model (from config + state_dict) in the module-level
    _SPLIT_WORKER dict so _solve_symbolic_split_worker can access it without
    re-loading the model for each combo. torch.set_num_threads(1) prevents
    inter-process thread contention.
    """
    torch.set_num_threads(1)
    model = F2FMLP(**config)
    model.load_state_dict(state_dict)
    model.eval()
    _SPLIT_WORKER.clear()
    _SPLIT_WORKER.update({
        "model": model,
        "image_flat": image_flat,
        "epsilon": epsilon,
        "k": k,
        "num_steps": num_steps,
        "keys": keys,
        "indices": set(int(i) for i in indices),
        "pixel_indices": np.asarray(indices, dtype=int),
        "base_lat": base_lat,
        "tight_bounds": tight_bounds,
        "label": label,
        "cert_only": cert_only,
        "singleton_bounds": singleton_bounds,
    })


def _solve_symbolic_split_worker(combo):
    """Worker function: solve one LP sub-problem for a fixed latency combination.

    combo is a tuple of latency assignments for the split pixels (those whose
    latency is fixed in this branch). The remaining perturbed pixels are still
    treated symbolically inside build_symbolic_relaxation_lp.

    If all perturbed pixels are fixed (depth == k), skip the LP and simulate
    the model exactly for this specific spike-timing assignment.
    """
    cfg = _SPLIT_WORKER
    keys = cfg["keys"]
    # Map each split pixel to its fixed latency for this branch.
    fixed = {keys[j]: combo[j] for j in range(len(keys))} if keys else {}
    label = cfg["label"]

    if len(fixed) == len(cfg["indices"]):
        # All k perturbed pixels are pinned — no LP needed, just simulate exactly.
        lats = cfg["base_lat"].copy()
        for pix, lat in fixed.items():
            lats[pix] = lat
        st = spike_train_from_latencies(lats, cfg["num_steps"])
        score, _, _ = cfg["model"].simulate_with_patterns(st)
        score_arr = np.array(score, dtype=np.float64)
        gap_joint = None
        if label is not None:
            # Compute exact margins for this spike pattern.
            gap_joint = np.full(len(score_arr), np.inf, dtype=np.float64)
            for c in range(len(score_arr)):
                if c != label:
                    gap_joint[c] = float(score_arr[label] - score_arr[c])
        return {
            "lb": score_arr, "ub": score_arr, "gap_joint": gap_joint,
            "n_lp_variables": 0, "n_lp_constraints": 0, "invalid_subproblem": False,
        }
    # Some pixels are still free — solve a partial LP with the fixed latencies.
    return build_symbolic_relaxation_lp(
        cfg["model"], cfg["image_flat"], cfg["epsilon"], cfg["k"], cfg["num_steps"],
        tight_bounds=cfg["tight_bounds"], fixed_latencies=fixed, label=label,
        cert_only=cfg["cert_only"], pixel_indices=cfg["pixel_indices"],
        singleton_bounds=cfg.get("singleton_bounds", False),
    )


# ---------------------------------------------------------------------------
# Core LP verification functions
# ---------------------------------------------------------------------------

def build_symbolic_relaxation_lp(model, image_flat: np.ndarray, epsilon: float,
                                 k: int, num_steps: int,
                                 tight_bounds: bool = False,
                                 fixed_latencies: dict[int, int] | None = None,
                                 label: int | None = None,
                                 cert_only: bool = False,
                                 pixel_indices: np.ndarray | None = None,
                                 singleton_bounds: bool = False,
                                 parallel_workers: int = 1,
                                 parallel_backend: str = "thread"):
    """Build and solve the triangle LP relaxation for the multi-layer F2F MLP.

    This is the central verification function. It constructs a linear program
    whose feasible set is a convex outer approximation of all spike-timing
    patterns consistent with the perturbation interval [lb_x, ub_x].

    === Variables ===

    1. Input spike variables  p[(pix, t)]  ∈ [0, 1]
       For each perturbed pixel 'pix' and each feasible latency t (those
       consistent with [lb_x[pix], ub_x[pix]]). Represents the probability
       (in the relaxation) that pixel pix fires at time t.

    2. Hidden spike variables  h_var(layer, i, t)  ∈ [0, 1]
       For each hidden neuron i in each layer, at each timestep t.

    3. Output spike variables  o_var(c, t)  ∈ [0, 1]
       For each output class c, at each timestep t.

    === Constraints ===

    At-most-once (AMO): Σ_t h_var(l, i, t) ≤ 1 for every hidden/output neuron.
    Input pixel sum: Σ_t p[(pix, t)] = 1 if silence is not feasible, else ≤ 1.
    First-fire relaxation: For each neuron at each timestep t, a triangle
      constraint encodes the relationship between the neuron's membrane
      potential and its spike variable (see add_first_fire_relaxation).

    === Objective ===

    For each competitor class c' ≠ label, solve:
        minimize  score[label] - score[c']
    where score[c] = Σ_t (T - t) * o_var(c, t).

    If the minimum (gap_joint) over all c' is > 0, the network is certified.

    pixel_indices must be provided by the caller (comp_snn_vs_ann.py); this
    function does not call select_pixels. k is kept for legacy compatibility.
    """

    # Extract weight matrices and biases from all layers.
    Ws = [fc.weight.detach().cpu().numpy() for fc in model.fcs]   # list of (out, in) arrays
    bs = [fc.bias.detach().cpu().numpy() for fc in model.fcs]     # list of (out,) arrays
    n_hidden_layers = len(model.hidden_sizes)
    hidden_sizes_list = model.hidden_sizes
    num_classes = Ws[-1].shape[0]

    # Retrieve beta (leak) and threshold from the first LIF neuron.
    beta = float(model.lifs[0].beta.item() if hasattr(model.lifs[0].beta, "item")
                 else model.lifs[0].beta)
    theta = float(model.lifs[0].threshold.item() if hasattr(model.lifs[0].threshold, "item")
                  else model.lifs[0].threshold)

    # Selected pixels whose latency is uncertain (the perturbation set).
    indices = np.asarray(pixel_indices, dtype=int) if pixel_indices is not None else np.empty(0, dtype=int)
    lb_x, ub_x = make_bounds(image_flat, indices, epsilon)   # per-pixel bounds

    # Nominal latency for each pixel under the clean (unperturbed) image.
    base_lat = latency_from_values(torch.from_numpy(image_flat).float(), num_steps).numpy()

    # -------------------------------------------------------------------------
    # Variable allocation
    # -------------------------------------------------------------------------
    # p_vars[(pix, t)] = index of the LP variable for pixel pix firing at time t.
    p_vars: dict[tuple[int, int], int] = {}

    # fixed_input[t, pix] = 1 if pixel pix definitely fires at time t (no LP variable needed).
    # Non-selected pixels with a finite latency are placed here directly.
    fixed_input = np.zeros((num_steps, image_flat.shape[0]), dtype=np.float64)

    # LP constraint accumulators.
    A_ub: list[np.ndarray] = []   # inequality constraint rows (A_ub @ x ≤ b_ub)
    b_ub: list[float] = []
    A_eq: list[np.ndarray] = []   # equality constraint rows (A_eq @ x = b_eq)
    b_eq: list[float] = []
    bounds: list[tuple[float, float]] = []  # per-variable [lb, ub]

    selected = set(int(i) for i in indices)

    # Assign fixed spike times for unperturbed pixels: they fire at their nominal latency.
    for pix, latency in enumerate(base_lat):
        if pix not in selected and latency < num_steps:
            # Non-selected pixel with a finite latency: fixed spike at time 'latency'.
            fixed_input[int(latency), pix] = 1.0

    if fixed_latencies is None:
        fixed_latencies = {}  # no latencies are pinned for this sub-problem


    multi_choices = False
    # Allocate LP variables for perturbed pixels.
    for pix in indices:
        if int(pix) in fixed_latencies:
            # This pixel's latency is pinned in this branch of the split tree.
            t_fix = fixed_latencies[int(pix)]
            if t_fix < num_steps:
                fixed_input[t_fix, pix] = 1.0  # treat as a fixed spike
            continue

        # Compute which timesteps are feasible given the pixel's ε-ball.
        choices = feasible_latencies(float(lb_x[pix]), float(ub_x[pix]), num_steps)
        #print(choices)
        if(len(choices) > 1):
            multi_choices = True
        finite_choices = [t for t in choices if t < num_steps]  # exclude the silent sentinel
        silent_feasible = num_steps in choices                   # can this pixel be silent?

        if len(finite_choices) == 1 and not silent_feasible:
            # Only one latency is feasible and silence is not possible: fix it.
            fixed_input[finite_choices[0], pix] = 1.0
            continue

        # Create one LP variable per feasible firing time.
        pix_var_ids = []
        for t in finite_choices:
            p_vars[(int(pix), int(t))] = len(bounds)
            pix_var_ids.append(len(bounds))
            bounds.append((0.0, 1.0))   # spike probability in [0, 1]

        # Add sum constraint for this pixel's spike variables.
        if pix_var_ids:
            row = np.zeros(len(bounds), dtype=np.float64)
            row[pix_var_ids] = 1.0
            if silent_feasible:
                # Sum ≤ 1: allows the pixel to be silent (no spike) as well.
                A_ub.append(row.copy())
                b_ub.append(1.0)
            else:
                # Sum = 1: pixel must fire at exactly one of its feasible times.
                A_eq.append(row.copy())
                b_eq.append(1.0)

    if(not multi_choices):
        #print("ERROR RAISED: SINGLE CHOICE")
        # raise ZeroDivisionError("Only one possible split")
        pass

    n_input_vars = len(bounds)   # number of input spike LP variables allocated so far

    # -------------------------------------------------------------------------
    # Allocate hidden and output spike variables.
    # Each (layer l, neuron i) gets one variable per timestep: h_var(l, i, t).
    # -------------------------------------------------------------------------
    hidden_offsets: list[int] = []   # base index for each hidden layer's variables
    cur_offset = n_input_vars
    for hs in hidden_sizes_list:
        hidden_offsets.append(cur_offset)
        cur_offset += hs * num_steps   # hs neurons × T timesteps
    output_offset = cur_offset         # base index for output spike variables
    n_vars = output_offset + num_classes * num_steps

    # All hidden and output spike variables are bounded to [0, 1].
    bounds.extend([(0.0, 1.0)] * (sum(hidden_sizes_list) * num_steps + num_classes * num_steps))

    def pad_rows(rows: list[np.ndarray]) -> list[np.ndarray]:
        """Extend constraint rows to the current total variable count n_vars."""
        padded = []
        for row in rows:
            if row.shape[0] == n_vars:
                padded.append(row)
            else:
                full = np.zeros(n_vars, dtype=np.float64)
                full[:row.shape[0]] = row
                padded.append(full)
        return padded

    # Existing rows were built before hidden/output variables were added; extend them.
    A_ub = pad_rows(A_ub)
    A_eq = pad_rows(A_eq)

    # Helper accessors for variable indices.
    def h_var(layer: int, i: int, t: int) -> int:
        """Index of the spike variable for hidden neuron i in 'layer' at time t."""
        return hidden_offsets[layer] + i * num_steps + t

    def o_var(c: int, t: int) -> int:
        """Index of the spike variable for output class c at time t."""
        return output_offset + c * num_steps + t

    def add_amo(var_fn, size: int):
        """Add at-most-once (AMO) constraints: Σ_t var_fn(i, t) ≤ 1 for each i."""
        for i in range(size):
            row = np.zeros(n_vars, dtype=np.float64)
            for t in range(num_steps):
                row[var_fn(i, t)] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)

    # AMO constraints for all hidden layers and the output layer (optional).
    if _USE_AMO_CONSTRAINTS:
        for layer in range(n_hidden_layers):
            add_amo(lambda i, t, layer=layer: h_var(layer, i, t), hidden_sizes_list[layer])
        add_amo(o_var, num_classes)

    # -------------------------------------------------------------------------
    # First-fire relaxation (triangle constraints)
    # -------------------------------------------------------------------------
    def add_first_fire_relaxation(var_idx: int, u_center: float, u_row: np.ndarray,
                                  previous_vars: list[int],
                                  u_lb_amo: float | None = None,
                                  u_ub_amo: float | None = None):
        """Add triangle constraints relating a spike variable to its membrane potential.

        The membrane potential at timestep t for neuron i is:
            u(t) = u_center + u_row @ lp_vars
        where u_center is the contribution from fixed (non-LP) inputs, and
        u_row carries the contributions from LP variables.

        The relaxation encodes the following:
          - If u_ub < θ (membrane can never reach threshold): spk[i, t] = 0.
          - Otherwise, two linear inequalities are added:
              Upper bound: u ≥ θ implies spk ≤ 1 (triangle upper edge)
              Lower bound: if u ≥ θ AND no earlier spike, spk ≥ linear function of u
            These form a triangle in the (u, spk) plane.

        previous_vars: indices of the same neuron's spike variables at t' < t.
        The lower-bound constraint subtracts previous spikes because a neuron
        that already fired at t' < t cannot fire again (at-most-once).
        """
        if tight_bounds:
            # Solve two LPs to get tight bounds on u (expensive, used rarely).
            u_lb = u_center + _solve_bound(u_row, A_ub, b_ub, A_eq, b_eq, bounds)
            u_ub = u_center + _solve_bound(u_row, A_ub, b_ub, A_eq, b_eq, bounds, maximize=True)
        elif u_lb_amo is not None:
            # Use the pre-computed at-most-once interval bounds.
            u_lb, u_ub = u_lb_amo, u_ub_amo
        else:
            # Cheap interval arithmetic over the variable bounds.
            u_lb, u_ub = _linear_interval(u_center, u_row, bounds)

        if u_ub < theta:
            # Membrane potential can never reach threshold: neuron cannot fire at t.
            # Force spk[i, t] = 0 with the constraint spk[i, t] ≤ 0.
            row = np.zeros(n_vars, dtype=np.float64)
            row[var_idx] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            return

        # Upper-triangle constraint: spk[i,t] ≤ (u - u_lb) / (θ - u_lb)
        # Rearranged: -u_row @ x + (θ - u_lb) * spk ≤ u_center - u_lb
        upper_denom = theta - u_lb
        if upper_denom > 1e-10:
            row = -u_row.copy()
            row[var_idx] += upper_denom
            A_ub.append(row)
            b_ub.append(u_center - u_lb)

        # Lower-triangle constraint: spk[i,t] ≥ (u - θ) / (u_ub - θ) - Σ_{t'<t} spk[i,t']
        # Rearranged: u_row @ x - (u_ub - θ) * spk - (u_ub - θ) * Σprev ≤ θ - u_center
        lower_denom = u_ub - theta
        if lower_denom > 1e-10:
            row = u_row.copy()
            row[var_idx] -= lower_denom
            for prev in previous_vars:
                row[prev] -= lower_denom   # subtract earlier spike contributions
            A_ub.append(row)
            b_ub.append(theta - u_center)

    # -------------------------------------------------------------------------
    # Build first-layer membrane potential expressions.
    # -------------------------------------------------------------------------
    W0 = Ws[0]     # (hidden_size_0, input_size)
    b0 = bs[0]     # (hidden_size_0,)
    hs0 = hidden_sizes_list[0]

    # layer0_current_c[i, t] = bias contribution + fixed-input contribution
    # to neuron i's membrane at time t (from non-LP inputs).
    layer0_current_c = W0 @ fixed_input.T + b0[:, None]   # shape (hs0, T)

    # layer0_current_rows[i, t, :] = LP-variable coefficients for neuron i's
    # membrane potential at time t (from perturbed-pixel LP variables).
    layer0_current_rows = np.zeros((hs0, num_steps, n_vars), dtype=np.float64)
    for (pix, t), var_idx in p_vars.items():
        # W0[:, pix] is the weight column connecting pixel pix to all hidden neurons.
        layer0_current_rows[:, t, var_idx] = W0[:, pix]

    # Pre-index p_vars by pixel for fast lookup in the AMO-bounds loop.
    pix_to_choices: dict[int, list[tuple[int, int]]] = {}
    for (pix, tau), var_idx in p_vars.items():
        pix_to_choices.setdefault(int(pix), []).append((int(tau), var_idx))

    # -------------------------------------------------------------------------
    # Layer-by-layer triangle relaxation construction.
    # -------------------------------------------------------------------------
    # We track the earliest (t_min) and latest (t_max) possible firing times
    # for each neuron across layers, needed to bound the next layer's membrane.
    all_t_min: list[np.ndarray] = []   # t_min[layer][i] = earliest possible fire time
    all_t_max: list[np.ndarray] = []   # t_max[layer][i] = latest possible fire time
    all_t_def: list[np.ndarray] = []   # t_def[layer][i] = timestep at which firing is guaranteed

    for layer in range(n_hidden_layers):
        hs_L = hidden_sizes_list[layer]
        W_L = Ws[layer]
        b_L = bs[layer]

        # Initialize firing-time range arrays for this layer.
        t_min_L = np.full(hs_L, num_steps, dtype=int)   # T means "never fires"
        t_max_L = np.full(hs_L, -1, dtype=int)          # -1 means "no upper bound found"
        t_def_L = np.full(hs_L, num_steps, dtype=int)   # T means "not guaranteed to fire"

        # Loose AMO bounds on membrane potential (used when tight_bounds=False).
        ub_NR_L = np.full((hs_L, num_steps), -np.inf)
        lb_NR_L = np.full((hs_L, num_steps), np.inf)

        if layer == 0:
            t_min_prev = t_max_prev = t_def_prev = None
        else:
            # Previous layer's firing-time ranges, needed to bound the current layer's
            # membrane potential (since this layer receives spikes from the previous one).
            t_min_prev = all_t_min[layer - 1]
            t_max_prev = all_t_max[layer - 1]
            t_def_prev = all_t_def[layer - 1]
            hs_prev = hidden_sizes_list[layer - 1]

        for i in range(hs_L):
            for t in range(num_steps):
                # Compute u_center (constant part) and u_row (LP-variable part)
                # for neuron i's membrane potential at time t.
                # The LIF membrane at time t is: u(t) = Σ_{τ≤t} β^(t-τ) * (W @ spk_in(τ) + b)
                # which decays exponentially from past inputs.
                u_center = 0.0
                u_row = np.zeros(n_vars, dtype=np.float64)

                if layer == 0:
                    # First hidden layer: input spikes come from the pixel LP variables.
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)   # exponential decay from timestep tau to t
                        u_center += scale * layer0_current_c[i, tau]
                        u_row += scale * layer0_current_rows[i, tau]

                    # Compute AMO-based interval bounds on u (without solving an LP).
                    # At most one pixel fires, and its weight might be positive or negative.
                    u_lb_amo = u_center
                    u_ub_amo = u_center
                    for pix, choices_pix in pix_to_choices.items():
                        # Only count pixels that could fire at or before time t.
                        at_or_before = [(tau, vid) for (tau, vid) in choices_pix if tau <= t]
                        if not at_or_before:
                            continue
                        # Largest scale factor: the earliest possible firing time gives
                        # the most decayed contribution (recall: earlier = smaller t-tau).
                        max_scale = max(beta ** (t - tau) for (tau, _) in at_or_before)
                        w = W_L[i, pix]   # weight connecting this pixel to neuron i
                        if w >= 0.0:
                            u_ub_amo += w * max_scale
                        else:
                            u_lb_amo += w * max_scale
                else:
                    # Deeper hidden layer: input spikes come from the previous layer's
                    # hidden spike LP variables h_var(layer-1, kk, tau).
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)
                        u_center += scale * b_L[i]   # bias accumulated over tau steps
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)
                        for kk in range(hs_prev):
                            u_row[h_var(layer - 1, kk, tau)] += scale * W_L[i, kk]

                    # AMO bounds for deeper layers: bound the spike contribution from each
                    # previous-layer neuron kk by its worst-case weight and timing.
                    u_lb_amo = u_center
                    u_ub_amo = u_center
                    for kk in range(hs_prev):
                        if t_min_prev[kk] > t or t_max_prev[kk] < 0:
                            continue  # neuron kk cannot fire at or before t
                        if (singleton_bounds
                                and t_def_prev[kk] <= t
                                and t_min_prev[kk] == t_max_prev[kk] == t_def_prev[kk]):
                            # Neuron kk fires at exactly one known timestep AND its
                            # spike variable is pinned to 1 by a singleton equality
                            # (added below when singleton_bounds is on). Only then is
                            # it valid to treat the contribution as exact in BOTH
                            # bounds. Without that pin the LP merely forces
                            # spk ≥ (u-θ)/(u_ub-θ) < 1, so an exact contribution would
                            # make u_lb_amo too high / u_ub_amo too low and render the
                            # downstream triangle constraints infeasible. Mirrors the
                            # output-layer guard below.
                            t_fire = t_def_prev[kk]
                            exact_scale = beta ** (t - t_fire)
                            u_lb_amo += W_L[i, kk] * exact_scale
                            u_ub_amo += W_L[i, kk] * exact_scale
                            continue
                        # Neuron kk fires at some unknown timestep in [t_min, t_max].
                        # Worst case: it fires as late as possible (smallest decay = max contribution).
                        latest = min(t, t_max_prev[kk])
                        max_scale = beta ** (t - latest)
                        w = W_L[i, kk]
                        if w >= 0.0:
                            u_ub_amo += w * max_scale
                        else:
                            u_lb_amo += w * max_scale

                # Store the AMO bounds for this neuron-timestep.
                ub_NR_L[i, t] = u_ub_amo
                lb_NR_L[i, t] = u_lb_amo

                # Add the triangle relaxation constraints for h_var(layer, i, t).
                # The AMO membrane bounds are only valid if the at-most-once
                # firing they assume is actually enforced in the LP. For layer 0
                # this is guaranteed by the always-present input-sum constraints
                # (Σ_τ spk[pix,τ] ≤ 1); for deeper layers it requires the optional
                # hidden-spike AMO constraints. When neither holds, the LP can fire
                # an upstream neuron at many timesteps and push u outside
                # [u_lb_amo, u_ub_amo], making the triangle constraints infeasible.
                # Fall back to the sound (looser) box interval in that case.
                amo_sound = (layer == 0) or _USE_AMO_CONSTRAINTS
                add_first_fire_relaxation(
                    h_var(layer, i, t), u_center, u_row,
                    [h_var(layer, i, tau) for tau in range(t)],  # earlier spikes of neuron i
                    u_lb_amo=u_lb_amo if amo_sound else None,
                    u_ub_amo=u_ub_amo if amo_sound else None,
                )

        # -------------------------------------------------------------------------
        # Compute firing-time bounds for this layer (used by the next layer above).
        # -------------------------------------------------------------------------
        for i in range(hs_L):
            # Earliest possible fire time: first t where u_ub ≥ θ (could reach threshold).
            for t in range(num_steps):
                if ub_NR_L[i, t] >= theta:
                    t_min_L[i] = t
                    break
            # Guaranteed fire time: first t where u_lb ≥ θ (always reaches threshold).
            for t in range(num_steps):
                if lb_NR_L[i, t] >= theta:
                    t_def_L[i] = t
                    break
            t_def = t_def_L[i]
            if t_def < num_steps:
                # If firing is guaranteed, the latest it can happen is t_def.
                t_max_L[i] = t_def
            elif t_min_L[i] < num_steps:
                # Firing is possible but not guaranteed; find the latest feasible time.
                for t in range(num_steps - 1, -1, -1):
                    if ub_NR_L[i, t] >= theta:
                        t_max_L[i] = t
                        break

        # -------------------------------------------------------------------------
        # Enforce the firing window on the LP variables.
        # -------------------------------------------------------------------------
        # A neuron cannot spike after its latest possible firing time t_max. This
        # is implicit for NON-guaranteed neurons (t_max is the last step with
        # ub_NR >= θ, so the triangle's own force-zero already pins later spikes
        # to 0), but NOT for guaranteed neurons: there t_max == t_def, yet the
        # free (non-resetting) membrane upper bound ub_NR commonly stays >= θ for
        # every step after t_def, so no force-zero is emitted on (t_def, T).
        #
        # The next layer's membrane upper bound (u_ub_amo) is computed assuming
        # this neuron fires no later than t_max == t_def. If the late-step spike
        # variables are left free, the LP can place the spike after t_def with a
        # larger decay factor, pushing the downstream membrane ABOVE u_ub_amo —
        # an invalid bound that makes the LP spuriously infeasible. This bites
        # harder at low k, where tighter input bounds make more neurons
        # "guaranteed" and thus hit this gap.
        #
        # Pinning h_var(layer, i, τ) = 0 for τ > t_max is sound: in any real
        # execution the neuron fires at most once, no later than t_max, so these
        # variables are 0 in every true trajectory the relaxation must contain.
        for i in range(hs_L):
            if t_max_L[i] < 0:
                continue  # neuron never fires; its spikes are already forced to 0
            for tau in range(t_max_L[i] + 1, num_steps):
                bounds[h_var(layer, i, tau)] = (0.0, 0.0)

        # Optional tighter bounds for neurons that fire at exactly one timestep.
        if singleton_bounds:
            for i in range(hs_L):
                if (t_def_L[i] < num_steps
                        and t_min_L[i] == t_max_L[i] == t_def_L[i]):
                    # Neuron i fires at exactly t_def_L[i] under all perturbations.
                    # Add an equality constraint to pin its spike variable.
                    t_fire = t_def_L[i]
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[h_var(layer, i, t_fire)] = 1.0
                    A_eq.append(row)
                    b_eq.append(1.0)

        all_t_min.append(t_min_L)
        all_t_max.append(t_max_L)
        all_t_def.append(t_def_L)

    # -------------------------------------------------------------------------
    # Output layer triangle relaxation.
    # -------------------------------------------------------------------------
    t_min_last = all_t_min[-1]
    t_max_last = all_t_max[-1]
    t_def_last = all_t_def[-1]
    hs_last = hidden_sizes_list[-1]
    W_out = Ws[-1]     # (num_classes, hs_last)
    b_out = bs[-1]     # (num_classes,)
    last_layer_idx = n_hidden_layers - 1

    for c in range(num_classes):
        for t in range(num_steps):
            # Output neuron c's membrane potential at time t:
            # u(t) = Σ_{τ≤t} β^(t-τ) * (W_out[c,:] @ h_spk(τ) + b_out[c])
            u_center = 0.0
            u_row = np.zeros(n_vars, dtype=np.float64)
            for tau in range(t + 1):
                scale = beta ** (t - tau)
                u_center += scale * b_out[c]
                for i in range(hs_last):
                    u_row[h_var(last_layer_idx, i, tau)] += scale * W_out[c, i]

            # AMO bounds for the output layer (same logic as the deeper hidden layers).
            u_lb_amo = u_center
            u_ub_amo = u_center
            for i in range(hs_last):
                if t_min_last[i] > t or t_max_last[i] < 0:
                    continue
                if (singleton_bounds
                        and t_def_last[i] <= t
                        and t_min_last[i] == t_max_last[i] == t_def_last[i]):
                    t_fire = t_def_last[i]
                    exact_scale = beta ** (t - t_fire)
                    u_lb_amo += W_out[c, i] * exact_scale
                    u_ub_amo += W_out[c, i] * exact_scale
                    continue
                latest = min(t, t_max_last[i])
                max_scale = beta ** (t - latest)
                w = W_out[c, i]
                if w >= 0.0:
                    u_ub_amo += w * max_scale
                else:
                    u_lb_amo += w * max_scale

            # Output-layer membrane bounds depend on at-most-once firing of the
            # last hidden layer, which is only enforced when AMO constraints are
            # on. Otherwise fall back to the sound box interval (see the layer
            # loop above for the full rationale).
            add_first_fire_relaxation(
                o_var(c, t), u_center, u_row, [o_var(c, tau) for tau in range(t)],
                u_lb_amo=u_lb_amo if _USE_AMO_CONSTRAINTS else None,
                u_ub_amo=u_ub_amo if _USE_AMO_CONSTRAINTS else None,
            )

    # -------------------------------------------------------------------------
    # Objective rows: score[c] = Σ_t (T - t) * o_var(c, t)
    # -------------------------------------------------------------------------
    # score_rows[c, :] = coefficient vector for class c's F2F score over all LP vars.
    score_rows = np.zeros((num_classes, n_vars), dtype=np.float64)
    for c in range(num_classes):
        for t in range(num_steps):
            score_rows[c, o_var(c, t)] = float(num_steps - t)

    # Build the final sparse LP matrices from the accumulated constraint lists.
    lp_matrices = _prepare_lp_matrices(A_ub, b_ub, A_eq, b_eq)

    # Populate the failure-diagnostic context so that any [LP-fail] message
    # printed by _solve_bound_prepared can identify the exact problem instance.
    _LP_CONTEXT.update({
        "pixel_indices": [int(i) for i in indices],
        "pixel_values": image_flat[indices].tolist() if len(indices) > 0 else [],
        "label": label,
        "epsilon": epsilon,
        "fixed_latencies": fixed_latencies if fixed_latencies else {},
    })

    # -------------------------------------------------------------------------
    # Optional self-check: verify the NOMINAL (unperturbed) trajectory is LP-feasible.
    # -------------------------------------------------------------------------
    # The centre of the ε-ball is a genuine network execution, so in a SOUND
    # relaxation it MUST satisfy every constraint and bound. If it violates one,
    # that constraint/bound is over-tight (excludes a real trajectory) — the
    # direct cause of spurious LP infeasibility. Enabled by SNNV_CHECK_NOMINAL=1.
    if os.environ.get("SNNV_CHECK_NOMINAL"):
        inv_pvars = {vid: (pix, t) for (pix, t), vid in p_vars.items()}

        def _decode_var(j: int) -> str:
            if j in inv_pvars:
                pix, t = inv_pvars[j]
                return f"in(pix={pix},t={t})"
            if j >= output_offset:
                rel = j - output_offset
                return f"o(c={rel // num_steps},t={rel % num_steps})"
            for L in range(n_hidden_layers):
                base = hidden_offsets[L]
                size = hidden_sizes_list[L] * num_steps
                if base <= j < base + size:
                    rel = j - base
                    ii, tt = rel // num_steps, rel % num_steps
                    return (f"h(L={L},i={ii},t={tt}) "
                            f"[t_min={all_t_min[L][ii]},t_max={all_t_max[L][ii]},"
                            f"t_def={all_t_def[L][ii]}]")
            return f"var#{j}"

        # Build the concrete nominal latency assignment (respecting any fixed
        # latencies pinned by the split branch), then simulate to get spikes.
        nom_lat = base_lat.astype(int).copy()
        for pix, tt in (fixed_latencies or {}).items():
            nom_lat[int(pix)] = int(tt)
        xc = np.zeros(n_vars, dtype=np.float64)
        for (pix, t), vid in p_vars.items():
            if int(nom_lat[pix]) == t:
                xc[vid] = 1.0
        st = spike_train_from_latencies(nom_lat, num_steps)
        _, hid, outp = model.simulate_with_patterns(st)   # hid:(T,Σhidden) outp:(T,classes)
        for L in range(n_hidden_layers):
            col0 = int(sum(hidden_sizes_list[:L]))
            for ii in range(hidden_sizes_list[L]):
                for tt in range(num_steps):
                    xc[h_var(L, ii, tt)] = float(hid[tt, col0 + ii])
        for c in range(num_classes):
            for tt in range(num_steps):
                xc[o_var(c, tt)] = float(outp[tt, c])

        tol = 1e-6
        ctx = _LP_CONTEXT
        hdr = (f"image_idx={ctx.get('image_idx', '?')} label={label} "
               f"eps={epsilon} k={len(indices)} pix={[int(i) for i in indices]}")
        viols: list[str] = []
        for j, (lo, hi) in enumerate(bounds):
            if xc[j] < lo - tol or xc[j] > hi + tol:
                viols.append(f"    BOUND  {_decode_var(j)}  value={xc[j]:.4g} "
                             f"not in [{lo:.4g},{hi:.4g}]")
        A_ub_arr = np.asarray(A_ub) if A_ub else np.zeros((0, n_vars))
        A_eq_arr = np.asarray(A_eq) if A_eq else np.zeros((0, n_vars))
        if len(A_ub_arr):
            ub_val = A_ub_arr @ xc
            for r in np.where(ub_val > np.asarray(b_ub) + tol)[0]:
                nz = np.nonzero(A_ub_arr[r])[0]
                terms = ", ".join(f"{A_ub_arr[r, j]:+.3g}*{_decode_var(j)}" for j in nz[:6])
                viols.append(f"    UB#{r}  lhs={ub_val[r]:.4g} > rhs={b_ub[r]:.4g}  "
                             f"[{terms}{' ...' if len(nz) > 6 else ''}]")
        if len(A_eq_arr):
            eq_val = A_eq_arr @ xc
            for r in np.where(np.abs(eq_val - np.asarray(b_eq)) > tol)[0]:
                nz = np.nonzero(A_eq_arr[r])[0]
                terms = ", ".join(f"{A_eq_arr[r, j]:+.3g}*{_decode_var(j)}" for j in nz[:6])
                viols.append(f"    EQ#{r}  lhs={eq_val[r]:.4g} != rhs={b_eq[r]:.4g}  "
                             f"[{terms}{' ...' if len(nz) > 6 else ''}]")
        if viols:
            print(f"[NOMINAL-INFEASIBLE] {hdr}  ({len(viols)} violations)")
            for line in viols[:20]:
                print(line)
            if len(viols) > 20:
                print(f"    ... ({len(viols) - 20} more)")

    if _DEBUG_LP:
        mats = lp_matrices if not isinstance(lp_matrices, _HighsPreparedLP) \
               else lp_matrices.lp_matrices
        A_ub_m, _, A_eq_m, _ = mats
        n_ub = A_ub_m.shape[0] if A_ub_m is not None else 0
        n_eq = A_eq_m.shape[0] if A_eq_m is not None else 0
        ctx = _LP_CONTEXT
        pidx = ctx.get("pixel_indices", [])
        pval = ctx.get("pixel_values", [])
        fixed = ctx.get("fixed_latencies") or {}
        print(f"  [LP-debug] image_idx={ctx.get('image_idx', '?')}  "
              f"label={ctx.get('label', '?')}  epsilon={ctx.get('epsilon', '?')}  "
              f"n_vars={n_vars}  n_ub={n_ub}  n_eq={n_eq}")
        print(f"    perturbed pixels (in selection order): {pidx}")
        print(f"    pixel values at those positions:       "
              f"{[round(v, 4) for v in pval]}")
        if fixed:
            print(f"    fixed latencies (split branch):        {fixed}")

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Solve for output score bounds (lb_y, ub_y) if requested.
    # -------------------------------------------------------------------------
    lb_y = None
    ub_y = None
    if not (cert_only and label is not None):
        lb_y = np.zeros(num_classes, dtype=np.float64)
        ub_y = np.zeros(num_classes, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Solve for the certification gap: min_c' score[label] - score[c'].
    # -------------------------------------------------------------------------
    gap_joint = None
    if label is not None:
        gap_joint = np.full(num_classes, np.inf)

    # Build the full list of LP tasks: each is (result_array, index, objective, maximize).
    # All tasks share the same lp_matrices and bounds (read-only), so they are
    # independent and safe to run in parallel threads (linprog releases the GIL).
    tasks = []
    if lb_y is not None:
        for c in range(num_classes):
            tasks.append((lb_y, c, score_rows[c],                          False))
            tasks.append((ub_y, c, score_rows[c],                          True))
    if gap_joint is not None:
        for c_prime in range(num_classes):
            if c_prime != label:
                tasks.append((gap_joint, c_prime, score_rows[label] - score_rows[c_prime], False))

    if parallel_workers > 1 and len(tasks) > 1 and parallel_backend != "process":
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            futures = {
                executor.submit(_solve_bound_prepared, obj, lp_matrices, bounds, maximize): (arr, idx)
                for arr, idx, obj, maximize in tasks
            }
            for future in futures:
                arr, idx = futures[future]
                arr[idx] = future.result()
    else:
        for arr, idx, obj, maximize in tasks:
            arr[idx] = _solve_bound_prepared(obj, lp_matrices, bounds, maximize)

    runtime = time.perf_counter() - t0

    total_hidden_spike_vars = sum(hs * num_steps for hs in hidden_sizes_list)
    return {
        "lb": lb_y, "ub": ub_y, "gap_joint": gap_joint,
        "runtime_s": runtime,
        "n_input_vars": n_input_vars,
        "n_hidden_spike_vars": total_hidden_spike_vars,
        "n_output_spike_vars": num_classes * num_steps,
        "n_lp_variables": n_vars,
        "n_lp_constraints": len(A_ub) + len(A_eq),
        "t_min_h": all_t_min[-1].tolist(),   # earliest firing times for last hidden layer
        "t_max_h": all_t_max[-1].tolist(),   # latest firing times for last hidden layer
        "invalid_subproblem": False,
    }


def build_symbolic_relaxation_lp_split(
    model, image_flat: np.ndarray, epsilon: float,
    k: int, num_steps: int,
    split_depth: int = 0,
    label: int | None = None,
    tight_bounds: bool = False,
    parallel_workers: int = 1,
    cert_only: bool = False,
    parallel_backend: str = "thread",
    split_strategy: str = "selected",
    pixel_indices: np.ndarray | None = None,
    singleton_bounds: bool = False,
) -> dict:
    """Symbolic split verification: branch on the first split_depth pixels' latencies.

    This function enumerates all combinations of feasible latency assignments for
    the top split_depth pixels (ordered by split_strategy) and solves one LP per
    branch. The final bounds are the element-wise min/max over all branch results.

    If all k pixels are fixed in a branch (split_depth == k), the model is
    simulated exactly for that latency assignment (no LP needed).

    Parallelism: branches are dispatched to a thread/process pool when
    parallel_workers > 1.
    """
    indices = np.asarray(pixel_indices, dtype=int) if pixel_indices is not None else np.empty(0, dtype=int)
    lb_x, ub_x = make_bounds(image_flat, indices, epsilon)
    num_classes = model.fcs[-1].out_features

    # Determine the split ordering for the first split_depth pixels.
    ordered_indices = order_split_indices(model, indices, lb_x, ub_x, num_steps, split_strategy)
    split_indices = ordered_indices[:split_depth]   # pixels whose latency will be branched

    # For each split pixel, enumerate all feasible latency values.
    pix_choices_split = {}
    for pix in split_indices:
        choices = feasible_latencies(float(lb_x[pix]), float(ub_x[pix]), num_steps)
        pix_choices_split[int(pix)] = choices

    # Generate all Cartesian-product combinations of fixed latency assignments.
    from itertools import product as iproduct
    keys = sorted(pix_choices_split)   # pixel indices in a deterministic order
    combos = list(iproduct(*[pix_choices_split[p] for p in keys])) if keys else [()]

    # Accumulators for merging results across all branches.
    lb_merged: np.ndarray | None = None
    ub_merged: np.ndarray | None = None
    gap_joint_merged: np.ndarray | None = None
    if label is not None:
        # Start with +inf; take element-wise min over branches (worst case per competitor).
        gap_joint_merged = np.full(num_classes, np.inf, dtype=np.float64)
    n_lp_vars_total = 0
    n_lp_constr_total = 0
    invalid_subproblem = False

    base_lat = latency_from_values(torch.from_numpy(image_flat).float(), num_steps).numpy()
    simulate_lock = threading.Lock()  # protect model.simulate_with_patterns in threaded mode

    def solve_combo(combo):
        """Solve one branch: fix latencies for split pixels, free the rest."""
        fixed = {keys[j]: combo[j] for j in range(len(keys))} if keys else {}
        if len(fixed) == len(indices):
            # All pixels are pinned — exact simulation, no LP.
            lats = base_lat.copy()
            for pix, lat in fixed.items():
                lats[pix] = lat
            st = spike_train_from_latencies(lats, num_steps)
            with simulate_lock:
                score, _, _ = model.simulate_with_patterns(st)
            score_arr = np.array(score, dtype=np.float64)
            gap_joint = None
            if label is not None and gap_joint_merged is not None:
                gap_joint = np.full(num_classes, np.inf, dtype=np.float64)
                for c in range(len(score_arr)):
                    if c != label:
                        gap = float(score_arr[label] - score_arr[c])
                        gap_joint[c] = min(gap_joint[c], gap)
            return {
                "lb": score_arr, "ub": score_arr, "gap_joint": gap_joint,
                "n_lp_variables": 0, "n_lp_constraints": 0, "invalid_subproblem": False,
            }
        # Partially fixed: solve the relaxation LP with the fixed latencies.
        return build_symbolic_relaxation_lp(
            model, image_flat, epsilon, k, num_steps,
            tight_bounds=tight_bounds, fixed_latencies=fixed, label=label,
            cert_only=cert_only, pixel_indices=indices, singleton_bounds=singleton_bounds,
            parallel_workers=parallel_workers, parallel_backend=parallel_backend,
        )

    n_combos = len(combos)
    if parallel_workers > 1 and n_combos > 1 and parallel_backend == "process":
        # Process pool: serialize the model and state for the worker initializer.
        config = {
            "input_size": model.fcs[0].in_features,
            "hidden_sizes": [fc.out_features for fc in model.fcs[:-1]],
            "num_classes": model.fcs[-1].out_features,
            "beta": float(model.lifs[0].beta.item() if hasattr(model.lifs[0].beta, "item")
                          else model.lifs[0].beta),
            "threshold": float(model.lifs[0].threshold.item()
                               if hasattr(model.lifs[0].threshold, "item")
                               else model.lifs[0].threshold),
            "num_steps": num_steps,
        }
        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        # Use "fork" on POSIX (faster) and "spawn" on Windows (required).
        mp_context = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        with ProcessPoolExecutor(
            max_workers=int(parallel_workers), mp_context=mp_context,
            initializer=_init_symbolic_split_worker,
            initargs=(config, state_dict, image_flat, epsilon, k, num_steps, keys, indices,
                      base_lat, tight_bounds, label, cert_only, singleton_bounds),
        ) as executor:
            combo_results = list(_tqdm(
                executor.map(_solve_symbolic_split_worker, combos),
                total=n_combos, desc="      combos", leave=False,
            ))
    elif parallel_workers > 1 and n_combos > 1:
        # Thread pool: simpler than process pool, avoids serialization cost.
        # Suitable because scipy LP releases the GIL.
        from concurrent.futures import as_completed as _as_completed
        combo_results = [None] * n_combos
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            futures = {executor.submit(solve_combo, combo): i for i, combo in enumerate(combos)}
            with _tqdm(total=n_combos, desc="      combos", leave=False) as cbar:
                for future in _as_completed(futures):
                    combo_results[futures[future]] = future.result()
                    cbar.update(1)
    else:
        # Sequential: iterate over all branches without parallelism.
        combo_results = []
        for combo in _tqdm(combos, desc="      combos", leave=False):
            combo_results.append(solve_combo(combo))

    # -------------------------------------------------------------------------
    # Merge branch results: take element-wise min for lower bounds (most
    # conservative), element-wise max for upper bounds (widest interval),
    # and element-wise min for gap_joint (worst-case margin).
    # -------------------------------------------------------------------------
    for r in combo_results:
        if r["lb"] is not None and r["ub"] is not None and (
            np.any(np.isinf(r["lb"])) or np.any(np.isinf(r["ub"]))
        ):
            invalid_subproblem = True
            break
        if r["lb"] is not None and r["ub"] is not None:
            if lb_merged is None:
                lb_merged = r["lb"].copy()
                ub_merged = r["ub"].copy()
            else:
                lb_merged = np.minimum(lb_merged, r["lb"])  # worst-case lower bound
                ub_merged = np.maximum(ub_merged, r["ub"])  # worst-case upper bound
        if label is not None and gap_joint_merged is not None and r.get("gap_joint") is not None:
            gap_joint_merged = np.minimum(gap_joint_merged, r["gap_joint"])  # worst-case gap
        n_lp_vars_total += r["n_lp_variables"]
        n_lp_constr_total += r["n_lp_constraints"]

    return {
        "lb": None if invalid_subproblem else lb_merged,
        "ub": None if invalid_subproblem else ub_merged,
        "gap_joint": gap_joint_merged,
        "n_cases": len(combos),          # number of LP sub-problems solved
        "invalid_subproblem": invalid_subproblem,
        "n_lp_variables": n_lp_vars_total,
        "n_lp_constraints": n_lp_constr_total,
        "n_input_vars": 0,
        "n_hidden_spike_vars": 0,
        "n_output_spike_vars": 0,
        "runtime_s": 0.0,
        "split_pixel_order": split_indices.tolist(),
    }


def _gap_from_result(result, label):
    """Extract the certification gap from an LP result dict.

    Prefers gap_joint (directly certified margin) when available.
    Falls back to lb_y[label] - max(ub_y[not label]) if gap_joint is missing.

    Returns +inf if certified with large margin, negative if not certified.
    """
    gap_joint = result.get("gap_joint")
    if gap_joint is not None:
        competitors = [c for c in range(len(gap_joint)) if c != label]
        if competitors and np.all(np.isfinite(gap_joint[competitors])):
            # gap > 0: label class beats every competitor under all perturbations.
            return float(np.min(gap_joint[competitors]))
    # Fallback: use per-class score bounds.
    lb_y = result["lb"]
    ub_y = result["ub"]
    if lb_y is None or ub_y is None:
        return float("-inf")  # LP failed or invalid
    return float(lb_y[label] - np.max(np.delete(ub_y, label)))


def verify_symbolic_sample(model, image_flat: np.ndarray, label: int, epsilon: float,
                           k: int, num_steps: int,
                           tight_bounds: bool = False,
                           split_depth: int = 0,
                           max_depth_cap: int | None = None,
                           parallel_workers: int = 1,
                           cert_only: bool = False,
                           parallel_backend: str = "thread",
                           split_strategy: str = "selected",
                           pixel_indices: np.ndarray | None = None,
                           track_depth: bool = False,
                           singleton_bounds: bool = False):
    """Attempt to certify a sample by iterating over split depths until certified.

    Starts at depth 0 (single LP over all k pixels). If the gap is not positive,
    tries depth 1 (branch on 1 pixel), depth 2, ..., up to max_depth.

    For split_depth > 0, each branch fixes one pixel's latency and solves a
    smaller LP for the remaining pixels. Higher depth = more branches = slower
    but tighter. Stops as soon as gap > 0 (certified).

    When called from SNNVerifier.verify with split_depth=0 and max_depth_cap=0,
    this is the pure depth-0 LP (Stage 1 of the two-stage strategy).
    """
    t0 = time.perf_counter()
    max_depth = split_depth if split_depth > 0 else min(
        len(pixel_indices) if pixel_indices is not None else k,
        max_depth_cap if max_depth_cap is not None else (
            len(pixel_indices) if pixel_indices is not None else k
        )
    )
    depth_reached = 0
    for depth in range(max_depth + 1):
        depth_reached = depth
        if depth == 0:
            # Single LP covering all k perturbed pixels simultaneously.
            result = build_symbolic_relaxation_lp(
                model, image_flat, epsilon, k, num_steps, tight_bounds=tight_bounds,
                label=label, cert_only=cert_only, pixel_indices=pixel_indices,
                singleton_bounds=singleton_bounds,
                parallel_workers=parallel_workers, parallel_backend=parallel_backend,
            )
        else:
            # Branch on the first 'depth' pixels; solve one LP per branch.
            result = build_symbolic_relaxation_lp_split(
                model, image_flat, epsilon, k, num_steps,
                split_depth=depth, label=label, tight_bounds=tight_bounds,
                parallel_workers=parallel_workers, cert_only=cert_only,
                parallel_backend=parallel_backend, split_strategy=split_strategy,
                pixel_indices=pixel_indices, singleton_bounds=singleton_bounds,
            )
        gap = _gap_from_result(result, label)
        if track_depth:
            elapsed = time.perf_counter() - t0
            status = "certified" if gap > 0.0 else "not certified"
            print(f"    depth {depth}/{max_depth}  gap={gap:.3f}  {status}  ({elapsed:.1f}s)")
        if gap > 0.0:
            break   # certified — no need to go deeper
    runtime = time.perf_counter() - t0
    lb_y = result["lb"]
    ub_y = result["ub"]
    if lb_y is None or ub_y is None:
        bound_width = float("nan")
        lb_list = None
        ub_list = None
    else:
        bound_width = float(np.mean(ub_y - lb_y))
        lb_list = lb_y.tolist()
        ub_list = ub_y.tolist()
    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "label": int(label),
        "mode": "symbolic",
        "tight_bounds": bool(tight_bounds),
        "gap": gap,
        "certified": gap > 0.0,
        "bound_width": bound_width,
        "runtime_s": runtime,
        "lp_runtime_s": result["runtime_s"],
        "depth_reached": int(depth_reached),
        "max_depth": int(max_depth),
        "pixel_order": result.get("split_pixel_order", []),
        "n_cases": int(result.get("n_cases", 1)),
        "n_lambda": result["n_lp_variables"],
        "n_patterns": result["n_lp_constraints"],
        "n_input_vars": result["n_input_vars"],
        "n_hidden_spike_vars": result["n_hidden_spike_vars"],
        "n_output_spike_vars": result["n_output_spike_vars"],
        "n_lp_variables": result["n_lp_variables"],
        "n_lp_constraints": result["n_lp_constraints"],
        "lb": lb_list,
        "ub": ub_list,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(rows, epsilons, ks, mode="full", split_strategy=None):
    """Aggregate per-sample verification rows into summary statistics.

    Groups rows by (epsilon, k) and computes certification rates, median/mean
    gaps, timing statistics, and depth statistics.
    """
    summary = []
    for epsilon in epsilons:
        for k in ks:
            # Select rows matching this (epsilon, k, mode, strategy) combination.
            group = [r for r in rows if r["mode"] == mode
                     and math.isclose(r["epsilon"], epsilon) and r["k"] == k
                     and (split_strategy is None or r.get("split_strategy") == split_strategy)]
            if not group:
                continue
            cert = sum(r["certified"] for r in group)
            gaps = np.array([r["gap"] for r in group], dtype=float)
            times = np.array([r["runtime_s"] for r in group], dtype=float)
            widths = np.array([r["bound_width"] for r in group], dtype=float)
            lambdas = np.array([r.get("n_lambda", 0) for r in group], dtype=float)
            patterns = np.array([r.get("n_patterns", 0) for r in group], dtype=float)
            # explicit_coverage: fraction of MC samples within the LP bounds.
            explicit_coverage = (
                100.0 * sum(bool(r.get("explicit_coverage") or False) for r in group) / len(group)
                if any("explicit_coverage" in r for r in group) else 100.0
            )
            mc_coverage = (
                float(np.mean([r.get("mc_coverage_pct", np.nan) for r in group]))
                if any("mc_coverage_pct" in r for r in group) else 100.0
            )
            if any("depth_reached" in r for r in group):
                depths = [r.get("depth_reached", 0) for r in group]
                depth0_count = int(sum(d == 0 for d in depths))
                split_count = int(sum(d > 0 for d in depths))
                median_depth = float(np.median(depths))
            else:
                depths = [0] * len(group)
                depth0_count = split_count = 0
                median_depth = 0.0
            entry = {
                "epsilon": epsilon, "k": k, "samples": len(group),
                "mc_coverage_pct": mc_coverage,
                "explicit_coverage_pct": explicit_coverage,
                "certified_pct": 100.0 * cert / len(group),
                "unknown_pct": 100.0 * (len(group) - cert) / len(group),
                "median_gap": float(np.median(gaps)),
                "median_time_s": float(np.median(times)),
                "median_width": float(np.median(widths)),
                "median_lambda": float(np.median(lambdas)),
                "median_patterns": float(np.median(patterns)),
                "depth0_count": depth0_count,
                "split_count": split_count,
                "median_depth": median_depth,
            }
            if mode == "symbolic":
                certs = [int(r.get("certified", False)) for r in group]
                entry["time_depth"] = [[float(t), int(d), c]
                                       for t, d, c in zip(times.tolist(), depths, certs)]
            else:
                entry["times_s"] = times.tolist()
            summary.append(entry)
    return summary


def summarize_depth0_exhaustive(rows, epsilons, ks):
    """Aggregate results from the two-stage depth-0 + exhaustive-fallback strategy.

    For each (epsilon, k) group, tracks how many samples were certified at
    depth-0 alone vs. required the exhaustive fallback, and their timing.

    The rows list contains a mix of 'symbolic' rows (Stage 1, mode='symbolic')
    and 'full' rows (Stage 2, mode='full'). Both are keyed by image_idx.
    """
    summary = []
    for epsilon in epsilons:
        for k in ks:
            # Depth-0 LP rows: mode='symbolic', split_strategy='depth0'.
            sym_rows = {
                r["image_idx"]: r for r in rows
                if r["mode"] == "symbolic" and r.get("split_strategy") == "depth0"
                and math.isclose(r["epsilon"], epsilon) and r["k"] == k
            }
            # Exhaustive fallback rows: mode='full'.
            full_rows = {
                r["image_idx"]: r for r in rows
                if r["mode"] == "full"
                and math.isclose(r["epsilon"], epsilon) and r["k"] == k
            }
            all_idxs = set(sym_rows) | set(full_rows)
            if not all_idxs:
                continue

            cert_depth0 = cert_exhaust = cert_neither = 0
            times_depth0 = []
            times_exhaust = []

            for idx in sorted(all_idxs):
                sym = sym_rows.get(idx)
                full = full_rows.get(idx)
                if sym:
                    times_depth0.append(sym["runtime_s"])
                if sym and sym["certified"]:
                    cert_depth0 += 1       # certified at Stage 1
                elif full:
                    times_exhaust.append(full["runtime_s"])
                    if full["certified"]:
                        cert_exhaust += 1  # certified at Stage 2
                    else:
                        cert_neither += 1  # neither stage could certify
                else:
                    cert_neither += 1

            n = len(all_idxs)
            summary.append({
                "epsilon": epsilon, "k": k, "samples": n,
                "cert_depth0_pct": 100.0 * cert_depth0 / n,
                "cert_exhaustive_pct": 100.0 * cert_exhaust / n,
                "cert_total_pct": 100.0 * (cert_depth0 + cert_exhaust) / n,
                "uncertified_pct": 100.0 * cert_neither / n,
                "median_time_depth0_s": float(np.median(times_depth0)) if times_depth0 else float("nan"),
                "median_time_exhaust_s": float(np.median(times_exhaust)) if times_exhaust else float("nan"),
            })
    return summary


# ---------------------------------------------------------------------------
# Row caching helpers
# ---------------------------------------------------------------------------

def _row_cache_key(image_idx, epsilon, k, mode, split_strategy=None):
    """Compute a hashable cache key for one verification result row.

    The cache key uniquely identifies a (sample, epsilon, k, verification mode,
    split strategy) combination, allowing re-runs to skip already-computed rows.
    """
    return (image_idx, round(float(epsilon), 10), int(k), mode, split_strategy)


def load_existing_rows(output_dir: Path) -> dict:
    """Load previously computed verification rows from disk into a lookup dict.

    Checks both results.json (final output) and rows_checkpoint.json (partial
    checkpoint from an interrupted run). Returns a dict keyed by _row_cache_key.
    """
    cache = {}
    for fname in ("results.json", "rows_checkpoint.json"):
        path = output_dir / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            for r in data.get("rows", []):
                key = _row_cache_key(r["image_idx"], r["epsilon"], r["k"],
                                     r["mode"], r.get("split_strategy"))
                cache[key] = r
        except Exception as exc:
            print(f"warning: could not load cache from {path} ({exc}); skipping")
    return cache


# ---------------------------------------------------------------------------
# SNNVerifier — public class interface
# ---------------------------------------------------------------------------

class SNNVerifier:
    """Train and verify an F2F SNN. Pixel selection is handled externally by the caller.

    Typical usage (from comp_snn_vs_ann.py):
        verifier = SNNVerifier(hidden_sizes=[128, 64], num_steps=20, beta=0.9,
                               threshold=1.0, output_dir="experiments/outputs/snn")
        verifier.train(train_ds, test_ds, epochs=5, lr=5e-4, ...)
        result = verifier.verify(image_flat, indices, epsilon=0.05, label=3, ...)
    """

    def __init__(self, hidden_sizes: list[int], num_steps: int, beta: float,
                 threshold: float, output_dir: str | Path):
        self.hidden_sizes = list(hidden_sizes)
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        self.model: F2FMLP | None = None
        self._input_size: int | None = None
        self._num_classes: int | None = None

    def load_checkpoint(self) -> bool:
        """Load the trained model from the saved checkpoint file.

        Returns True if the checkpoint existed and was loaded, False otherwise.
        Supports both the new format (with 'config' key) and the legacy format.
        """
        ckpt = self.output_dir / "snn_checkpoint.pt"
        if not ckpt.exists():
            return False
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "config" in data:
            input_size = data["config"]["input_size"]
            num_classes = data["config"]["num_classes"]
        else:
            # Legacy checkpoint: infer input_size and num_classes from the state dict.
            sd = data["model_state_dict"]
            input_size = sd["fcs.0.weight"].shape[1]
            last_w = sorted([k for k in sd if k.startswith("fcs.") and k.endswith(".weight")],
                            key=lambda k: int(k.split(".")[1]))[-1]
            num_classes = sd[last_w].shape[0]
        model = F2FMLP(
            input_size=input_size, hidden_sizes=self.hidden_sizes,
            num_classes=num_classes, beta=self.beta,
            threshold=self.threshold, num_steps=self.num_steps,
        )
        model.load_state_dict(data["model_state_dict"])
        self.model = model.eval()
        self._input_size = input_size
        self._num_classes = num_classes
        return True

    def train(self, train_ds, test_ds, epochs: int, lr: float,
              train_limit: int, batch_size: int) -> dict:
        """Train the SNN. Saves checkpoint to output_dir. Returns train_summary.

        Infers input_size and num_classes from the first batch of training data.
        Runs one evaluation pass on the full test set after each epoch.
        """
        input_size = train_ds[0][0].numel()   # total number of pixels (flattened)

        # Infer num_classes by scanning a small batch of labels.
        sample_loader = DataLoader(Subset(train_ds, list(range(min(256, len(train_ds))))),
                                   batch_size=256)
        labels_seen = set()
        for _, lbls in sample_loader:
            for l in lbls:
                labels_seen.add(int(l))
        num_classes = max(labels_seen) + 1

        self._input_size = input_size
        self._num_classes = num_classes
        ckpt = self.output_dir / "snn_checkpoint.pt"

        model = F2FMLP(
            input_size=input_size, hidden_sizes=self.hidden_sizes,
            num_classes=num_classes, beta=self.beta,
            threshold=self.threshold, num_steps=self.num_steps,
        )

        # Cap the training set at train_limit samples to allow quick experiments.
        subset = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        summary = {"epochs": epochs, "train_limit": len(subset)}

        for epoch in range(epochs):
            model.train()
            correct = total = 0
            loss_sum = 0.0
            for images, labels in loader:
                # Convert pixel images to latency-coded spike trains before forwarding.
                spikes = encode_batch(images, self.num_steps)
                scores = model(spikes)
                loss = criterion(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                correct += (scores.argmax(dim=1) == labels).sum().item()
                total += labels.numel()
            summary[f"epoch_{epoch+1}_loss"] = loss_sum / max(1, len(loader))
            summary[f"epoch_{epoch+1}_train_acc"] = 100.0 * correct / max(1, total)

            # Evaluate on the full test set each epoch.
            model.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    spikes = encode_batch(images, self.num_steps)
                    scores = model(spikes)
                    t_correct += (scores.argmax(dim=1) == labels).sum().item()
                    t_total += labels.numel()
            summary[f"epoch_{epoch+1}_test_acc"] = 100.0 * t_correct / max(1, t_total)
            print(f"epoch {epoch+1}: loss={summary[f'epoch_{epoch+1}_loss']:.4f}"
                  f"  train_acc={summary[f'epoch_{epoch+1}_train_acc']:.1f}%"
                  f"  test_acc={summary[f'epoch_{epoch+1}_test_acc']:.1f}%")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_summary": summary,
            "config": {"input_size": input_size, "num_classes": num_classes},
        }, ckpt)
        self.model = model.eval()
        return summary

    @torch.no_grad()
    def test_accuracy(self, test_ds, batch_size: int = 256) -> float:
        """Evaluate classification accuracy on the test set. Returns percentage."""
        assert self.model is not None, "call train() or load_checkpoint() first"
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        correct = total = 0
        for images, labels in loader:
            spikes = encode_batch(images, self.num_steps)
            scores = self.model(spikes)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
        return 100.0 * correct / max(1, total)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> int:
        """Return the predicted class label for a single image tensor."""
        assert self.model is not None
        spikes = encode_batch(image.unsqueeze(0), self.num_steps)
        return int(self.model(spikes)[0].argmax().item())

    @torch.no_grad()
    def scores(self, image: torch.Tensor) -> np.ndarray:
        """Return raw F2F scores for a single image (shape: num_classes)."""
        assert self.model is not None
        spikes = encode_batch(image.unsqueeze(0), self.num_steps)
        return self.model(spikes)[0].cpu().numpy()

    def verify(self, image_flat: np.ndarray, indices: np.ndarray, epsilon: float,
               label: int, split_depth: int = -1,
               strategy: str = "choice-influence",
               parallel_workers: int = 1,
               parallel_backend: str = "thread",
               mc_samples: int = 0,
               singleton_bounds: bool = False,
               tight_bounds: bool = False,
               track_depth: bool = False,
               image_idx: int | None = None,
               eq_constraints: bool = True,
               debug_lp: bool = False,
               amo: bool = False) -> dict:
        """Certify one sample. indices and epsilon are passed in by the caller.

        split_depth=-1 (default): Two-stage strategy.
          Stage 1: depth-0 LP (single LP, all k pixels symbolic).
          Stage 2: exhaustive fallback if Stage 1 fails (enumerate all latency
                   combos, exact simulation per combo).

        split_depth=0: Depth-0 LP only (no fallback).

        split_depth=N>0: Symbolic split to depth N (branch-and-bound LP tree,
                         no exhaustive simulation).

        Returns a dict with keys 'symbolic' (Stage 1 result) and 'exhaustive'
        (Stage 2 result, or None if Stage 1 certified or split_depth != -1).
        """
        assert self.model is not None, "call train() or load_checkpoint() first"
        # Seed the failure-diagnostic context with the image index so that any
        # [LP-fail] printed during this call identifies the right sample.
        _LP_CONTEXT["image_idx"] = image_idx
        global _USE_EQ_CONSTRAINTS, _DEBUG_LP, _USE_AMO_CONSTRAINTS
        _USE_EQ_CONSTRAINTS = eq_constraints
        _DEBUG_LP = debug_lp
        _USE_AMO_CONSTRAINTS = amo
        k = len(indices)
        rng = np.random.default_rng(0)

        # Optional Monte Carlo sanity check: verify that LP bounds cover MC outputs.
        mc_outputs = None
        if mc_samples > 0:
            mc_outputs = monte_carlo_outputs(
                self.model, image_flat, indices, epsilon,
                self.num_steps, mc_samples, rng,
            )

        if split_depth == -1:
            # Stage 1: depth-0 LP (max_depth_cap=0 prevents depth iteration).
            symbolic = verify_symbolic_sample(
                    self.model, image_flat, label, epsilon, k, self.num_steps,
                    tight_bounds=tight_bounds, split_depth=0, max_depth_cap=0,
                    parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                    split_strategy=strategy, track_depth=track_depth,
                    pixel_indices=indices, singleton_bounds=singleton_bounds,
                )
            '''
            try:
                symbolic = verify_symbolic_sample(
                    self.model, image_flat, label, epsilon, k, self.num_steps,
                    tight_bounds=tight_bounds, split_depth=0, max_depth_cap=0,
                    parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                    split_strategy=strategy, track_depth=track_depth,
                    pixel_indices=indices, singleton_bounds=singleton_bounds,
                )
            except ZeroDivisionError as error:
                # This is when it is bounded to a single possible pixel pattern, and the LP would throw an error.
                # Catches that, and the exhaustive is exactly the same as the symbolic and very fast.
                print("\nCaught: ", error)
                symbolic = self._verify_exhaustive(image_flat, indices, epsilon, label)
            '''    
            symbolic["split_strategy"] = "depth0"  # tag to distinguish from split rows
            if mc_outputs is not None:
                symbolic["mc_coverage_pct"] = bounds_cover_outputs(symbolic, mc_outputs)
                symbolic["mc_samples"] = mc_samples

            if not symbolic["certified"]:
                # Stage 2: exhaustive fallback — enumerate all latency combinations.
                exh = self._verify_exhaustive(image_flat, indices, epsilon, label)
                if mc_outputs is not None:
                    exh["mc_coverage_pct"] = bounds_cover_outputs(exh, mc_outputs)
                    exh["mc_samples"] = mc_samples
                return {"symbolic": symbolic, "exhaustive": exh}
            return {"symbolic": symbolic, "exhaustive": None}

        else:
            # Symbolic split: no exhaustive fallback, just the LP tree.
            # When split_depth=0 is passed explicitly, cap at depth 0 so the
            # function doesn't iterate to higher depths (its else-branch in the
            # max_depth computation would otherwise allow up to k splits).
            result = verify_symbolic_sample(
                self.model, image_flat, label, epsilon, k, self.num_steps,
                tight_bounds=tight_bounds, split_depth=split_depth,
                max_depth_cap=0 if split_depth == 0 else None,
                parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                split_strategy=strategy, track_depth=track_depth,
                pixel_indices=indices, singleton_bounds=singleton_bounds,
            )
            result["split_strategy"] = strategy
            if mc_outputs is not None:
                result["mc_coverage_pct"] = bounds_cover_outputs(result, mc_outputs)
                result["mc_samples"] = mc_samples
            return {"symbolic": result, "exhaustive": None}

    def _verify_exhaustive(self, image_flat: np.ndarray, indices: np.ndarray,
                           epsilon: float, label: int) -> dict:
        """Exhaustive fallback: enumerate ALL feasible spike-timing combinations.

        For each combination of latency assignments in the Cartesian product
            product(feasible_latencies(lb[i], ub[i], T) for i in indices)
        simulate the network exactly and collect the output scores.

        Certification: label score minus max competitor score > 0 for every combo.

        This is exact (no relaxation) but exponential in k and the number of
        feasible latencies per pixel. Suitable for small k (1-4) only.
        """
        lb, ub = make_bounds(image_flat, indices, epsilon)
        base_lat = latency_from_values(torch.from_numpy(image_flat).float(), self.num_steps).numpy()

        # Feasible latency sets for each perturbed pixel.
        choices = [feasible_latencies(float(lb[i]), float(ub[i]), self.num_steps) for i in indices]

        # Total number of combinations to evaluate (product of set sizes).
        n_lambda = int(np.prod([len(c) for c in choices], dtype=np.int64)) if choices else 1

        outputs = []
        t0 = time.perf_counter()
        for combo in itertools.product(*choices):
            lat = base_lat.copy()
            lat[indices] = np.array(combo, dtype=np.int64)   # pin each pixel's latency
            spike_train = spike_train_from_latencies(lat, self.num_steps)
            score, _, _ = self.model.simulate_with_patterns(spike_train)
            outputs.append(score)
        runtime = time.perf_counter() - t0

        Y = np.stack(outputs, axis=0)   # shape (n_lambda, num_classes)
        lb_y = Y.min(axis=0)            # worst-case lower bound per class
        ub_y = Y.max(axis=0)            # worst-case upper bound per class

        # Gap: minimum margin of the true class over all competitors, worst case.
        gap = float(lb_y[label] - np.max(np.delete(ub_y, label)))
        return {
            "epsilon": float(epsilon),
            "k": int(len(indices)),
            "label": int(label),
            "mode": "full",      # distinguishes from the symbolic LP rows
            "n_lambda": n_lambda,
            "gap": gap,
            "certified": gap > 0.0,
            "bound_width": float(np.mean(ub_y - lb_y)),
            "runtime_s": runtime,
            "lb": lb_y.tolist(),
            "ub": ub_y.tolist(),
        }


# ---------------------------------------------------------------------------
# Standalone main (thin wrapper around SNNVerifier for isolated runs via comp)
# ---------------------------------------------------------------------------

MEDMNIST_DATASETS = {
    "pathmnist", "octmnist", "pneumoniamnist", "dermamnist",
    "retinamnist", "breastmnist", "bloodmnist", "tissuemnist",
    "organamnist", "organcmnist", "organsmnist",
}


def main():
    all_datasets = ["mnist", "fashion-mnist", "cifar10"] + sorted(MEDMNIST_DATASETS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/outputs/snn")
    parser.add_argument("--dataset", default="pathmnist", choices=all_datasets)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    # get_datasets lives in comp_snn_vs_ann.py (the canonical location post-refactor).
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from comp_snn_vs_ann import get_datasets

    torch.manual_seed(0)
    np.random.seed(0)

    verifier = SNNVerifier(
        hidden_sizes=args.hidden_sizes,
        num_steps=args.num_steps,
        beta=args.beta,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    train_ds, test_ds, input_size, num_classes = get_datasets(args.dataset)
    print(f"dataset={args.dataset}  input_size={input_size}  num_classes={num_classes}")

    if not args.retrain and verifier.load_checkpoint():
        print("loaded existing checkpoint")
    else:
        verifier.train(train_ds, test_ds,
                       epochs=args.epochs, lr=args.lr,
                       train_limit=args.train_limit, batch_size=args.batch_size)

    acc = verifier.test_accuracy(test_ds, args.batch_size)
    print(f"test_acc={acc:.1f}%")


if __name__ == "__main__":
    main()
