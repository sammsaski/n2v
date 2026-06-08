# snn_comparison.py — Reference

## Overview

`snn_comparison.py` trains and verifies a **First-to-Fire (F2F) Leaky Integrate-and-Fire (LIF) SNN** using a sound step-hull relaxation over input-latency variables. The public entry point is `SNNVerifier`. Everything else is internal machinery that it orchestrates.

The verifier implements the adaptive latency-splitting algorithm (Algorithm 1): it starts from a single relaxed symbolic subproblem over the full input set and refines by splitting feasible input-latency regions only when the current relaxation cannot certify the pairwise class margins.

---

## Core Concepts

### F2F Latency Encoding (E_F2F)

Continuous normalized inputs **x** ∈ [0,1]^n are converted to spike trains via a monotonically decreasing latency function. The integer firing timestep for pixel i is:

```
L_Q(x_i) = round((T-1) * (1 - x_i)) + 1,   x_i > 0
L_Q(x_i) = ∞  (silent),                      x_i = 0
```

- Bright pixel (x=1.0) → fires at t=1 (earliest)
- Mid pixel (x=0.5) → fires near t=T/2
- Zero-valued pixel → **silent** — never fires

The resulting spike train S_0 ∈ {0,1}^(T×n_x) has at most one spike per input neuron. This is a defining property of F2F latency coding: every input neuron fires at most once per inference. The implementation indexes timesteps from 0; the latency formula becomes `round((T-1)*(1-x))` in 0-based form, with silent sentinel `T`.

### Has-Fired Mask (At-Most-Once Property)

During the forward pass, a `fired` mask enforces the single-spike-per-neuron property: once a neuron fires at timestep t, its spike output is zeroed for all t' > t. This applies to all hidden and output neurons — it is not an additional constraint but a defining characteristic of F2F latency coding. The same property is encoded in the LP as the at-most-once (AMO) constraint Σ_t spk[i,t] ≤ 1.

### F2F Score Map (D_F2F) and Decoding

The F2F class score is defined by the first finite output spike time t^out_c:

```
y_c(x) = T + 1 - t^out_c(x),   if t^out_c < ∞
y_c(x) = 0,                     if t^out_c = ∞  (no spike)
```

Earlier output spikes produce larger scores; a no-spike class receives score zero. In implementation (0-indexed), this accumulates as:

```
score[c] = Σ_t (T - t) * output_spike[c, t]
```

The predicted class is `argmax_c y_c(x)`. The strict robustness condition requires `y_c* > y_c` for all c ≠ c*; exact ties violate the specification. Training minimizes `CrossEntropyLoss` applied directly to these scores.

### Spike Pattern and Effective Synapses

For a fixed F2F input spike train and fixed downstream spike pattern σ = (S_1, ..., S_L), every firing decision in the SNN is determined and the decoded output is affine in the input. The effective weight matrix at layer i and timestep t is:

```
W^eff_i[t] = W_i ⊙ s_{i-1}[t]^T
```

which selects only the columns corresponding to neurons that fired at t. Once σ is fixed, all effective weight matrices are determined and the finite-horizon SNN computation reduces to affine propagation — the piecewise-affine structure that the spiking star set framework exploits.

### Step-Hull Relaxation

For each neuron-time decision (i, t), the verifier computes LP bounds [U, Ū] on the no-reset membrane potential U (the pre-threshold quantity before the current spike and subtractive reset are applied). Three cases:

- **Ū < ϑ**: spike is fixed to 0 (neuron cannot fire)
- **U ≥ ϑ**: spike is fixed to 1 (neuron always fires)
- **U < ϑ < Ū**: ambiguous — a relaxed spike variable σ ∈ [0,1] is introduced with the step-hull constraints:

```
T[U,Ū] = { (U, σ) | 0 ≤ σ ≤ 1,  σ ≤ (U - U)/(ϑ - U),  σ ≥ (U - ϑ)/(Ū - ϑ) }
```

This is the convex hull of the closed threshold graph and gives a sound outer approximation of the discontinuous firing rule. All constraints are linear and are appended to the augmented star predicate. The resulting relaxation is sound (Theorem 1): every reachable output is contained in the relaxed output star.

### Robustness Certification

The input set I is **certified robust** for ground-truth class c* if the pairwise margin lower bound is positive for all competing classes:

```
Δ^η_c = min_{y ∈ [[S^η_rel]]} (y_c* - y_c) > 0   for all c ≠ c*
```

A positive `gap` in the result dict is equivalent to certification. The verifier returns CERTIFIED only when every latency subproblem covering I has positive margin.

---

## Key Classes and Functions

### `F2FMLP`

The F2F LIF SNN model. A feedforward MLP where every layer is a `Linear + LIF` pair. Implements the finite-horizon LIF dynamics from Eq. (2)–(3) with subtractive reset and a has-fired mask.

```python
F2FMLP(
    input_size=784,       # input dimension n_x (flattened pixels)
    hidden_sizes=[128, 64],  # hidden layer widths [N_1, ..., N_{L-1}]
    num_classes=4,        # output classes n_c
    beta=0.9,             # one-step decay factor β (shared across layers)
    threshold=1.0,        # firing threshold ϑ = V_th
    num_steps=20          # time horizon T
)
```

Key methods:
- `forward(spike_train)` — returns F2F class scores y_c, shape `(B, n_c)`. Input spike train S_0 shape: `(B, n_x, T)`.
- `simulate_with_patterns(spike_train)` — returns `(scores, hidden_spikes, output_spikes)` for a single sample. Used by the full-depth (exact) latency-cell evaluator and Monte Carlo checks. `hidden_spikes` is the spike pattern σ restricted to hidden layers; shape `(T, total_hidden_neurons)`.

---

### `SNNVerifier`

The public class implementing the adaptive F2F verification algorithm. Owns the trained model and exposes train / load / verify.

```python
verifier = SNNVerifier(
    hidden_sizes=[128, 64],
    num_steps=20,         # time horizon T
    beta=0.9,             # decay factor β
    threshold=1.0,        # firing threshold ϑ
    output_dir="experiments/outputs/snn"
)
```

#### `train(train_ds, test_ds, epochs, lr, train_limit, batch_size) → dict`

Trains an `F2FMLP` from scratch using F2F-encoded inputs and CrossEntropyLoss on the F2F score map. Saves a checkpoint to `output_dir/snn_checkpoint.pt`.

- Infers `n_x` (input dimension) and `n_c` (number of classes) from the dataset automatically.
- Optimizer: Adam with cosine annealing LR decay and gradient clipping (`max_norm=1.0`).
- Returns a `summary` dict with per-epoch `loss`, `train_acc`, and `test_acc` entries.

#### `load_checkpoint() → bool`

Loads a previously saved checkpoint. Returns `True` if found, `False` otherwise. Must be called before `verify()` if not training.

#### `verify(image_flat, indices, epsilon, label, ...) → dict`

The main verification call. Constructs a sound reachable-set enclosure R̂_N(I) for the input set I = {x : ||x - x_0||_∞ ≤ ε at perturbed pixels} and determines whether I is certified robust for ground-truth class c*.

```python
result = verifier.verify(
    image_flat,           # np.ndarray, shape (n_x,), nominal input x_0 ∈ [0,1]^n_x
    indices,              # np.ndarray of int, perturbed pixel set K ⊆ {0,...,n_x-1}
    epsilon=0.05,         # perturbation radius ε
    label=2,              # ground-truth class c*
    split_depth=-1,       # depth d for adaptive latency splitting (see below)
    strategy="choice-influence",  # split order heuristic
    parallel_workers=8,   # thread-pool workers for parallel LP solves
    parallel_backend="thread",
    eq_constraints=True,  # use equality constraints in the augmented predicate
    amo=False,            # enable AMO-aware bounds
    singleton_bounds=True,  # enable singleton-definite pinning
)
```

Returns a dict with two keys:
- `"symbolic"` — result of the step-hull relaxation (relaxed subproblem)
- `"exhaustive"` — result of exact full-depth latency-cell evaluation, or `None` if the relaxation certified or `split_depth != -1`

Both sub-dicts share the same structure (see **Verification Result Dict** below).

**`split_depth` modes (depth d in Algorithm 1):**

| Value | Behavior |
|---|---|
| `-1` | Two-stage: depth-0 relaxation → full-depth exact evaluation if relaxation fails |
| `0` | Depth-0 relaxation only (D = ∅, single subproblem over full input star) |
| `N > 0` | Adaptive latency splitting to depth N; no full-depth fallback |

#### `test_accuracy(test_ds, batch_size) → float`

Evaluates classification accuracy on a dataset using F2F encoding and argmax decoding. Returns percentage.

#### `predict(image) → int`

Returns the predicted class (argmax of F2F scores) for a single image tensor.

#### `scores(image) → np.ndarray`

Returns raw F2F class scores y_c for a single image, shape `(n_c,)`.

---

## Verification Result Dict

Both `"symbolic"` and `"exhaustive"` sub-dicts contain:

| Key | Type | Description |
|---|---|---|
| `"certified"` | `bool` | True if I is certified robust for c* |
| `"gap"` | `float` | `min_{c≠c*} Δ^η_c` — minimum pairwise margin lower bound. Positive = CERTIFIED. |
| `"lb"` | `list[float]` | Sound lower bound on each class score y_c over I: `[y_c]` in R̂_N(I) |
| `"ub"` | `list[float]` | Sound upper bound on each class score y_c over I: `[ȳ_c]` in R̂_N(I) |
| `"gap_joint"` | `list[float]` | Per-competitor margin lower bound: `Δ^η_c = min(y_c* - y_c)` for each c ≠ c* |
| `"epsilon"` | `float` | Perturbation radius ε used |
| `"k"` | `int` | Number of perturbed pixels \|K\| |
| `"label"` | `int` | Ground-truth class c* |
| `"runtime_s"` | `float` | Wall time in seconds |
| `"n_lp_variables"` | `int` | Variables in the augmented predicate (symbolic only) |
| `"n_lp_constraints"` | `int` | Constraints in the augmented predicate (symbolic only) |
| `"n_lambda"` | `int` | Number of latency cells enumerated (exhaustive/full-depth only) |

The symbolic result also carries `"split_strategy"` (the split-order heuristic tag used).

---

## Encoding Utilities

### `latency_from_values(values, num_steps) → Tensor`

Applies the F2F encoder E_F2F to a flat pixel tensor in [0,1]. Returns integer firing timesteps (0-indexed). Silent pixels (value=0) receive the sentinel T (they never fire in the T-step horizon).

### `encode_batch(images, num_steps) → Tensor`

Converts a batch of images `(B, C, H, W)` to F2F spike trains S_0 of shape `(B, n_x, T)`. Used before every forward pass during training and inference.

### `spike_train_from_latencies(latencies, num_steps) → Tensor`

Builds a single-sample spike train `(n_x, T)` from a precomputed integer latency vector λ ∈ {1,...,T,∞}^n_x. Used by the full-depth latency-cell evaluator and Monte Carlo sanity checks to simulate the model for one specific input spike train S^(λ)_0.

### `make_bounds(image_flat, indices, epsilon) → (lb, ub)`

Constructs the L∞ perturbation ball: perturbed pixels get `[x_0 ± ε]` clipped to [0,1]; unperturbed pixels are pinned to their nominal value. This defines the input set I used by the verifier.

### `feasible_latencies(lb, ub, num_steps) → list[int]`

Returns the feasible latency set Λ_i(I) for a single pixel with value range [lb, ub] — the integer timesteps whose latency bin overlaps the perturbation interval. Includes the silent sentinel T if lb ≤ 0 (the pixel may become zero after perturbation).

### `effective_pixel_bounds(image_flat, indices, epsilon, num_steps) → (eff_lb, eff_ub)`

Returns the effective pixel-value bounds that the LP actually certifies over, accounting for latency quantization bin edges. Because each integer latency t corresponds to a bin of width 1/(T-1) in pixel-value space, the certified input region is at least as wide as [x_0 ± ε] on both sides. Used to give ANN verifiers a comparable input region for fair comparison.

---

## LP Internals (lower-level)

These are called internally by `verify()` but can be called directly if needed.

### `build_symbolic_relaxation_lp(model, image_flat, epsilon, k, num_steps, ...) → dict`

Builds and solves the relaxed subproblem for a single latency subproblem I_η (depth-0 when called with no fixed latencies, or a partially fixed branch). Propagates the input star through the step-hull relaxation layer by layer and timestep by timestep, introducing one relaxed spike variable σ ∈ [0,1] per ambiguous neuron-time decision. Computes:

1. **No-reset potential bounds** [U_j,t, Ū_j,t] via LP for each ambiguous neuron-time pair — two LP solves per decision.
2. **Score bounds** — sound enclosure R̂_N(I_η): `lb_y[c]` and `ub_y[c]` via LP over the augmented predicate. Two LP solves per class.
3. **Pairwise margin lower bounds** Δ^η_c — one LP per competing class c ≠ c* minimizing `y_c* - y_c` over the augmented predicate.

With `parallel_workers > 1` all score-bound and margin LPs are dispatched simultaneously to a `ThreadPoolExecutor` (scipy linprog releases the GIL).

### `build_symbolic_relaxation_lp_split(model, image_flat, epsilon, k, num_steps, split_depth, ...) → dict`

Implements latency splitting to depth d: branches on the first `split_depth` pixels (ordered by `strategy`), creating one latency subproblem I_η per feasible latency assignment of those pixels. Calls `build_symbolic_relaxation_lp` for each nonempty branch and takes the element-wise worst case over all branches (min of lower bounds, max of upper bounds). The final gap is the minimum margin across all branches.

With `parallel_workers > 1` and `parallel_backend="thread"`, branches are dispatched to a `ThreadPoolExecutor`.

### `verify_symbolic_sample(model, image_flat, label, epsilon, k, num_steps, ...) → dict`

Implements the adaptive loop of Algorithm 1: iterates `build_symbolic_relaxation_lp_split` from depth 0 upward, stopping as soon as the margin lower bound is positive (CERTIFIED). With `max_depth_cap=0`, only depth-0 is attempted (pure symbolic relaxation, no splitting).

### `order_split_indices(model, indices, lb_x, ub_x, num_steps, strategy) → np.ndarray`

Determines the split order over perturbed pixels K — which pixels are branched on first in Algorithm 1's deterministic pixel ordering. The choice-influence heuristic (recommended) prioritizes pixels that both introduce multiple feasible latencies and have large first-layer weight impact, skipping singleton-feasible pixels that need no splitting.

| Strategy | Description |
|---|---|
| `selected` | Original index order (no reordering) |
| `influence` | `Σ_j |W1[j, i]|` — large first-layer weight impact first |
| `choice` | `|Λ_i(I)|` — most feasible latency values first |
| `choice-influence` | `influence × max(|Λ_i(I)| − 1, 0)` — skips singleton-feasible pixels |
| `random` | Fixed-seed shuffle (for ablations) |

---

## Practical Tightening Flags

Two F2F-specific refinements tighten each relaxed subproblem without changing soundness.

### AMO-Aware Bounds (`--amo` / `amo=True`)

Under F2F encoding, each input neuron fires at most once, so the interval bound on no-reset potential U_j[t] uses the maximum per-pixel contribution over **feasible latencies** rather than their sum. Controlled by `_USE_AMO_CONSTRAINTS`. Adds explicit Σ_t spk[i,t] ≤ 1 constraints to the augmented predicate.

### Singleton-Definite Pinning (`--singleton-bounds` / `singleton_bounds=True`)

If the earliest and latest feasible first-spike times of a neuron j (inferred from membrane bounds) coincide at t*, then every feasible input makes neuron j fire at the same timestep. The relaxation adds the equality σ_j[t*] = 1, removing infeasible relaxed executions (such as the all-silent assignment) without excluding any true execution.

---

## Global Flags

Three module-level globals control LP behavior. They are set by `SNNVerifier.verify()` before each call and are read-only during LP solve time, making parallel thread-pool solves safe.

| Name | CLI flag | Default | Effect |
|---|---|---|---|
| `_USE_EQ_CONSTRAINTS` | `--equality-constraints` | `True` | Latency-cell constraints added as equalities vs. paired inequalities in the augmented predicate |
| `_USE_AMO_CONSTRAINTS` | `--amo` | `False` | Enables AMO-aware bounds (see above) |
| `_DEBUG_LP` | `--debug-lp` | `False` | Prints LP context (image, pixels, dimensions) before every solve |

---

## Checkpoint Format

Saved to `output_dir/snn_checkpoint.pt` as a `torch.save` dict:

```python
{
    "model_state_dict": ...,      # F2FMLP weights {W_i, b_i} for all layers
    "train_summary": {...},       # per-epoch loss/accuracy
    "config": {
        "input_size": int,        # n_x
        "num_classes": int,       # n_c
    }
}
```

`load_checkpoint()` also accepts the legacy format (no `"config"` key; infers n_x and n_c from the state dict shape).
