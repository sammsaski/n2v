# Spiking Neural Network Verification

## Overview

n2v supports formal reachability analysis for First-to-Fire (F2F) spiking neural networks (SNNs) via the `SpikingNeuralNetwork` class. The interface mirrors `NeuralNetwork` as closely as possible, but the underlying computation is fundamentally different: instead of propagating sets through layer operations, the SNN verifier builds and solves a linear programming (LP) relaxation of the SNN's spike-timing dynamics.

This guide covers:

- The new files and what each one does
- Training an SNN and saving it for verification
- Loading a trained SNN and calling `reach()`
- Configuring verification via `SNNReachConfig`
- Key differences between `SpikingNeuralNetwork` and `NeuralNetwork`

---

## Background: The F2F SNN Model

The supported model class is `F2FMLP` — a multi-layer perceptron using **First-to-Fire (F2F) latency coding**. Each input value is converted to a spike time: a smaller value fires earlier. Neurons use Leaky Integrate-and-Fire (LIF) dynamics with an **at-most-once** constraint (each neuron fires at most once per input). The class score for class `c` is the weighted sum of output spike times, where earlier spikes contribute more.

This model structure enables LP-based verification because the at-most-once constraint makes the firing pattern piecewise-linear over the input space.

---

## New Files

### `n2v/snn/model.py`

Contains `F2FMLP`, the snntorch-based SNN architecture. You do not typically instantiate this directly — `SNNVerifier.train()` creates and trains it, and `torch.load("snn_model.pt")` restores it.

### `n2v/snn/encoding.py`

Three functions for converting real-valued inputs into spike trains:

- `latency_from_values(x, num_steps)` — maps values in `[0, 1]` to integer spike times in `{1, ..., num_steps}`. Lower value → later spike (higher latency).
- `encode_batch(x_batch, num_steps)` — converts a `(B, D)` tensor of values into a `(T, B, D)` binary spike train tensor.
- `spike_train_from_latencies(latencies, num_steps)` — converts a latency array to a binary spike train.

These are used internally by `SpikingNeuralNetwork.forward()` and by `SNNVerifier`.

### `n2v/snn/lp.py`

The LP engine. The two functions you care about are:

- `build_symbolic_relaxation_lp(model, ...)` — constructs and solves a single LP relaxation over the input set (the "approx" / depth-0 method). Returns a dict with `"lb"` and `"ub"` arrays giving per-class output score bounds.
- `build_symbolic_relaxation_lp_split(model, ...)` — branch-and-bound enumeration of all feasible latency combinations for a set of input dimensions (the "exact" method).

You do not call these directly when using `SpikingNeuralNetwork` — `reach()` calls them for you.

### `n2v/snn/verifier.py`

Contains `SNNVerifier`, which handles training and standalone verification workflows. It is the research-facing interface. `SpikingNeuralNetwork` is the integration-facing interface built on top of the LP engine.

`SNNVerifier` also contains `monte_carlo_outputs` and `bounds_cover_outputs` for sampling-based sanity checks.

### `n2v/nn/spiking_neural_network.py`

The main user-facing file. Defines:

- `SNNReachConfig` — frozen dataclass for configuring `reach()`.
- `SpikingNeuralNetwork` — wraps an `F2FMLP` model, provides `forward()` and `reach()`.

---

## Quick Start

```python
import torch
import n2v
from n2v import SpikingNeuralNetwork, SNNReachConfig, Box, Star

# Load a trained SNN (saved by SNNVerifier.train())
model = torch.load("path/to/snn_model.pt")
snn = SpikingNeuralNetwork(model)

# Define an input set: a Box with per-dimension bounds
import numpy as np
lb = np.full((28 * 28, 1), 0.0)  # lower bounds, shape (dim, 1)
ub = np.full((28 * 28, 1), 1.0)  # upper bounds, shape (dim, 1)
input_box = Box(lb, ub)

# Compute output score bounds (over-approximation)
output_sets = snn.reach(input_box, method='approx')
out_box = output_sets[0]  # always a single Box
print("Output score lower bounds:", out_box.lb.flatten())
print("Output score upper bounds:", out_box.ub.flatten())
```

---

## Training an SNN

`SNNVerifier` trains an F2FMLP and saves it to disk. The saved file is what `SpikingNeuralNetwork` loads.

```python
from n2v.snn import SNNVerifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST("data/", train=True,  download=True, transform=transforms.ToTensor())
test_dataset  = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64)

verifier = SNNVerifier(
    output_dir="results/mnist_snn",
    num_steps=16,         # number of simulation timesteps (T)
    hidden_sizes=[256],   # hidden layer sizes
    num_epochs=5,
    lr=1e-3,
)

verifier.train(train_loader, test_loader)
# Saves to results/mnist_snn/:
#   snn_model.pt        <- full model object for SpikingNeuralNetwork
#   snn_checkpoint.pt   <- state dict + optimizer state
#   training_log.json
```

After training:

```python
model = torch.load("results/mnist_snn/snn_model.pt")
snn = SpikingNeuralNetwork(model)
```

The `snn_model.pt` file stores the **full model object** (not just weights). This is a `torch.save(model, ...)` save, not a state-dict save. Loading it requires the same Python environment (same snntorch version, same `F2FMLP` class available).

---

## Loading a Trained Model

```python
import torch
from n2v import SpikingNeuralNetwork

model = torch.load("snn_model.pt")

# Optional: validate that the model accepts the expected input size
snn = SpikingNeuralNetwork(model, input_size=784)  # raises ValueError on mismatch

# Or just let the wrapper infer it from model.fcs[0].in_features
snn = SpikingNeuralNetwork(model)

print(snn)
# SpikingNeuralNetwork(input_size=784, output_size=10, hidden_sizes=[256], num_steps=16)
```

The `input_size` argument is optional and only used to validate. The model's architecture determines the actual input/output sizes.

---

## Performing Reachability Analysis

### `reach()` signature

```python
snn.reach(input_set, method='approx', config=None, **kwargs)
```

- `input_set` — a `Star` or `Box`. No other set types are supported.
- `method` — `'approx'` (default) or `'exact'`. See below.
- `config` — an `SNNReachConfig` instance. Overrides `method` and all `**kwargs`.
- `**kwargs` — fields of `SNNReachConfig` passed directly. Cannot be combined with `config=`.

Returns `List[Box]` — always a single-element list containing one `Box` with per-class output score bounds, shape `(num_classes, 1)`.

### `method='approx'` — LP Relaxation

Constructs a single LP that relaxes the step (Heaviside) threshold function for each neuron at each timestep using a triangle relaxation. Solved once over the full input set. Fast, but the output box may be larger than the true reachable set.

**When to use**: Default choice. Use whenever you want a fast sound over-approximation, or as a preliminary check before running `'exact'`.

```python
output = snn.reach(input_star, method='approx')
```

### `method='exact'` — Latency Enumeration

Enumerates all feasible spike-time combinations for the symbolic input dimensions via branch-and-bound, solving one LP per combination. The union of all LP results is the exact reachable output set (up to the triangle relaxation on hidden neurons).

**When to use**: When you need the tightest possible bounds and are willing to pay exponential time in the number of symbolic input dimensions. Practical only for small input sets (few symbolic dimensions).

```python
output = snn.reach(input_star, method='exact')
```

For a `Box` or `Star` with `k` symbolic dimensions, `'exact'` solves up to `O(T^k)` LPs, where `T` is `num_steps`. Start with `'approx'` to check feasibility before running `'exact'`.

---

## Configuring Verification with `SNNReachConfig`

`SNNReachConfig` is a frozen dataclass that bundles all verification parameters.

```python
from n2v import SNNReachConfig

cfg = SNNReachConfig(
    method='approx',          # 'approx' or 'exact'
    parallel_workers=4,       # threads for LP solving (0 = use global config)
    tight_bounds=False,       # per-neuron LP bounds (slower, tighter)
    singleton_bounds=False,   # equality constraints for singleton timesteps
    split_strategy='choice-influence',  # how to order dimensions in 'exact'
    label=None,               # true class for certification gap computation
)

output = snn.reach(input_set, config=cfg)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `'approx'` | `'approx'` (single LP) or `'exact'` (full enumeration) |
| `parallel_workers` | `int` | `0` | Worker threads for LP solving. `0` defers to the global n2v config |
| `tight_bounds` | `bool` | `False` | Run per-neuron LP to compute tight firing-time bounds (slower but tighter) |
| `singleton_bounds` | `bool` | `False` | Add equality constraints where only one firing time is feasible |
| `split_strategy` | `str` | `'choice-influence'` | Dimension ordering for `'exact'`: `'selected'`, `'influence'`, `'choice'`, `'choice-influence'`, `'random'` |
| `label` | `int \| None` | `None` | If set, the LP also computes a certification gap `score[label] - max(score[others])` |

### Passing config vs kwargs

These two calls are equivalent:

```python
# Style 1: config object
cfg = SNNReachConfig(method='exact', parallel_workers=8, tight_bounds=True)
output = snn.reach(input_set, config=cfg)

# Style 2: bare kwargs
output = snn.reach(input_set, method='exact', parallel_workers=8, tight_bounds=True)
```

Passing both `config=` and extra `**kwargs` raises `ValueError`. When `config=` is given, its `method` field takes effect regardless of the `method` positional argument.

### Parallel workers

`parallel_workers=0` (the default) defers to the global n2v configuration:

```python
import n2v

# Use 8 threads automatically for problems with >= 10 symbolic dims
n2v.set_parallel('auto', n_workers=8, threshold=10)

# Or force parallel on for all SNN verification calls
n2v.set_parallel(True, n_workers=8)
```

Setting `parallel_workers` explicitly in the config overrides the global setting for that call only.

---

## Input Set Types

`SpikingNeuralNetwork.reach()` accepts `Star` and `Box` only.

### Box (Axis-Aligned Bounds)

```python
import numpy as np
from n2v import Box

lb = np.array([[0.0], [0.0], [0.3]])  # shape (dim, 1)
ub = np.array([[1.0], [0.5], [0.7]])  # shape (dim, 1)
box = Box(lb, ub)
output = snn.reach(box, method='approx')
```

For a Box, `reach()` reads `lb` and `ub` directly (no LP needed to extract bounds). All dimensions where `lb[i] < ub[i]` are treated as symbolic.

### Star (Polytope)

```python
from n2v import Star

star = Star.from_bounds(lb, ub)        # or a general Star with C, d
output = snn.reach(star, method='approx')
```

For a Star, `reach()` first solves `2 * dim` LPs (via `get_ranges()`) to compute the axis-aligned bounding box of the Star. The SNN LP then operates over these bounds. This is a sound over-approximation: the true reachable output set is contained in the returned Box, but the Star's non-axis-aligned geometry is not exploited.

**Note**: The SNN LP engine operates over axis-aligned boxes internally. A Star input is therefore over-approximated by its bounding box before being passed to the LP. If your Star is tightly bounded (low eccentricity), this works well. If your Star is highly elongated with tight diagonal constraints, the output Box may be conservative.

### Unsupported types

`Zono`, `Hexatope`, `Octatope`, `ImageStar`, and `ImageZono` are not supported — `reach()` raises `TypeError`. SNNs always operate on flat 1D inputs.

---

## Understanding the Output

`reach()` always returns `List[Box]` — a single-element list:

```python
output_sets = snn.reach(input_set)
assert len(output_sets) == 1
out_box = output_sets[0]

# out_box.lb: shape (num_classes, 1), lower bound on each class score
# out_box.ub: shape (num_classes, 1), upper bound on each class score
lb_scores = out_box.lb.flatten()  # shape (num_classes,)
ub_scores = out_box.ub.flatten()  # shape (num_classes,)
```

The bounds are over **output class scores** (weighted spike-time sums), not logits or probabilities.

### Using the output for verification

To check whether a given class `c` is always the top scorer:

```python
label = 3  # class we want to verify dominates

output = snn.reach(input_set, label=label)  # pass label= to get cert gap
# Alternatively, analyze the bounds manually:

out_box = snn.reach(input_set)[0]
lb_scores = out_box.lb.flatten()
ub_scores = out_box.ub.flatten()

# Class c is provably dominant if its lower bound exceeds all other upper bounds
other_classes = [i for i in range(len(lb_scores)) if i != label]
dominant = all(lb_scores[label] >= ub_scores[j] for j in other_classes)
print("Verified dominant:", dominant)
```

---

## Differences from `NeuralNetwork`

### At a Glance

| Feature | `NeuralNetwork` | `SpikingNeuralNetwork` |
|---|---|---|
| **Model type** | Any `nn.Module` traceable by `torch.fx` | `F2FMLP` (snntorch, F2F latency coding) |
| **`input_size` type** | `Optional[tuple]` | `Optional[int]` |
| **`layers` property** | Yes (lazy, via `torch.fx`) | No (F2FMLP is not fx-traceable) |
| **Supported input sets** | `Star`, `Zono`, `Box`, `Hexatope`, `Octatope`, `ImageStar`, `ImageZono` | `Star`, `Box` only |
| **`reach()` return type** | `List[<same type as input>]` | `List[Box]` always |
| **Number of output sets** | One per input set (can grow via splitting) | Always exactly one `Box` |
| **Default method** | `'exact'` | `'approx'` |
| **How reachability works** | Set propagation layer by layer | LP relaxation of spike-timing dynamics |
| **Training** | External (any PyTorch training loop) | `SNNVerifier.train()` |
| **Config conflict handling** | Raises if `config.method != method` | Config takes priority; no error |

### Input/Output Shapes

`NeuralNetwork.input_size` is a `tuple` representing the multi-dimensional input shape (e.g., `(1, 28, 28)` for an image). `SpikingNeuralNetwork.input_size` is a scalar `int` representing the flat 1D input dimension (e.g., `784`). SNNs always operate on flat vectors.

```python
# NeuralNetwork
nn_net = NeuralNetwork(model, input_size=(1, 28, 28))

# SpikingNeuralNetwork
snn_net = SpikingNeuralNetwork(model, input_size=784)
```

### Return Type

`NeuralNetwork.reach()` returns a list of the same set type as the input — if you pass a `Star`, you get back `List[Star]`. If you split (exact method), you may get multiple output Stars. `SpikingNeuralNetwork.reach()` always returns `List[Box]` with exactly one element, regardless of input type or method.

```python
# NeuralNetwork: output type matches input type
stars_out = nn_net.reach(star_in, method='exact')   # List[Star]
boxes_out = nn_net.reach(box_in,  method='approx')  # List[Box]

# SpikingNeuralNetwork: always a single Box
result = snn_net.reach(star_in,  method='approx')   # List[Box], len == 1
result = snn_net.reach(box_in,   method='exact')    # List[Box], len == 1
```

### How Reachability Works

`NeuralNetwork.reach()` propagates the input set forward through each layer of the network. For `'exact'`, it uses Star splitting at ReLU boundaries. For `'approx'`, it relaxes ReLU constraints.

`SpikingNeuralNetwork.reach()` does not do layer-by-layer propagation. Instead, it builds a single large LP that encodes all spike-timing constraints across all layers and all timesteps simultaneously. The LP variables represent fractional spike probabilities, and the LP objective gives tight bounds on the output class scores over the input set.

This means:
- SNN verification is a **single monolithic LP** (or a set of LPs for `'exact'`), not a sequential process.
- There is no concept of intermediate set representations between layers.
- The `layers` property is not available.

### Default Method

`NeuralNetwork.reach()` defaults to `method='exact'`. `SpikingNeuralNetwork.reach()` defaults to `method='approx'`. This reflects the cost structure: SNN `'exact'` involves exponential enumeration, while NN `'exact'` is standard Star splitting. **Always start with `'approx'` for SNN verification** and only switch to `'exact'` when you need tight bounds on a small input set.

### Config Conflict Behavior

In `NeuralNetwork`, if you pass `config=ReachConfig(method='approx')` while `method='exact'` is the default, a `TypeError` is raised because the two disagree. You must explicitly match them:

```python
# NeuralNetwork: must match method= and config.method
nn_net.reach(input_set, method='approx', config=ReachConfig(method='approx'))  # OK
nn_net.reach(input_set, config=ReachConfig(method='approx'))                   # TypeError!
```

In `SpikingNeuralNetwork`, the `config=` object is the authority. You never need to repeat the method:

```python
# SpikingNeuralNetwork: config wins
cfg = SNNReachConfig(method='exact')
snn_net.reach(input_set, config=cfg)          # correct, method='exact' from config
snn_net.reach(input_set, method='exact')      # also fine, no config
```

### No `layers` Property

`NeuralNetwork` extracts individual layers via `torch.fx.symbolic_trace` and exposes them via `net.layers`. `F2FMLP` cannot be traced by `torch.fx` because its forward pass contains data-dependent control flow (the at-most-once spike masking loop). `SpikingNeuralNetwork` does not have a `layers` property at all. Accessing it raises `AttributeError`.

### Training

`NeuralNetwork` wraps any pre-trained PyTorch model — training is entirely external. `SpikingNeuralNetwork` also wraps a pre-trained model, but training an F2FMLP correctly requires the snntorch-specific training loop in `SNNVerifier`. The two-phase workflow is:

```
Phase 1 (training):    SNNVerifier.train(train_loader, test_loader)
                       → saves snn_model.pt

Phase 2 (verification): model = torch.load("snn_model.pt")
                        snn = SpikingNeuralNetwork(model)
                        output = snn.reach(input_set)
```

---

## Global Configuration

`SpikingNeuralNetwork.reach()` respects the global n2v config for two independent operations:

1. **Bounding box computation** (Star inputs only) — uses `global_config.lp_solver`, `global_config.parallel_lp`, and `global_config.n_workers` to solve the `2 * dim` LPs that extract the Star's bounding box.
2. **LP enumeration parallelism** — uses `global_config.should_use_parallel()` and `global_config.get_n_workers()` to determine `parallel_workers` when `SNNReachConfig.parallel_workers == 0`.

```python
import n2v

# Parallel LP solving for high-dimensional inputs
n2v.set_parallel('auto', n_workers=8, threshold=10)

# Use fast LP solver for bounding box computation
n2v.set_lp_solver('linprog')

# Now SpikingNeuralNetwork picks up both settings automatically
output = snn.reach(star_input, method='approx')
```

---

## Common Patterns

### Checking robustness for a single input

```python
import torch
import numpy as np
from n2v import SpikingNeuralNetwork, Box

model = torch.load("snn_model.pt")
snn = SpikingNeuralNetwork(model)

# Build a perturbation box around a test image
x = test_image.flatten().numpy()  # shape (D,)
epsilon = 0.05
lb = np.clip(x - epsilon, 0.0, 1.0).reshape(-1, 1)
ub = np.clip(x + epsilon, 0.0, 1.0).reshape(-1, 1)
input_box = Box(lb, ub)

out_box = snn.reach(input_box, method='approx')[0]
lb_scores = out_box.lb.flatten()
ub_scores = out_box.ub.flatten()

true_label = 3
other_labels = [i for i in range(10) if i != true_label]
certified = all(lb_scores[true_label] >= ub_scores[j] for j in other_labels)
print("Certified robust:", certified)
```

### Tightening bounds with `tight_bounds=True`

```python
from n2v import SNNReachConfig

# Default: faster, looser
output_fast = snn.reach(input_set)

# Tight: slower, tighter
cfg = SNNReachConfig(method='approx', tight_bounds=True, singleton_bounds=True)
output_tight = snn.reach(input_set, config=cfg)
```

### Exact verification on a small perturbation set

```python
# Only practical for small k (few symbolic dimensions)
x = test_image.flatten().numpy()
epsilon = 0.01  # very small perturbation
lb = np.clip(x - epsilon, 0.0, 1.0).reshape(-1, 1)
ub = np.clip(x + epsilon, 0.0, 1.0).reshape(-1, 1)

# Perturbed dims: may be only a handful at this epsilon
input_box = Box(lb, ub)
output = snn.reach(input_box, method='exact')
```

### Star input (general polytope)

```python
from n2v import Star
import numpy as np

# Star defined by: x = c + V @ alpha, subject to A_pred @ alpha <= b_pred
c = np.zeros((784, 1))
V = 0.05 * np.eye(784)           # diagonal perturbation generators
star = Star(np.hstack([c, V]))    # or use Star.from_bounds(lb, ub)

output = snn.reach(star, method='approx')
```

The Star's axis-aligned bounding box is computed by solving `2 * 784` LPs. The SNN LP then operates over those bounds. For tight Stars (where the bounding box is close to the Star itself), results are similar to passing the equivalent Box directly.
