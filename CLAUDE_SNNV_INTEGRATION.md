# SNN Verification Integration into n2v
## Implementation Guide for Claude

---

## Part 1 — What We Are Building and Why (Read This First)

### The Big Picture

`n2v` is a formal verification library for neural networks. Its core idea is: given a **set** of possible inputs (a Star, Box, etc.), compute a **set** of guaranteed outputs. If every possible input classifies correctly, you have a robustness certificate.

Currently n2v verifies **standard ANNs** (MLPs, CNNs). This integration adds support for **spiking neural networks (SNNs)** — a different computational model where neurons communicate via discrete events (spikes) rather than continuous activations. The verification problem is fundamentally different, but the *interface* the user sees should be identical.

---

### What Is a Spiking Neural Network?

In a standard ANN, a neuron computes `y = ReLU(Wx + b)` — a real-valued output at each layer.

In the **First-to-Fire (F2F)** SNN used here:
- Time is discretized into T timesteps (e.g. T = 20).
- Each neuron fires **at most once** during the T timesteps (at-most-once constraint).
- Brighter/larger input values fire **earlier** (timestep 0 = earliest); darker/smaller values fire **later** (timestep T-1); zero values are **silent** (never fire).
- This encoding is called **latency coding**: information is carried in the *timing* of spikes, not their amplitude.

The network model used is `F2FMLP`: a multi-layer perceptron where each layer is a `nn.Linear` followed by a snntorch `Leaky` LIF neuron. `F2FMLP` lives in `external_snnv/snn_comparison.py` and uses snntorch for the LIF dynamics.

A **classification score** for class `c` is `score_c = Σ_t (T - t) × output_spike(c, t)` — earlier spikes for class `c` mean higher scores.

---

### Why Is SNN Verification Different?

For a standard ANN, n2v propagates a **star set** `⟨c, V, P⟩` through each layer. ReLU introduces case splits (neuron on/off), handled by LP or exact enumeration. The output is another star set.

For an SNN:
- The output depends on **when** each input fires, not its raw value.
- The firing time of a neuron depends on whether it accumulated enough input spikes from the previous layer — a **threshold crossing** event.
- This threshold crossing is a **Heaviside step function** (discontinuous), not a ReLU.
- The full input–output map involves a combinatorial number of possible spike timing patterns.

The **core insight** from the ATVA paper ("Robustness Verification of Spiking Neural Networks") is to relax this combinatorial problem into a **linear program**:

- For each input dimension `i` and each feasible timestep `t`, introduce a **fractional spike variable** `p[(i, t)] ∈ [0, 1]` representing the "probability" that dimension `i` fires at time `t`.
- Add **at-most-once** constraints: `Σ_t p[(i, t)] ≤ 1` per input.
- For each hidden neuron, apply a **step-hull (triangle) relaxation** — the tightest convex outer approximation of the threshold-crossing function.
- The output class scores become linear functions of the spike variables.
- Minimize/maximize each class score over this LP → guaranteed per-class score bounds.

This LP is called the **depth-0 relaxation** (Algorithm 1, d=0 in the paper). It is an **over-approximation**: the true reachable score set is a subset of the LP-derived Box.

For a **tighter (exact) result**, the paper describes **adaptive latency splitting**: fix one dimension's spike time to each feasible value, solve a sub-LP for each branch, and take the union. At full depth (d=k, where k = number of perturbed dimensions), every dimension is fixed and each sub-problem is an **exact simulation** — no LP needed. This gives the exact reachable output set.

---

### How the Fractional Spike Variables Connect to n2v's Star Set

n2v's `Star` set `⟨c, V, P⟩` represents a convex set of inputs: `x = c + Vα` subject to `Cα ≤ d`.

The SNN LP's fractional spike variables `p[(i, t)]` represent a convex relaxation of the **latency-timing space**: for input dimension `i`, which timestep is feasible given the value range `[lb_i, ub_i]`?

The connection is:
1. A `Star` or `Box` defines per-dimension value ranges `lb_i, ub_i` (via `get_ranges()` or directly from `.lb/.ub`).
2. For each dimension, `feasible_latencies(lb_i, ub_i, T)` returns the set of timesteps consistent with that value range.
3. The LP introduces `p[(i, t)]` variables **only for those feasible timesteps**.
4. This is the **spiking star set** from the paper: the LP's fractional spike variables are the continuous relaxation of the spike timing pattern set induced by the input perturbation set.

**Key takeaway**: The SNN LP already operates over a general convex input set, expressed through per-dimension bounds. It is not inherently tied to images or ε-perturbations — those were just the original application. The only thing needed is per-dimension `(lb, ub)`, which a `Star` or `Box` provides directly.

---

### How `SpikingNeuralNetwork` Fits Into n2v

```
n2v/
├── NeuralNetwork          ← wraps nn.Module, reach() → List[Star/Box/Zono]
└── SpikingNeuralNetwork   ← wraps snntorch nn.Module, reach() → List[Box]

Both expose:
  __init__(model: nn.Module, input_size=None)
  forward(x: Tensor) → Tensor
  reach(input_set: Star | Box, method='approx'|'exact', **kwargs) → List[Box]
  __repr__()
```

`SpikingNeuralNetwork.reach()` returns `List[Box]` (not `List[Star]`) because the SNN verification LP produces **axis-aligned output bounds** — it gives a lower and upper bound per output class, which is exactly a `Box`. There is no basis for constructing a tighter set type here.

The `layers` property that `NeuralNetwork` exposes (via `torch.fx.symbolic_trace`) is **not available** for `F2FMLP`: the SNN's time loop and data-dependent at-most-once masking make it untraceable. This is documented analogously to how `NeuralNetwork` documents untraceable models.

---

## Part 2 — Current Codebase Structure

### `external_snnv/snn_comparison.py` (the only file, ~2400 lines)

This monolithic file contains everything. Its logical sections, with approximate line ranges:

| Lines | Content |
|-------|---------|
| 1–37 | Imports |
| 38–164 | `F2FMLP` — the SNN model class (snntorch-based) |
| 165–228 | Latency encoding: `latency_from_values`, `encode_batch`, `spike_train_from_latencies` |
| 229–328 | LP input helpers: `make_bounds`, `feasible_latencies` |
| 329–343 | Global LP state globals: `_LP_CONTEXT`, `_USE_EQ_CONSTRAINTS`, `_DEBUG_LP`, `_USE_AMO_CONSTRAINTS` |
| 344–414 | `order_split_indices` — pixel ordering for split strategy |
| 415–541 | Monte Carlo helpers: `monte_carlo_outputs`, `bounds_cover_outputs` |
| 542–642 | `_HighsPreparedLP` — HiGHS model caching for batch LP solves |
| 643–848 | LP worker functions: `_init_symbolic_split_worker`, `_solve_symbolic_split_worker` |
| 849–1575 | `build_symbolic_relaxation_lp` — the core LP construction (~726 lines) |
| 1576–1747 | `build_symbolic_relaxation_lp_split` — branch-and-bound LP tree |
| 1748–1769 | `_gap_from_result` — extract certification gap from LP result dict |
| 1770–1865 | `verify_symbolic_sample` — Algorithm 1: iterates depths until certified |
| 1866–1994 | `summarize`, `summarize_depth0_exhaustive` — result aggregation helpers |
| 1995–2036 | Row caching helpers: `_row_cache_key`, `load_existing_rows` |
| 2037–2348 | `SNNVerifier` — high-level train/verify class |
| 2349+ | `__main__` entry point |

### n2v (existing, do NOT modify internals)

The files you must NOT change (only add imports to `__init__.py` files):
- `n2v/sets/star.py` — `Star` class with `get_ranges()`, `estimate_ranges()`
- `n2v/sets/box.py` — `Box` class with `.lb`, `.ub` (shape `(n, 1)`)
- `n2v/nn/neural_network.py` — `NeuralNetwork` (read as template)
- `n2v/nn/reach.py` — `ReachConfig`, `reach_pytorch_model`
- All other existing `n2v/` files

---

## Part 3 — Target File Structure

After the integration, the repository looks like this:

```
n2v/
├── __init__.py                        ← MODIFY: add SpikingNeuralNetwork, SNNVerifier
├── sets/                              ← unchanged
├── nn/
│   ├── __init__.py                    ← MODIFY: add SpikingNeuralNetwork, SNNReachConfig
│   ├── neural_network.py              ← unchanged (template to mirror)
│   ├── reach.py                       ← unchanged
│   └── spiking_neural_network.py      ← NEW: SpikingNeuralNetwork + SNNReachConfig
└── snn/                               ← NEW subpackage
    ├── __init__.py                    ← NEW
    ├── model.py                       ← NEW: F2FMLP (from snn_comparison.py)
    ├── encoding.py                    ← NEW: latency encoding utils
    ├── lp.py                          ← NEW: LP engine (MODIFIED: input_bounds param)
    └── verifier.py                    ← NEW: SNNVerifier + summary helpers

external_snnv/
└── snn_comparison.py                  ← keep as-is (or update to import from n2v.snn)
```

---

## Part 4 — Implementation Steps

Work through these steps in order. Each step is self-contained.

---

### Step 1: Create `n2v/snn/model.py`

**Source**: Copy `F2FMLP` from `snn_comparison.py` lines 38–164.

**Imports needed at top of file**:
```python
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
```

**Content**: The `F2FMLP` class verbatim. Do not change any logic.

`F2FMLP.__init__` signature for reference:
```python
def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int,
             beta: float = 0.9, threshold: float = 1.0, num_steps: int = 20)
```

Key attributes the LP engine accesses on this model:
- `model.fcs` — `nn.ModuleList` of `nn.Linear` layers
- `model.lifs` — `nn.ModuleList` of `snn.Leaky` neurons
- `model.num_steps` — integer T
- `model.fcs[-1].out_features` — num_classes
- `model.fcs[0].in_features` — input_size
- `model.simulate_with_patterns(spike_train)` — exact simulation method

---

### Step 2: Create `n2v/snn/encoding.py`

**Source**: Copy from `snn_comparison.py` lines 165–228.

**Functions to include**:
```python
def latency_from_values(values: torch.Tensor, num_steps: int) -> torch.Tensor
def encode_batch(images: torch.Tensor, num_steps: int) -> torch.Tensor
def spike_train_from_latencies(latencies: np.ndarray, num_steps: int) -> torch.Tensor
```

**Imports needed**:
```python
import numpy as np
import torch
```

Do not change any logic.

---

### Step 3: Create `n2v/snn/lp.py` (the critical file)

**Source**: Copy from `snn_comparison.py` lines 229–414 and 542–1865 (skipping 415–541, which are `monte_carlo_outputs` / `bounds_cover_outputs` — those go in `verifier.py`). Plus all relevant imports.

**Imports needed** (collect all that the functions in this range use):
```python
import math
import multiprocessing
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog as scipy_linprog
from scipy.sparse import csr_matrix

try:
    import highspy
    _HAS_HIGHSPY = True
except ImportError:
    _HAS_HIGHSPY = False

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = lambda x, **kw: x

from n2v.snn.model import F2FMLP
from n2v.snn.encoding import latency_from_values, spike_train_from_latencies
```

**Functions to include** (all verbatim except the modifications noted below):
- `make_bounds`
- `feasible_latencies`
- Global state: `_LP_CONTEXT`, `_USE_EQ_CONSTRAINTS`, `_DEBUG_LP`, `_USE_AMO_CONSTRAINTS`
- `order_split_indices`
- `_HighsPreparedLP`
- `_init_symbolic_split_worker`, `_solve_symbolic_split_worker`
- `build_symbolic_relaxation_lp` ← **MODIFY** (see Step 3a)
- `build_symbolic_relaxation_lp_split` ← **MODIFY** (see Step 3b)
- `_gap_from_result`
- `verify_symbolic_sample`

**Do NOT include** `monte_carlo_outputs` or `bounds_cover_outputs` in `lp.py`. These are used exclusively by `SNNVerifier.verify()` and belong in `verifier.py`.

#### Step 3a: Modify `build_symbolic_relaxation_lp`

This function currently starts:
```python
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
```

**Change 1**: Add `input_bounds` as the last keyword parameter:
```python
def build_symbolic_relaxation_lp(model, image_flat: np.ndarray, epsilon: float,
                                  k: int, num_steps: int,
                                  tight_bounds: bool = False,
                                  fixed_latencies: dict[int, int] | None = None,
                                  label: int | None = None,
                                  cert_only: bool = False,
                                  pixel_indices: np.ndarray | None = None,
                                  singleton_bounds: bool = False,
                                  parallel_workers: int = 1,
                                  parallel_backend: str = "thread",
                                  input_bounds: tuple[np.ndarray, np.ndarray] | None = None):
```

**Change 2**: Find the line inside the function body that reads:
```python
lb_x, ub_x = make_bounds(image_flat, indices, epsilon)
```
Replace it with:
```python
if input_bounds is not None:
    lb_x, ub_x = input_bounds
else:
    lb_x, ub_x = make_bounds(image_flat, indices, epsilon)
```

That is the **only** change to this function. All other logic is untouched.

#### Step 3b: Modify `build_symbolic_relaxation_lp_split`

This function currently starts:
```python
def build_symbolic_relaxation_lp_split(model, image_flat: np.ndarray, epsilon: float,
                                        k: int, num_steps: int,
                                        split_depth: int = 1,
                                        label: int | None = None,
                                        ...):
```

**Change 1**: Add `input_bounds` as the last keyword parameter (same as above).

**Change 2**: Find the line inside the function body:
```python
lb_x, ub_x = make_bounds(image_flat, indices, epsilon)
```
Replace with the same conditional:
```python
if input_bounds is not None:
    lb_x, ub_x = input_bounds
else:
    lb_x, ub_x = make_bounds(image_flat, indices, epsilon)
```

**Change 3**: Inside the nested `solve_combo` function, find the call to `build_symbolic_relaxation_lp`:
```python
return build_symbolic_relaxation_lp(
    model, image_flat, epsilon, k, num_steps,
    tight_bounds=tight_bounds, fixed_latencies=fixed, label=label,
    cert_only=cert_only, pixel_indices=indices, singleton_bounds=singleton_bounds,
    parallel_workers=parallel_workers, parallel_backend=parallel_backend,
)
```
Add `input_bounds=input_bounds` to this call:
```python
return build_symbolic_relaxation_lp(
    model, image_flat, epsilon, k, num_steps,
    tight_bounds=tight_bounds, fixed_latencies=fixed, label=label,
    cert_only=cert_only, pixel_indices=indices, singleton_bounds=singleton_bounds,
    parallel_workers=parallel_workers, parallel_backend=parallel_backend,
    input_bounds=input_bounds,
)
```

**Change 4**: Add a process-pool guard at the very top of the function body, before any other logic:
```python
if input_bounds is not None and parallel_backend == 'process':
    raise ValueError(
        "input_bounds is not supported with parallel_backend='process'. "
        "Use parallel_backend='thread' (the default for SpikingNeuralNetwork)."
    )
```

**Rationale for Changes 2–3**: When `build_symbolic_relaxation_lp_split` spawns sub-LPs for partially-pinned branches, those sub-LPs must use the same per-dimension bounds as the parent LP. The `fixed_latencies` dict handles the pinned dimensions; the `input_bounds` array continues to provide bounds for the remaining symbolic dimensions. Passing `input_bounds` through ensures consistency.

**Rationale for Change 4**: The process pool path serializes state into worker processes via `_init_symbolic_split_worker`. That function's signature and the `initargs` tuple do not include `input_bounds`, so the worker would silently fall back to `make_bounds(image_flat, epsilon)` — giving wrong results. A hard error is safer than silent corruption. `SpikingNeuralNetwork.reach()` always passes `parallel_backend='thread'`, so this guard never triggers in the normal integration path; it only protects against someone calling the LP functions directly with incompatible arguments.

---

### Step 4: Create `n2v/snn/verifier.py`

**Source**: Copy from `snn_comparison.py`:
- Lines 415–541: `monte_carlo_outputs`, `bounds_cover_outputs` ← **place here, not in lp.py**
- Lines 1866–2348: `summarize`, `summarize_depth0_exhaustive`, `_row_cache_key`, `load_existing_rows`, `SNNVerifier`

**Imports needed**:
```python
import itertools
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import n2v.snn.lp as _lp_module          # needed for global flag mutation — see below
from n2v.snn.model import F2FMLP
from n2v.snn.encoding import encode_batch, latency_from_values, spike_train_from_latencies
from n2v.snn.lp import (
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
    verify_symbolic_sample,
    make_bounds,
    feasible_latencies,
    _LP_CONTEXT,                           # dict — in-place mutation works across modules
)
```

#### Critical: Global flag mutation across modules

`SNNVerifier.verify()` currently contains this block:
```python
_LP_CONTEXT["image_idx"] = image_idx
global _USE_EQ_CONSTRAINTS, _DEBUG_LP, _USE_AMO_CONSTRAINTS
_USE_EQ_CONSTRAINTS = eq_constraints
_DEBUG_LP = debug_lp
_USE_AMO_CONSTRAINTS = amo
```

After the split, this **will not work as written**. The `global` statement makes `_USE_EQ_CONSTRAINTS` etc. global in `verifier.py`'s own namespace — it does not modify the variables in `lp.py` where the LP functions read them. This is a Python scoping rule: `global x` only affects the current module.

**Fix**: Replace the `global` block entirely with module-level attribute assignment on `_lp_module`:

```python
# Replace:
#   global _USE_EQ_CONSTRAINTS, _DEBUG_LP, _USE_AMO_CONSTRAINTS
#   _USE_EQ_CONSTRAINTS = eq_constraints
#   _DEBUG_LP = debug_lp
#   _USE_AMO_CONSTRAINTS = amo
# With:
_LP_CONTEXT["image_idx"] = image_idx      # dict mutation — works fine across modules
_lp_module._USE_EQ_CONSTRAINTS = eq_constraints
_lp_module._DEBUG_LP = debug_lp
_lp_module._USE_AMO_CONSTRAINTS = amo
```

`_LP_CONTEXT["image_idx"] = image_idx` works across modules because it mutates the dict object in-place — the import gives a reference to the same dict. The three boolean flags need the explicit `_lp_module.` prefix because reassigning them (not mutating) only affects the local binding without the module reference.

#### Modify `SNNVerifier.train()`

After the existing `torch.save({...}, ckpt)` call, add one line:

```python
# Existing checkpoint save (keep exactly as-is):
torch.save({
    "model_state_dict": model.state_dict(),
    "train_summary": summary,
    "config": {"input_size": input_size, "num_classes": num_classes},
}, ckpt)

# New: save the full model object for SpikingNeuralNetwork loading
torch.save(model, self.output_dir / "snn_model.pt")

self.model = model.eval()
return summary
```

**Why**: `torch.save(model, path)` serializes the entire model object (architecture + weights), so `torch.load(path)` reconstructs it without needing to know `hidden_sizes`, `beta`, `threshold`, or `num_steps`. This mirrors the standard PyTorch loading pattern that `NeuralNetwork` users are familiar with.

These are the **only two changes** to `SNNVerifier`. Everything else is verbatim.

---

### Step 5: Create `n2v/snn/__init__.py`

```python
"""
Spiking neural network verification subpackage.

Provides the F2FMLP model, latency encoding utilities, LP-based reachability
engine, and the SNNVerifier training/verification class.
"""

from n2v.snn.model import F2FMLP
from n2v.snn.encoding import latency_from_values, encode_batch, spike_train_from_latencies
from n2v.snn.lp import (
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
    verify_symbolic_sample,
    make_bounds,
    feasible_latencies,
)
from n2v.snn.verifier import SNNVerifier, monte_carlo_outputs, bounds_cover_outputs

__all__ = [
    "F2FMLP",
    "latency_from_values",
    "encode_batch",
    "spike_train_from_latencies",
    "build_symbolic_relaxation_lp",
    "build_symbolic_relaxation_lp_split",
    "verify_symbolic_sample",
    "make_bounds",
    "feasible_latencies",
    "SNNVerifier",
    "monte_carlo_outputs",
    "bounds_cover_outputs",
]
```

---

### Step 6: Create `n2v/nn/spiking_neural_network.py`

This is the main new user-facing file. It mirrors `neural_network.py` structurally and integrates with n2v's global config.

```python
"""
Spiking Neural Network wrapper for verification.

Wraps snntorch F2FMLP models to enable reachability analysis using
n2v's set-based interface (Star, Box).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.sets.box import Box
from n2v.config import config as global_config
from n2v.snn.encoding import encode_batch
from n2v.snn.lp import (
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
)


@dataclass(frozen=True)
class SNNReachConfig:
    """
    Configuration for SpikingNeuralNetwork.reach().

    Mirrors ReachConfig in n2v/nn/reach.py. Either pass this as
    ``config=SNNReachConfig(...)`` or pass the fields as bare kwargs to
    reach(). The two styles are mutually exclusive.

    Attributes:
        method: 'approx' (depth-0 LP relaxation, Algorithm 1 d=0) or
            'exact' (full latency enumeration, Algorithm 1 d=k).
        parallel_workers: Number of parallel LP branch workers (thread pool).
            0 means defer to n2v's global config (set_parallel / set_parallel).
        tight_bounds: If True, use tighter LP bounds at the cost of more
            LP variables.
        singleton_bounds: If True, pin dimensions whose feasible latency
            set has exactly one element (zero overhead, slightly tighter).
        split_strategy: Dimension ordering for 'exact' enumeration.
            One of 'random', 'influence', 'choice', 'choice-influence'.
        label: If set, also compute the certification gap (score[label] minus
            max competitor score). Does not affect the score bounds themselves.
    """
    method: Literal['approx', 'exact'] = 'approx'
    parallel_workers: int = 0
    tight_bounds: bool = False
    singleton_bounds: bool = False
    split_strategy: str = 'choice-influence'
    label: Optional[int] = None

    def __post_init__(self):
        if self.method not in ('approx', 'exact'):
            raise ValueError(
                f"SNNReachConfig.method must be 'approx' or 'exact', "
                f"got {self.method!r}"
            )
        if self.parallel_workers < 0:
            raise ValueError(
                f"SNNReachConfig.parallel_workers must be >= 0, "
                f"got {self.parallel_workers}"
            )
        valid_strategies = ('random', 'influence', 'choice', 'choice-influence')
        if self.split_strategy not in valid_strategies:
            raise ValueError(
                f"SNNReachConfig.split_strategy must be one of "
                f"{valid_strategies}, got {self.split_strategy!r}"
            )


def _validate_snn_reach_config(
    method: str,
    config: Optional[SNNReachConfig],
    **kwargs,
) -> SNNReachConfig:
    """
    Reconcile config= and bare kwargs for reach() dispatch.

    Mirrors _validate_reach_config in n2v/nn/reach.py. The two styles
    (config object vs bare kwargs) are mutually exclusive.
    """
    if config is not None and kwargs:
        raise TypeError(
            "Pass either a SNNReachConfig object via config= or bare kwargs, "
            "not both."
        )
    if config is not None:
        if not isinstance(config, SNNReachConfig):
            raise TypeError(
                f"config must be an SNNReachConfig instance, got {type(config)}"
            )
        return config
    return SNNReachConfig(method=method, **kwargs)


def _set_to_bounds(
    input_set: Union[Star, Box],
    lp_solver: str = 'default',
    parallel: bool = False,
    n_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-dimension (lb, ub) arrays from a Star or Box.

    For Star: calls get_ranges() which runs 2*dim LP solves to compute the
    exact axis-aligned bounding box. All three n2v LP config arguments
    (lp_solver, parallel, n_workers) are forwarded so that n2v's global
    config (set_lp_solver, set_parallel) is fully respected for this step.
    The SNN LP then operates over this bounding box — a sound over-approximation
    of the star because the bounding box contains the star.

    For Box: reads .lb and .ub directly — exact, no LP needed.

    Args:
        input_set: Star or Box input perturbation set.
        lp_solver: LP solver for Star.get_ranges(). Comes from n2v global config.
        parallel: Whether to use parallel LP solving for Star.get_ranges().
        n_workers: Number of workers for parallel Star.get_ranges().

    Returns:
        lb: 1-D float64 array of shape (n_dims,)
        ub: 1-D float64 array of shape (n_dims,)
    """
    if isinstance(input_set, Star):
        lb, ub = input_set.get_ranges(
            lp_solver=lp_solver,
            parallel=parallel,
            n_workers=n_workers,
        )   # shape (dim, 1) each
    elif isinstance(input_set, Box):
        lb, ub = input_set.lb, input_set.ub                   # shape (dim, 1) each
    else:
        raise TypeError(
            f"SpikingNeuralNetwork.reach() requires a Star or Box input set, "
            f"got {type(input_set).__name__}."
        )
    return lb.flatten().astype(np.float64), ub.flatten().astype(np.float64)


class SpikingNeuralNetwork:
    """
    Spiking Neural Network wrapper for formal verification.

    Wraps a snntorch nn.Module (specifically F2FMLP) to enable reachability
    analysis using n2v's set-based interface. The interface is intentionally
    identical to NeuralNetwork.

    The model must expose:
        - .fcs:       nn.ModuleList of nn.Linear layers
        - .lifs:      nn.ModuleList of snntorch.Leaky neurons
        - .num_steps: int, number of timesteps T

    F2FMLP satisfies all of these. Other snntorch MLP architectures will work
    provided they expose the same attributes.

    Loading:
        Load a trained model via torch.load on the full model object saved by
        SNNVerifier.train(), then wrap it — exactly as with NeuralNetwork:

            model = torch.load("snn_model.pt")
            net = SpikingNeuralNetwork(model, input_size=(784,))

    Note on torch.fx tracing:
        F2FMLP cannot be symbolically traced (time loop + data-dependent
        at-most-once masking). The .layers property is therefore not available.
        This is the same constraint NeuralNetwork documents for untraceable
        models: probabilistic / LP-based methods do not need tracing.

    Note on global config:
        SpikingNeuralNetwork.reach() respects n2v.set_parallel() and
        n2v.set_lp_solver() exactly as NeuralNetwork does. When
        SNNReachConfig.parallel_workers is 0 (the default), the number of
        LP workers is determined by the global config.
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Optional[tuple] = None,
    ) -> None:
        """
        Initialize SpikingNeuralNetwork wrapper.

        Args:
            model: A snntorch nn.Module exposing .fcs, .lifs, .num_steps.
            input_size: Expected input size (excluding batch dim). Used to
                validate the model via a forward pass if provided.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module")

        for attr in ("fcs", "lifs", "num_steps"):
            if not hasattr(model, attr):
                raise TypeError(
                    f"model must expose .{attr} (required by the SNN LP engine). "
                    f"Use F2FMLP or a compatible snntorch architecture."
                )

        self.model = model
        self.model.eval()
        self.input_size = input_size
        self.output_size = None

        if input_size is not None:
            self._validate_input_size(input_size)

    def _validate_input_size(self, input_size: tuple) -> None:
        """Validate input size via a dry forward pass using encoded spikes."""
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, *input_size)
                spikes = encode_batch(dummy, self.model.num_steps)
                output = self.model(spikes)
                self.output_size = tuple(output.shape[1:])
        except Exception as e:
            raise ValueError(
                f"Model forward pass failed with input_size={input_size}: {e}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode input as latency-coded spikes, then run the SNN.

        Args:
            x: Raw input tensor (batch, *input_dims). Values should be in [0, 1].

        Returns:
            Score tensor (batch, num_classes).
        """
        with torch.no_grad():
            spikes = encode_batch(x, self.model.num_steps)
            return self.model(spikes)

    def reach(
        self,
        input_set: Union[Star, Box],
        method: str = 'approx',
        **kwargs,
    ) -> List[Box]:
        """
        Reachability analysis: compute guaranteed per-class output score bounds.

        Args:
            input_set: Input perturbation set. Star or Box.
                - Box: exact per-dimension bounds, most efficient.
                - Star: bounding box computed via get_ranges() (exact LP bounds,
                  requires 2*dim LP solves). The SNN LP then operates over the
                  star's bounding box — sound but potentially conservative.
            method:
                'approx': Depth-0 LP relaxation (Algorithm 1, d=0 from the ATVA
                    paper). Fast. Returns a sound over-approximation.
                'exact': Full latency enumeration (Algorithm 1, d=k). Exponential
                    in the number of symbolic dimensions. Returns tight bounds
                    for Box inputs; the bounding-box approximation still applies
                    to Star inputs at the input stage.
            **kwargs: SNNReachConfig fields as bare kwargs, OR
                ``config=SNNReachConfig(...)`` — not both.

        Returns:
            List containing a single Box.
            output[0].lb shape: (num_classes, 1) — per-class lower score bound.
            output[0].ub shape: (num_classes, 1) — per-class upper score bound.

        Example:
            >>> model = torch.load("snn_model.pt")
            >>> net = SpikingNeuralNetwork(model, input_size=(784,))
            >>> lb = np.zeros((784, 1))
            >>> ub = np.full((784, 1), 0.1)
            >>> out = net.reach(Box(lb, ub), method='approx')[0]
            >>> out = net.reach(Box(lb, ub), method='exact',
            ...                 config=SNNReachConfig(parallel_workers=4))
        """
        cfg = _validate_snn_reach_config(method, kwargs.pop("config", None), **kwargs)

        # Resolve parallel_workers: 0 means defer to n2v global config.
        parallel_workers = cfg.parallel_workers
        if parallel_workers == 0:
            n_sym_est = input_set.dim if hasattr(input_set, 'dim') else 1
            if global_config.should_use_parallel(n_sym_est):
                parallel_workers = global_config.get_n_workers(n_sym_est)
            else:
                parallel_workers = 1

        # Extract per-dimension bounds, forwarding full n2v global config for Stars.
        lb_arr, ub_arr = _set_to_bounds(
            input_set,
            lp_solver=global_config.lp_solver,
            parallel=global_config.parallel_lp,
            n_workers=global_config.n_workers,
        )

        midpoint = (lb_arr + ub_arr) / 2.0
        symbolic_dims = np.where(ub_arr - lb_arr > 1e-12)[0]
        n_symbolic = len(symbolic_dims)
        num_steps = self.model.num_steps

        if method == 'approx':
            result = build_symbolic_relaxation_lp(
                model=self.model,
                image_flat=midpoint,
                epsilon=0.0,
                k=n_symbolic,
                num_steps=num_steps,
                tight_bounds=cfg.tight_bounds,
                label=cfg.label,
                cert_only=False,
                pixel_indices=symbolic_dims,
                singleton_bounds=cfg.singleton_bounds,
                parallel_workers=parallel_workers,
                parallel_backend='thread',
                input_bounds=(lb_arr, ub_arr),
            )
        elif method == 'exact':
            result = build_symbolic_relaxation_lp_split(
                model=self.model,
                image_flat=midpoint,
                epsilon=0.0,
                k=n_symbolic,
                num_steps=num_steps,
                split_depth=n_symbolic,
                label=cfg.label,
                tight_bounds=cfg.tight_bounds,
                cert_only=False,
                pixel_indices=symbolic_dims,
                singleton_bounds=cfg.singleton_bounds,
                parallel_workers=parallel_workers,
                parallel_backend='thread',
                split_strategy=cfg.split_strategy,
                input_bounds=(lb_arr, ub_arr),
            )
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"SpikingNeuralNetwork.reach() supports 'approx' and 'exact'."
            )

        lb_y = result.get('lb')
        ub_y = result.get('ub')

        if lb_y is None or ub_y is None:
            raise RuntimeError(
                "SNN LP returned no bounds (invalid subproblem). "
                "Ensure the input set has values in [0, 1] and num_steps > 0."
            )

        return [Box(
            lb=np.array(lb_y, dtype=np.float64).reshape(-1, 1),
            ub=np.array(ub_y, dtype=np.float64).reshape(-1, 1),
        )]

    def __repr__(self) -> str:
        num_classes = (
            self.model.fcs[-1].out_features
            if hasattr(self.model, 'fcs') and len(self.model.fcs) > 0
            else '?'
        )
        return (
            f"SpikingNeuralNetwork("
            f"num_steps={self.model.num_steps}, "
            f"num_classes={num_classes}, "
            f"input_size={self.input_size})"
        )
```

---

### Step 7: Modify `n2v/nn/__init__.py`

Current content:
```python
from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import ReachConfig

__all__ = ["NeuralNetwork", "ReachConfig"]
```

Updated content:
```python
from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import ReachConfig
from n2v.nn.spiking_neural_network import SpikingNeuralNetwork, SNNReachConfig

__all__ = ["NeuralNetwork", "ReachConfig", "SpikingNeuralNetwork", "SNNReachConfig"]
```

---

### Step 8: Modify `n2v/__init__.py`

Add to the imports block:
```python
from n2v.nn import NeuralNetwork, ReachConfig, SpikingNeuralNetwork, SNNReachConfig
from n2v.snn import SNNVerifier, F2FMLP
```

Add to `__all__`:
```python
"SpikingNeuralNetwork",
"SNNReachConfig",
"SNNVerifier",
"F2FMLP",
```

---

## Part 5 — Design Decisions and Why

### Why `input_bounds` instead of a new function?

The LP engine (`build_symbolic_relaxation_lp`) is ~726 lines. Duplicating it to create a `from_bounds` variant would create a maintenance burden and divergence risk. Adding a single `input_bounds` optional parameter at the top — two lines — is surgical and backward-compatible. All existing callers (SNNVerifier, verify_symbolic_sample) continue to work unchanged.

### Why `image_flat=midpoint` when using `input_bounds`?

The `image_flat` argument is still used in `build_symbolic_relaxation_lp` for one thing after we bypass `make_bounds`: computing `base_lat = latency_from_values(image_flat, num_steps)`. This gives the **nominal latency** for each dimension. For fixed dimensions (lb == ub), this is used as their exact spike time. For symbolic dimensions, `base_lat` is overwritten in the split enumeration logic. Using the midpoint `(lb + ub) / 2` as the nominal point is correct for fixed dimensions (lb == ub means midpoint == lb == ub) and harmless for symbolic dimensions.

### Why is `epsilon=0.0` passed when using `input_bounds`?

With `input_bounds` provided, the `make_bounds` call inside the LP function is bypassed entirely. The `epsilon` argument is therefore never read. We pass `0.0` as a valid placeholder to satisfy the function signature without breaking any other logic.

### Why `get_ranges()` for Star and not `estimate_ranges()`?

`get_ranges()` computes the **exact** axis-aligned bounding box of the star via LP (one LP per dimension). `estimate_ranges()` is a fast over-approximation using the zonotope generator structure — it is wider. Since the SNN LP already over-approximates the true reachable set via the step-hull relaxation, we do not want to add additional conservatism at the input stage. `get_ranges()` gives the tightest possible input bounds, which leads to tighter output bounds.

Trade-off: for high-dimensional inputs, `get_ranges()` requires `2 * dim` LP solves. For a 784-dimensional MNIST input, this is 1568 LPs. This may be acceptable since the SNN LP itself is far more expensive. If performance is a concern, the user can pre-compute a Box from the Star before calling `reach()`. `Star` does not have a `to_box()` method — use `get_ranges()` directly:
```python
lb, ub = input_star.get_ranges()           # shape (dim, 1) each
box = Box(lb, ub)
net.reach(box, method='approx')
```

### Why is the output always `List[Box]`?

The LP engine produces scalar lower/upper bounds per output class. There is no basis for a tighter representation (no basis vectors, no predicate structure). `Box` is the appropriate output type. This is why the return type is `List[Box]` rather than `List[Star]` as in `NeuralNetwork.reach()`.

### Why no `layers` property?

`F2FMLP` has a time loop (`for t in range(num_steps)`) and data-dependent control flow (the at-most-once spike mask). These patterns are explicitly documented as untraceable by `torch.fx`. The LP engine does not use layer-by-layer propagation via `torch.fx`; it introspects `model.fcs` and `model.lifs` directly. Providing a broken `layers` property would be misleading.

### Why does `method='exact'` use `build_symbolic_relaxation_lp_split` with `split_depth=n_symbolic`?

At `split_depth = n_symbolic`, the split function enumerates all combinations of feasible latencies across all symbolic dimensions (Cartesian product). For each fully-pinned combination, it calls exact simulation via `model.simulate_with_patterns` — no LP required. This is Algorithm 1 at full depth d=k from the ATVA paper. The result is the exact reachable output set for the given input bounds.

This is equivalent to `SNNVerifier._verify_exhaustive()` but expressed through the existing split infrastructure, which means it benefits from the parallel execution support already built in.

### Why is n2v's `solve_lp_batch` not used for the SNN LP?

n2v provides `n2v.utils.lpsolver.solve_lp_batch`, which builds a HiGHS model once and solves multiple objectives — the same caching pattern used by the SNN's `_HighsPreparedLP`. However, `solve_lp_batch` only accepts inequality constraints (`Ax ≤ b`) plus variable bounds. The SNN LP requires **equality constraints** (`Σ_t p[(i,t)] = 1` for constrained inputs, AMO constraints, etc.) in addition to inequalities. Converting equalities to pairs of inequalities would double the constraint count and hurt solver performance. The SNN LP also has a more intricate constraint structure (triangle relaxation, cross-layer spike variables) than `solve_lp_batch` is designed for. Keeping the SNN's own `_HighsPreparedLP` is the correct choice.

### How `SpikingNeuralNetwork` respects n2v's global config

n2v exposes `set_parallel(enabled, n_workers)` and `set_lp_solver(solver)` as global config. `SpikingNeuralNetwork.reach()` integrates with this in two places:

1. **`Star.get_ranges()` call**: `get_ranges()` accepts `lp_solver`, `parallel`, and `n_workers` parameters. Pass n2v's global config values to this call so the bounding-box LP solves use the user's configured solver and parallelism.

2. **`SNNReachConfig.parallel_workers` default**: When the user has not explicitly set `parallel_workers` (i.e., it is 1), `reach()` should check `global_config.should_use_parallel(n_symbolic)` and fall back to `global_config.get_n_workers(n_symbolic)`. This means `n2v.set_parallel(True, n_workers=8)` automatically accelerates SNN verification too.

These integrations must be added to `spiking_neural_network.py` in the `reach()` method — see the corrected code in Step 6 below.

---

## Part 6 — Complete Workflow Example

```python
import torch
import numpy as np
import n2v
from n2v import SpikingNeuralNetwork, SNNReachConfig
from n2v.sets import Box, Star
from n2v.snn import SNNVerifier, F2FMLP

# ── Training (done once, separate from verification) ──────────────────────────
verifier = SNNVerifier(
    hidden_sizes=[128, 64],
    num_steps=20,
    beta=0.9,
    threshold=1.0,
    output_dir="experiments/snn",
)
verifier.train(train_ds, test_ds, epochs=10, lr=5e-4,
               train_limit=10000, batch_size=128)
# Saves: experiments/snn/snn_checkpoint.pt  (state dict, existing format unchanged)
#        experiments/snn/snn_model.pt        (full model object, NEW)

# ── Loading and wrapping for verification ─────────────────────────────────────
# Identical pattern to NeuralNetwork: load the model, then wrap it.
model = torch.load("experiments/snn/snn_model.pt")   # full model, no arch params needed
net = SpikingNeuralNetwork(model, input_size=(784,))
print(net)  # SpikingNeuralNetwork(num_steps=20, num_classes=10, input_size=(784,))

# ── Reachability with a Box ────────────────────────────────────────────────────
image = torch.load("sample.pt").numpy().flatten()    # shape (784,)
epsilon = 0.05

lb = np.clip(image - epsilon, 0.0, 1.0).reshape(-1, 1)
ub = np.clip(image + epsilon, 0.0, 1.0).reshape(-1, 1)
input_box = Box(lb, ub)

# Approx: depth-0 LP, fast, over-approximate
output_boxes = net.reach(input_box, method='approx')
print(output_boxes[0].lb.T)  # lower score bound per class, shape (1, num_classes)
print(output_boxes[0].ub.T)  # upper score bound per class

# Exact: full enumeration, slow, exact for Box inputs
output_boxes_exact = net.reach(input_box, method='exact',
                               config=SNNReachConfig(parallel_workers=4))

# ── Global config is respected ────────────────────────────────────────────────
# parallel_workers=0 (the default) defers to n2v's global config.
n2v.set_parallel(True, n_workers=8)
output_boxes = net.reach(input_box, method='approx')  # uses 8 workers automatically

# ── Reachability with a Star ───────────────────────────────────────────────────
# Stars come from prior ANN layer propagation or are constructed directly.
input_star = Star.from_bounds(lb, ub)   # box-shaped star for simplicity

# Internally calls input_star.get_ranges(lp_solver=global_config.lp_solver)
# to extract the axis-aligned bounding box, then runs the SNN LP over that box.
output_boxes_star = net.reach(input_star, method='approx')

# If you want to pre-compute the bounding box yourself (avoids re-running
# get_ranges() if you call reach() multiple times with the same star):
lb_star, ub_star = input_star.get_ranges()   # shape (784, 1) each
input_box_from_star = Box(lb_star, ub_star)
output_boxes_pre = net.reach(input_box_from_star, method='approx')

# ── With a label for certification gap ────────────────────────────────────────
cfg = SNNReachConfig(method='approx', label=3)
result = net.reach(input_box, config=cfg)
# To check certification manually:
# lb_label = result[0].lb[3, 0]
# ub_others = np.delete(result[0].ub.flatten(), 3).max()
# certified = lb_label - ub_others > 0
```

---

## Part 7 — Checklist Before Marking Complete

### File creation
- [ ] `n2v/snn/model.py` — F2FMLP copied verbatim, all imports present
- [ ] `n2v/snn/encoding.py` — three encoding functions copied verbatim
- [ ] `n2v/snn/lp.py` — all LP functions copied, `input_bounds` added to two functions, passed through in `solve_combo`
- [ ] `n2v/snn/verifier.py` — SNNVerifier + helpers copied, `torch.save(model, ...)` added to `train()`
- [ ] `n2v/snn/__init__.py` — exports all public symbols
- [ ] `n2v/nn/spiking_neural_network.py` — `SNNReachConfig`, `_validate_snn_reach_config`, `_set_to_bounds`, `SpikingNeuralNetwork` all present

### Existing file modifications (only `__init__.py` additions)
- [ ] `n2v/nn/__init__.py` — `SpikingNeuralNetwork` and `SNNReachConfig` added
- [ ] `n2v/__init__.py` — `SpikingNeuralNetwork`, `SNNReachConfig`, `SNNVerifier`, `F2FMLP` added to both imports and `__all__`

### LP engine changes (in `n2v/snn/lp.py`)
- [ ] `monte_carlo_outputs` and `bounds_cover_outputs` are NOT in `lp.py` — they go in `verifier.py`
- [ ] `build_symbolic_relaxation_lp`: `input_bounds` param added to signature as last keyword arg
- [ ] `build_symbolic_relaxation_lp`: `make_bounds` call replaced with conditional that checks `input_bounds is not None`
- [ ] `build_symbolic_relaxation_lp_split`: same `input_bounds` param added to signature
- [ ] `build_symbolic_relaxation_lp_split`: same `make_bounds` conditional replacement
- [ ] `build_symbolic_relaxation_lp_split`: `input_bounds=input_bounds` added to the `build_symbolic_relaxation_lp(...)` call inside `solve_combo`
- [ ] `build_symbolic_relaxation_lp_split`: guard at top of function body raises `ValueError` if `input_bounds is not None and parallel_backend == 'process'`

### SNNVerifier changes (in `n2v/snn/verifier.py`)
- [ ] `monte_carlo_outputs` and `bounds_cover_outputs` copied from `snn_comparison.py` lines 415–541 into `verifier.py`
- [ ] `verifier.py` imports `import n2v.snn.lp as _lp_module` (for flag mutation)
- [ ] `verifier.py` imports `_LP_CONTEXT` from `n2v.snn.lp` (dict, in-place mutation is safe)
- [ ] `SNNVerifier.verify()`: `global _USE_EQ_CONSTRAINTS, _DEBUG_LP, _USE_AMO_CONSTRAINTS` block replaced with `_lp_module._USE_EQ_CONSTRAINTS = ...` etc.
- [ ] `SNNVerifier.train()`: `torch.save(model, self.output_dir / "snn_model.pt")` added immediately after existing `torch.save({...}, ckpt)` line

### SpikingNeuralNetwork correctness (in `n2v/nn/spiking_neural_network.py`)
- [ ] `encode_batch` imported at module level from `n2v.snn.encoding` (not deferred inside methods — no circular import risk)
- [ ] `SNNReachConfig` has `__post_init__` validating method, parallel_workers >= 0, split_strategy
- [ ] `SNNReachConfig.parallel_workers` defaults to `0` (not 1) — defers to global config
- [ ] `_set_to_bounds` signature has `lp_solver`, `parallel`, `n_workers` params
- [ ] `_set_to_bounds` passes all three to `Star.get_ranges(lp_solver=..., parallel=..., n_workers=...)`
- [ ] `reach()` calls `_set_to_bounds(input_set, lp_solver=global_config.lp_solver, parallel=global_config.parallel_lp, n_workers=global_config.n_workers)`
- [ ] `reach()` resolves `parallel_workers=0` via `global_config.should_use_parallel()` / `get_n_workers()`
- [ ] `reach()` imports `from n2v.config import config as global_config`
- [ ] `reach()` raises `TypeError` for non-Star/Box inputs (from `_set_to_bounds`)
- [ ] `reach()` raises `RuntimeError` when LP returns `None` bounds
- [ ] `reach()` returns `List[Box]` with lb/ub shapes `(num_classes, 1)`
- [ ] `Star.to_box()` is NOT referenced anywhere — use `Box(lb, ub)` after `lb, ub = star.get_ranges()`
