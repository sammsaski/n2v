# SNNV Integration Log

Record of all changes made to integrate SNN verification into n2v.

---

## Phase 1 — Architecture Design and Implementation

### New subpackage: `n2v/snn/`

Split from `external_snnv/snn_comparison.py` into a proper Python subpackage.

**`n2v/snn/model.py`**
- Contains `F2FMLP`: snntorch-based First-to-Fire MLP
- Verbatim from `snn_comparison.py` lines 38–164; no logic changes
- Key attributes: `num_steps`, `num_classes`, `hidden_sizes`, `fcs` (ModuleList), `lifs` (ModuleList)
- `forward(spike_train)`: takes `(B, D, T)` spike train, returns `(B, num_classes)` class scores
- `simulate_with_patterns()`: returns `(scores, hidden_spikes, output_spikes)` for the verifier

**`n2v/snn/encoding.py`**
- `latency_from_values(values, num_steps)`: maps `[0,1]` values to integer spike times; larger → earlier; zero → silent
- `encode_batch(images, num_steps)`: converts `(B, D)` tensor to `(B, D, T)` binary spike train
- `spike_train_from_latencies(latencies, num_steps)`: builds `(D, T)` spike train from latency array

**`n2v/snn/lp.py`** (~1510 lines)
- Core LP engine: verbatim from `snn_comparison.py` lines 229–414 and 542–1865
- Key changes from source:
  - Added `input_bounds: tuple[np.ndarray, np.ndarray] | None = None` parameter to `build_symbolic_relaxation_lp` and `build_symbolic_relaxation_lp_split`
  - When `input_bounds` is provided, bypasses `make_bounds(image_flat, indices, epsilon)` and uses per-dimension lb/ub arrays directly — this is the bridge from n2v sets to the LP engine
  - Added process-pool guard: `input_bounds` cannot be used with `parallel_backend='process'` (raises `ValueError`)
  - `from itertools import product as iproduct` moved to top-level imports
  - `import warnings` and `import math` removed (unused in this module after split)
  - `_tqdm` try/except import retained (used at lines 1333, 1343, 1350)

**`n2v/snn/verifier.py`**
- Contains `SNNVerifier`, `monte_carlo_outputs`, `bounds_cover_outputs`
- Verbatim from `snn_comparison.py` lines 415–541 and 1866–2348
- Key changes from source:
  - `import n2v.snn.lp as _lp_module` added
  - In `SNNVerifier.verify()`, replaced `global _USE_EQ_CONSTRAINTS` etc. with module-attribute assignment (`_lp_module._USE_EQ_CONSTRAINTS = eq_constraints`) — required because after the module split, the `global` statement would only affect `verifier.py`'s namespace, not `lp.py`'s
  - `SNNVerifier.train()` now calls `torch.save(model, self.output_dir / "snn_model.pt")` in addition to the checkpoint save — produces the full model file that `SpikingNeuralNetwork` loads
  - `_tqdm` try/except block removed (unused in `verifier.py`; lives in `lp.py`)

**`n2v/snn/__init__.py`**
- Exports: `F2FMLP`, `latency_from_values`, `encode_batch`, `spike_train_from_latencies`, `build_symbolic_relaxation_lp`, `build_symbolic_relaxation_lp_split`, `verify_symbolic_sample`, `make_bounds`, `feasible_latencies`, `SNNVerifier`, `monte_carlo_outputs`, `bounds_cover_outputs`

---

### New wrapper class: `n2v/nn/spiking_neural_network.py`

**`SNNReachConfig`** (frozen dataclass)
- Fields: `method` ('approx'|'exact'), `parallel_workers` (0=global config), `singleton_bounds`, `split_strategy`, `label`
- Validated in `__post_init__`
- Note: `tight_bounds` was initially included but removed (see Phase 5)

**`SpikingNeuralNetwork`**
- Constructor: `SpikingNeuralNetwork(model, input_size=None)` — mirrors `NeuralNetwork`
- `input_size` is `Optional[int]` (flat 1D), not `Optional[tuple]` like `NeuralNetwork`
- `forward(x)`: takes `(B, D)` or `(D,)` tensor, encodes to spike train, returns `(B, num_classes)` scores
- `reach(input_set, method='exact', config=None, **kwargs)`: returns `List[Box]` (always 1 element)
- `_set_to_bounds()` helper: extracts per-dimension lb/ub from Star (via `get_ranges()`) or Box (via `.lb/.ub`), returns flat float64 arrays

Key design decisions:
- Returns `List[Box]` always (not the input set type) — LP engine produces axis-aligned score bounds
- `method='approx'`: depth-0 LP relaxation (single LP, `build_symbolic_relaxation_lp`)
- `method='exact'`: full latency enumeration (`build_symbolic_relaxation_lp_split`, split_depth = n_symbolic)
- `parallel_backend='thread'` hardcoded — process pools incompatible with `input_bounds` parameter
- Bounds extraction happens before parallel worker resolution (Box has no `.dim` attribute)
- config= is the authority when passed — does not raise on method mismatch (better UX than `NeuralNetwork`)

---

### Modified existing files

**`n2v/nn/__init__.py`**
- Added: `from n2v.nn.spiking_neural_network import SpikingNeuralNetwork, SNNReachConfig`
- Added: `SpikingNeuralNetwork`, `SNNReachConfig` to `__all__`
- Later modified: wrapped in `try/except ImportError` guard (see Phase 3)

**`n2v/__init__.py`**
- Added: `from n2v.snn import SNNVerifier, F2FMLP`
- Added: SNN items to `__all__`
- Later modified: wrapped in `try/except ImportError` guard (see Phase 3)

---

## Phase 2 — Review and Bug Fixes

### Unused imports removed
- `n2v/snn/lp.py`: removed `import math`, `import warnings`
- `n2v/snn/verifier.py`: removed dead `_tqdm` try/except block (tqdm is used in lp.py, not verifier.py)
- `n2v/nn/spiking_neural_network.py`: removed `from pathlib import Path`, `verify_symbolic_sample`, `from n2v.snn.model import F2FMLP`

### Config dispatch bug fixed
- `_validate_snn_reach_config`: removed erroneous `config.method != method` check. When `config=` is passed it is the authority; the `method` positional parameter is only used when building `SNNReachConfig(method=method)` from kwargs.
- `reach()`: dispatch now uses `cfg.method` (not `method`) — fixes incorrect routing when user passes `config=SNNReachConfig(method='exact')` without explicitly setting `method='exact'`.

### Box dimension bug fixed
- `reach()` originally used `input_set.dim if hasattr(input_set, 'dim') else 1` for parallel worker estimation. `Box` has no `.dim` attribute, so Box inputs always got `n_sym_est=1`, disabling parallelism regardless of actual dimension.
- Fixed by reordering: bounds extraction (`_set_to_bounds`) now runs before parallel worker resolution, so `n_dims = len(lb_arr)` is used for both Star and Box inputs.

### Error message consistency
- `SpikingNeuralNetwork.__init__`: "model" → "Model" to match `NeuralNetwork`

---

## Phase 3 — Testing Infrastructure

### Dependency declaration

**`pyproject.toml`**
- Added `[project.optional-dependencies] snn = ["snntorch>=0.6"]`
- Users install with: `pip install n2v[snn]`

### Import guard (fixes `import n2v` breaking without snntorch)

**`n2v/nn/__init__.py`**
```python
try:
    from n2v.nn.spiking_neural_network import SpikingNeuralNetwork, SNNReachConfig
    _SNN_AVAILABLE = True
except ImportError:
    _SNN_AVAILABLE = False
```

**`n2v/__init__.py`**
```python
try:
    from n2v.nn import SpikingNeuralNetwork, SNNReachConfig
    from n2v.snn import SNNVerifier, F2FMLP
    _SNN_AVAILABLE = True
except ImportError:
    _SNN_AVAILABLE = False
```

Without these guards, `import n2v` failed with `ModuleNotFoundError: No module named 'snntorch'` for any user who had not installed `n2v[snn]`.

### CI update

**`.github/workflows/tests.yml`**
- Changed `pip install -e ".[dev]"` → `pip install -e ".[dev,snn]"` so snntorch is installed in CI

### Test suite

All SNN tests skip gracefully if snntorch is not installed via `pytest.importorskip('snntorch')`.

**`tests/unit/snn/__init__.py`** — empty package marker

**`tests/unit/snn/conftest.py`** — shared fixtures
- `tiny_model`: `F2FMLP(input_size=4, hidden_sizes=[8], num_classes=3, num_steps=8)` — small enough for LP tests to run in milliseconds
- `snn_wrapper`: `SpikingNeuralNetwork(tiny_model)`
- `tiny_box`: 4-D Box, all dimensions symbolic
- `tiny_star`: 4-D Star equivalent to `tiny_box`
- `partial_box`: 4-D Box with only 2 symbolic dims (for `'exact'` method tests)
- `partial_star`: 4-D Star with same 2-symbolic-dim bounds as `partial_box` (for Star+exact path tests)

**`tests/unit/snn/test_model.py`**
- `TestF2FMLPConstruction`: default and custom params, multi-layer, layer sizes, nn.Module instance
- `TestF2FMLPForward`: output shape (single/batch), scores non-negative, deterministic, all-zero spike → zero scores, float32 output
- `TestF2FMLPSimulate`: output shapes, binary spikes, scores match `forward()`, accepts batch dim

**`tests/unit/snn/test_encoding.py`**
- `TestLatencyFromValues`: max→fires-first, zero→silent, negative→silent, midpoint, monotone ordering, all-silent input, output dtype, batch shape preserved
- `TestEncodeBatch`: output shape, binary values, non-silent fires exactly once, silent never fires, high value fires early, spike timestep matches latency
- `TestSpikeTrainFromLatencies`: output shape, binary values, fires at correct time, silent sentinel, returns Tensor, consistent with `encode_batch`

**`tests/unit/snn/test_snn_reach_config.py`**
- `TestSNNReachConfigDefaults`: all field defaults
- `TestSNNReachConfigValidation`: invalid method/parallel_workers/split_strategy raise; valid values accepted
- `TestSNNReachConfigValidStrategies`: all 5 valid strategies parametrised
- `TestSNNReachConfigImmutability`: frozen, equality, inequality
- `TestSNNReachConfigFieldCombinations`: all fields set at once

**`tests/unit/snn/test_spiking_neural_network.py`**
- `TestSpikingNeuralNetworkConstruction`: valid, eval mode, TypeError for non-module, input_size match/mismatch, inference from model, repr, no `layers` attribute
- `TestSpikingNeuralNetworkForward`: 2D/1D input shapes, scores non-negative, float32, no gradient
- `TestSpikingNeuralNetworkReachStructure`: returns list, single Box, correct output shape, lb≤ub, scores≥0
- `TestSpikingNeuralNetworkReachInputTypes`: Box, Star, Star+exact (exercises `get_ranges()` → LP split path), unsupported type raises TypeError
- `TestSpikingNeuralNetworkReachMethods`: approx, exact, exact tighter than approx, invalid method raises
- `TestSpikingNeuralNetworkReachConfig`: config object, config.method priority, config+kwargs raises TypeError, kwargs style, label kwarg
- `TestSpikingNeuralNetworkReachOptions`: `singleton_bounds=True` accepted and sound (lb≤ub, lb≥0)
- `TestSpikingNeuralNetworkReachDimCheck`: wrong dimension raises ValueError

**`tests/soundness/test_soundness_snn.py`**
- Core invariant: for 300 random samples from the input set, all true model scores must lie within `reach()` bounds (with tolerance 1e-4)
- `TestSoundnessBoxInput`: approx/all-symbolic, approx/partial-symbolic, exact/two-symbolic
- `TestSoundnessStarInput`: approx/star-from-bounds, exact/star-from-bounds (exercises full `Star.get_ranges()` → `input_bounds` → LP split path)
- `TestExactTighterThanApprox`: exact lb ≥ approx lb and exact ub ≤ approx ub
- `TestSoundnessOptions`: `singleton_bounds=True` soundness check
- `TestSoundnessPointInput`: degenerate lb==ub case; single forward pass score must be within bounds

---

## Phase 4 — Documentation

**`docs/user-guide/spiking-neural-networks.md`** (created)
- Background on F2FMLP and F2F latency coding
- Description of all new files and their roles
- Quick start example
- Training with `SNNVerifier` walkthrough
- Loading a trained model
- `reach()` signature, both methods explained
- `SNNReachConfig` field table
- Input set types (Box and Star), including the Star bounding-box limitation
- Output format and how to use it for certification
- Detailed difference table vs `NeuralNetwork`
- Global config integration
- Common usage patterns

**Note**: `docs/user-guide/index.rst` and `docs/api/index.rst` have not yet been updated to include the SNN docs in the Sphinx toctree. This is a remaining gap.

---

## Phase 5 — Bug Fixes, Consistency Pass, and Test Additions

### Consistency fixes vs `NeuralNetwork`

**`output_size` type**: `SpikingNeuralNetwork.output_size` was initially `int`. Changed to `tuple`
(`(model.fcs[-1].out_features,)`) to match `NeuralNetwork`'s `Optional[tuple]` return type.

**Config+kwargs error type**: `_validate_snn_reach_config` raised `ValueError` when both `config=`
and kwargs were provided. Changed to `TypeError` to match `NeuralNetwork.reach()`.

**Default `reach()` method**: `SpikingNeuralNetwork.reach()` defaulted to `method='approx'`.
Changed to `method='exact'` to match `NeuralNetwork.reach()`.

Note: `SNNReachConfig.method` default is still `'approx'` — this is the internal config object
default, not the user-facing reach() default. A bare `SNNReachConfig()` is conservative by design.

### `tight_bounds` removal

`tight_bounds` was an unfinished feature (not in the paper) with a latent crash bug:
`_solve_bound_prepared` applies `b_ub_vec = b_ub_vec + 1e-5` but `b_ub_vec` is `None` when
the LP has no inequality constraints (only triggered on the `tight_bounds=True` code path, which
builds a different, tighter LP).

Resolution: removed `tight_bounds` entirely from `SNNReachConfig` and hardcoded
`tight_bounds=False` in both LP calls inside `reach()`. The bug in `_solve_bound_prepared` is
left as-is (the triggering code path no longer exists in the public API).

### Additional tests added

**`tests/unit/snn/conftest.py`**: Added `partial_star` fixture — a 4-D Star with the same
2-symbolic-dim bounds as `partial_box`. Used to test the `Star + method='exact'` code path.

**`tests/unit/snn/test_spiking_neural_network.py`**:
- `test_reach_with_star_exact` (in `TestSpikingNeuralNetworkReachInputTypes`): exercises the
  `Star.get_ranges()` → `input_bounds` → `build_symbolic_relaxation_lp_split` path end-to-end.
- `TestSpikingNeuralNetworkReachOptions`: two tests for `singleton_bounds=True` — accepted without
  error and produces valid (lb ≤ ub, lb ≥ 0) bounds.

**`tests/soundness/test_soundness_snn.py`**:
- `test_exact_star_from_bounds` (in `TestSoundnessStarInput`): soundness check for the
  Star+exact path — 300 random samples must all fall within the LP bounds.
- `TestSoundnessOptions.test_singleton_bounds_sound`: soundness check for `singleton_bounds=True`.

### Code documentation improvements

**`n2v/snn/lp.py`**: Added comment at the `b_ub_vec + 1e-5` line explaining the non-None
assumption and why it holds (triangle relaxation always produces ub constraints for hidden neurons
when `tight_bounds=False`).

**`n2v/snn/model.py`**: Added comment at `simulate_with_patterns` line 132 noting the `[0]` batch
index assumes a single sample.

---

## Remaining Gaps

| Gap | Status |
|---|---|
| `docs/user-guide/index.rst` toctree entry | Not done |
| `docs/api/spiking-neural-network.rst` (Sphinx autoclass) | Not done |
| `docs/api/index.rst` SNN card | Not done |
| README.md mention of SNN | Not done |
| Examples directory for SNN | Not done |
| Integration test for full `SNNVerifier.train()` → `SpikingNeuralNetwork.reach()` pipeline | Not done (requires dataset + several minutes; belongs in a slow/benchmark suite) |
| Fix `b_ub_vec + 1e-5` crash for zero-hidden-layer networks | Latent bug; not triggered by current API (F2FMLP always has ≥1 hidden layer) |
