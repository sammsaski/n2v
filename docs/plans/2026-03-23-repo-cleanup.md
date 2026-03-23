# Repository Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove experimental hexatope code (alpha-hexatope, differentiable solver), enforce code style across the library, update documentation, and prepare the repository for remote push.

**Architecture:** Five sequential phases — surgical removal of experimental code, paper alignment verification, code style enforcement across `n2v/` library, documentation updates, and a final sweep of examples and dead references.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, pytest

---

## Phase 1: Experimental Code Removal

### Task 1.1: Delete experimental files and directories

**Files:**
- Delete: `n2v/alpha_hexatope/` (entire directory)
- Delete: `n2v/utils/differentiable_solver.py`
- Delete: `examples/AlphaHexatope/` (entire directory)
- Delete: `examples/Octatope/profile_differentiable_solver.py`
- Delete: `tests/unit/alpha_hexatope/` (entire directory)
- Delete: `tests/soundness/test_soundness_differentiable.py`
- Delete: `tests/unit/utils/test_differentiable_solver_v2.py`
- Delete: `docs/differentiable_solver.md`
- Delete: `docs/plans/2026-03-12-hexoct-soundness-fix.md`
- Delete: `docs/plans/2026-03-12-hexoct-soundness-fix-design.md`
- Delete: `docs/plans/2026-03-12-solver-api-and-examples.md`
- Delete: `docs/plans/2026-03-18-differentiable-solver-rewrite.md`
- Delete: `docs/plans/2026-03-18-differentiable-solver-rewrite-design.md`

**Step 1:** Delete all listed files and directories.

**Step 2:** Verify deleted files no longer exist:
```bash
test ! -d n2v/alpha_hexatope && test ! -f n2v/utils/differentiable_solver.py && echo "OK"
```

### Task 1.2: Clean `n2v/__init__.py`

**Files:**
- Modify: `n2v/__init__.py`

**Step 1:** Remove line 17 (`from n2v import alpha_hexatope`).

**Step 2:** Remove `set_device` from the config import on line 18. Result:
```python
from n2v.config import config, set_parallel, set_lp_solver, get_config
```

**Step 3:** Remove `"alpha_hexatope"` and `"set_device"` from `__all__` list.

### Task 1.3: Clean `n2v/config.py`

**Files:**
- Modify: `n2v/config.py`

**Step 1:** Remove `self._device = 'cpu'` from `__init__` (line 24).

**Step 2:** Remove the `device` property and setter (lines 90-103).

**Step 3:** Remove `self._device = 'cpu'` from `reset()` (line 146).

**Step 4:** Remove `device='{self.device}'` from `__repr__` (line 154).

**Step 5:** Remove the entire `set_device()` function (lines 209-238).

**Step 6:** Remove `'device': config.device` from `get_config()` (line 249).

### Task 1.4: Clean `n2v/nn/neural_network.py`

**Files:**
- Modify: `n2v/nn/neural_network.py`

**Step 1:** Remove the `verify()` method (lines 163-185).

**Step 2:** Remove the `verify_robustness()` method (lines 187-210).

**Step 3:** In the `reach()` docstring, remove the `exact-differentiable` method option from the Hexatope/Octatope section (line 124). The Hexatope/Octatope section should read:
```
For Hexatope/Octatope:
    - 'approx': Over-approximate reachability
```

Note: Hexatope/Octatope only support 'approx' through the dispatcher (exact splitting produces multiple over-approximate sets but is labeled 'approx' in the dispatcher). Remove the 'exact' option too since the dispatcher functions `_reach_layer_hexatope` and `_reach_layer_octatope` only call the approx variants.

### Task 1.5: Clean `n2v/utils/lpsolver.py`

**Files:**
- Modify: `n2v/utils/lpsolver.py`

**Step 1:** Remove the `solve_dcs_differentiable()` function and its section comment (lines 321-395).

**Step 2:** Remove "Differentiable solver for DCS/UTVPI constraints (delegates to DCSOptimizer)" from the module docstring (line 8).

**Step 3:** Remove the torch import try/except block (lines 16-20) if torch is not used elsewhere in this file. Check first — if `TORCH_AVAILABLE` is referenced elsewhere in the file, keep it.

### Task 1.6: Clean `n2v/sets/hexatope.py` — remove differentiable solver references

**Files:**
- Modify: `n2v/sets/hexatope.py`

**Step 1:** Remove the try-import block for `solve_dcs_differentiable` and `HAS_DIFFERENTIABLE_SOLVER` (around lines 30-35).

**Step 2:** Remove `_get_ranges_differentiable()` method (around lines 429-470).

**Step 3:** Remove the `solver == 'differentiable'` branch in `get_ranges()` (around lines 354-355).

**Step 4:** Remove `_optimize_dcs_differentiable()` method (around lines 712-742).

**Step 5:** Remove the `'differentiable'` option from `optimize_linear()` solver validation. Update to: `solver in ('lp', 'mcf')`.

**Step 6:** Remove the differentiable branch in `_dcs_bounding_box()` (around lines 859-897).

**Step 7:** Update all docstrings that mention `solver='differentiable'` as an option.

### Task 1.7: Clean `n2v/sets/octatope.py` — remove differentiable solver references

**Files:**
- Modify: `n2v/sets/octatope.py`

Same pattern as Task 1.6 but for octatope:

**Step 1:** Remove try-import block for differentiable solver.

**Step 2:** Remove `_get_ranges_differentiable()` method.

**Step 3:** Remove `solver == 'differentiable'` branch in `get_ranges()`.

**Step 4:** Remove `_optimize_utvpi_differentiable()` method.

**Step 5:** Update `optimize_linear()` solver validation to `('lp', 'mcf')`.

**Step 6:** Remove differentiable branch in `_utvpi_bounding_box()`.

**Step 7:** Update docstrings.

### Task 1.8: Check for remaining references and run tests

**Step 1:** Grep the entire repo for stale references:
```bash
grep -r "alpha_hexatope\|AlphaHexatope\|AlphaOctatope\|differentiable_solver\|DCSOptimizer\|solve_dcs_differentiable\|set_device\|HAS_DIFFERENTIABLE_SOLVER\|exact-differentiable" --include="*.py" n2v/ tests/ examples/
```
Fix any remaining references found.

**Step 2:** Run the full test suite:
```bash
pytest tests/ -x -q
```
All tests should pass. Fix any failures caused by the removal.

---

## Phase 2: Hexatope/Octatope Paper Verification

### Task 2.1: Verify MCF solver works end-to-end

**Step 1:** Run hexatope/octatope unit tests:
```bash
pytest tests/unit/sets/test_hexatope.py tests/unit/sets/test_octatope.py -v
```

**Step 2:** Run hexatope/octatope layer ops tests:
```bash
pytest tests/unit/layer_ops/test_hexatope_octatope_reachability.py tests/unit/layer_ops/test_relu_hexatope_octatope.py -v
```

**Step 3:** Run hexatope/octatope soundness tests:
```bash
pytest tests/soundness/test_soundness_hexoct_relu.py -v
```

All should pass with LP and MCF solvers only.

### Task 2.2: Audit hexatope/octatope docstrings for paper references

**Files:**
- Review: `n2v/sets/hexatope.py`
- Review: `n2v/sets/octatope.py`

**Step 1:** Confirm all key methods reference the correct theorem/algorithm from Bak et al. (2024):
- `affine_map()` → Theorem 4 (hexatope) / Theorem 6 (octatope)
- `optimize_linear()` → Theorem 5 / Theorem 7
- `intersect_half_space()` → Algorithm 5.1
- Class-level docstring → Theorem 3 (hexatope definition) / Theorem 6 (octatope definition)

**Step 2:** Remove any references to the differentiable solver from docstrings or comments.

---

## Phase 3: Code Style Enforcement

Apply research-code-dev.md conventions to all files in `n2v/`. Do NOT touch `tests/`, `examples/`, or `third_party/`.

### Task 3.1: Add logging to `n2v/nn/reach.py`

**Files:**
- Modify: `n2v/nn/reach.py`

**Step 1:** Add at top of file after imports:
```python
import logging

logger = logging.getLogger(__name__)
```

**Step 2:** Convert all print statements:
- `print(f'Layer {i+1}...')` → `logger.info(f'Layer {i+1}...')`
- `print(f'  Output: ...')` → `logger.debug(f'  Output: ...')`
- `print(f'  Processing: ...')` → `logger.debug(f'  Processing: ...')`
- `print(f"  Exceeded {max_stars}...")` → `logger.info(f"  Exceeded {max_stars}...")`
- `print(f"  Exceeded {timeout_per_layer}...")` → `logger.info(f"  Exceeded {timeout_per_layer}...")`

### Task 3.2: Add logging to `n2v/nn/layer_ops/relu_reach.py`

**Files:**
- Modify: `n2v/nn/layer_ops/relu_reach.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)` at top.

**Step 2:** Convert:
- `print(f'Exact ReLU_{neuron_idx}...')` → `logger.debug(f'Exact ReLU_{neuron_idx}...')`
- `print(f'  ⚡ Processing...')` → `logger.info(f'Processing {len(input_stars)} Stars in parallel...')`
- `print(f'  Error processing Star: {e}')` → `logger.error(f'Error processing Star: {e}')`

### Task 3.3: Add logging to `n2v/nn/layer_ops/leakyrelu_reach.py`

**Files:**
- Modify: `n2v/nn/layer_ops/leakyrelu_reach.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert:
- `print(f'Exact LeakyReLU_{neuron_idx}...')` → `logger.debug(f'Exact LeakyReLU_{neuron_idx}...')`

### Task 3.4: Add logging to `n2v/nn/layer_ops/maxpool2d_reach.py`

**Files:**
- Modify: `n2v/nn/layer_ops/maxpool2d_reach.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert:
- `print(f'There are splits...')` → `logger.debug(f'There are splits...')`
- `print(f'Split {len(output_stars)}...')` → `logger.debug(f'Split {len(output_stars)}...')`
- `print(f'{new_pred_count} new variables...')` → `logger.debug(f'{new_pred_count} new variables...')`

### Task 3.5: Add logging to `n2v/probabilistic/verify.py`

**Files:**
- Modify: `n2v/probabilistic/verify.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert all 15 print statements. Use `logger.info()` for step headers (e.g., "Step 1: Generating..."), `logger.info()` for summary results (coverage, confidence), and `logger.debug()` for intermediate details.

### Task 3.6: Add logging to `n2v/probabilistic/surrogates/clipping_block.py`

**Files:**
- Modify: `n2v/probabilistic/surrogates/clipping_block.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert:
- `print(f"Projected {i + 1}/{m} samples")` → `logger.debug(f"Projected {i + 1}/{m} samples")`
- `print(f"Processing batch...")` → `logger.debug(f"Processing batch...")`

### Task 3.7: Add logging to `n2v/probabilistic/dimensionality/deflation_pca.py`

**Files:**
- Modify: `n2v/probabilistic/dimensionality/deflation_pca.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert:
- `print(f"Finding component...")` → `logger.debug(f"Finding component...")`
- `print(f"  Variance...")` → `logger.debug(f"  Variance...")`
- `print(f"  Converged...")` → `logger.debug(f"  Converged...")`
- `print(f"  Max iterations reached")` → `logger.warning(f"Max iterations reached")`

### Task 3.8: Add logging to `n2v/sets/hexatope.py`

**Files:**
- Modify: `n2v/sets/hexatope.py`

**Step 1:** Add `import logging` and `logger = logging.getLogger(__name__)`.

**Step 2:** Convert:
- `print(f"MCF solver error: {e}, returning None")` → `logger.warning(f"MCF solver error: {e}, returning None")`

### Task 3.9: Add type hints across `n2v/` library

**Files:**
- Modify: `n2v/config.py` — add `-> None` return type to all setters
- Modify: `n2v/nn/reach.py` — add type hints to helper functions
- Modify: `n2v/utils/bounds_precomputation.py` — add type hints to internal functions
- Modify: `n2v/nn/layer_ops/reduce_reach.py` — add return type hints to helpers
- Modify: remaining files with missing type hints per audit

For each file: read the file, identify functions missing type hints on parameters or return types, add them. Focus on public API first, then internal helpers.

### Task 3.10: Add missing docstrings across `n2v/` library

**Files:**
- Modify: `n2v/sets/hexatope.py` — 3 methods missing docstrings
- Modify: `n2v/sets/octatope.py` — 4 methods missing docstrings
- Modify: `n2v/sets/image_star.py` — 2 methods missing docstrings
- Modify: remaining files with missing docstrings per audit

For each file: read the file, identify public/complex methods missing docstrings, add Google-style docstrings with Args/Returns/Raises. Include paper references where applicable.

### Task 3.11: Run tests to verify style changes don't break anything

**Step 1:**
```bash
pytest tests/ -x -q
```
All tests should pass.

---

## Phase 4: Documentation Update

### Task 4.1: Update `README.md`

**Files:**
- Modify: `README.md`

**Step 1:** Read the full README.

**Step 2:** Remove all references to:
- Alpha-hexatope / AlphaHexatope / AlphaOctatope
- Differentiable solver / `solver='differentiable'`
- `set_device()` / device configuration
- `verify()` and `verify_robustness()` methods on NeuralNetwork

**Step 3:** Ensure hexatope/octatope sections mention LP and MCF solvers only.

### Task 4.2: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Step 1:** Remove `set_device` from config functions.

**Step 2:** Remove `alpha_hexatope` from exports and module references.

**Step 3:** Remove `exact-differentiable` from method options.

**Step 4:** Update any API examples that reference removed functionality.

### Task 4.3: Update `docs/theoretical_foundations.md`

**Files:**
- Modify: `docs/theoretical_foundations.md`

**Step 1:** Read the file.

**Step 2:** Remove or trim sections about the differentiable solver and alpha-hexatope.

**Step 3:** Ensure hexatope/octatope sections reference Bak et al. (2024) correctly.

### Task 4.4: Update `docs/development_status.md`

**Files:**
- Modify: `docs/development_status.md`

**Step 1:** Read the file.

**Step 2:** Update feature inventory to reflect that alpha-hexatope and differentiable solver have been removed.

**Step 3:** Update any roadmap items related to these features.

### Task 4.5: Update `examples/Octatope/README.md`

**Files:**
- Modify: `examples/Octatope/README.md`

**Step 1:** Read the file.

**Step 2:** Remove references to `profile_differentiable_solver.py` and `solver='differentiable'`.

**Step 3:** Update usage examples to show LP and MCF solvers only.

### Task 4.6: Update `examples/README.md`

**Files:**
- Modify: `examples/README.md`

**Step 1:** Read the file.

**Step 2:** Remove AlphaHexatope section.

**Step 3:** Update Octatope section if it references differentiable solver.

---

## Phase 5: Examples & Final Sweep

### Task 5.1: Clean Octatope example scripts

**Files:**
- Modify: `examples/Octatope/compare_tiny.py`
- Modify: `examples/Octatope/compare_small.py`
- Modify: `examples/Octatope/compare_mnist.py`

**Step 1:** Read each file.

**Step 2:** Remove any `solver='differentiable'` references or comparisons.

**Step 3:** Ensure scripts use only `solver='lp'` or `solver='mcf'`.

### Task 5.2: Spot-check ACASXu and vnncomp examples

**Files:**
- Review: `examples/ACASXu/run_instance.py`
- Review: `examples/ACASXu/verify_acasxu.py`
- Review: `examples/vnncomp/prepare_instance.py`
- Review: `examples/vnncomp/run_instance.py`

**Step 1:** Read each file and verify no references to removed functionality.

**Step 2:** Fix any stale imports or API calls.

### Task 5.3: Spot-check remaining examples

**Files:**
- Review: `examples/simple_verification.py`
- Review: `examples/MNIST/` notebooks
- Review: `examples/ProbVer/` scripts

**Step 1:** Read and verify no references to removed functionality.

### Task 5.4: Final grep sweep

**Step 1:** Scan for dead references across entire repo:
```bash
grep -r "alpha_hexatope\|AlphaHexatope\|AlphaOctatope\|differentiable_solver\|DCSOptimizer\|set_device\|HAS_DIFFERENTIABLE_SOLVER\|exact-differentiable" --include="*.py" --include="*.md" .
```

**Step 2:** Scan for absolute paths:
```bash
grep -rn "/home/\|/Users/\|/tmp/" --include="*.py" --include="*.md" . | grep -v third_party | grep -v ".git/"
```

**Step 3:** Scan for sensitive strings:
```bash
grep -rn "api_key\|API_KEY\|secret\|SECRET\|password\|PASSWORD" --include="*.py" . | grep -v third_party | grep -v test
```

**Step 4:** Fix any issues found.

### Task 5.5: Run full test suite — final verification

**Step 1:**
```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```
All tests should pass.

**Step 2:** Verify test count is reasonable (expect ~670 tests, minus the removed differentiable/alpha-hexatope tests).
