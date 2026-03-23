# Repository Cleanup and Preparation Design

**Date:** 2026-03-23
**Goal:** Remove experimental code, enforce code style, update documentation, and prepare repository for remote push.

## 1. Problem Statement

The n2v repository has accumulated experimental code (alpha-hexatope optimization, differentiable solver) that is not showing promise and should be removed. The codebase also needs style enforcement per the project's research-code-dev conventions, documentation updates, and a final sweep for sensitive content before pushing to remote.

## 2. Scope

### In Scope
- Remove alpha-hexatope module, differentiable solver, and all references
- Keep hexatope/octatope set classes and their dispatcher integration (LP/MCF solvers)
- Apply research-code-dev.md style conventions to `n2v/` library code
- Convert print statements to logging in library code (not examples)
- Update all documentation to reflect changes
- Verify examples work with current API
- Scan for absolute paths, secrets, dead imports

### Out of Scope
- `third_party/onnx2torch` (submodule, do not modify)
- `tests/` (no style enforcement — tests are internal)
- `examples/` (print statements acceptable, light touch only)

## 3. Phase 1: Experimental Code Removal

### Delete Entirely
- `n2v/alpha_hexatope/` (8 files)
- `n2v/utils/differentiable_solver.py`
- `examples/AlphaHexatope/` (entire directory)
- `examples/Octatope/profile_differentiable_solver.py`
- `tests/unit/alpha_hexatope/` (7 test files)
- `tests/soundness/test_soundness_differentiable.py`
- `tests/unit/utils/test_differentiable_solver_v2.py`
- `docs/differentiable_solver.md`
- All `docs/plans/` files related to hexatope/differentiable solver

### Surgical Edits
- `n2v/__init__.py` — remove `alpha_hexatope` import/export
- `n2v/sets/hexatope.py` — remove differentiable solver imports, methods, branches; update solver validation to `('lp', 'mcf')`
- `n2v/sets/octatope.py` — same as hexatope
- `n2v/utils/lpsolver.py` — remove `solve_dcs_differentiable()`
- `n2v/nn/neural_network.py` — remove `exact-differentiable` method, alpha-hexatope verify path
- `n2v/config.py` — remove `device` property and `set_device()` (only used by differentiable solver)

### Keep Intact
- Hexatope/Octatope set classes with LP and MCF solver backends
- Dispatcher integration (`_reach_layer_hexatope`, `_reach_layer_octatope`)
- Layer ops: `linear_hexatope/octatope`, `relu_hexatope/octatope`, `flatten_hexatope/octatope`
- All non-differentiable tests for hexatope/octatope

## 4. Phase 2: Hexatope/Octatope Paper Verification

Confirm implementation aligns with Bak et al. (2024) "The hexatope and octatope abstract domains for neural network verification" (FMSD 64:178-199).

Pre-audit finding: **98% alignment**. All theorems (3-7), Algorithm 5.1, and mathematical definitions match. Only divergence was the differentiable solver extension (being removed).

After removal, verify MCF solver still works end-to-end via existing tests.

## 5. Phase 3: Code Style Enforcement

**Scope:** `n2v/` library code only.

### Print → Logging (33 statements, 9 files)
- Each file gets `logger = logging.getLogger(__name__)`
- Progress/status → `logger.info()`, errors → `logger.warning()`/`logger.error()`
- Verbose iteration output → `logger.debug()`
- Heaviest files: `probabilistic/verify.py` (15), `nn/reach.py` (6)

### Type Hints (~80 missing)
- Public API functions first, then internal helpers
- Biggest gaps: `config.py`, `reach.py`, `bounds_precomputation.py`, `reduce_reach.py`

### Docstrings (~18 missing)
- Google-style with Args/Returns/Raises
- Paper references where applicable
- Biggest gaps: `hexatope.py`, `octatope.py`, `image_star.py`

### Already Clean (no changes needed)
- Imports at top of file
- `pathlib.Path` used consistently
- `try/except` blocks justified

## 6. Phase 4: Documentation Update

### Update
- `README.md` — remove alpha-hexatope, differentiable solver, `set_device()` references
- `CLAUDE.md` — update exports, config functions, layer support table
- `docs/theoretical_foundations.md` — trim differentiable solver sections
- `docs/development_status.md` — update feature inventory

### Delete
- `docs/differentiable_solver.md`
- All 5 plan files in `docs/plans/` related to hexatope/differentiable solver

## 7. Phase 5: Examples & Final Sweep

### Examples
- `examples/Octatope/` — remove `solver='differentiable'` references from remaining scripts
- `examples/ACASXu/`, `examples/vnncomp/` — spot-check API consistency
- `examples/MNIST/` — check notebooks still run
- `examples/simple_verification.py` — sanity check

### Final Sweep
- Grep for dead references: `alpha_hexatope`, `differentiable_solver`, `DCSOptimizer`, `set_device`, `AlphaHexatope`, `AlphaOctatope`
- Scan for absolute paths, sensitive strings, dead imports
- Run full test suite
