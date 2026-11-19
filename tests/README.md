# n2v test suite

The tests exist to fail when n2v is wrong. A test earns its place by naming a bug it would catch that no other test catches; a test that cannot fail, that pins a known wrong answer, or that was loosened to pass is a bad test. Tolerance is never added in the unsound direction.

## Layout

| Directory | What it checks | Oracle |
|---|---|---|
| `contracts/` | the public promises: import layering, reach coverage, the set protocol, specs, the verify entry, verdicts, the probabilistic guarantees, outward rounding of smooth operations | the design document, read with `ast` or exercised through public names only; values are checked by the oracles below |
| `_oracles/` | not n2v: the three oracles the tests are checked against, each with a self-test that plants bugs in a correct enclosure and watches the oracle reject them | `rational` (exact image and membership in `fractions.Fraction`), `interval_proof` (containment proof for smooth operations, mpmath intervals), `membership` (one LP on the set's own rows; it reads Star, ImageStar, Zono, ImageZono, Box, Hexatope with its anchor pinned to 0 and Octatope, and solves in scaled variables, rechecking the solver's point, so a predicate box as narrow as the solver tolerance stays reliable) |
| `soundness/` | enclosure soundness of every reach kernel, pre-pass, certificate and spec check, plus witness validity; guards for fixed soundness bugs, one file per component | onnxruntime samples, exact arithmetic (`fractions.Fraction`, mpmath) and LP membership against the constraint rows, never the code's own helpers |
| `unit/` | internal behaviour that nothing above reaches, light touch | small analytic cases |
| `integration/` | small real models end to end, the competition runner | onnxruntime replay |
| `differential/` | the VNN-LIB parsers against the external corpus, reference parsers and a golden snapshot | the corpus (`corpus` tier) |

`cell_support.py`, beside `known_bugs.py`, holds the shared parts of a soundness cell (input sets of every type, sample points, ONNX and torch reference models with their replays, reach wrappers that refuse an empty result and a reach that changed its input set, the coverage assertion) and is self-tested by `test_cell_support.py`.

A failing verdict path is a bug to fix, or a strict `xfail` naming the step of the plan that fixes it (`xfail_strict = true`: a fixed bug fails the test until its marker is removed). The markers are in `known_bugs.py`: `known_bug(step, defect)` for a defect a plan step fixes, `deferred_bug(number, defect)` for one the owner deferred. Every marked case belongs to a row of the defect registry (`.claude/research/34-API-Redesign/test-audit/DEFECTS.md`), which states the defect, the cases that show it and the proposed fix; the change that fixes a defect removes its markers and updates its row. Each marker names the one exception the test may fail with today, so the marker absorbs the documented failure and nothing else; a precondition inside such a test uses `pytest.fail`, which the marker does not absorb, never `assert`.

## Tiers and markers

The default run takes about 35 minutes on one process with four threads:

```bash
pytest                 # the default tier: everything not marked slow, gpu or corpus
pytest -m slow         # long flow training, large Monte Carlo sweeps, the 300-sample softmax cell
pytest -m gpu          # tests that need a CUDA device
pytest -m corpus       # tests that need external data: the VNN-COMP corpus, reference checkouts
pytest -m ""           # everything
```

Selecting `gpu` or `corpus` explicitly turns a missing-resource skip into a failure, so a requested tier can never pass vacuously. The certified GPU bound engine is deliberately not behind the `gpu` marker: its soundness guards take the `engine_device` fixture and run on CPU torch everywhere, and on CUDA where it exists.

Warnings are errors. Each `ignore` in `pytest.ini` is message-based, names what emits it and, for n2v's own warnings, the plan step that removes it.

## Test independence

Two autouse fixtures in `conftest.py` keep tests independent of their order:

- `n2v.config` (parallel LP, worker count, LP solver) is snapshotted before each test and restored after it; a test that changed it without restoring it fails. A test that needs another configuration sets it and resets it itself (`config.reset()` in a teardown, or `try/finally`).
- `random`, `numpy` and `torch` are seeded from the test id and their state is restored afterwards, so reseeding inside a test leaks nowhere.

## Running

```bash
pytest tests/soundness/test_soundness_relu_cells.py    # one file
pytest -k "relu and exact"                             # by name
pytest --durations=50                                  # the slow end of the tier
pytest -m slow tests/unit/probabilistic/flow           # one tier in one directory
```

The default tier needs `onnx`, `onnxruntime` (pinned to the VNN-COMP grading version in `pyproject.toml`), `torch`, `cvxpy`, `highspy` and the `third_party/onnx2torch` editable install; a missing one fails the suite rather than skipping it. The optional gates that remain are for genuinely optional packages (`plotly`, `scikit-image`, `beartype`, the official `vnnlib` package).

## Writing a test

- Say in the docstring which bug the test catches. If you cannot, the test is probably not needed.
- Soundness is one-sided: the computed set must contain the true value, with zero slack against exact references, a pad of the `1e-12` class against float64 references and a derived per-cell bound against float32 onnxruntime samples. Never a two-sided `allclose` on a bound, never slack that forgives an inward error.
- Check membership against the constraint rows (`Star.contains(method="lp")` or the exact oracle), not the bounding box: a wrong row leaves the box unchanged.
- Use inputs that can trigger the bug: asymmetric image shapes such as `(2, 3, 5)` with more than one channel, non-uniform bounds, a star with a binding non-box row, inputs on both sides of every kink.
- Prefer a deterministic injected fault (a stubbed solver, a patched bound) to a search for a host where the fault happens to occur; a test must not skip when the host is well behaved.
- A missing tracked asset is a broken checkout: raise, never skip. Skip only for the optional packages above, and mark external data `corpus` and CUDA `gpu`.
- Before accepting a new soundness test, break the kernel once (drop a row, flip a sign, reverse a split) and watch the test fail.

### A soundness cell

A soundness cell is one (operation, set type, method, option), checked by a test built from the helpers in `cell_support.py`; read its module docstring first, since its layout convention (a tensor is channel, height, width; an image set stores height, width, channel; a flat set keeps the tensor's own order) is the one every cell follows. A cell has these parts:

1. An independent oracle: the exact image in rationals (`_oracles/rational.py`) for piecewise-linear operations, the containment proof (`_oracles/interval_proof.py`) for smooth ones, and on every cell sample points replayed on onnxruntime or on torch in float64 (`replay_onnx`, `replay_torch`) whose membership the LP oracle decides (`assert_covered`). Never n2v's own helpers, never two n2v paths compared with each other as the verdict.
2. Inputs that can trigger the bug: an image with more than one channel and height different from width, such as (2, 3, 5), with its own interval per pixel (`pixel_box`); coordinates on both sides of every kink and one of zero width (`flat_box`); a Star with a binding row that is not a box row (`constrained_star`, `constrained_image_star`); several input sets at once (`split_cover`, `image_split_cover`); asymmetric kernels, strides, pads and dilations.
3. Membership against the set's own rows, one-sided, with an image output converted to the stored order explicitly (`assert_covered(..., out_chw=(C, H, W))`): no slack against exact references, a pad of the `1e-12` class against float64 ones, `LP_TOL` for the LP check, never a two-sided `allclose` on a bound.
4. Composition: affine, then the operation, then affine, checked the same way, and through the model entry point (`reach_model`) when the option is only reachable there.
5. Exactness where the method claims it, in both directions: no true output outside the result, and no vertex of a returned set without a preimage.
6. Refusal: a combination the library does not support must raise `NotImplementedError`, and the test asserts the raise; `reach_model` and `reach_one_layer` never catch it, and fail on an empty result. They also fail, through `pytest.fail` so that no marker absorbs it, when the reach changed an input set in anything but the ranges the set caches about itself (`state_lb`, `state_ub`): a kernel that works on its input's arrays in place answers correctly once and wrongly for the second reader of the same set, which in a graph is the other branch of a residual connection.
7. A docstring that names the bug the test catches.
