# n2v test suite

The tests exist to fail when n2v is wrong. A test earns its place by naming a bug it would catch that no other test catches; a test that cannot fail, that pins a known wrong answer, or that was loosened to pass is a bad test. Tolerance is never added in the unsound direction.

## Layout

| Directory | What it checks | Oracle |
|---|---|---|
| `contracts/` | the public promises: import layering today; the reach coverage, set protocol, spec, verify and witness contracts as the API redesign lands | the design document, read with `ast` or exercised through public names only |
| `soundness/` | enclosure soundness of every reach kernel, pre-pass, certificate and spec check, plus witness validity; guards for fixed soundness bugs, one file per component | onnxruntime samples, exact arithmetic (`fractions.Fraction`, mpmath) and LP membership against the constraint rows, never the code's own helpers |
| `unit/` | internal behaviour that nothing above reaches, light touch | small analytic cases |
| `integration/` | small real models end to end, the competition runner | onnxruntime replay |
| `differential/` | the VNN-LIB parsers against the external corpus, reference parsers and a golden snapshot | the corpus (`corpus` tier) |

A failing verdict path is a bug to fix, or a strict `xfail` naming the step of the plan that fixes it (`xfail_strict = true`: a fixed bug fails the test until its marker is removed).

## Tiers and markers

The default run is the fast tier, about four and a half minutes on one process:

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
pytest tests/soundness/test_soundness_relu.py          # one file
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
