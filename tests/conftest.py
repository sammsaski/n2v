"""Top-level pytest configuration shared across every test subdirectory.

The set-validity assertion helpers are monkeypatched onto the ``pytest``
module so tests can call ``pytest.assert_star_valid(...)`` etc. They live
here (the rootdir conftest) rather than only under ``tests/unit/`` so they
are available to EVERY invocation — ``pytest tests/unit/``,
``pytest tests/integration/``, and ``pytest tests/soundness/`` alike.

CI runs those three directories as separate ``pytest`` commands, so a
helper registered only under ``tests/unit/`` is absent when
``tests/integration/`` runs on its own (a full ``pytest tests/`` run
happens to work because ``tests/unit/conftest.py`` loads first and sets
the attribute process-wide). Registering here closes that gap.

NOTE: ``tests/unit/conftest.py`` (and the ``layer_ops`` / ``sets`` unit
conftests) still define their own copies of these helpers. Those are now
redundant with this file; consolidating them is a safe follow-up cleanup.

This file also holds the suite-wide infrastructure:

* two autouse fixtures that keep tests independent of each other: one
  snapshots ``n2v.config`` and fails a test that changed it without restoring
  it, the other seeds the global RNGs from the test id and restores their
  state afterwards;
* the ``engine_device`` fixture that runs the certified GPU bound engine on
  CPU torch (the path CI takes) and, where a CUDA device exists, on CUDA;
* a report hook that turns a skip into a failure when the ``gpu`` or
  ``corpus`` tier was selected explicitly with ``-m`` but the resource is
  missing, so a requested tier can never pass vacuously.
"""

import random
import re
import zlib

import numpy as np
import pytest
import torch

from n2v.config import config as _global_config


def assert_star_valid(star):
    """Assert that a Star set is valid."""
    assert star.V is not None
    assert star.C is not None
    assert star.d is not None
    assert star.V.shape[1] == star.nVar + 1
    assert star.C.shape[1] == star.nVar


def assert_zono_valid(zono):
    """Assert that a Zonotope is valid."""
    assert zono.c is not None
    assert zono.V is not None
    assert len(zono.c.shape) == 2
    assert zono.c.shape[1] == 1


def assert_image_star_valid(img_star):
    """Assert that an ImageStar is valid."""
    # ImageStar has 4D V: (H, W, C, nVar+1), so we check differently than Star
    assert img_star.V is not None
    assert img_star.C is not None
    assert img_star.d is not None
    assert img_star.V.ndim == 4, f"ImageStar V should be 4D, got {img_star.V.ndim}D"
    assert img_star.V.shape[3] == img_star.nVar + 1
    assert img_star.C.shape[1] == img_star.nVar
    assert img_star.height > 0
    assert img_star.width > 0
    assert img_star.num_channels > 0
    assert img_star.V.shape[0] == img_star.height
    assert img_star.V.shape[1] == img_star.width
    assert img_star.V.shape[2] == img_star.num_channels
    assert img_star.dim == img_star.height * img_star.width * img_star.num_channels


def assert_hexatope_valid(hexatope):
    """Assert that a Hexatope is valid."""
    assert hexatope.center is not None
    assert hexatope.generators is not None
    assert hexatope.dcs is not None
    assert hexatope.generators.shape[0] == hexatope.dim
    assert hexatope.generators.shape[1] == hexatope.nVar


def assert_octatope_valid(octatope):
    """Assert that an Octatope is valid."""
    assert octatope.center is not None
    assert octatope.generators is not None
    assert octatope.utvpi is not None
    assert octatope.generators.shape[0] == octatope.dim
    assert octatope.generators.shape[1] == octatope.nVar


# Make helpers available via the pytest module namespace.
pytest.assert_star_valid = assert_star_valid
pytest.assert_zono_valid = assert_zono_valid
pytest.assert_image_star_valid = assert_image_star_valid
pytest.assert_hexatope_valid = assert_hexatope_valid
pytest.assert_octatope_valid = assert_octatope_valid


# ---------------------------------------------------------------------------
# Opt-in tiers: an explicitly requested tier must not pass by skipping
# ---------------------------------------------------------------------------

_TIER_MARKERS = ("gpu", "corpus")


def pytest_configure(config):
    expr = config.getoption("markexpr") or ""
    config._n2v_explicit_tiers = {
        name for name in _TIER_MARKERS
        if re.search(rf"(?<!not )\b{name}\b", expr)}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if not rep.skipped or hasattr(rep, "wasxfail"):
        return
    explicit = getattr(item.config, "_n2v_explicit_tiers", set())
    hit = [m for m in _TIER_MARKERS
           if m in explicit and item.get_closest_marker(m) is not None]
    if hit:
        reason = (rep.longrepr[2] if isinstance(rep.longrepr, tuple)
                  else str(rep.longrepr))
        rep.outcome = "failed"
        rep.longrepr = (
            f"the {hit} tier was selected explicitly with -m, but the test "
            f"skipped ({reason}). A missing resource is an error when its tier "
            f"is requested; run without that -m to deselect the tier instead.")


# ---------------------------------------------------------------------------
# Test independence: global config and global RNGs
# ---------------------------------------------------------------------------

def _config_snapshot():
    c = _global_config
    return {"parallel_lp": c.parallel_lp, "n_workers": c.n_workers,
            "auto_parallel": c.auto_parallel,
            "parallel_threshold": c.parallel_threshold,
            "lp_solver": c.lp_solver}


def _config_restore(snap):
    c = _global_config
    c.parallel_lp = snap["parallel_lp"]
    c.n_workers = snap["n_workers"]
    c.auto_parallel = snap["auto_parallel"]
    c.parallel_threshold = snap["parallel_threshold"]
    c.lp_solver = snap["lp_solver"]


@pytest.fixture(autouse=True)
def _isolate_global_config():
    """Fail a test that leaves ``n2v.config`` changed, and restore it anyway.

    ``set_parallel`` and ``set_lp_solver`` are process-wide. A test (or the
    code it calls) that switches them on and does not switch them back changes
    how every later test solves its LPs, which made at least one regression
    guard's verdict depend on test order (MUTATIONS.md, row C9). A test that
    needs a different configuration sets it and restores it itself
    (``try/finally`` or ``config.reset()`` in a teardown)."""
    before = _config_snapshot()
    yield
    after = _config_snapshot()
    _config_restore(before)
    changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    if changed:
        pytest.fail(
            f"the test changed n2v.config without restoring it: {changed}",
            pytrace=False)


@pytest.fixture(autouse=True)
def _seed_global_rngs(request):
    """Seed ``random``, ``numpy`` and ``torch`` from the test id, then restore
    their previous state, so a test's random draws depend on nothing but the
    test itself and reseeding inside a test leaks nowhere."""
    seed = zlib.crc32(request.node.nodeid.encode()) & 0x7FFFFFFF
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


# ---------------------------------------------------------------------------
# The certified GPU bound engine on CPU torch and on CUDA
# ---------------------------------------------------------------------------

def _cuda_usable():
    try:
        from n2v.utils.lpsolver_gpu import gpu_available
        return bool(gpu_available())
    except Exception:  # noqa: BLE001 - no torch, no CUDA
        return False


needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no CUDA device")


@pytest.fixture(params=[
    pytest.param("cpu", id="cpu-torch"),
    pytest.param("cuda", id="cuda", marks=[pytest.mark.gpu, needs_cuda]),
])
def engine_device(request, monkeypatch):
    """Run the certified GPU bound engine (``dual_gpu`` bounds, prune and
    witnesses) on CPU torch and, where a CUDA device exists, on CUDA.

    ``device=None`` resolves through ``n2v.utils.lpsolver_gpu.gpu_available``,
    so patching it to False on a CUDA host forces the CPU-torch path that a
    CPU-only CI takes. Soundness guards of that engine must use this fixture
    rather than a CUDA-only skip: the engine is the production default on
    CUDA hosts and must be exercised on every host."""
    if request.param == "cpu":
        monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)
    return request.param
