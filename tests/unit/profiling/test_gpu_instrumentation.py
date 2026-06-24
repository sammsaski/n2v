"""GPU-path profiler instrumentation (PT-2 batched-GPU LP).

Pins that device time is captured via CUDA events and surfaced on the Record /
rollup / exports, that ``device`` is tagged, and that the GPU LP solve counts
``n_gpu_lp_solves``. Skipped without CUDA. The CPU profiler invariants
(no-op-when-disabled, non-interference) are covered elsewhere; here we only add
the GPU-specific surface.
"""

import numpy as np
import pytest

from n2v.profiling import add_gpu_time, profile, region

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def test_add_gpu_time_is_noop_when_disabled():
    """No active profile -> add_gpu_time does nothing and never raises."""
    add_gpu_time(1.23)  # must not raise


def test_add_gpu_time_accumulates_on_current_region():
    """Reported device time lands on the open region and rolls up (no torch)."""
    with profile(level="operation") as p:
        with region("k"):
            add_gpu_time(0.5)
            add_gpu_time(0.25)
    rec = p.find("k")[0]
    assert rec.gpu_time == pytest.approx(0.75)
    assert rec.device == "cuda"
    assert p.rollup()["gpu_time"] == pytest.approx(0.75)


def test_gpu_time_in_exports():
    with profile(level="operation") as p:
        with region("k"):
            add_gpu_time(0.1)
    d = p.to_dict()
    assert d["root"]["children"][0]["gpu_time"] == pytest.approx(0.1)
    assert d["root"]["children"][0]["device"] == "cuda"
    header = p.to_csv().splitlines()[0]
    assert "gpu_time" in header and "device" in header


@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
def test_gpu_lp_solve_records_device_time_and_counter():
    from n2v.sets import Star

    rng = np.random.default_rng(0)
    V = rng.standard_normal((10, 5))
    C = rng.standard_normal((3, 4))
    d = (np.abs(C).sum(1) + rng.uniform(1, 2, 3)).reshape(-1, 1)
    plb = rng.uniform(-1, 0, (4, 1))
    pub = plb + rng.uniform(0.5, 2, (4, 1))
    s = Star(V, C, d, plb, pub)

    import n2v
    n2v.set_gpu_lp(True)
    try:
        with profile(level="operation") as p:
            s.get_ranges()
    finally:
        n2v.set_gpu_lp(False)

    ro = p.rollup()
    assert ro["gpu_time"] > 0.0
    assert ro["totals"].get("n_gpu_lp_solves") == 2 * s.dim
    solve = p.find("pdhg_solve")[0]
    assert solve.gpu_time > 0.0
    assert solve.device == "cuda"
