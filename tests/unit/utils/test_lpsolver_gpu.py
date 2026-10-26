"""
Unit tests for n2v.utils.lpsolver_gpu that run without CUDA.

  * Profiling never touches CUDA: with profiling active, a CPU-torch solve
    must not call ``torch.cuda.Event`` or ``torch.cuda.synchronize`` (the
    removed GPU-timing hook did, and crashed on CPU-only hosts).
  * Device resolution: ``None`` picks CUDA only when a CUDA device is usable;
    an explicit device is kept.
"""

import numpy as np
import torch

import n2v.profiling
import n2v.utils.lpsolver_gpu as lpg
from n2v.utils.lpsolver_gpu import solve_lp_population_gpu


def _cuda_timing_used(*args, **kwargs):
    raise AssertionError("CUDA timing called on a CPU solve")


def test_profiled_cpu_solve_never_calls_cuda_timing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _cuda_timing_used)
    monkeypatch.setattr(torch.cuda, "synchronize", _cuda_timing_used)
    with n2v.profiling.profile():
        bounds = solve_lp_population_gpu(
            A=[[[1.0, 1.0]]], b=[[1.0]],
            lb=np.zeros((1, 2)), ub=np.ones((1, 2)),
            C=np.eye(2)[None], minimize_flags=[False, False],
            max_iters=50, device="cpu")
    # certified upper bounds on max x_i over {x0 + x1 <= 1, [0, 1]^2}: the
    # true maximum is 1 for each i
    assert bounds.shape == (1, 2)
    assert np.all(bounds >= 1.0 - 1e-12)


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(lpg, "gpu_available", lambda: False)
    assert lpg._resolve_device(None) == "cpu"
    assert lpg._resolve_device("cuda") == "cuda"
