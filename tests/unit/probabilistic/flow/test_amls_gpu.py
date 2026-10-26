"""Tests for GPU acceleration of AMLS.

These tests verify that AMLS produces equivalent results on CPU and GPU
(when CUDA is available), and that the device autodetection / explicit
override path is wired through consistently.

The tests are skipped automatically when CUDA is not available; the
``test_amls_default_device_autodetects`` test runs on either backend
since it exercises only the autodetection logic.
"""
from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.amls import (
    amls_estimate_halfspace_mass,
    amls_certify_spec,
)
from n2v.sets.halfspace import HalfSpace


_CUDA_AVAILABLE = torch.cuda.is_available()


# ---------- Identity flow stub (mirrors test_amls.py) ----------


class _IdentityFlow(torch.nn.Module):
    """Trivial 'flow' whose inverse is the identity, but device-aware.

    Wraps as nn.Module so .to(device) works, and respects input device
    on inverse() so we can detect missed device threading.
    """

    def __init__(self):
        super().__init__()
        # A registered buffer follows .to(device) so we can introspect
        # which device the flow currently lives on.
        self.register_buffer('_marker', torch.zeros(1))
        self.velocity_field = None

    def inverse(self, z, **_kw):
        # Echo input device so the caller's device is preserved.
        return z

    def forward(self, y, **_kw):
        return y


# ---------- GPU vs CPU equivalence ----------


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason='CUDA not available')
def test_amls_gpu_smoke_runs_and_returns_result():
    """Smoke: GPU run completes and returns a valid AMLSResult."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=3,
        seed=0, device='cuda',
    )
    assert res.detected_unsafe is True
    assert res.levels_used >= 1
    assert isinstance(res.worst_y, np.ndarray)
    assert res.worst_y.shape == (2,)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason='CUDA not available')
def test_amls_gpu_matches_cpu_on_easy_unsafe():
    """AMLS on GPU yields the same detected/levels as CPU for an easy case.

    'Easy' means level 0 already detects ``U`` so MCMC ordering does
    not influence the verdict.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    r_cpu = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=300, max_levels=5, n_mcmc_steps=3,
        seed=0, device='cpu',
    )
    r_gpu = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=300, max_levels=5, n_mcmc_steps=3,
        seed=0, device='cuda',
    )
    assert r_cpu.detected_unsafe == r_gpu.detected_unsafe
    assert r_cpu.levels_used == r_gpu.levels_used
    # On the easy case, both should detect at level 0 so pi_hat is the
    # empirical bulk fraction; with the SAME numpy seed both backends
    # draw the same z, so pi_hat is bit-equal.
    assert r_cpu.pi_hat == pytest.approx(r_gpu.pi_hat, rel=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason='CUDA not available')
def test_amls_gpu_matches_cpu_on_undetectable():
    """AMLS on GPU agrees with CPU when ``U`` is unreachable (no detect)."""
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    r_cpu = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=5,
        seed=0, device='cpu',
    )
    r_gpu = amls_estimate_halfspace_mass(
        flow, hs,
        n_samples_per_level=200, max_levels=5, n_mcmc_steps=5,
        seed=0, device='cuda',
    )
    assert r_cpu.detected_unsafe == r_gpu.detected_unsafe
    assert r_cpu.detected_unsafe is False
    # For an unreachable target, both runs exhaust max_levels.
    assert r_cpu.levels_used == r_gpu.levels_used


# ---------- Default device autodetection ----------


def test_amls_default_device_autodetects_to_available_backend():
    """When ``device`` is unspecified, AMLS picks GPU iff CUDA available.

    Sentinel: monkeypatch ``torch.tensor.to`` is too invasive; we instead
    monkeypatch ``torch.cuda.is_available`` and observe that the flow
    object's marker buffer is moved to the expected device.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    # Force CPU branch.
    with mock.patch('torch.cuda.is_available', return_value=False):
        amls_estimate_halfspace_mass(
            flow, hs,
            n_samples_per_level=50, max_levels=2, n_mcmc_steps=1,
            seed=0,  # device unspecified -> autodetect
        )
        assert flow._marker.device.type == 'cpu'

    if _CUDA_AVAILABLE:
        flow_gpu = _IdentityFlow()
        # Force GPU branch.
        with mock.patch('torch.cuda.is_available', return_value=True):
            amls_estimate_halfspace_mass(
                flow_gpu, hs,
                n_samples_per_level=50, max_levels=2, n_mcmc_steps=1,
                seed=0,  # device unspecified -> autodetect
            )
            assert flow_gpu._marker.device.type == 'cuda'


@pytest.mark.gpu
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason='CUDA not available')
def test_amls_certify_spec_threads_device_kwarg():
    """``amls_certify_spec`` should accept ``device`` and thread it into
    each per-HalfSpace AMLS call.
    """
    flow = _IdentityFlow()
    hs = HalfSpace(
        np.array([[-1.0, 0.0]]), np.array([[10.0]]),
    )
    res = amls_certify_spec(
        flow, [[hs]],
        n_samples_per_level=50, max_levels=3, n_mcmc_steps=1,
        seed=0, device='cuda',
    )
    assert res.detected_any is True
    # The flow buffer should have been moved to cuda.
    assert flow._marker.device.type == 'cuda'
