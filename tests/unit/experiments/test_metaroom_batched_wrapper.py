"""Soundness check for the ``_GenericONNXWrapper`` batched-inference fix.

Some VNN-COMP ONNX exports (notably ``metaroom_2023``) embed a literal
``Reshape`` op whose target shape is ``(1, -1)``, which collapses the
batch dim. Before the fix, the wrapper detected this and fell back to a
per-sample loop, making the wrapper an O(N) bottleneck on every flow
training / scenario forward.

The fix patches the offending ``OnnxConstant`` from ``(1, -1)`` to
``(-1, K)`` (where K is the per-sample flat size measured during a
batch=1 probe). This test verifies:

  1. Metaroom: the patch eliminates ``batch_loop`` (fast path);
  2. Outputs at batch=1 are bit-exact identical to the unpatched model;
  3. Outputs at batch>1 match the per-sample loop oracle to float32
     precision (rel diff <= 1e-5; the only source of disagreement is
     conv accumulation order, which is float32-noise level).

Skipped if the metaroom ONNX file is not on disk.

------------------------------------------------------------------------
INFRASTRUCTURE DEPENDENCY (read before running in CI):
------------------------------------------------------------------------
This test requires the VNN-COMP 2025 benchmark repo to be present at::

    ~/v/other/VNNCOMP/vnncomp2025_benchmarks/

specifically the metaroom_2023 ONNX file referenced by ``_METAROOM_GZ``
below. The ``metaroom_onnx_path`` fixture (see below) calls
``pytest.skip()`` when this file is absent, so on machines without the
VNN-COMP repo every test in this module silently skips. This is
intentional -- we don't want the test suite to crash on developer boxes
that don't have the (large) external benchmark data -- but it also
means CI gets no signal from this file unless the repo is mounted.

This is currently the ONLY test in the repository that depends on real
VNN-COMP benchmark data. Every other VNN-COMP-flavored test (e.g.
``tests/integration/test_vnncomp_runner.py``) builds synthetic ONNX
inside ``tmp_path`` and is fully self-contained.

To exercise this test in CI, the VNN-COMP 2025 benchmark repo would
need to be mounted (or at least the single ``metaroom_2023`` ONNX
checked into a fixtures directory) so that ``_METAROOM_GZ`` resolves.
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest
import torch

# External benchmark data: the opt-in ``corpus`` tier. When that tier is
# selected explicitly, a missing file fails instead of skipping.
pytestmark = pytest.mark.corpus

# Path to a representative metaroom ONNX (.gz). If absent, we skip.
_METAROOM_GZ = Path.home() / (
    'v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/metaroom_2023/'
    'onnx/6cnn_tz_35_5_no_custom_OP.onnx.gz'
)
_METAROOM_FLAT_DIM = 5376  # 3 * 32 * 56


@pytest.fixture(scope='module')
def metaroom_onnx_path(tmp_path_factory):
    """Decompress the metaroom ONNX once per module; skip if missing."""
    if not _METAROOM_GZ.exists():
        pytest.skip(f'metaroom ONNX not found at {_METAROOM_GZ}')
    out = tmp_path_factory.mktemp('metaroom') / 'model.onnx'
    with gzip.open(_METAROOM_GZ, 'rb') as f, open(out, 'wb') as g:
        shutil.copyfileobj(f, g)
    return out


def test_patch_eliminates_batch_loop(metaroom_onnx_path):
    """``_detect_input_shape`` should report ``batch_loop=False`` after
    the reshape constant is rewritten."""
    from n2v.utils.model_loader import load_onnx
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        _detect_input_shape,
    )

    model = load_onnx(str(metaroom_onnx_path))
    shape, batch_loop, _ = _detect_input_shape(model, _METAROOM_FLAT_DIM)
    assert shape == (3, 32, 56)
    assert batch_loop is False, (
        'The reshape patch should have made the inner module batch-friendly.')


def test_batch_one_bit_exact(metaroom_onnx_path):
    """Batch=1 forward through the patched model must match the
    unpatched model bit-exactly (no floating point change at batch=1)."""
    from n2v.utils.model_loader import load_onnx
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        _GenericONNXWrapper, _detect_input_shape,
    )

    m_unpatched = load_onnx(str(metaroom_onnx_path))

    m_patched = load_onnx(str(metaroom_onnx_path))
    shape, batch_loop, _ = _detect_input_shape(m_patched, _METAROOM_FLAT_DIM)
    assert batch_loop is False
    w_patched = _GenericONNXWrapper(m_patched, input_shape=shape, batch_loop=False)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 56)
    with torch.no_grad():
        y_unpatched = m_unpatched(x)
        y_patched = w_patched(x)
    assert torch.equal(y_unpatched, y_patched), (
        f'Batch=1 should be bit-exact identical; max diff '
        f'{(y_unpatched - y_patched).abs().max()}')


@pytest.mark.parametrize('N', [2, 5, 17, 32])
def test_batched_matches_per_sample_loop(metaroom_onnx_path, N):
    """Batch>1 forward through the patched wrapper must match the
    per-sample loop oracle to float32 precision (rel diff <= 1e-5).

    The only source of any non-zero diff is conv accumulation order at
    batch>1 vs batch=1, which is bounded by float32 ULP × #flops. We
    cap relative diff at 1e-5 to leave generous margin while still
    catching any algorithmic regression.
    """
    from n2v.utils.model_loader import load_onnx
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        _GenericONNXWrapper, _detect_input_shape,
    )

    # Oracle: unpatched model, forced batch_loop wrapper.
    m_oracle = load_onnx(str(metaroom_onnx_path))
    w_oracle = _GenericONNXWrapper(m_oracle, input_shape=(3, 32, 56),
                                    batch_loop=True)

    # Patched: detect_input_shape rewrites reshape consts; wrapper batches.
    m_new = load_onnx(str(metaroom_onnx_path))
    shape, batch_loop, _ = _detect_input_shape(m_new, _METAROOM_FLAT_DIM)
    assert batch_loop is False
    w_new = _GenericONNXWrapper(m_new, input_shape=shape, batch_loop=False)

    torch.manual_seed(123 + N)
    x_flat = torch.randn(N, _METAROOM_FLAT_DIM)
    with torch.no_grad():
        y_oracle = w_oracle(x_flat)
        y_new = w_new(x_flat)

    assert y_oracle.shape == (N, 20)
    assert y_new.shape == (N, 20)

    abs_diff = (y_oracle - y_new).abs().max().item()
    mag = y_oracle.abs().max().item()
    rel = abs_diff / max(mag, 1e-12)
    # 1e-5 is comfortably above observed conv-batching ULP (~1e-7) but
    # tight enough to flag any real soundness regression.
    assert rel < 1e-5, (
        f'Patched batched output diverges from per-sample oracle: '
        f'N={N} max_abs_diff={abs_diff:.3e} mag={mag:.3e} rel={rel:.3e}')
