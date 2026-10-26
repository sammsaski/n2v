"""Regressions for ``n2v.utils.falsify``: the counterexample a caller gets back
is the model's batch-1 output at the returned point.

Square scored candidates with one batched forward and returned that batch's
row. A batched float32 forward can differ from the batch-1 forward of the same
point in the last bits, so a point on the boundary of the unsafe region was
returned as SAT although its batch-1 (and onnxruntime) replay is safe. The
models below make the batched rows read lower than the batch-1 output on
purpose, so the disagreement does not depend on the BLAS in use.
"""

import importlib
import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.halfspace import HalfSpace
from n2v.utils import falsify
from n2v.utils.falsify import _falsify_square

# a plain ``import n2v.utils.falsify`` resolves to the re-exported function
fz = importlib.import_module("n2v.utils.falsify")


class _BatchSkew(nn.Module):
    """y = sum(x); rows of a batch larger than one read ``shift`` lower."""

    def __init__(self, shift=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.shift = shift

    def forward(self, x):
        y = self.scale * x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True)
        return y - self.shift if x.shape[0] > 1 else y


def _point_box(x0):
    x0 = np.asarray(x0, dtype=np.float32).astype(np.float64)
    return x0.copy(), x0.copy()


def test_square_rechecks_batched_hit_at_batch_one():
    """Zero-width box at x0 with sum(x0) = 1.5: batch-1 y = 1.5, batched rows
    1.499. Unsafe y <= 1.4995 holds only for the batched rows."""
    lb, ub = _point_box([0.5, 1.0])
    hs = HalfSpace(np.array([[1.0]]), np.array([[1.4995]]))
    model = _BatchSkew()
    assert _falsify_square(model, lb, ub, hs, n_iters=200, batch=16,
                           seed=0) == (2, None)
    assert falsify(model, lb, ub, hs, method='square', n_iters=200, batch=16,
                   seed=0) == (2, None)


def test_square_returns_batch_one_output():
    lb, ub = np.zeros(2), np.ones(2)
    hs = HalfSpace(np.array([[1.0]]), np.array([[0.2]]))    # unsafe: y <= 0.2
    res, (x, y) = _falsify_square(_BatchSkew(), lb, ub, hs, n_iters=2000,
                                  batch=16, seed=0)
    assert res == 0
    assert y[0] == pytest.approx(np.float32(x).sum(), abs=1e-6)  # batch 1
    assert y[0] <= 0.2


def test_falsify_rechecks_returned_witness_at_batch_one(monkeypatch):
    """An attack's reported output is not trusted: the wrapper recomputes it
    at the returned point even when the point is already inside the box."""
    lb, ub = np.zeros(2), np.ones(2)
    x_in = np.array([0.5, 0.25])                          # sum = 0.75
    monkeypatch.setattr(fz, "_falsify_impl",
                        lambda *a, **k: (0, (x_in, np.array([0.1]))))
    unsafe_low = HalfSpace(np.array([[1.0]]), np.array([[0.5]]))
    assert falsify(_BatchSkew(), lb, ub, unsafe_low) == (2, None)
    unsafe_high = HalfSpace(np.array([[-1.0]]), np.array([[-0.5]]))
    res, (x, y) = falsify(_BatchSkew(), lb, ub, unsafe_high)
    assert res == 0 and np.array_equal(x, x_in) and y[0] == pytest.approx(0.75)


def test_falsify_signature_and_docs_are_public():
    """Autodoc shows the real parameters and docstring, not *args/**kwargs."""
    params = inspect.signature(falsify).parameters
    assert list(params)[:6] == ['model', 'lb', 'ub', 'property', 'method',
                                'seed']
    assert params['method'].default == 'random'
    assert falsify.__name__ == 'falsify'
    assert 'Attempt to find a counterexample' in falsify.__doc__
