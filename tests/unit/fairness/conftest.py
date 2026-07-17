"""Shared fixtures + helpers for the group-fairness test suite.

The code under test lives in ``examples/FairN2V/group_fairness/`` (put on
``sys.path`` below, following the pattern of the other example-code tests
in this suite, e.g. ``tests/unit/test_prepare_instance.py``).

The suite is split into two soundness layers (see README.md in this directory):

  * the statistical / logical core -- Clopper-Pearson intervals and the verdict
    logic -- which must be sound *given* the population model. These tests use no
    flow and no data files; they are pure math.
  * the plumbing + end-to-end recovery, which exercises the flow sampler and the
    classifier against a KNOWN distribution with an analytic ground-truth rate.

To keep the core tests self-contained and deterministic we build synthetic
`DatasetAdapter`s here rather than loading the bundled datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# group_fairness/ (the modules under test) and FairN2V/ (adapter.py) on the path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FAIRN2V_DIR = _REPO_ROOT / 'examples' / 'FairN2V'
_GF_DIR = _FAIRN2V_DIR / 'group_fairness'
for _p in (str(_GF_DIR), str(_FAIRN2V_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import DatasetAdapter

SEED = 12345


class ThresholdNet(torch.nn.Module):
    """Deterministic stand-in net with an analytic favorable rate.

    Emits logits ``[x[:, col], 0.5]``. With ``class_type='max'`` and
    ``positive_class=0`` the favorable prediction is exactly ``{x[col] > 0.5}`` --
    a rule whose probability under a known feature distribution is analytic, so a
    test can check the pipeline recovers it.
    """

    def __init__(self, col: int = 0):
        super().__init__()
        self.col = col

    def forward(self, x):
        c0 = x[:, self.col]
        c1 = torch.full_like(c0, 0.5)
        return torch.stack([c0, c1], dim=1)


def make_adapter(X, y, sensitive_features, sensitive_encoding, *, net=None,
                 positive_class=0, class_type='max', group_names=None):
    """Assemble a synthetic DatasetAdapter from raw arrays."""
    X = np.asarray(X, dtype=np.float64)
    dim = X.shape[0]
    sens = list(sensitive_features)
    return DatasetAdapter(
        name='synthetic',
        X=X,
        y=np.asarray(y, dtype=int),
        min_values=np.zeros(dim),
        max_values=np.ones(dim),
        net=net if net is not None else ThresholdNet(col=0),
        sensitive_features=sens,
        perturbable_features=[i for i in range(dim) if i not in sens],
        sensitive_encoding=sensitive_encoding,
        output_size=2,
        class_type=class_type,
        positive_class=positive_class,
        group_names=group_names,
    )


@pytest.fixture
def binary_adapter():
    """Two groups via a binary sensitive column (col 2); decision feature col 0."""
    rng = np.random.default_rng(SEED)
    n = 240
    group = np.array([0] * (n // 2) + [1] * (n // 2))
    x0 = rng.uniform(0, 1, n)                 # decision feature
    x1 = rng.uniform(0, 1, n)                 # filler
    X = np.vstack([x0, x1, group.astype(float)])          # (3, n)
    y = rng.integers(0, 2, n)                 # true labels (both present per group)
    return make_adapter(X, y, sensitive_features=[2], sensitive_encoding='binary',
                        group_names=['A', 'B'])


@pytest.fixture
def onehot_adapter():
    """Three groups via a one-hot sensitive block (cols 2,3,4); decision col 0."""
    rng = np.random.default_rng(SEED)
    per = 90
    n = 3 * per
    g = np.repeat([0, 1, 2], per)
    onehot = np.zeros((3, n))
    onehot[g, np.arange(n)] = 1.0
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    X = np.vstack([x0, x1, onehot])                       # (5, n)
    y = rng.integers(0, 2, n)
    return make_adapter(X, y, sensitive_features=[2, 3, 4], sensitive_encoding='onehot',
                        group_names=['A', 'B', 'C'])


@pytest.fixture
def known_rate_adapter():
    """Two groups with analytic favorable rates under ThresholdNet ({x0 > 0.5}):

      group 0: x0 ~ U[0, 1]     -> rate 1/2
      group 1: x0 ~ U[0.25, 1]  -> rate (1-0.5)/(1-0.25) = 2/3

    ratio = 0.5 / 0.667 = 0.75 < 0.8, so demographic parity is truly UNFAIR.
    Returns (adapter, (rate0, rate1)).
    """
    rng = np.random.default_rng(SEED)
    n = 3000
    x0 = np.concatenate([rng.uniform(0.0, 1.0, n), rng.uniform(0.25, 1.0, n)])
    x1 = rng.uniform(0, 1, 2 * n)
    group = np.array([0] * n + [1] * n)
    X = np.vstack([x0, x1, group.astype(float)])          # (3, 2n)
    y = (x0 > 0.5).astype(int)
    adapter = make_adapter(X, y, sensitive_features=[2], sensitive_encoding='binary',
                           group_names=['g0', 'g1'])
    return adapter, (0.5, 2.0 / 3.0)
