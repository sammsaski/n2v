"""Plumbing correctness of the flow population + classifier (the pieces between
the net and the verdict). These do not prove soundness of the guarantee -- they
check that the sampler and classifier do exactly what the verdict logic assumes:
the classifier reproduces the net's argmin/argmax decision, group labels and the
label mask select the right rows, and every synthetic draw has its sensitive
columns pinned to the intended group.
"""
import numpy as np
import pytest
import torch

from tests.unit.fairness.conftest import SEED
from flow_population import FlowGroupModel, _group_labels, make_favorable_classifier


def test_classifier_matches_manual_decision(binary_adapter):
    classify = make_favorable_classifier(binary_adapter)
    V = binary_adapter.X.T
    got = classify(V)
    logits = binary_adapter.net.forward(torch.as_tensor(V, dtype=torch.float32))
    preds = logits.argmax(dim=1)                      # class_type == 'max'
    manual = (preds == binary_adapter.positive_class).cpu().numpy()
    assert np.array_equal(got, manual)


def test_group_labels_binary(binary_adapter):
    lab = _group_labels(binary_adapter)
    assert np.array_equal(lab, (binary_adapter.X[2] > 0.5).astype(int))


def test_group_labels_onehot(onehot_adapter):
    lab = _group_labels(onehot_adapter)
    assert np.array_equal(lab, onehot_adapter.X[[2, 3, 4], :].argmax(axis=0))


@pytest.mark.slow
def test_sensitive_pinning_binary(binary_adapter):
    model = FlowGroupModel(binary_adapter, n_epochs=10, seed=SEED)
    for a in (0, 1):
        s = model.sample(a, 64)
        assert np.allclose(s[:, 2], float(a))         # sensitive col pinned to group
        assert (s >= 0).all() and (s <= 1).all()      # clamped to [0, 1]


@pytest.mark.slow
def test_sensitive_pinning_onehot(onehot_adapter):
    model = FlowGroupModel(onehot_adapter, n_epochs=10, seed=SEED)
    for a in (0, 1, 2):
        block = model.sample(a, 48)[:, [2, 3, 4]]
        assert np.allclose(block[:, a], 1.0)          # active category set
        assert np.allclose(block.sum(axis=1), 1.0)    # valid one-hot


@pytest.mark.slow
def test_label_mask_selects_cells(binary_adapter):
    """A true-label mask restricts each group's flow to that (group, label) cell,
    so group_sizes match the real per-cell counts."""
    mask = binary_adapter.y == 1
    lab = _group_labels(binary_adapter)
    model = FlowGroupModel(binary_adapter, mask=mask, n_epochs=8, seed=SEED)
    expected = [int(((lab == a) & mask).sum()) for a in range(model.k)]
    assert list(model.group_sizes) == expected
