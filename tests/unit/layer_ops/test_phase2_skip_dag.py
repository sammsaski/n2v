"""Phase 2 sanity tests: LayerScale, DropPath, frozen-skip and DAG ops."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    layerscale_reach,
    drop_path_reach,
    add_with_frozen_skip_reach,
    concat_with_frozen_skip_reach,
    dag_add_reach,
    dag_concat_reach,
    concat2d_reach,
    selective_feature_fusion_reach,
)
from n2v.nn.layers import (
    LayerScale,
    DropPath,
    AddWithFrozenSkip,
    ConcatWithFrozenSkip,
)


@pytest.fixture
def small_box():
    lb = np.array([[-1.0], [0.0], [1.0]])
    ub = np.array([[0.0], [1.0], [2.0]])
    return Box(lb, ub)


def test_layerscale_box(small_box):
    layer = LayerScale(dim=3, init_value=2.0)
    out = layerscale_reach.layerscale_box(layer, [small_box])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), 2.0 * small_box.lb.flatten())
    np.testing.assert_allclose(out[0].ub.flatten(), 2.0 * small_box.ub.flatten())


def test_drop_path_is_identity(small_box):
    layer = DropPath(drop_prob=0.5)
    layer.eval()
    out = drop_path_reach.drop_path_box(layer, [small_box])
    np.testing.assert_array_equal(out[0].lb, small_box.lb)
    np.testing.assert_array_equal(out[0].ub, small_box.ub)


def test_add_with_frozen_skip(small_box):
    layer = AddWithFrozenSkip(skip=torch.tensor([1.0, 2.0, 3.0]))
    out = add_with_frozen_skip_reach.add_with_frozen_skip_box(layer, [small_box])
    np.testing.assert_allclose(out[0].lb.flatten(), small_box.lb.flatten() + np.array([1, 2, 3]))


def test_concat_with_frozen_skip(small_box):
    """T1-4: single-token append. Input flat dim equals skip flat dim;
    output dim = input + skip = 6 (legacy semantics preserved).

    PR-1 audit N8: tighten from ``dim == 6`` only to the EXACT lb/ub
    values. The prior assertion would pass a reach helper that
    duplicated the skip, swapped lb/ub, or stacked in the wrong order.
    Pin the canonical layout: input first, skip second.
    """
    layer = ConcatWithFrozenSkip(skip=torch.tensor([[7.0], [8.0], [9.0]]))
    out = concat_with_frozen_skip_reach.concat_with_frozen_skip_box(layer, [small_box])
    assert out[0].dim == 6
    # Audit N8: pin the exact concat ordering and skip values.
    expected_lb = np.concatenate(
        [np.asarray(small_box.lb).flatten(),
         np.array([7.0, 8.0, 9.0])],
    )
    expected_ub = np.concatenate(
        [np.asarray(small_box.ub).flatten(),
         np.array([7.0, 8.0, 9.0])],
    )
    np.testing.assert_allclose(
        np.asarray(out[0].lb).flatten(), expected_lb, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(out[0].ub).flatten(), expected_ub, atol=1e-9,
    )


def test_concat_with_frozen_skip_multi_token_raises():
    """T1-4 (audit medium): the multi-token case (input flat dim !=
    skip flat dim) is not yet recoverable from flat data alone. The
    reach raises rather than silently producing the previous mis-labelled
    [all_x, skip] vstack that verified the wrong operation. Until an
    explicit shape attribute lands on the wrapper, single-token is the
    supported case.
    """
    layer = ConcatWithFrozenSkip(skip=torch.tensor([100.0]), dim=-1)
    box = Box(
        np.array([[0.0], [1.0], [2.0], [3.0]]),
        np.array([[10.0], [11.0], [12.0], [13.0]]),
    )
    with pytest.raises(NotImplementedError, match="multi-token"):
        concat_with_frozen_skip_reach.concat_with_frozen_skip_box(layer, [box])


def test_concat_with_frozen_skip_wrapper_forward_3d_input():
    """T1-4 wrapper fix: ConcatWithFrozenSkip.forward must broadcast a
    feature-shaped (D,) skip up to a (B, L, D_x) input's full rank.

    Pre-fix the forward did only one unsqueeze, so a (D,) skip became
    (B, D) (rank 2) while x was rank 3 -> torch.cat raised RuntimeError.
    Now the skip is unsqueezed iteratively to match x.ndim and expanded.
    """
    layer = ConcatWithFrozenSkip(skip=torch.tensor([100.0, 200.0]), dim=-1)
    layer.eval()
    x = torch.zeros((1, 5, 3))  # B=1, L=5, D_x=3
    y = layer(x)
    # Output is (B, L, D_x + D_skip) = (1, 5, 5).
    assert y.shape == (1, 5, 5)
    # Last two features per token are the skip values.
    expected_skip = torch.tensor([100.0, 200.0])
    for li in range(5):
        assert torch.allclose(y[0, li, 3:], expected_skip)


def test_dag_add_two_streams(small_box):
    other = Box(np.array([[1.0], [1.0], [1.0]]), np.array([[2.0], [2.0], [2.0]]))
    out = dag_add_reach.dag_add_box([small_box], [[other]])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), small_box.lb.flatten() + np.array([1, 1, 1]))


def test_dag_concat_two_streams(small_box):
    other = Box(np.array([[10.0]]), np.array([[11.0]]))
    out = dag_concat_reach.dag_concat_box([small_box], [[other]])
    assert out[0].dim == 4


def test_concat2d_two_streams(small_box):
    other = Box(np.array([[-3.0]]), np.array([[-2.0]]))
    out = concat2d_reach.concat2d_box([small_box], [[other]])
    assert out[0].dim == 4


def test_sff_two_streams(small_box):
    other = Box(np.array([[-5.0], [-5.0], [-5.0]]), np.array([[-4.0], [-4.0], [-4.0]]))
    out = selective_feature_fusion_reach.selective_feature_fusion_box(
        [small_box], [[other]]
    )
    np.testing.assert_allclose(
        out[0].lb.flatten(), np.array([-5.0, -5.0, -5.0])
    )
    np.testing.assert_allclose(
        out[0].ub.flatten(), np.array([0.0, 1.0, 2.0])
    )
