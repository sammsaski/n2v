"""Phase 3 sanity tests: attention reachability shapes and basic soundness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    softmax_attention_reach,
    causal_mask_reach,
    sparsemax_reach,
    relative_attention_bias_t5_reach,
    relative_position_bias_table_reach,
    linear_attention_reach,
    dispatcher,
)
from n2v.nn.layers import RelativeAttentionBiasT5, RelativePositionBiasTable


def _box_2d(rows: int, cols: int) -> Box:
    n = rows * cols
    return Box(np.zeros((n, 1)), np.ones((n, 1)))


def test_softmax_attention_box():
    q = _box_2d(3, 4)
    k = _box_2d(3, 4)
    v = _box_2d(3, 4)
    out = softmax_attention_reach.softmax_attention_box([q], [k], [v], l_q=3, d_v=4)
    assert len(out) == 1
    # Sum-to-one + values in [0, 1] implies output bounded by [0, 1] in each dim.
    assert np.all(out[0].lb >= 0.0 - 1e-9)
    assert np.all(out[0].ub <= 1.0 + 1e-9)


def test_sparsemax_box():
    b = Box(np.array([[0.0], [1.0], [2.0]]), np.array([[1.0], [2.0], [3.0]]))
    out = sparsemax_reach.sparsemax_box([b])
    np.testing.assert_array_equal(out[0].lb, np.zeros((3, 1)))
    np.testing.assert_array_equal(out[0].ub, np.ones((3, 1)))


def test_linear_attention_box_d_v_gt_1_raises():
    """Copilot review / math audit: for d_v > 1 the implementation omits
    the cross-feature terms of phi(k).T @ v (it computes only the
    diagonal) -- a different function. The helper must refuse it; the
    sound d_v == 1 regime is pinned by the T1-9 four-corner test in
    test_phase1_activations_norms.py.
    """
    q = _box_2d(2, 4)
    k = _box_2d(2, 4)
    v = _box_2d(2, 4)
    with pytest.raises(NotImplementedError, match="cross-feature"):
        linear_attention_reach.linear_attention_box([q], [k], [v], l_q=2, d_v=4)


def test_linear_attention_box_legacy_raises():
    """Single-input call was unsound and should fail loudly."""
    b = Box(np.array([[-1.0], [0.0], [1.0]]), np.array([[0.0], [1.0], [2.0]]))
    with pytest.raises(NotImplementedError, match="Q/K/V"):
        linear_attention_reach.linear_attention_box([b])


def test_relative_bias_t5_box_exact_translation():
    """The wrapper forward is ``logits + bias``; the Box reach is the
    EXACT translation by the constant bias (Copilot review: replaced
    the loose [min(table), max(table)] envelope that discarded the
    input set).
    """
    torch.manual_seed(0)
    layer = RelativeAttentionBiasT5(num_buckets=8, max_distance=16, n_heads=2)
    with torch.no_grad():
        layer.relative_attention_bias.weight.data.uniform_(-0.5, 0.5)
    # Flattened (n_heads=2, L=2, L=2) logits: dim 8.
    lb = np.linspace(0.0, 0.7, 8).reshape(-1, 1)
    ub = lb + 0.25
    out = relative_attention_bias_t5_reach.relative_attention_bias_t5_box(
        layer, [Box(lb, ub)])
    assert out[0].dim == 8
    bias = layer.bias(2, 2).detach().numpy().astype(np.float64).reshape(-1, 1)
    np.testing.assert_allclose(out[0].lb, lb + bias, atol=1e-12)
    np.testing.assert_allclose(out[0].ub, ub + bias, atol=1e-12)
    # Forward correspondence: logits + bias for a sample is contained.
    x = (lb + ub).flatten() / 2.0
    with torch.no_grad():
        y = layer(torch.from_numpy(x.reshape(1, 2, 2, 2)).float()).numpy().flatten()
    assert np.all(out[0].lb.flatten() - 1e-5 <= y)
    assert np.all(y <= out[0].ub.flatten() + 1e-5)


def test_relative_bias_t5_box_non_square_raises():
    layer = RelativeAttentionBiasT5(num_buckets=8, n_heads=2)
    b = Box(np.zeros((6, 1)), np.ones((6, 1)))  # per-head dim 3: not L*L
    with pytest.raises(ValueError, match="not a square"):
        relative_attention_bias_t5_reach.relative_attention_bias_t5_box(layer, [b])


def test_relative_position_bias_table_box_exact_translation():
    """Exact constant translation by the Swin bias (Copilot review)."""
    torch.manual_seed(0)
    layer = RelativePositionBiasTable(window_size=2, n_heads=2)
    with torch.no_grad():
        layer.bias_table.data.uniform_(-0.3, 0.3)
    # Flattened (n_heads=2, W*W=4, W*W=4) logits: dim 32.
    lb = np.linspace(-0.5, 0.5, 32).reshape(-1, 1)
    ub = lb + 0.1
    out = relative_position_bias_table_reach.relative_position_bias_table_box(
        layer, [Box(lb, ub)])
    assert out[0].dim == 32
    bias = layer.bias().detach().numpy().astype(np.float64).reshape(-1, 1)
    np.testing.assert_allclose(out[0].lb, lb + bias, atol=1e-12)
    np.testing.assert_allclose(out[0].ub, ub + bias, atol=1e-12)


def test_relative_position_bias_table_dim_mismatch_raises():
    layer = RelativePositionBiasTable(window_size=2, n_heads=2)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))  # bias size is 32
    with pytest.raises(ValueError, match="does not match the bias size"):
        relative_position_bias_table_reach.relative_position_bias_table_box(
            layer, [b])


def test_relative_position_bias_table_round_trip_through_dispatcher():
    """T1-2 (audit high): RelativePositionBiasTable was grouped with
    RelativeAttentionBiasT5 in a tuple-isinstance branch and routed to the
    T5 reach helper, which reads layer.relative_attention_bias.weight --
    the Swin table stores its parameter as `bias_table`. End-to-end
    dispatch raised AttributeError. This test pins the dispatcher route to
    the dedicated relative_position_bias_table_reach module for Box, Star,
    and Zono.
    """
    layer = RelativePositionBiasTable(window_size=2, n_heads=2)
    layer.eval()
    with torch.no_grad():
        layer.bias_table.data.uniform_(-0.3, 0.3)

    n = 32  # flattened (n_heads=2, 4, 4) logits
    bias = layer.bias().detach().numpy().astype(np.float64).reshape(-1)
    for ctor in (
        lambda: Box(np.zeros((n, 1)), np.ones((n, 1))),
        lambda: Star.from_bounds(np.zeros((n, 1)), np.ones((n, 1))),
        # Copilot review: construct the Zono via from_bounds -- the
        # previous test passed an out-of-contract 3-D generator array.
        lambda: Zono.from_bounds(np.zeros((n, 1)), np.ones((n, 1))),
    ):
        s = ctor()
        out = dispatcher.reach_layer(layer, [s], "approx")
        assert len(out) == 1
        assert out[0].dim == n
        # Exact translation: bounds are the input bounds + bias.
        if isinstance(out[0], Box):
            np.testing.assert_allclose(
                out[0].lb.flatten(), bias, atol=1e-12)
            np.testing.assert_allclose(
                out[0].ub.flatten(), 1.0 + bias, atol=1e-12)


def test_relative_attention_bias_t5_round_trip_through_dispatcher():
    """T1-2 sibling test: the T5 bias still routes correctly to its own
    helper after the split. Catches accidental routing to the Swin helper.
    """
    layer = RelativeAttentionBiasT5(num_buckets=4, n_heads=2)
    layer.eval()
    n = 8  # flattened (n_heads=2, L=2, L=2) logits
    b = Box(np.zeros((n, 1)), np.ones((n, 1)))
    out = dispatcher.reach_layer(layer, [b], "approx")
    assert len(out) == 1
    assert out[0].dim == n
    bias = layer.bias(2, 2).detach().numpy().astype(np.float64).reshape(-1)
    np.testing.assert_allclose(out[0].lb.flatten(), bias, atol=1e-12)
    np.testing.assert_allclose(out[0].ub.flatten(), 1.0 + bias, atol=1e-12)
