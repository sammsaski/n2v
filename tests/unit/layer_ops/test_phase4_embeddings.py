"""Phase 4 sanity tests: embeddings, positional encodings, tokens."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import (
    embedding_reach,
    positional_encoding_reach,
    rope_reach,
    cls_token_reach,
    distillation_token_reach,
    segment_embedding_reach,
    dispatcher,
)
from n2v.nn.layers import (
    PositionalEncoding,
    RoPE,
    CLSToken,
    DistillationToken,
    SegmentEmbedding,
)


def test_embedding_box():
    """4 token indices × 4 embed_dim = 16 output features."""
    layer = nn.Embedding(num_embeddings=10, embedding_dim=4)
    layer.eval()
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))  # 4 tokens
    out = embedding_reach.embedding_box(layer, [b])
    assert out[0].dim == 16


def test_positional_encoding_box():
    layer = PositionalEncoding(dim=4, max_len=16)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = positional_encoding_reach.positional_encoding_box(layer, [b])
    assert out[0].dim == 4


def test_rope_box():
    layer = RoPE(dim=4, max_len=16)
    b = Box(np.zeros((8, 1)), np.ones((8, 1)))
    out = rope_reach.rope_box(layer, [b])
    assert out[0].dim == 8


def test_cls_token_prepends_dim():
    layer = CLSToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = cls_token_reach.cls_token_box(layer, [b])
    assert out[0].dim == 8


def test_distillation_token_prepends_dim():
    layer = DistillationToken(dim=4)
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    out = distillation_token_reach.distillation_token_box(layer, [b])
    assert out[0].dim == 8


def test_segment_embedding_round_trip_through_dispatcher():
    """T1-3 (audit high): SegmentEmbedding had a complete reach
    implementation and exported wrapper, but no dispatcher isinstance
    branch -> running it through the dispatcher raised NotImplementedError
    end-to-end. This test pins the dispatcher routes for all three set
    types.
    """
    num_segments = 3
    dim = 4  # embedding dim
    layer = SegmentEmbedding(num_segments=num_segments, dim=dim)
    layer.eval()
    # 2 tokens; segment_ids selects rows of the embedding table.
    n_tokens = 2
    flat = n_tokens * dim
    segment_ids = torch.tensor([0, 2])

    for set_type, ctor, dispatch_method in (
        (Box, lambda: Box(np.zeros((flat, 1)), np.ones((flat, 1))), "approx"),
        (
            Star,
            lambda: Star.from_bounds(
                np.zeros((flat, 1)), np.ones((flat, 1)),
            ),
            "approx",
        ),
        (
            Zono,
            lambda: Zono(
                np.zeros((flat, 1)),
                np.eye(flat).astype(np.float64).reshape(flat, flat, 1),
            ),
            "approx",
        ),
    ):
        s = ctor()
        out = dispatcher.reach_layer(
            layer, [s], dispatch_method, segment_ids=segment_ids,
        )
        assert len(out) == 1, set_type.__name__
        assert out[0].dim == flat, (
            f"{set_type.__name__}: reach out dim {out[0].dim} != {flat}"
        )


def test_segment_embedding_dispatch_passes_through_segment_ids():
    """Sanity: kwargs.get('segment_ids') is forwarded from dispatcher to
    the reach helper. Without segment_ids the reach helper raises
    ValueError; the dispatcher must surface that without swallowing it."""
    layer = SegmentEmbedding(num_segments=3, dim=2)
    layer.eval()
    b = Box(np.zeros((4, 1)), np.ones((4, 1)))
    with pytest.raises(ValueError, match="segment_ids"):
        dispatcher.reach_layer(layer, [b], "approx")  # no segment_ids kwarg
