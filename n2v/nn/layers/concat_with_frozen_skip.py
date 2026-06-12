"""Concat the running activation with a stored (frozen) skip tensor."""

import torch
import torch.nn as nn


class ConcatWithFrozenSkip(nn.Module):
    """Concatenate a frozen (buffer) skip tensor to the current activation.

    Supports broadcasting a feature-shaped skip up to ``x``'s full rank.
    For a 3D input ``(B, L, D)`` with ``dim=-1`` and a feature-shaped skip
    of shape ``(D,)`` or ``(1, D)``, the skip is unsqueezed and expanded to
    match the leading dims so the ``torch.cat`` call along ``dim=-1`` is
    well-formed. Documented contract: every dim except ``self.dim`` must
    match (or be broadcastable to) ``x`` after expansion.
    """

    def __init__(self, skip: torch.Tensor, dim: int = -1):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("skip", skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip
        # T1-4 (audit medium): the previous version only added ONE leading
        # unsqueeze, so a (D,) skip on a (B, L, D) input became (B, D) (rank
        # 2) while x was rank 3 -- torch.cat then raised "Tensors must have
        # same number of dimensions". The documented contract is to
        # broadcast a feature-shaped skip up to x's full rank.
        while skip.ndim < x.ndim:
            skip = skip.unsqueeze(0)
        # Now skip and x have the same rank. Expand only SINGLETON dims
        # to match x; any non-concat dim that is neither 1 nor equal to
        # x's size is not broadcastable -- fail fast with a clear message
        # instead of letting ``expand`` raise a cryptic runtime error
        # (Copilot review).
        concat_axis = self.dim if self.dim >= 0 else self.dim + x.ndim
        expand_shape = list(skip.shape)
        for i in range(x.ndim):
            if i == concat_axis:
                continue
            if expand_shape[i] == x.shape[i]:
                continue
            if expand_shape[i] == 1:
                expand_shape[i] = x.shape[i]
            else:
                raise ValueError(
                    f"ConcatWithFrozenSkip: skip shape "
                    f"{tuple(self.skip.shape)} is not broadcastable to "
                    f"input shape {tuple(x.shape)} along dim {i} "
                    f"(size {expand_shape[i]} vs {x.shape[i]}; only "
                    f"singleton dims are expanded). Every dim except "
                    f"the concat dim ({self.dim}) must match or be 1."
                )
        if tuple(expand_shape) != tuple(skip.shape):
            skip = skip.expand(*expand_shape)
        return torch.cat([x, skip], dim=self.dim)
