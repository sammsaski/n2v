"""
Velocity field network for flow matching.

Maps (t, y) -> v_t(y), the velocity at time t and position y.
"""

import math

import torch
import torch.nn as nn


# Chunk size for DiTLiteVelocityField.forward. PyTorch's CUDA efficient
# attention kernel cannot produce valid seed/offset outputs when the batch
# size exceeds 65535; we chunk well below that to keep a safety margin.
_DIT_MAX_BATCH = 32768


class _ResidualBlock(nn.Module):
    """Residual block.

    Without layer_norm: ``x + Linear(act(Linear(x)))``.
    With layer_norm:    ``x + Linear(act(LN(Linear(x))))``, where LN is
    a LayerNorm over the hidden dimension.
    """

    def __init__(self, hidden: int, act_cls, layer_norm: bool = False):
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden) if layer_norm else None
        self.act = act_cls()
        self.lin2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = self.lin1(x)
        if self.ln is not None:
            h = self.ln(h)
        return x + self.lin2(self.act(h))


class VelocityField(nn.Module):
    """
    MLP velocity field for flow matching.

    Input: time t (batch,) and position y (batch, dim).
    Output: velocity v (batch, dim).

    Args:
        dim: Spatial dimensionality of the data.
        hidden: Hidden layer width.
        n_layers: Total number of layers (including input and output).
        activation: Activation function ('silu' or 'relu'). Default 'silu'.
        time_embed: How to encode time into the network input. 'concat'
            (default) concatenates the raw scalar t. 'sinusoidal' expands
            t into a 32-dim embedding using 16 log-spaced frequencies.
        residual: If True, build the middle layers as residual blocks
            ``x + Linear(act(Linear(x)))`` instead of plain
            ``Linear -> act``. The input and output projections are
            unchanged. Default False.
        layer_norm: If True, insert a LayerNorm at the start of each
            residual block, yielding ``x + Linear(act(LN(Linear(x))))``.
            Requires ``residual=True``; otherwise raises ``ValueError``.
            Default False.
        zero_init_output: If True, zero-initialize both the weight and
            bias of the final output ``Linear(hidden, dim)`` so that
            ``forward(t, y)`` returns the zero vector for every input
            at initialization. Standard flow matching trick: starts the
            flow as the identity map and lets training perturb away
            from it. Default False.
    """

    def __init__(self, dim: int, hidden: int = 128, n_layers: int = 4,
                 activation: str = 'silu',
                 time_embed: str = 'concat',
                 residual: bool = False,
                 layer_norm: bool = False,
                 zero_init_output: bool = False):
        super().__init__()
        valid_act = ('relu', 'silu')
        if activation not in valid_act:
            raise ValueError(
                f"activation must be one of {valid_act}, got '{activation}'"
            )
        valid_te = ('concat', 'sinusoidal')
        if time_embed not in valid_te:
            raise ValueError(
                f"time_embed must be one of {valid_te}, got '{time_embed}'"
            )
        if layer_norm and not residual:
            raise ValueError(
                "layer_norm=True requires residual=True"
            )
        self.time_embed = time_embed
        self.residual = residual
        self.layer_norm = layer_norm
        self.dim = dim

        if time_embed == 'concat':
            in_dim = dim + 1
        else:
            # Sinusoidal: 16 frequencies -> 32-dim embedding
            self.n_freqs = 16
            freqs = torch.exp(
                torch.linspace(0.0, math.log(1000.0), self.n_freqs)
            )
            self.register_buffer('sinusoidal_freqs', freqs)
            in_dim = dim + 2 * self.n_freqs

        act_cls = nn.ReLU if activation == 'relu' else nn.SiLU
        layers = [nn.Linear(in_dim, hidden), act_cls()]
        for _ in range(n_layers - 2):
            if residual:
                layers += [
                    _ResidualBlock(hidden, act_cls, layer_norm=layer_norm)
                ]
            else:
                layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

        self.zero_init_output = zero_init_output
        if zero_init_output:
            final_linear = self.net[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Optional output standardization buffers. When set (by train_flow
        # with standardize_outputs=True), forward() whitens incoming y and
        # de-whitens its output velocity so inference-time callers see data
        # in the original (unwhitened) space. Default None = no-op pass-through
        # (byte-identical to pre-change forward).
        self.register_buffer('y_mean', None)
        self.register_buffer('y_std', None)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity v_t(y).

        Args:
            t: (batch,) or scalar, time in [0, 1].
            y: (batch, dim) positions.

        Returns:
            (batch, dim) velocities.
        """
        # Whiten input if standardization buffers are registered. During
        # training these are None, so this path is a no-op and the model
        # sees whatever (pre-whitened) data train_flow feeds it.
        if self.y_mean is not None:
            y_work = (y - self.y_mean) / self.y_std
        else:
            y_work = y

        if t.dim() == 0:
            t = t.expand(y_work.shape[0])
        if self.time_embed == 'concat':
            ty = torch.cat([t.unsqueeze(1), y_work], dim=1)
        else:
            # Outer product t x freqs -> (batch, n_freqs)
            args = t.unsqueeze(1) * self.sinusoidal_freqs.unsqueeze(0)
            emb = torch.cat([args.sin(), args.cos()], dim=1)
            ty = torch.cat([emb, y_work], dim=1)
        out = self.net(ty)

        # De-whiten velocity: y_white = (y - m) / s  =>  dy/dt = s * dy_white/dt.
        # So the original-space velocity is v_white * std (multiply, not divide).
        if self.y_mean is not None:
            out = out * self.y_std
        return out


class DiTLiteVelocityField(nn.Module):
    """Small transformer backbone for flow matching.

    Treats each output coordinate as a token. Time is injected via a
    sinusoidal embedding added to every token. Intended to validate the
    transformer approach at low dim before using it at higher dim in
    future work.
    """

    def __init__(self, dim: int, hidden: int = 64,
                 n_blocks: int = 2, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.coord_embed = nn.Embedding(dim, hidden)

        # Sinusoidal time embedding projected to hidden
        self.n_freqs = 16
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), self.n_freqs))
        self.register_buffer('time_freqs', freqs)
        self.time_proj = nn.Linear(2 * self.n_freqs, hidden)

        # Per-coordinate scalar-to-hidden projection (value injection)
        self.value_proj = nn.Linear(1, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=4 * hidden, batch_first=True,
            activation='gelu',
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.out_proj = nn.Linear(hidden, 1)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(y.shape[0])
        batch = y.shape[0]

        # Small batches: single forward pass (no behavior change for
        # typical training batches).
        if batch <= _DIT_MAX_BATCH:
            return self._forward_chunk(t, y)

        # Large batches: chunked forward to avoid the CUDA efficient
        # attention kernel's 65535 batch cap. Used by the MC volume
        # estimator which pushes hundreds of thousands of points at once.
        out_chunks = []
        for start in range(0, batch, _DIT_MAX_BATCH):
            end = start + _DIT_MAX_BATCH
            out_chunks.append(self._forward_chunk(t[start:end], y[start:end]))
        return torch.cat(out_chunks, dim=0)

    def _forward_chunk(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Run one chunk through the transformer backbone."""
        batch = y.shape[0]

        # (batch, dim, 1) -> (batch, dim, hidden) value tokens
        values = self.value_proj(y.unsqueeze(-1))

        # (dim,) coordinate indices -> (batch, dim, hidden) coord embeddings
        coord_ids = torch.arange(self.dim, device=y.device)
        coord_tokens = self.coord_embed(coord_ids).unsqueeze(0).expand(
            batch, -1, -1
        )

        # (batch, 2*n_freqs) sinusoidal time -> (batch, hidden) -> broadcast
        args = t.unsqueeze(1) * self.time_freqs.unsqueeze(0)
        t_emb = torch.cat([args.sin(), args.cos()], dim=1)
        t_tokens = self.time_proj(t_emb).unsqueeze(1)  # (batch, 1, hidden)

        tokens = values + coord_tokens + t_tokens  # (batch, dim, hidden)
        tokens = self.blocks(tokens)  # self-attention over the dim tokens
        return self.out_proj(tokens).squeeze(-1)  # (batch, dim)
