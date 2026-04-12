"""
Velocity field network for flow matching.

Maps (t, y) -> v_t(y), the velocity at time t and position y.
"""

import torch
import torch.nn as nn


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
    """

    def __init__(self, dim: int, hidden: int = 128, n_layers: int = 4,
                 activation: str = 'silu'):
        super().__init__()
        valid = ('relu', 'silu')
        if activation not in valid:
            raise ValueError(
                f"activation must be one of {valid}, got '{activation}'"
            )
        act_cls = nn.ReLU if activation == 'relu' else nn.SiLU
        layers = [nn.Linear(dim + 1, hidden), act_cls()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity v_t(y).

        Args:
            t: (batch,) or scalar, time in [0, 1].
            y: (batch, dim) positions.

        Returns:
            (batch, dim) velocities.
        """
        if t.dim() == 0:
            t = t.expand(y.shape[0])
        ty = torch.cat([t.unsqueeze(1), y], dim=1)
        return self.net(ty)
