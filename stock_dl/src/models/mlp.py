from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)])
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
