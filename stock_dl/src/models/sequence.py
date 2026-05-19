from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionRegressor(nn.Module):
    """Transformer encoder for short stock feature sequences."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        last = h[:, -1]
        return self.head(torch.cat([pooled, last], dim=-1)).squeeze(-1)
