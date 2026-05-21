from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Learnable attention pooling over time steps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, max(d_model // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(d_model // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        if return_weights:
            return pooled, weights
        return pooled


class GRUTransformerRegressor(nn.Module):
    """Bi-GRU + Transformer encoder model from the v2 project plan."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 64,
        gru_hidden: int | None = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        hidden = int(gru_hidden or max(d_model // 2, 1))
        gru_out = hidden * 2

        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Identity() if gru_out == d_model else nn.Linear(gru_out, d_model)

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
        self.pool = TemporalAttentionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h = self.input_proj(self.input_norm(x)) + self.pos[:, : x.size(1)]
        h, _ = self.gru(h)
        h = self.gru_proj(h)
        h = self.encoder(h)
        if return_attention:
            pooled, weights = self.pool(h, return_weights=True)
            return self.head(pooled).squeeze(-1), weights
        pooled = self.pool(h)
        return self.head(pooled).squeeze(-1)
