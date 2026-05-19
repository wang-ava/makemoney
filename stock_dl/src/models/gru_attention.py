from __future__ import annotations

import torch
from torch import nn


class GRUAttentionRegressor(nn.Module):
    """GRU with Self-Attention for stock prediction.

    GRU provides efficient temporal modeling with gating mechanisms,
    while self-attention captures long-range dependencies.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        nhead: int = 4,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model * 2)

        self.query_proj = nn.Linear(d_model * 2, d_model * 2)
        self.key_proj = nn.Linear(d_model * 2, d_model * 2)
        self.value_proj = nn.Linear(d_model * 2, d_model * 2)

        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        gru_out, _ = self.gru(h)

        q = self.query_proj(gru_out)
        k = self.key_proj(gru_out)
        v = self.value_proj(gru_out)
        attn_out, _ = self.self_attn(q, k, v)
        attn_out = self.attn_norm(gru_out + attn_out)

        attn_weights = torch.softmax(self.temporal_attention(attn_out).squeeze(-1), dim=1)
        pooled = (attn_out * attn_weights.unsqueeze(-1)).sum(dim=1)

        last = attn_out[:, -1]

        combined = torch.cat([pooled, last], dim=-1)

        return self.head(combined).squeeze(-1)
