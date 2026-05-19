from __future__ import annotations

import torch
from torch import nn


class BiLSTMAttentionRegressor(nn.Module):
    """Bidirectional LSTM with Multi-Head Attention for stock prediction.

    BiLSTM captures bidirectional temporal dependencies while attention mechanism
    focuses on the most informative time steps for prediction.
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

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model * 2)

        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model * 2, 1),
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
        batch_size = x.size(0)

        h = self.input_proj(x)

        lstm_out, _ = self.lstm(h)

        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out, need_weights=True
        )
        attn_out = self.attn_norm(lstm_out + attn_out)

        time_weights = torch.softmax(self.temporal_attn(attn_out).squeeze(-1), dim=1)
        pooled = (attn_out * time_weights.unsqueeze(-1)).sum(dim=1)

        last = attn_out[:, -1]

        combined = torch.cat([pooled, last], dim=-1)

        return self.head(combined).squeeze(-1)
