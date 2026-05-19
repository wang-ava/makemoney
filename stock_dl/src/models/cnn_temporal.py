from __future__ import annotations

import torch
from torch import nn


class TemporalCNNRegressor(nn.Module):
    """Temporal Convolutional Network with Dilated Convolutions.

    Uses dilated 1D convolutions to capture multi-scale temporal patterns
    efficiently without the computational cost of attention mechanisms.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.15,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)

        layers = []
        receptive_field = 1
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            in_channels = d_model if i > 0 else d_model
            out_channels = d_model

            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            receptive_field += (kernel_size - 1) * dilation

        self.conv_layers = nn.ModuleList(layers)

        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        h = h.transpose(1, 2)

        for conv in self.conv_layers:
            h = conv(h) + nn.functional.pad(h, (0, h.size(-1) - h.size(-1)), mode="constant")[:, :, :1] if h.size(-1) < self.seq_len else h
            h = conv(h)

        h = h.transpose(1, 2)

        attn_weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)
        pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)

        last = h[:, -1]

        combined = torch.cat([pooled, last], dim=-1)

        return self.head(combined).squeeze(-1)


class InceptionBlock(nn.Module):
    """Inception module with multiple kernel sizes for diverse pattern capture."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.15):
        super().__init__()

        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1x1(x)
        b3 = self.branch3x3(x)
        b5 = self.branch5x5(x)
        bp = self.branch_pool(x)

        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.dropout(out)
        return out


class InceptionTimeRegressor(nn.Module):
    """InceptionTime architecture adapted for stock prediction.

    Combines multiple kernel sizes to capture patterns at different time scales.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        num_inception_blocks: int = 3,
        dropout: float = 0.15,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)

        self.inception_blocks = nn.ModuleList([
            InceptionBlock(d_model, d_model, dropout)
            for _ in range(num_inception_blocks)
        ])

        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h.transpose(1, 2)

        for block in self.inception_blocks:
            h = block(h) + h

        h = h.transpose(1, 2)

        attn_weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)
        pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)

        last = h[:, -1]

        combined = torch.cat([pooled, last], dim=-1)

        return self.head(combined).squeeze(-1)
