from __future__ import annotations

from typing import Any

from torch import nn

from .mlp import MLPRegressor
from .sequence import TemporalAttentionRegressor
from .lstm_attention import BiLSTMAttentionRegressor
from .gru_attention import GRUAttentionRegressor
from .cnn_temporal import TemporalCNNRegressor, InceptionTimeRegressor
from .gru_transformer import GRUTransformerRegressor


def build_model(model_cfg: dict[str, Any], n_features: int, seq_len: int) -> tuple[nn.Module, bool]:
    name = model_cfg.get("name", "mlp").lower()
    if name == "mlp":
        hidden = model_cfg.get("hidden_dims", [128, 64])
        dropout = float(model_cfg.get("dropout", 0.2))
        return MLPRegressor(seq_len * n_features, hidden, dropout), True

    if name in {"temporal_attention", "transformer", "sequence"}:
        return (
            TemporalAttentionRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                nhead=int(model_cfg.get("nhead", 4)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
                dropout=float(model_cfg.get("dropout", 0.15)),
                head_hidden=int(model_cfg.get("head_hidden", 128)),
            ),
            False,
        )

    if name in {"bilstm_attention", "bilstm", "lstm_attention", "lstm"}:
        return (
            BiLSTMAttentionRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.15)),
                nhead=int(model_cfg.get("nhead", 4)),
                head_hidden=int(model_cfg.get("head_hidden", 128)),
            ),
            False,
        )

    if name in {"gru_attention", "gru"}:
        return (
            GRUAttentionRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.15)),
                nhead=int(model_cfg.get("nhead", 4)),
                head_hidden=int(model_cfg.get("head_hidden", 128)),
            ),
            False,
        )

    if name in {"gru_transformer", "gru_trans", "hybrid_gru_transformer"}:
        return (
            GRUTransformerRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                nhead=int(model_cfg.get("nhead", 4)),
                num_layers=int(model_cfg.get("num_layers", 2)),
                dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                head_hidden=int(model_cfg.get("head_hidden", 64)),
                gru_hidden=int(model_cfg.get("gru_hidden", max(int(model_cfg.get("d_model", 128)) // 2, 1))),
            ),
            False,
        )

    if name in {"temporal_cnn", "cnn", "tcnn"}:
        return (
            TemporalCNNRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                num_layers=int(model_cfg.get("num_layers", 4)),
                kernel_size=int(model_cfg.get("kernel_size", 3)),
                dropout=float(model_cfg.get("dropout", 0.15)),
                head_hidden=int(model_cfg.get("head_hidden", 128)),
            ),
            False,
        )

    if name in {"inception_time", "inception"}:
        return (
            InceptionTimeRegressor(
                n_features=n_features,
                seq_len=seq_len,
                d_model=int(model_cfg.get("d_model", 128)),
                num_inception_blocks=int(model_cfg.get("num_inception_blocks", 3)),
                dropout=float(model_cfg.get("dropout", 0.15)),
                head_hidden=int(model_cfg.get("head_hidden", 128)),
            ),
            False,
        )

    raise ValueError(f"Unknown model.name: {name}")


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[nn.Module, bool]:
    model_cfg = ckpt.get("model_cfg")
    if model_cfg is None:
        model_cfg = {
            "name": "mlp",
            "hidden_dims": ckpt.get("hidden_dims", [128, 64]),
            "dropout": ckpt.get("dropout", 0.2),
        }
    model, flatten = build_model(
        model_cfg,
        n_features=len(ckpt["feat_cols"]),
        seq_len=int(ckpt["seq_len"]),
    )
    return model, flatten
