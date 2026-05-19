from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any


class EnsembleRegressor(nn.Module):
    """Ensemble of multiple models with weighted averaging.

    Loads multiple pre-trained models and combines their predictions
    using learnable or fixed weights to reduce variance and improve stability.
    """

    def __init__(
        self,
        models: list[nn.Module] | None = None,
        weights: list[float] | None = None,
        method: str = "weighted",
    ):
        super().__init__()
        self.models = nn.ModuleList(models) if models else nn.ModuleList()
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models) if self.models else []
        else:
            self.weights = weights

        if method == "learnable" and models:
            total_features = sum(m.n_features if hasattr(m, "n_features") else 1 for m in models)
            self.weight_head = nn.Sequential(
                nn.Linear(total_features, 32),
                nn.ReLU(),
                nn.Linear(32, len(models)),
            )
        else:
            self.weight_head = None

    def add_model(self, model: nn.Module, weight: float = 1.0) -> None:
        self.models.append(model)
        self.weights.append(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.models:
            raise ValueError("No models in ensemble")

        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                preds.append(pred)

        if self.method == "weighted":
            preds_tensor = torch.stack(preds, dim=0)
            weights = torch.tensor(self.weights, device=x.device, dtype=x.dtype)
            weights = weights / weights.sum()
            final_pred = (preds_tensor * weights.view(-1, 1)).sum(dim=0)
        elif self.method == "average":
            final_pred = torch.stack(preds).mean(dim=0)
        elif self.method == "learnable" and self.weight_head is not None:
            features = torch.cat([p.unsqueeze(-1) for p in preds], dim=-1)
            raw_weights = self.weight_head(features.mean(dim=1))
            weights = torch.softmax(raw_weights, dim=-1)
            preds_tensor = torch.stack(preds, dim=0).transpose(0, 1)
            final_pred = (preds_tensor * weights.unsqueeze(1)).sum(dim=1)
        else:
            final_pred = torch.stack(preds).mean(dim=0)

        return final_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def load_ensemble_from_checkpoints(
    checkpoint_paths: list[Path | str],
    device: torch.device,
    weights: list[float] | None = None,
) -> EnsembleRegressor:
    """Load multiple model checkpoints into an ensemble."""
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).parent.parent))
    from src.models.factory import build_model_from_checkpoint

    ensemble = EnsembleRegressor(weights=weights)
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model, _ = build_model_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        ensemble.add_model(model)

    return ensemble


class StackingRegressor(nn.Module):
    """Two-level stacking ensemble with a meta-learner."""

    def __init__(
        self,
        base_models: list[nn.Module],
        meta_hidden: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)

        n_base = len(base_models)
        self.meta_learner = nn.Sequential(
            nn.Linear(n_base, meta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden, meta_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():
                pred = model(x)
                base_preds.append(pred)

        meta_features = torch.stack(base_preds, dim=1)
        return self.meta_learner(meta_features).squeeze(-1)


class BootstrapEnsemble(nn.Module):
    """Bootstrap aggregating ensemble for reduced variance."""

    def __init__(
        self,
        model_class: type[nn.Module],
        n_models: int = 5,
        model_kwargs: dict | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            model_class(**(model_kwargs or {}))
            for _ in range(n_models)
        ])

        self.head = nn.Sequential(
            nn.Linear(n_models, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        for model in self.models:
            pred = model(x)
            preds.append(pred)

        meta_features = torch.stack(preds, dim=1)
        return self.head(meta_features).squeeze(-1)
