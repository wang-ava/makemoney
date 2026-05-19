from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def top_bottom_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_frac: float = 0.2,
    margin: float = 0.0,
) -> torch.Tensor:
    """Encourage high-realized-return stocks to receive higher scores than low ones."""
    n = target.numel()
    if n < 8:
        return pred.new_tensor(0.0)
    k = max(2, int(n * top_frac))
    if 2 * k > n:
        k = max(1, n // 2)
    order = torch.argsort(target)
    low = order[:k]
    high = order[-k:]
    spread = pred[high].mean() - pred[low].mean()
    return F.softplus(margin - spread)


def direction_loss(pred: torch.Tensor, raw_return: torch.Tensor) -> torch.Tensor:
    labels = (raw_return > 0).float()
    return F.binary_cross_entropy_with_logits(pred, labels)


def label_smoothed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smoothing: float = 0.05,
) -> torch.Tensor:
    """Label-smoothed MSE loss to reduce overfitting.

    Shifts targets towards zero (the cross-sectional mean) by a small amount.
    """
    if smoothing <= 0:
        return F.mse_loss(pred, target)
    smoothed_target = target * (1 - smoothing)
    return F.mse_loss(pred, smoothed_target)


def focal_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    top_frac: float = 0.2,
) -> torch.Tensor:
    """Focal ranking loss focuses on hard-to-rank samples.

    Samples with larger prediction error receive higher weight,
    helping the model focus on stocks it gets wrong.
    """
    n = target.numel()
    if n < 8:
        return pred.new_tensor(0.0)

    sorted_target, indices = torch.sort(target, descending=True)
    sorted_pred = pred[indices]

    position_weights = torch.arange(1, n + 1, device=pred.device, dtype=pred.dtype).float()
    position_weights = (position_weights / n) ** gamma
    position_weights = 1 - position_weights

    pred_diffs = sorted_pred[:-1] - sorted_pred[1:]
    target_diffs = sorted_target[:-1] - sorted_target[1:]

    correct_order = (pred_diffs * target_diffs > 0).float()
    loss = (1 - correct_order) * position_weights[:-1]
    return loss.mean()


def listwise_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Listwise ranking loss using softmax over all samples.

    Encourages the model to assign higher scores to stocks with higher returns.
    """
    n = target.numel()
    if n < 2:
        return pred.new_tensor(0.0)

    logits = pred / temperature
    log_probs = F.log_softmax(logits, dim=0)

    target_probs = F.softmax(target / temperature, dim=0)
    loss = -(target_probs * log_probs).sum()

    return loss


def composite_signal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    raw_return: torch.Tensor,
    base_loss: nn.Module,
    rank_weight: float = 0.0,
    direction_weight: float = 0.0,
    rank_top_frac: float = 0.2,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = base_loss(pred, target)

    if label_smoothing > 0:
        base = label_smoothed_loss(pred, target, smoothing=label_smoothing)

    if focal_gamma > 0:
        focal = focal_ranking_loss(pred, target, gamma=focal_gamma)
    else:
        focal = pred.new_tensor(0.0)

    rank = top_bottom_rank_loss(pred, target, top_frac=rank_top_frac) if rank_weight > 0 else pred.new_tensor(0.0)
    direct = direction_loss(pred, raw_return) if direction_weight > 0 else pred.new_tensor(0.0)

    total = base + rank_weight * rank + direction_weight * direct + focal * float(focal_gamma > 0) * 0.1
    return total, {
        "base_loss": float(base.detach().cpu()),
        "rank_loss": float(rank.detach().cpu()),
        "direction_loss": float(direct.detach().cpu()),
        "focal_loss": float(focal.detach().cpu()) if focal_gamma > 0 else 0.0,
    }
