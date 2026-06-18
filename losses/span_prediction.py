"""JEPA-inspired span representation loss: cosine distance between a
predicted span representation (from masked context) and its target
representation (from the unmasked sequence, EMA-teacher branch) — same
family as BYOL's loss, adapted to text spans instead of image crops.

Includes an optional VICReg-style variance term as a second line of
defense against representation collapse: v1 of this loss (cosine
distance alone, paired with a stop-gradient-only target with no EMA
teacher) collapsed in practice (measured ~0.9997 mean pairwise cosine
similarity across unrelated sentences). The variance term directly
penalizes any embedding dimension whose per-batch std drops below
`variance_target`, independent of whatever the EMA teacher fixes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanPredictionLoss(nn.Module):
    def __init__(self, variance_weight: float = 0.0, variance_target: float = 1.0):
        super().__init__()
        self.variance_weight = variance_weight
        self.variance_target = variance_target

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(predicted, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cosine_loss = (1 - (pred_norm * target_norm).sum(dim=-1)).mean()

        if self.variance_weight <= 0:
            return cosine_loss

        std_per_dim = predicted.std(dim=0)
        variance_penalty = F.relu(self.variance_target - std_per_dim).mean()
        return cosine_loss + self.variance_weight * variance_penalty


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = SpanPredictionLoss()

    batch, dim = 8, 16
    target = F.normalize(torch.randn(batch, dim), dim=-1)

    identical_loss = loss_fn(target, target.clone())
    random_loss = loss_fn(F.normalize(torch.randn(batch, dim), dim=-1), target)
    print(f"loss when predicted == target exactly: {identical_loss.item():.4f} (should be ~0)")
    print(f"loss with unrelated random prediction: {random_loss.item():.4f} (should be higher)")
    assert identical_loss.item() < random_loss.item()

    # Isolate the variance term itself -- comparing *total* loss between a
    # collapsed and a healthy prediction would be confounded by each one's
    # unrelated per-row cosine alignment to a random target.
    collapsed_predicted = torch.zeros(batch, dim)  # zero std across the batch -- maximally collapsed
    healthy_predicted = torch.randn(batch, dim)  # std ~1 across the batch -- healthy
    collapsed_penalty = F.relu(1.0 - collapsed_predicted.std(dim=0)).mean()
    healthy_penalty = F.relu(1.0 - healthy_predicted.std(dim=0)).mean()
    print(f"variance penalty for a collapsed (zero-std) prediction: {collapsed_penalty.item():.4f} (should be ~1, max penalty)")
    print(f"variance penalty for a healthy (std~1) prediction:      {healthy_penalty.item():.4f} (should be ~0)")
    assert collapsed_penalty.item() > healthy_penalty.item()
