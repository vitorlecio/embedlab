"""JEPA-inspired span representation loss: cosine distance between a
predicted span representation (from masked context) and its target
representation (from the unmasked sequence, stop-gradient) — same family
as BYOL's loss, adapted to text spans instead of image crops.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanPredictionLoss(nn.Module):
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1 - (predicted * target).sum(dim=-1)).mean()


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
