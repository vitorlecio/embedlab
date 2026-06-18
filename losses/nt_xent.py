"""NT-Xent (normalized temperature-scaled cross entropy) contrastive loss.

Same formulation used by SimCLR/SimCSE: given N anchor-positive embedding
pairs, treat the matching positive as the only correct class among all N
positives in the batch (in-batch negatives), and average the loss computed
in both directions (anchor->positive and positive->anchor).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """z_anchor, z_positive: (batch, dim), row i in each is a matching pair."""
        if z_anchor.shape != z_positive.shape:
            raise ValueError(f"shape mismatch: {z_anchor.shape} vs {z_positive.shape}")

        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        sim_matrix = z_anchor @ z_positive.T / self.temperature  # (batch, batch)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss_a2p = F.cross_entropy(sim_matrix, labels)
        loss_p2a = F.cross_entropy(sim_matrix.T, labels)
        return (loss_a2p + loss_p2a) / 2


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = NTXentLoss(temperature=0.05)

    batch, dim = 8, 16
    z = F.normalize(torch.randn(batch, dim), dim=-1)

    identical_loss = loss_fn(z, z.clone())
    random_loss = loss_fn(z, F.normalize(torch.randn(batch, dim), dim=-1))
    print(f"loss when anchor == positive exactly: {identical_loss.item():.4f} (should be near 0)")
    print(f"loss with unrelated random positives:  {random_loss.item():.4f} (should be higher)")
    assert identical_loss.item() < random_loss.item()
