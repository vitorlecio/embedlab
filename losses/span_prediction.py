"""JEPA-inspired span representation loss: cosine distance between a
predicted span representation (from masked context) and its target
representation (from the unmasked sequence, EMA-teacher branch) — same
family as BYOL's loss, adapted to text spans instead of image crops.

Includes two optional VICReg-style anti-collapse terms:
  - variance: penalizes any embedding dimension whose per-batch std drops
    below `variance_target`.
  - covariance: penalizes correlation *between* dimensions (squared
    off-diagonal covariance).

Both were added incrementally after collapse was measured in practice:
v1 (cosine loss alone, stop-gradient target, no EMA teacher) collapsed
to ~0.9997 mean pairwise cosine similarity across unrelated sentences.
v2 added an EMA teacher + the variance term alone, which *still*
collapsed (~0.996) and scored even worse on STS-B -- PCA on the
embeddings showed why: a single principal component explained ~52% of
variance (vs. ~13% for frozen BERT), meaning the variance term was
satisfied (healthy per-dimension std) while every embedding still
pointed in nearly the same *direction*, just at different magnitudes.
Marginal variance alone can't catch that; only an explicit covariance
term — which directly discourages dimensions from moving together —
targets it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanPredictionLoss(nn.Module):
    def __init__(
        self,
        variance_weight: float = 0.0,
        variance_target: float = 1.0,
        covariance_weight: float = 0.0,
    ):
        super().__init__()
        self.variance_weight = variance_weight
        self.variance_target = variance_target
        self.covariance_weight = covariance_weight

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(predicted, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        loss = (1 - (pred_norm * target_norm).sum(dim=-1)).mean()

        if self.variance_weight > 0:
            std_per_dim = predicted.std(dim=0)
            variance_penalty = F.relu(self.variance_target - std_per_dim).mean()
            loss = loss + self.variance_weight * variance_penalty

        if self.covariance_weight > 0:
            loss = loss + self.covariance_weight * self._covariance_penalty(predicted)

        return loss

    @staticmethod
    def _covariance_penalty(predicted: torch.Tensor) -> torch.Tensor:
        batch, dim = predicted.shape
        centered = predicted - predicted.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(batch - 1, 1)
        off_diag_sq_sum = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag_sq_sum / dim


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

    # Isolate the covariance term: a rank-1 "single ray" collapse (every row
    # is a scalar multiple of the same direction) has healthy per-dimension
    # variance but should still get a large covariance penalty, since all
    # dimensions move in lockstep. Use a larger batch here than above --
    # with few samples relative to dim, the *sample* covariance matrix is
    # dominated by estimation noise even for genuinely independent
    # dimensions (real training uses batch_size=32 vs. hidden_size=768, an
    # even more extreme ratio -- worth keeping in mind as a caveat on how
    # reliable this term's per-batch gradient signal actually is).
    large_batch, dim = 512, 16
    direction = torch.randn(dim)
    scales = torch.randn(large_batch, 1)
    rank1_predicted = scales * direction  # high std per dimension, but rank-1
    decorrelated_predicted = torch.randn(large_batch, dim)
    rank1_cov_penalty = SpanPredictionLoss._covariance_penalty(rank1_predicted)
    decorrelated_cov_penalty = SpanPredictionLoss._covariance_penalty(decorrelated_predicted)
    print(f"covariance penalty for a rank-1 (single-ray) prediction: {rank1_cov_penalty.item():.4f} (should be high)")
    print(f"covariance penalty for a decorrelated prediction:        {decorrelated_cov_penalty.item():.4f} (should be much lower)")
    assert rank1_cov_penalty.item() > decorrelated_cov_penalty.item()
