"""JEPA-inspired self-supervised encoder: mask a span of tokens and predict
its *representation* from context, rather than predicting the masked
tokens themselves (BERT's MLM) or contrasting labeled pairs (NT-Xent).

Architecture: a shared backbone produces both branches.
  - Context branch: span tokens replaced with [MASK], run through the
    backbone + a small predictor head -> predicted span representation.
  - Target branch: original (unmasked) sequence run through the same
    backbone with gradients stopped -> target span representation.
The predictor + stop-gradient asymmetry (BYOL/JEPA-style) is what keeps
this from trivially collapsing to a constant — there is no separate EMA
teacher network here, which is a simplification worth noting if collapse
shows up in the downstream STS-B eval.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

from models.encoder import TextEncoder


def sample_span_mask(attention_mask: torch.Tensor, mask_span_ratio: float, num_spans: int = 1) -> torch.Tensor:
    """Returns a bool mask (batch, seq_len), True at masked span position(s).
    Spans are sampled within each sequence's real tokens, excluding the
    [CLS] (position 0) and [SEP] (last real position) tokens."""
    batch, seq_len = attention_mask.shape
    span_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    lengths = attention_mask.sum(dim=1)

    for i in range(batch):
        length = int(lengths[i].item())
        maskable = max(length - 2, 1)  # real tokens, excluding [CLS]/[SEP]
        span_len = max(1, min(round(maskable * mask_span_ratio), maskable))

        for _ in range(num_spans):
            max_start = max(1, length - 1 - span_len)
            start = random.randint(1, max_start) if max_start > 1 else 1
            span_mask[i, start : start + span_len] = True

    return span_mask


class JEPAEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean"):
        super().__init__()
        self.encoder = TextEncoder(model_name=model_name, pooling=pooling)
        hidden = self.encoder.hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.mask_token_id = self.encoder.tokenizer.mask_token_id

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, span_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted, target) pooled span representations, each
        shape (batch, hidden_size)."""
        masked_input_ids = input_ids.clone()
        masked_input_ids[span_mask] = self.mask_token_id

        context_hidden = self.encoder.backbone(input_ids=masked_input_ids, attention_mask=attention_mask).last_hidden_state
        predicted = self.predictor(self._masked_pool(context_hidden, span_mask))

        with torch.no_grad():
            target_hidden = self.encoder.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            target = self._masked_pool(target_hidden, span_mask)

        return predicted, target

    @staticmethod
    def _masked_pool(hidden_states: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        mask = span_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


if __name__ == "__main__":
    jepa = JEPAEncoder()
    sentences = [
        "A man is playing a guitar on the street.",
        "The stock market crashed yesterday after the announcement.",
    ]
    batch = jepa.encoder.tokenizer(sentences, padding=True, truncation=True, max_length=32, return_tensors="pt")
    span_mask = sample_span_mask(batch["attention_mask"], mask_span_ratio=0.15)
    print("span_mask sum per row:", span_mask.sum(dim=1).tolist())

    predicted, target = jepa(batch["input_ids"], batch["attention_mask"], span_mask)
    print("predicted shape:", predicted.shape, "target shape:", target.shape)
