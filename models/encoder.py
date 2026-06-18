"""Base transformer encoder with a hand-written pooling head.

The backbone (AutoModel) is pretrained, but pooling and the sentence-embedding
forward pass are implemented explicitly here rather than relying on
sentence-transformers' Pooling module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean"):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"Unsupported pooling: {pooling!r}")
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size: int = self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns sentence embeddings of shape (batch, hidden_size)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if self.pooling == "cls":
            return token_embeddings[:, 0]

        return self._mean_pool(token_embeddings, attention_mask)

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)  # (batch, seq_len, 1)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(self, sentences: list[str], max_length: int = 128, device: str | torch.device = "cpu") -> torch.Tensor:
        """Convenience method: tokenize raw strings and return embeddings."""
        self.eval()
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        return self.forward(batch["input_ids"], batch["attention_mask"])


if __name__ == "__main__":
    encoder = TextEncoder()
    sentences = [
        "A man is playing a guitar.",
        "Someone is performing music on an instrument.",
        "The stock market crashed yesterday.",
    ]
    embeddings = encoder.encode(sentences)
    print("embeddings shape:", embeddings.shape)
    sims = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings)
    print("cosine sim to sentence[0]:", sims.tolist())
