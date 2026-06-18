"""Siamese wrapper: runs both sentences of a pair through the same TextEncoder
(shared weights), producing the (anchor, positive) embedding pairs that
NT-Xent expects.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoder import TextEncoder


class SiameseEncoder(nn.Module):
    def __init__(self, encoder: TextEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_1 = self.encoder(input_ids_1, attention_mask_1)
        emb_2 = self.encoder(input_ids_2, attention_mask_2)
        return emb_1, emb_2

    def encode_pairs(
        self,
        sentences_1: list[str],
        sentences_2: list[str],
        max_length: int = 128,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for eval/notebook use: tokenize raw string pairs
        and return their embeddings."""
        tokenizer = self.encoder.tokenizer
        batch_1 = tokenizer(sentences_1, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        batch_2 = tokenizer(sentences_2, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        return self.forward(batch_1["input_ids"], batch_1["attention_mask"], batch_2["input_ids"], batch_2["attention_mask"])


if __name__ == "__main__":
    encoder = TextEncoder()
    siamese = SiameseEncoder(encoder)

    sentences_1 = ["A man is playing a guitar.", "The stock market crashed yesterday."]
    sentences_2 = ["Someone is performing music on an instrument.", "A man is playing a guitar."]

    with torch.no_grad():
        emb_1, emb_2 = siamese.encode_pairs(sentences_1, sentences_2)
    print("emb_1 shape:", emb_1.shape, "emb_2 shape:", emb_2.shape)

    sims = torch.nn.functional.cosine_similarity(emb_1, emb_2)
    print("pairwise cosine similarity:", sims.tolist())
