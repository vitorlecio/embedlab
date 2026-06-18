"""Supervised contrastive training loop (Siamese + NT-Xent on STS-B).

Usage:
    uv run python -m training.train_contrastive
    uv run python -m training.train_contrastive --max-steps 5   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from data.prepare import load_config, prepare_sts_b
from losses.nt_xent import NTXentLoss
from models.encoder import TextEncoder
from models.siamese import SiameseEncoder

ROOT = Path(__file__).resolve().parent.parent


class PositivePairDataset(Dataset):
    """Only positive-labeled pairs (label == 1) are used: NT-Xent needs
    matched anchor/positive pairs and draws negatives implicitly from the
    rest of the batch, so the explicit negative-labeled rows aren't needed
    as separate training examples here.
    """

    def __init__(self, df):
        self.sentences_1 = df.loc[df["label"] == 1, "sentence1"].tolist()
        self.sentences_2 = df.loc[df["label"] == 1, "sentence2"].tolist()

    def __len__(self) -> int:
        return len(self.sentences_1)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.sentences_1[idx], self.sentences_2[idx]


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train(config: dict, max_steps: int | None = None) -> Path:
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    data_cfg = config["data"]
    splits = prepare_sts_b(
        data_cfg["sts_b_dataset"],
        ROOT / data_cfg["cache_dir"],
        positive_threshold=data_cfg["positive_threshold"],
        negative_threshold=data_cfg["negative_threshold"],
    )
    train_dataset = PositivePairDataset(splits["train"])
    print(f"[train] {len(train_dataset)} positive training pairs")

    train_cfg = config["training"]
    dataloader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    encoder = TextEncoder(model_name=config["model"]["backbone"], pooling=config["model"]["pooling"]).to(device)
    siamese = SiameseEncoder(encoder)
    loss_fn = NTXentLoss(temperature=train_cfg["temperature"])

    optimizer = build_optimizer(encoder, train_cfg["learning_rate"], train_cfg["weight_decay"])
    total_steps = len(dataloader) * train_cfg["num_epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    encoder.train()
    step = 0
    start = time.time()
    for epoch in range(train_cfg["num_epochs"]):
        for sentences_1, sentences_2 in dataloader:
            optimizer.zero_grad()
            emb_1, emb_2 = siamese.encode_pairs(list(sentences_1), list(sentences_2), max_length=config["model"]["max_length"], device=device)
            loss = loss_fn(emb_1, emb_2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 10 == 0 or step == 1:
                elapsed = time.time() - start
                print(f"[train] epoch {epoch} step {step}/{total_steps} loss={loss.item():.4f} ({elapsed:.1f}s elapsed)")

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    checkpoint_dir = ROOT / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "contrastive_encoder.pt"
    torch.save(encoder.state_dict(), checkpoint_path)
    print(f"[train] saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the supervised contrastive (Module 1) encoder")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (for smoke testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
