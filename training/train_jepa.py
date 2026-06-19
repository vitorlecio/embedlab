"""Self-supervised JEPA-inspired training loop: mask a span of tokens,
predict its representation from context.

Usage:
    uv run python -m training.train_jepa
    uv run python -m training.train_jepa --max-steps 5   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from data.prepare import load_config, prepare_sts_b
from losses.span_prediction import SpanPredictionLoss
from models.jepa_encoder import JEPAEncoder, sample_span_mask
from training.utils import build_optimizer

ROOT = Path(__file__).resolve().parent.parent


class UnlabeledSentenceDataset(Dataset):
    """Plain sentences from STS-B's train split (sentence1 + sentence2,
    deduplicated) — scores/labels are deliberately ignored, since this
    module's whole point is training without labeled pairs, for a fair
    comparison against Module 1's contrastive approach on the same
    underlying sentences."""

    def __init__(self, df):
        self.sentences = sorted(set(df["sentence1"].tolist()) | set(df["sentence2"].tolist()))

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


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
    train_dataset = UnlabeledSentenceDataset(splits["train"])
    print(f"[train] {len(train_dataset)} unlabeled sentences")

    train_cfg = config["training"]
    jepa_cfg = config["jepa"]
    dataloader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    model = JEPAEncoder(model_name=config["model"]["backbone"], pooling=config["model"]["pooling"]).to(device)
    loss_fn = SpanPredictionLoss(
        variance_weight=jepa_cfg["variance_weight"],
        variance_target=jepa_cfg["variance_target"],
        covariance_weight=jepa_cfg["covariance_weight"],
    )

    optimizer = build_optimizer(model, train_cfg["learning_rate"], train_cfg["weight_decay"])
    total_steps = len(dataloader) * train_cfg["num_epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    model.train()
    step = 0
    start = time.time()
    for epoch in range(train_cfg["num_epochs"]):
        for sentences in dataloader:
            optimizer.zero_grad()
            batch = model.encoder.tokenizer(
                list(sentences),
                padding=True,
                truncation=True,
                max_length=config["model"]["max_length"],
                return_tensors="pt",
            ).to(device)
            span_mask = sample_span_mask(batch["attention_mask"], jepa_cfg["mask_span_ratio"], jepa_cfg["num_masked_spans"])

            predicted, target = model(batch["input_ids"], batch["attention_mask"], span_mask)
            loss = loss_fn(predicted, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.update_target_backbone(jepa_cfg["ema_momentum"])

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
    checkpoint_path = checkpoint_dir / "jepa_encoder.pt"
    torch.save(model.encoder.state_dict(), checkpoint_path)
    print(f"[train] saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the JEPA-inspired self-supervised (Module 2) encoder")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (for smoke testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
