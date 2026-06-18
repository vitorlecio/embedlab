"""Spearman correlation eval on STS-B, comparing baselines against the
trained contrastive encoder.

Usage:
    uv run python -m evaluation.sts_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel

from data.prepare import load_config, prepare_sts_b
from models.encoder import TextEncoder

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def encode_all(encoder: TextEncoder, sentences: list[str], max_length: int, device, batch_size: int = 64) -> torch.Tensor:
    encoder.eval()
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        embeddings.append(encoder.encode(batch, max_length=max_length, device=device).cpu())
    return torch.cat(embeddings, dim=0)


def spearman_for_encoder(encoder: TextEncoder, df, max_length: int, device) -> float:
    emb_1 = encode_all(encoder, df["sentence1"].tolist(), max_length, device)
    emb_2 = encode_all(encoder, df["sentence2"].tolist(), max_length, device)
    sims = F.cosine_similarity(emb_1, emb_2).numpy()
    rho, _ = spearmanr(sims, df["score"].to_numpy())
    return float(rho)


def build_random_encoder(model_name: str, pooling: str, seed: int | None = None) -> TextEncoder:
    """Same architecture as the pretrained backbone, but randomly initialized
    (no pretraining, no fine-tuning) — the bottom-of-the-scale baseline.

    Seeds torch right before initializing the random weights so this baseline
    is reproducible run-to-run (unlike the pretrained/checkpoint encoders,
    its weights are otherwise drawn fresh every call).
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder = TextEncoder(model_name=model_name, pooling=pooling)
    encoder.backbone = AutoModel.from_config(encoder.backbone.config)
    return encoder


def spearman_for_sbert(model_name: str, df) -> float:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb_1 = model.encode(df["sentence1"].tolist(), convert_to_tensor=True)
    emb_2 = model.encode(df["sentence2"].tolist(), convert_to_tensor=True)
    sims = F.cosine_similarity(emb_1, emb_2).cpu().numpy()
    rho, _ = spearmanr(sims, df["score"].to_numpy())
    return float(rho)


def main() -> dict[str, float]:
    parser = argparse.ArgumentParser(description="Evaluate STS-B Spearman correlation across baselines")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    data_cfg = config["data"]
    splits = prepare_sts_b(
        data_cfg["sts_b_dataset"],
        ROOT / data_cfg["cache_dir"],
        positive_threshold=data_cfg["positive_threshold"],
        negative_threshold=data_cfg["negative_threshold"],
    )
    df = splits[args.split]
    print(f"[eval] {args.split} split: {len(df)} pairs")

    backbone = config["model"]["backbone"]
    pooling = config["model"]["pooling"]
    max_length = config["model"]["max_length"]

    results: dict[str, float] = {}

    print("[eval] random encoder...")
    results["random"] = spearman_for_encoder(build_random_encoder(backbone, pooling, seed=config["training"]["seed"]).to(device), df, max_length, device)

    print("[eval] frozen pretrained BERT...")
    results["frozen_bert"] = spearman_for_encoder(TextEncoder(model_name=backbone, pooling=pooling).to(device), df, max_length, device)

    checkpoint_dir = ROOT / config["output"]["checkpoint_dir"]
    for name, filename in [("contrastive", "contrastive_encoder.pt"), ("jepa", "jepa_encoder.pt")]:
        checkpoint_path = checkpoint_dir / filename
        if checkpoint_path.exists():
            print(f"[eval] {name} (Module {'1' if name == 'contrastive' else '2'}) encoder from {checkpoint_path}...")
            trained_encoder = TextEncoder(model_name=backbone, pooling=pooling)
            trained_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
            results[name] = spearman_for_encoder(trained_encoder.to(device), df, max_length, device)
        else:
            print(f"[eval] no checkpoint at {checkpoint_path}, skipping {name} encoder")

    print("[eval] SBERT baseline (all-MiniLM-L6-v2)...")
    results["sbert"] = spearman_for_sbert("all-MiniLM-L6-v2", df)

    print(f"\nSpearman correlation on STS-B {args.split} split:")
    for name, rho in results.items():
        print(f"  {name:>12}: {rho:.4f}")

    results_dir = ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"sts_eval_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[eval] saved results to {out_path}")

    return results


if __name__ == "__main__":
    main()
