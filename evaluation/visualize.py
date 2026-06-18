"""UMAP visualization of the embedding space across encoders.

For a sample of STS-B positive pairs, projects both sentences of each pair
to 2D per encoder. Matching colors mark the same pair, so a tighter visual
clustering of same-colored points indicates the encoder pulls paraphrases
closer together in embedding space.

Usage:
    uv run python -m evaluation.visualize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import umap

from data.prepare import load_config, prepare_sts_b
from evaluation.sts_eval import build_random_encoder, encode_all
from models.encoder import TextEncoder

# matplotlib must be imported after `datasets` (pulled in via data.prepare) —
# importing it first causes a native segfault on Windows when umap/numba and
# datasets/pyarrow are both loaded in the same process (LLVM symbol clash).
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def sample_pairs(df, n_pairs: int, seed: int):
    positive = df[df["label"] == 1].reset_index(drop=True)
    n_pairs = min(n_pairs, len(positive))
    return positive.sample(n=n_pairs, random_state=seed).reset_index(drop=True)


def pair_embeddings(encoder: TextEncoder, pairs_df, max_length: int, device) -> np.ndarray:
    """Returns an array of shape (2 * n_pairs, hidden_dim): all sentence1
    embeddings followed by all sentence2 embeddings, in matching pair order."""
    emb_1 = encode_all(encoder, pairs_df["sentence1"].tolist(), max_length, device)
    emb_2 = encode_all(encoder, pairs_df["sentence2"].tolist(), max_length, device)
    return torch.cat([emb_1, emb_2], dim=0).numpy()


def project_umap(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n_neighbors = max(2, min(15, len(embeddings) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=seed)
    return reducer.fit_transform(embeddings)


def plot_umap_grid(projections: dict[str, np.ndarray], n_pairs: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(projections), figsize=(5 * len(projections), 5))
    if len(projections) == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, n_pairs))

    for ax, (name, coords) in zip(axes, projections.items()):
        sent1_coords, sent2_coords = coords[:n_pairs], coords[n_pairs:]
        for i in range(n_pairs):
            ax.plot(
                [sent1_coords[i, 0], sent2_coords[i, 0]],
                [sent1_coords[i, 1], sent2_coords[i, 1]],
                color=colors[i], alpha=0.4, linewidth=1, zorder=1,
            )
        ax.scatter(sent1_coords[:, 0], sent1_coords[:, 1], color=colors, marker="o", zorder=2)
        ax.scatter(sent2_coords[:, 0], sent2_coords[:, 1], color=colors, marker="^", zorder=2)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].scatter([], [], color="black", marker="o", label="sentence1")
    axes[0].scatter([], [], color="black", marker="^", label="sentence2")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("UMAP projection of STS-B positive pairs (matching colors = same pair)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[visualize] saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP visualization of the embedding space across encoders")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--n-pairs", type=int, default=25)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config["training"]["seed"]

    data_cfg = config["data"]
    splits = prepare_sts_b(
        data_cfg["sts_b_dataset"],
        ROOT / data_cfg["cache_dir"],
        positive_threshold=data_cfg["positive_threshold"],
        negative_threshold=data_cfg["negative_threshold"],
    )
    pairs_df = sample_pairs(splits[args.split], args.n_pairs, seed)
    print(f"[visualize] sampled {len(pairs_df)} positive pairs from {args.split}")

    backbone = config["model"]["backbone"]
    pooling = config["model"]["pooling"]
    max_length = config["model"]["max_length"]

    encoders = {
        "random": build_random_encoder(backbone, pooling),
        "frozen_bert": TextEncoder(model_name=backbone, pooling=pooling),
    }
    checkpoint_path = ROOT / config["output"]["checkpoint_dir"] / "contrastive_encoder.pt"
    if checkpoint_path.exists():
        print(f"[visualize] loading contrastive checkpoint from {checkpoint_path}")
        contrastive_encoder = TextEncoder(model_name=backbone, pooling=pooling)
        contrastive_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        encoders["contrastive"] = contrastive_encoder
    else:
        print(f"[visualize] no checkpoint at {checkpoint_path}, skipping contrastive encoder")

    projections = {}
    for name, encoder in encoders.items():
        print(f"[visualize] computing + projecting embeddings for {name}...")
        embeddings = pair_embeddings(encoder.to(device), pairs_df, max_length, device)
        projections[name] = project_umap(embeddings, seed)

    results_dir = ROOT / config["output"]["results_dir"]
    plot_umap_grid(projections, len(pairs_df), results_dir / f"umap_{args.split}.png")


if __name__ == "__main__":
    main()
