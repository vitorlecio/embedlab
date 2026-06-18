"""Download and preprocess STS-B for contrastive training and evaluation.

Usage:
    uv run python -m data.prepare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_sts_b(
    dataset_id: str,
    cache_dir: Path,
    positive_threshold: float = 3.5,
    negative_threshold: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Download STS-B and return train/validation/test as DataFrames with
    columns: sentence1, sentence2, score (0-5 float), label (binarized 0/1).
    """
    raw = load_dataset(dataset_id, cache_dir=str(cache_dir))

    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            continue
        df = raw[split_name].to_pandas()[["sentence1", "sentence2", "score"]].copy()
        df["sentence1"] = df["sentence1"].str.strip()
        df["sentence2"] = df["sentence2"].str.strip()
        df["score"] = df["score"].astype(float)
        # Binarize for contrastive pair construction: high similarity -> positive
        # pair, low similarity -> negative pair. Mid-range (ambiguous) rows are
        # kept with label=-1 and should be excluded from pair-based training but
        # retained for Spearman correlation eval, which uses the raw `score`.
        df["label"] = -1
        df.loc[df["score"] >= positive_threshold, "label"] = 1
        df.loc[df["score"] <= negative_threshold, "label"] = 0
        splits[split_name] = df.reset_index(drop=True)

    return splits


def load_domain_corpus(domain_corpus_dir: Path) -> list[str]:
    """Load unlabeled sentences for JEPA self-supervised pretraining.

    Populate `domain_corpus_dir` with .txt files (one sentence/passage per
    line) later, e.g. exported from the RAG agent project's document store.
    Returns an empty list with a warning if the directory has no data yet.
    """
    if not domain_corpus_dir.exists():
        print(f"[prepare] domain corpus dir not found: {domain_corpus_dir} (skipping)")
        return []

    lines: list[str] = []
    for txt_file in sorted(domain_corpus_dir.glob("*.txt")):
        lines.extend(l.strip() for l in txt_file.read_text(encoding="utf-8").splitlines() if l.strip())

    if not lines:
        print(f"[prepare] domain corpus dir is empty: {domain_corpus_dir} (skipping)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STS-B + domain corpus for EmbedLab")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override processed data output dir")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    cache_dir = ROOT / data_cfg["cache_dir"]
    out_dir = args.out_dir or (cache_dir / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] downloading {data_cfg['sts_b_dataset']} ...")
    splits = prepare_sts_b(
        data_cfg["sts_b_dataset"],
        cache_dir,
        positive_threshold=data_cfg["positive_threshold"],
        negative_threshold=data_cfg["negative_threshold"],
    )

    for split_name, df in splits.items():
        out_path = out_dir / f"sts_b_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        print(f"[prepare] {split_name}: {len(df)} rows -> {out_path} (positive={n_pos}, negative={n_neg})")

    domain_corpus_dir = ROOT / data_cfg["domain_corpus_dir"]
    domain_sentences = load_domain_corpus(domain_corpus_dir)
    print(f"[prepare] domain corpus: {len(domain_sentences)} sentences")


if __name__ == "__main__":
    main()
