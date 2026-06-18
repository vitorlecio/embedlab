# EmbedLab — From Contrastive Learning to RAG Retrieval

A from-scratch PyTorch implementation of contrastive text representation learning, benchmarked against SBERT, extended with a JEPA-inspired self-supervised objective, and evaluated end-to-end through a RAG retrieval pipeline.

Most people use embeddings as a black box. This project opens the box: we implement contrastive text representation learning from scratch, extend it with a JEPA-inspired self-supervised objective, and measure whether the choice of training signal actually matters for downstream RAG retrieval quality. Spoiler: it does, and the why is interesting.

> **Status:** in progress. Module 1 (contrastive) code is complete and smoke-tested on CPU; the real training run happens on a free Colab GPU (local machine has no CUDA GPU), then evaluation comes next. See the [progress](#progress) section below.

## The three modules

1. **Supervised contrastive (Siamese + NT-Xent)** — fine-tune `bert-base-uncased` with mean pooling on STS-B sentence pairs. *Research question: how much does contrastive fine-tuning move the embedding space relative to frozen BERT?*
2. **JEPA-inspired self-supervised extension** — same backbone, mask a span of tokens and predict its *representation* (not its tokens) from context, no labels needed. *Research question: can a self-supervised objective produce a comparable embedding space without labeled pairs?*
3. **RAG retrieval benchmark** — swap the ChromaDB encoder in the [RAG Agent project](#) between the Module 1 encoder, the Module 2 encoder, and SBERT, reusing that project's P@k eval harness. *Research question: does the training objective choice matter for downstream retrieval, and by how much?*

## Results

| Encoder | STS-B Spearman ρ | P@k (RAG retrieval) | Latency |
|---|---|---|---|
| Random encoder | TBD | TBD | TBD |
| Frozen BERT (no fine-tuning) | TBD | TBD | TBD |
| Contrastive (Module 1) | TBD | TBD | TBD |
| JEPA-inspired (Module 2) | TBD | TBD | TBD |
| SBERT (`all-MiniLM-L6-v2`) | TBD | TBD | TBD |

UMAP plots of the embedding space (random → frozen BERT → contrastive → JEPA-inspired) will be added here once training is in place.

## Progress

- [x] `data/prepare.py` — downloads STS-B (`mteb/stsbenchmark-sts`), binarizes pairs into positive/negative/ambiguous by similarity score
- [x] `models/encoder.py` — `TextEncoder`: pretrained backbone (`AutoModel`) + hand-written mean-pooling head, forward pass verified
- [x] `notebooks/01_contrastive_learning.ipynb` — explainer notebook, currently covering data prep + encoder forward pass
- [x] `losses/nt_xent.py` — NT-Xent contrastive loss, sanity-checked
- [x] `models/siamese.py` — siamese wrapper, verified shapes + gradient flow
- [x] `training/train_contrastive.py` — supervised contrastive training loop, smoke-tested on CPU (~50s/step — too slow locally for a full run)
- [x] `notebooks/colab_train_contrastive.ipynb` — self-contained Colab notebook for the actual full training run on a free GPU
- [ ] Run the real Module 1 training on Colab, bring the checkpoint back
- [ ] `evaluation/sts_eval.py` — Spearman ρ on STS-B + baselines (random, frozen BERT, SBERT)
- [ ] `evaluation/visualize.py` — UMAP plots
- [ ] Module 2 (JEPA-inspired self-supervised encoder)
- [ ] Module 3 (RAG retrieval benchmark)

## Repo structure

```
embedlab/
├── README.md
├── data/prepare.py             # Download & preprocess STS-B + domain corpus
├── models/
│   ├── encoder.py              # Base transformer encoder + pooling (from scratch)
│   ├── siamese.py              # Siamese network wrapper
│   └── jepa_encoder.py         # Span-prediction self-supervised encoder
├── losses/
│   ├── nt_xent.py              # NT-Xent contrastive loss
│   └── span_prediction.py      # JEPA-inspired span representation loss
├── training/
│   ├── train_contrastive.py    # Supervised contrastive training loop
│   └── train_jepa.py           # Self-supervised training loop
├── evaluation/
│   ├── sts_eval.py             # Spearman correlation on STS-B
│   ├── retrieval_eval.py       # P@k evaluation (plugs into RAG agent)
│   └── visualize.py            # UMAP plots of embedding space
├── notebooks/                  # Self-contained explainer notebooks (LinkedIn writeups)
└── configs/default.yaml        # Hyperparameters
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run python -m data.prepare
uv run python -m models.encoder
uv run python -m training.train_contrastive --max-steps 5   # smoke test only
```

The local dev machine has no CUDA GPU, so `train_contrastive.py` is for smoke-testing the loop (use `--max-steps`), not full runs. Actual training happens in `notebooks/colab_train_contrastive.ipynb` on a free Colab GPU — download the resulting checkpoint into `checkpoints/contrastive_encoder.pt` to use it locally for evaluation.

## Connection to the RAG Agent project

Retrieval evaluation (Module 3) uses the eval harness from the [RAG Agent project](#) — its labeled query→chunk pairs and P@k logic are reused here rather than rebuilt. That project's README links back here: "Retrieval encoder benchmarked in EmbedLab."
