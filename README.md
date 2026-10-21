# HaluGate

A hallucination guardrail benchmark: what do cheap detectors really detect?
It compares a lexical overlap detector, DistilBERT classifiers and an NLI
cross-encoder on HaluEval (QA) and RAGTruth (QA), against a response-length
baseline, and reports AUC with grouped bootstrap intervals, within-model AUC
and per-response latency.

The question, the arms, the metrics and what may not change after seeing test
results are fixed in [`paper/protocol.md`](paper/protocol.md). Numbers are in
the paper, not here, so they cannot go stale.

## Layout

| Path | Contents |
|---|---|
| `src/` | detectors (`strategies/`), data loading, splits, metrics, statistics, reporting |
| `scripts/` | training, repeated latency, truncation report, manifest, exploratory analyses |
| `configs/config.yaml` | datasets, arms, parameters, seed |
| `paper/protocol.md` | the protocol and its dated amendments |
| `tests/` | unit tests |
| `data/`, `results/`, `models/` | not in git: raw data, outputs, trained weights |

## Setup

Requires Docker with the NVIDIA container toolkit (the results were produced
on WSL2 with an RTX 5060 Laptop GPU).

```bash
docker compose build halugate
docker compose up -d halugate
docker compose exec halugate pytest
```

The image installs every package at the version in `requirements-lock.txt`
and starts from a base image pinned by digest.

## Data

Place the raw files here (they are not redistributed):

- `data/raw/halueval_qa.jsonl`: the QA file of HaluEval, one JSON object per
  line with `question`, `knowledge`, `right_answer`, `hallucinated_answer`.
- `data/raw/ragtruth/response.jsonl` and `data/raw/ragtruth/source_info.jsonl`:
  the files of the RAGTruth dataset. Only the QA task is used.

`scripts/write_manifest.py` records the SHA-256 of each file.

## Reproducing the results

```bash
# 1. Fine-tune the two DistilBERT checkpoints (HaluEval, then RAGTruth QA)
docker compose exec halugate python scripts/train_distilbert.py
docker compose exec halugate python scripts/train_distilbert.py \
    --dataset ragtruth_qa --output-dir models/distilbert-ragtruth

# 2. Accuracy, calibration, bootstrap intervals, frontier
docker compose exec halugate python -m src.evaluate

# 3. Repeated, controlled latency (plugged in, high-performance power plan)
docker compose exec halugate python scripts/latency_repeats.py
docker compose exec halugate python scripts/latency_repeats.py \
    --datasets ragtruth_qa --sample random

# 4. Share of pairs truncated by the DistilBERT arms
docker compose exec halugate python scripts/truncation_report.py

# 5. Record what produced the results (commit first)
docker compose exec halugate python scripts/write_manifest.py --strict
```

Tables are written to `results/tables/`, figures to `results/figures/`.

## What is and is not reproducible

- Splits, bootstrap resamples and the order of the latency repetitions are
  seeded from `random_seed` in the config.
- Package versions, the base image and the Hub revisions of the NLI model are
  pinned. `results/manifest.json` lists them next to the commit and the hashes
  of the data and the trained weights.
- Fine-tuning on a GPU in fp16 is not bit-for-bit reproducible. The trained
  weights are therefore part of the artefacts: compare their SHA-256 with the
  manifest rather than expecting a retrained model to match.
- Latency depends on the machine. The manifest and
  `latency_environment_<run_id>.json` record the GPU, driver and library
  versions; absolute numbers are only comparable on the same hardware.