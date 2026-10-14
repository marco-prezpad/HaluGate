#!/usr/bin/env python3
"""
train_distilbert.py

Fine-tunes DistilBERT as a binary sequence-pair classifier (context,
response) -> hallucinated/not, on HaluEval's train split, and saves the
result to models/distilbert-halu -- the path DistilBERTDetector expects by
default (see configs/config.yaml's strategy_params.distilbert.model_path).


Resumable training: uses Hugging Face Trainer's own checkpointing
(save_strategy="steps", saved under --output-dir). If training is
interrupted, re-running this script with the same --output-dir
auto-resumes from the latest checkpoint instead of starting over.

Usage:
    python scripts/train_distilbert.py --config configs/config.yaml
    python scripts/train_distilbert.py --config configs/config.yaml --epochs 5

Author: Marco Pérez Padilla
Date:   06-10-2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import get_dataset_loader  # noqa: E402
from src.evaluate import load_config  # noqa: E402
from src.splitting import split_dataset  # noqa: E402
from src.tokenization import encode_pairs  # noqa: E402
from src.training_data import samples_to_training_arrays  # noqa: E402
from src.utils import configure_logging, set_global_seed  # noqa: E402

DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_NUM_EPOCHS = 3
DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_CHECKPOINT_SAVE_STEPS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for HaluGate.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--dataset", type=str, default="halueval", help="Dataset name to train on."
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "distilbert-halu",
        help="Where the fine-tuned model is saved (and Trainer checkpoints, "
        "during training, under output-dir/checkpoint-<step>).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_CHECKPOINT_SAVE_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.get("random_seed", 42))
    logger = configure_logging()

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    data_root = Path(config["data_root"])
    split_ratios = config.get("splits", {"train": 0.7, "validation": 0.15, "test": 0.15})
    random_seed = config.get("random_seed", 42)

    logger.info("Loading dataset '%s'", args.dataset)
    loader = get_dataset_loader(args.dataset, data_root)
    samples = loader.load()

    splits = split_dataset(samples, split_ratios, random_seed)
    logger.info(
        "Split sizes -- train=%d validation=%d (test=%d, unused here)",
        len(splits.train), len(splits.validation), len(splits.test),
    )

    train_contexts, train_responses, train_labels = samples_to_training_arrays(splits.train)
    val_contexts, val_responses, val_labels = samples_to_training_arrays(splits.validation)

    logger.info("Loading tokenizer and base model '%s'", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return encode_pairs(
            tokenizer, batch["context"], batch["response"], args.max_sequence_length
        )

    train_dataset = Dataset.from_dict(
        {"context": train_contexts, "response": train_responses, "labels": train_labels}
    ).map(_tokenize, batched=True)
    eval_dataset = Dataset.from_dict(
        {"context": val_contexts, "response": val_responses, "labels": val_labels}
    ).map(_tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resume automatically if a Trainer checkpoint already exists under
    # output_dir.
    existing_checkpoint = _find_latest_checkpoint(args.output_dir)
    if existing_checkpoint is not None:
        logger.info("Resuming training from checkpoint: %s", existing_checkpoint)
    trainer.train(resume_from_checkpoint=existing_checkpoint)

    final_model_dir = args.output_dir
    logger.info("Saving final fine-tuned model to %s", final_model_dir)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    logger.info("Done.")


def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the path to the most recent Trainer checkpoint under
    ``output_dir``, or ``None`` if there isn't one yet.
    """
    if not output_dir.exists():
        return None

    checkpoint_dirs = [
        path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return None

    def _step_number(path: Path) -> int:
        return int(path.name.removeprefix("checkpoint-"))

    latest = max(checkpoint_dirs, key=_step_number)
    return str(latest)


if __name__ == "__main__":
    main()