#!/usr/bin/env python3
"""
inspect_ragtruth.py

Prints a compact structural summary of the raw RAGTruth files: counts per
task, split and quality flag, positive rate and responses per source for
one task, plus one example source record and one labelled response. It
reads the files only; no detector is run.

Usage:
    python scripts/inspect_ragtruth.py [--data-dir data/raw/ragtruth] [--task QA]

Author: Marco Pérez Padilla
Date:   10-10-2026
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from statistics import median
from typing import Any

MAX_PREVIEW_CHARS = 200
MAX_PREVIEW_ITEMS = 3


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def preview(value: Any) -> Any:
    """Shorten long strings and lists so one record stays readable."""
    if isinstance(value, str):
        if len(value) <= MAX_PREVIEW_CHARS:
            return value
        return f"{value[:MAX_PREVIEW_CHARS]}... [{len(value)} chars]"
    if isinstance(value, dict):
        return {key: preview(item) for key, item in value.items()}
    if isinstance(value, list):
        shown = [preview(item) for item in value[:MAX_PREVIEW_ITEMS]]
        if len(value) > MAX_PREVIEW_ITEMS:
            shown.append(f"... [{len(value)} items]")
        return shown
    return value


def text_length(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(json.dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the raw RAGTruth files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ragtruth"))
    parser.add_argument("--task", type=str, default="QA")
    args = parser.parse_args()

    sources = {
        record["source_id"]: record for record in read_jsonl(args.data_dir / "source_info.jsonl")
    }
    responses = list(read_jsonl(args.data_dir / "response.jsonl"))

    print(f"sources: {len(sources)}   responses: {len(responses)}")
    print("task types:", dict(Counter(s.get("task_type") for s in sources.values())))
    print("splits:", dict(Counter(r.get("split") for r in responses)))
    print("quality:", dict(Counter(r.get("quality") for r in responses)))
    print("models:", dict(Counter(r.get("model") for r in responses)))

    missing = [r for r in responses if r["source_id"] not in sources]
    print(f"responses whose source_id is missing from source_info: {len(missing)}")

    selected = [
        r for r in responses
        if r["source_id"] in sources and sources[r["source_id"]].get("task_type") == args.task
    ]
    print(f"\n=== task {args.task}: {len(selected)} responses ===")
    if not selected:
        print("no responses for this task; check the task names printed above")
        return

    for split_name in sorted({r.get("split") for r in selected}, key=str):
        in_split = [r for r in selected if r.get("split") == split_name]
        positives = [r for r in in_split if r.get("labels")]
        per_source = Counter(r["source_id"] for r in in_split)
        print(
            f"split={split_name}: responses={len(in_split)} sources={len(per_source)} "
            f"positive={len(positives)} ({len(positives) / len(in_split):.1%}) "
            f"responses/source={dict(Counter(per_source.values()))}"
        )
        print(f"  quality: {dict(Counter(r.get('quality') for r in in_split))}")

    print(
        "\nmedian chars -> response:",
        int(median(len(r["response"]) for r in selected)),
        " source_info:",
        int(median(text_length(sources[r["source_id"]]["source_info"]) for r in selected)),
    )

    example_source = sources[selected[0]["source_id"]]
    print(f"\n--- example {args.task} source record (shortened) ---")
    print(json.dumps(preview(example_source), indent=2, ensure_ascii=False))

    labelled = next((r for r in selected if r.get("labels")), None)
    print(f"\n--- example labelled {args.task} response (shortened) ---")
    print(json.dumps(preview(labelled), indent=2, ensure_ascii=False) if labelled else "none")


if __name__ == "__main__":
    main()