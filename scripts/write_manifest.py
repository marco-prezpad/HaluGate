#!/usr/bin/env python3
"""
write_manifest.py

Writes results/manifest.json: the code state, configuration, protocol,
installed packages, lock file, raw data, trained weights and Hub model
revisions behind a set of results (see src/manifest.py). Run it right
after the final evaluation, with the tree committed, and keep the file with
the results.

With --strict the exit code is 1 when anything is not pinned down (dirty
tree, packages that differ from the lock, a Hub model without a revision,
missing data or weights), so it can gate the final runs.

Usage:
    python scripts/write_manifest.py [--strict]

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import get_dataset_loader  # noqa: E402
from src.evaluate import load_config  # noqa: E402
from src.latency_repeats import collect_environment  # noqa: E402
from src.manifest import (  # noqa: E402
    build_manifest,
    default_hub_cache,
    manifest_warnings,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the manifest of a run configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--protocol", type=Path, default=Path("paper/protocol.md"))
    parser.add_argument("--lock", type=Path, default=Path("requirements-lock.txt"))
    parser.add_argument("--output", type=Path, default=None, help="default: results/manifest.json")
    parser.add_argument("--strict", action="store_true", help="exit 1 if anything is not pinned")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = Path(config["data_root"])
    output = args.output or Path(config["results_root"]) / "manifest.json"
    datasets_raw = {
        name: get_dataset_loader(name, data_root).raw_path for name in config["datasets"]
    }

    manifest = build_manifest(
        config, args.config, args.protocol, args.lock,
        collect_environment(), default_hub_cache(), datasets_raw,
    )
    write_manifest(manifest, output)

    environment = manifest["environment"]
    print(f"manifest written to {output}")
    print(f"  commit   {environment.get('git_commit')} (dirty: {environment.get('git_dirty')})")
    print(f"  python   {environment['python']} | torch {environment['packages'].get('torch')}"
          f" | transformers {environment['packages'].get('transformers')}")
    print(f"  seed     {manifest['config']['random_seed']}")
    for name, files in manifest["datasets"].items():
        print(f"  data     {name}: {', '.join(f'{k} {v[:12]}' for k, v in (files or {}).items())}")
    for strategy, model in manifest["local_models"].items():
        weights = {k: v for k, v in (model["files"] or {}).items() if k.startswith("model.")}
        print(f"  weights  {strategy}: {', '.join(f'{k} {v[:12]}' for k, v in weights.items())}")
    for name, model in manifest["hub_models"].items():
        print(
            f"  hub      {name}: main={model['refs_main']} "
            f"pinned={model['configured_revisions']} cached={model['snapshots']}"
         )
    for problem in (manifest["lock"] or {}).get("mismatches", []):
        print(f"  lock     {problem}")

    warnings = manifest_warnings(manifest)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if warnings and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()