"""
manifest.py

Everything needed to say, next to a set of results, exactly what produced
them: the code (git commit and whether the tree was clean), the
configuration and the protocol (SHA-256), the installed packages and their
agreement with the lock file, the raw data files (SHA-256), the trained
weights (SHA-256) and the revision of every model taken from the Hugging
Face Hub.

Hashing and reading are separate from the script (scripts/write_manifest.py)
so each piece can be tested on a temporary directory.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import codecs
import hashlib
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any

HASH_CHUNK_BYTES = 1 << 20
WEIGHT_SUFFIXES = (".safetensors", ".bin")
MODEL_SUPPORT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)
# Written by the Trainer next to the weights; it pickles the run's arguments,
# not the model, so it is left out of the hash.
NOT_WEIGHTS = ("training_args.bin",)
# Models the pipeline starts from when it trains (scripts/train_distilbert.py).
BASE_MODELS = ("distilbert-base-uncased",)
LOCK_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(\S+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_any_bom(path: Path) -> str:
    """Read a text file honouring a UTF-8 or UTF-16 BOM.

    Editors on Windows (PowerShell ``>``, "Unicode" save in Notepad) write
    UTF-16; without this a plain ``read_text()`` raises
    ``UnicodeDecodeError`` on the BOM before the lock file can be parsed.
    """
    data = path.read_bytes()
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    return data.decode("utf-8")


def hash_dataset(raw_path: Path) -> dict[str, str] | None:
    """SHA-256 of a dataset's raw file, or of every ``*.jsonl`` in its
    directory (RAGTruth ships two). ``None`` when the path does not exist.
    """
    if raw_path.is_file():
        return {raw_path.name: sha256_file(raw_path)}
    if raw_path.is_dir():
        return {path.name: sha256_file(path) for path in sorted(raw_path.glob("*.jsonl"))}
    return None


def hash_model_dir(model_dir: Path) -> dict[str, str] | None:
    """SHA-256 of the weights, config and tokenizer files at the top level of a
    saved model. Trainer checkpoints (subdirectories) are ignored. ``None``
    when the directory does not exist.
    """
    if not model_dir.is_dir():
        return None
    return {
        path.name: sha256_file(path)
        for path in sorted(model_dir.iterdir())
        if path.is_file()
        and path.name not in NOT_WEIGHTS
        and (path.suffix in WEIGHT_SUFFIXES or path.name in MODEL_SUPPORT_FILES)
    }


def default_hub_cache() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def hub_revisions(model_name: str, cache_dir: Path) -> dict[str, Any]:
    """Revisions of a Hub model present in the local cache: the commit that
    ``main`` pointed to when it was last fetched and every snapshot stored.
    A name without an organisation (``distilbert-base-uncased``) is also
    looked up under the organisation the Hub moved it to.
    """
    folders = [cache_dir / f"models--{model_name.replace('/', '--')}"]
    if "/" not in model_name:
        folders += sorted(cache_dir.glob(f"models--*--{model_name}"))
    folder = next((candidate for candidate in folders if candidate.is_dir()), None)
    if folder is None:
        return {"refs_main": None, "snapshots": []}

    refs_main = folder / "refs" / "main"
    snapshots = folder / "snapshots"
    return {
        "refs_main": refs_main.read_text().strip() if refs_main.is_file() else None,
        "snapshots": (
            sorted(path.name for path in snapshots.iterdir()) if snapshots.is_dir() else []
        ),
    }


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_packages() -> dict[str, str]:
    """Every installed distribution, by normalised name. A package installed
    in two places (the system one and an upgraded copy) is reported at the
    version found first on ``sys.path``, the one that gets imported.
    """
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            packages.setdefault(normalize_name(name), distribution.version)
    return dict(sorted(packages.items()))


def parse_lock(text: str) -> dict[str, str]:
    """Pinned ``name==version`` lines of a lock file; comments, options
    (``--extra-index-url``) and anything that is not an exact pin are ignored.
    """
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
        match = LOCK_LINE.match(line)
        if match:
            pins[normalize_name(match.group(1))] = match.group(2)
    return pins


def lock_mismatches(pins: dict[str, str], installed: dict[str, str]) -> list[str]:
    """Pinned packages that are missing or at another version. Installed
    packages that the lock does not list are not a mismatch.
    """
    problems = []
    for name, version in sorted(pins.items()):
        if name not in installed:
            problems.append(f"{name}=={version}: not installed")
        elif installed[name] != version:
            problems.append(f"{name}=={version}: installed {installed[name]}")
    return problems


def configured_strategies(config: dict[str, Any]) -> list[str]:
    names = list(config.get("strategies", []))
    for strategies in config.get("strategies_by_dataset", {}).values():
        names.extend(strategies)
    return list(dict.fromkeys(names))


def build_manifest(
    config: dict[str, Any],
    config_path: Path,
    protocol_path: Path,
    lock_path: Path,
    environment: dict[str, Any],
    hub_cache: Path,
    datasets_raw: dict[str, Path],
    packages: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The manifest of one run configuration.

    ``datasets_raw`` maps each dataset name to its raw file or directory and
    ``environment`` is the output of ``collect_environment``.
    """
    packages = installed_packages() if packages is None else packages
    params = config.get("strategy_params", {})

    local_models: dict[str, Any] = {}
    hub_models: dict[str, Any] = {}
    for strategy in configured_strategies(config):
        strategy_params = params.get(strategy, {})
        if "model_path" in strategy_params:
            model_path = strategy_params["model_path"]
            local_models[strategy] = {
                "model_path": model_path,
                "files": hash_model_dir(Path(model_path)),
            }
        if "model_name" in strategy_params:
            name = strategy_params["model_name"]
            previous = hub_models.get(name, {"used_by": [], "configured_revisions": []})
            hub_models[name] = {
                "used_by": sorted({*previous["used_by"], strategy}),
                "configured_revisions": sorted(
                    {*previous["configured_revisions"], strategy_params.get("revision")},
                    key=str,
                ),
                **hub_revisions(name, hub_cache),
            }
    for name in BASE_MODELS:
        hub_models.setdefault(
            name,
            {
                "used_by": ["training"],
                "configured_revisions": [None],
                **hub_revisions(name, hub_cache),
            },
        )

    lock = None
    if lock_path.is_file():
        lock = {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
            "mismatches": lock_mismatches(
                parse_lock(read_text_any_bom(lock_path)), packages
            ),
        }

    return {
        "environment": environment,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "random_seed": config.get("random_seed"),
            "bootstrap_resamples": config.get("bootstrap_resamples"),
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path) if protocol_path.is_file() else None,
        },
        "lock": lock,
        "datasets": {name: hash_dataset(path) for name, path in datasets_raw.items()},
        "local_models": local_models,
        "hub_models": hub_models,
        "packages": packages,
    }


def manifest_warnings(manifest: dict[str, Any]) -> list[str]:
    """Reasons why the manifest does not pin the run down completely."""
    warnings: list[str] = []
    environment = manifest["environment"]
    if environment.get("git_commit") is None:
        warnings.append("no git commit recorded")
    if environment.get("git_dirty"):
        warnings.append("uncommitted changes: the commit does not describe the code that ran")

    lock = manifest["lock"]
    if lock is None:
        warnings.append("no lock file: package versions are not pinned")
    elif lock["mismatches"]:
        warnings.append(f"{len(lock['mismatches'])} packages differ from the lock file")

    if manifest["protocol"]["sha256"] is None:
        warnings.append("protocol file not found")
    for name, files in manifest["datasets"].items():
        if not files:
            warnings.append(f"dataset {name}: raw files not found")
    for strategy, model in manifest["local_models"].items():
        if not model["files"]:
            warnings.append(f"{strategy}: no model files in {model['model_path']}")
    for name, model in manifest["hub_models"].items():
        if model["used_by"] == ["training"]:
            continue
        revisions = model["configured_revisions"]
        if None in revisions:
            warnings.append(f"{name}: no revision pinned in the config")
        pinned = [revision for revision in revisions if revision is not None]
        if len(pinned) > 1:
            warnings.append(f"{name}: strategies pin different revisions: {', '.join(pinned)}")
        for revision in pinned:
            if model["snapshots"] and revision not in model["snapshots"]:
                warnings.append(f"{name}: pinned revision {revision} is not in the local cache")
    return warnings


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")