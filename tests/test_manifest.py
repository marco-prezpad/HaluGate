"""
test_manifest.py

Covers the hashing of data and weights, the lookup of Hub revisions in a
cache laid out like Hugging Face's, the comparison of the installed
packages with a lock file, and the manifest built from a small temporary
project.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from src.manifest import (
    build_manifest,
    configured_strategies,
    hash_dataset,
    hash_model_dir,
    hub_revisions,
    installed_packages,
    lock_mismatches,
    manifest_warnings,
    normalize_name,
    parse_lock,
    sha256_file,
    write_manifest,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class TestSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")

        assert sha256_file(path) == _sha(b"abc")

    def test_reads_files_larger_than_one_chunk(self, tmp_path: Path) -> None:
        content = b"x" * ((1 << 20) * 2 + 17)
        path = tmp_path / "big.bin"
        path.write_bytes(content)

        assert sha256_file(path) == _sha(content)


class TestHashDataset:
    def test_a_file_is_hashed_under_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "halueval_qa.jsonl"
        path.write_bytes(b"1")

        assert hash_dataset(path) == {"halueval_qa.jsonl": _sha(b"1")}

    def test_a_directory_hashes_its_jsonl_files_only(self, tmp_path: Path) -> None:
        (tmp_path / "response.jsonl").write_bytes(b"r")
        (tmp_path / "source_info.jsonl").write_bytes(b"s")
        (tmp_path / "README.md").write_bytes(b"ignored")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "other.jsonl").write_bytes(b"ignored")

        assert hash_dataset(tmp_path) == {
            "response.jsonl": _sha(b"r"),
            "source_info.jsonl": _sha(b"s"),
        }

    def test_a_missing_path_is_none(self, tmp_path: Path) -> None:
        assert hash_dataset(tmp_path / "nope") is None


class TestHashModelDir:
    def test_hashes_weights_config_and_tokenizer_only(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"w")
        (tmp_path / "config.json").write_bytes(b"c")
        (tmp_path / "vocab.txt").write_bytes(b"v")
        (tmp_path / "training_args.bin").write_bytes(b"varies between runs")
        (tmp_path / "notes.txt").write_bytes(b"x")
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-200" / "model.safetensors").write_bytes(b"old")
        (tmp_path / "folder.bin").mkdir()

        assert hash_model_dir(tmp_path) == {
            "config.json": _sha(b"c"),
            "model.safetensors": _sha(b"w"),
            "vocab.txt": _sha(b"v"),
        }

    def test_the_weights_hash_changes_with_the_weights(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"one")
        first = hash_model_dir(tmp_path)
        (tmp_path / "model.safetensors").write_bytes(b"two")

        assert hash_model_dir(tmp_path) != first

    def test_a_missing_directory_is_none(self, tmp_path: Path) -> None:
        assert hash_model_dir(tmp_path / "nope") is None


def _cache_model(cache: Path, folder: str, main: str | None, snapshots: list[str]) -> None:
    root = cache / folder
    for snapshot in snapshots:
        (root / "snapshots" / snapshot).mkdir(parents=True)
    if main is not None:
        (root / "refs").mkdir(parents=True)
        (root / "refs" / "main").write_text(main + "\n")


class TestHubRevisions:
    def test_reads_main_and_every_snapshot(self, tmp_path: Path) -> None:
        _cache_model(tmp_path, "models--cross-encoder--nli", "bbb", ["bbb", "aaa"])

        assert hub_revisions("cross-encoder/nli", tmp_path) == {
            "refs_main": "bbb",
            "snapshots": ["aaa", "bbb"],
        }

    def test_a_model_that_is_not_cached_has_no_revision(self, tmp_path: Path) -> None:
        assert hub_revisions("org/missing", tmp_path) == {"refs_main": None, "snapshots": []}

    def test_a_name_without_organisation_is_found_under_the_one_it_moved_to(
        self, tmp_path: Path
    ) -> None:
        _cache_model(tmp_path, "models--distilbert--distilbert-base-uncased", "ccc", ["ccc"])

        assert hub_revisions("distilbert-base-uncased", tmp_path)["refs_main"] == "ccc"

    def test_a_name_with_organisation_is_not_matched_loosely(self, tmp_path: Path) -> None:
        _cache_model(tmp_path, "models--other--nli", "ddd", ["ddd"])

        assert hub_revisions("cross-encoder/nli", tmp_path)["refs_main"] is None


class TestLock:
    LOCK = """\
--extra-index-url https://download.pytorch.org/whl/cu128
# a comment
numpy==1.26.4
typing_extensions==4.12.2  # inline comment
torch==2.11.0+cu128
Pillow[extra]==10.0.0
some-package @ git+https://example.org/x.git
loose>=1.0
-e .
"""

    def test_parses_exact_pins_only(self) -> None:
        assert parse_lock(self.LOCK) == {
            "numpy": "1.26.4",
            "typing-extensions": "4.12.2",
            "torch": "2.11.0+cu128",
            "pillow": "10.0.0",
        }

    @pytest.mark.parametrize(
        "name, expected",
        [("Typing_Extensions", "typing-extensions"), ("a.b_c-d", "a-b-c-d"), ("NumPy", "numpy")],
    )
    def test_names_are_normalised(self, name: str, expected: str) -> None:
        assert normalize_name(name) == expected

    def test_agreement_has_no_mismatch(self) -> None:
        assert lock_mismatches({"numpy": "1.26.4"}, {"numpy": "1.26.4", "extra": "1"}) == []

    def test_reports_another_version_and_a_missing_package(self) -> None:
        problems = lock_mismatches(
            {"numpy": "1.26.4", "torch": "2.11.0+cu128", "scipy": "1.0"},
            {"numpy": "2.0.0", "torch": "2.11.0"},
        )

        assert problems == [
            "numpy==1.26.4: installed 2.0.0",
            "scipy==1.0: not installed",
            "torch==2.11.0+cu128: installed 2.11.0",
        ]

    def test_installed_packages_are_normalised_and_include_this_test_runner(self) -> None:
        packages = installed_packages()

        assert "pytest" in packages
        assert all(name == normalize_name(name) for name in packages)

    def test_a_package_installed_twice_is_reported_at_the_version_found_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Distribution:
            def __init__(self, name: str, version: str) -> None:
                self.metadata = {"Name": name}
                self.version = version

        monkeypatch.setattr(
            metadata,
            "distributions",
            lambda: [_Distribution("pip", "26.2.1"), _Distribution("Pip", "22.0.2")],
        )

        assert installed_packages() == {"pip": "26.2.1"}

def test_configured_strategies_are_the_union_without_repeats() -> None:
    config = {
        "strategies": ["lexical", "nli"],
        "strategies_by_dataset": {"d": ["lexical", "nli_chunked"], "e": ["length"]},
    }

    assert configured_strategies(config) == ["lexical", "nli", "nli_chunked", "length"]


CONFIG: dict[str, Any] = {
    "random_seed": 42,
    "bootstrap_resamples": 1000,
    "strategies": ["distilbert", "nli"],
    "strategies_by_dataset": {"ragtruth_qa": ["distilbert_indomain", "nli_chunked"]},
    "strategy_params": {
        "distilbert": {"model_path": "MODEL_A"},
        "distilbert_indomain": {"model_path": "MODEL_B"},
        "nli": {"model_name": "cross-encoder/nli", "revision": "rev1"},
        "nli_chunked": {"model_name": "cross-encoder/nli", "revision": "rev1"},
    },
}
CLEAN_ENVIRONMENT = {"git_commit": "abc", "git_dirty": False, "python": "3.10"}


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Any]:
    models = {}
    for key in ("MODEL_A", "MODEL_B"):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(key.encode())
        models[key] = str(directory)
    config = json.loads(json.dumps(CONFIG))
    config["strategy_params"]["distilbert"]["model_path"] = models["MODEL_A"]
    config["strategy_params"]["distilbert_indomain"]["model_path"] = models["MODEL_B"]

    (tmp_path / "config.yaml").write_text("seed: 42")
    (tmp_path / "protocol.md").write_text("protocol")
    (tmp_path / "halueval.jsonl").write_text("data")
    cache = tmp_path / "hub"
    _cache_model(cache, "models--cross-encoder--nli", "rev1", ["rev1"])
    return {"config": config, "root": tmp_path, "cache": cache}


def _build(project: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    root = project["root"]
    arguments: dict[str, Any] = {
        "config": project["config"],
        "config_path": root / "config.yaml",
        "protocol_path": root / "protocol.md",
        "lock_path": root / "requirements-lock.txt",
        "environment": dict(CLEAN_ENVIRONMENT),
        "hub_cache": project["cache"],
        "datasets_raw": {"halueval": root / "halueval.jsonl"},
        "packages": {"numpy": "1.26.4"},
    }
    arguments.update(overrides)
    return build_manifest(**arguments)


class TestBuildManifest:
    def test_records_config_protocol_and_data_hashes(self, project: dict[str, Any]) -> None:
        manifest = _build(project)

        assert manifest["config"]["sha256"] == _sha(b"seed: 42")
        assert manifest["config"]["random_seed"] == 42
        assert manifest["protocol"]["sha256"] == _sha(b"protocol")
        assert manifest["datasets"] == {"halueval": {"halueval.jsonl": _sha(b"data")}}

    def test_records_the_weights_of_every_local_model(self, project: dict[str, Any]) -> None:
        manifest = _build(project)

        assert manifest["local_models"]["distilbert"]["files"] == {
            "model.safetensors": _sha(b"MODEL_A")
        }
        assert manifest["local_models"]["distilbert_indomain"]["files"] == {
            "model.safetensors": _sha(b"MODEL_B")
        }

    def test_a_hub_model_shared_by_two_strategies_appears_once(
        self, project: dict[str, Any]
    ) -> None:
        model = _build(project)["hub_models"]["cross-encoder/nli"]

        assert model["used_by"] == ["nli", "nli_chunked"]
        assert (model["configured_revisions"], model["refs_main"]) == (["rev1"], "rev1")

    def test_the_base_model_of_the_training_is_listed(self, project: dict[str, Any]) -> None:
        assert _build(project)["hub_models"]["distilbert-base-uncased"]["used_by"] == ["training"]

    def test_without_a_lock_file_there_is_no_lock_entry(self, project: dict[str, Any]) -> None:
        assert _build(project)["lock"] is None

    def test_a_lock_file_is_hashed_and_compared_with_the_packages(
        self, project: dict[str, Any]
    ) -> None:
        lock = project["root"] / "requirements-lock.txt"
        lock.write_text("numpy==2.0.0\n")

        entry = _build(project)["lock"]

        assert entry["sha256"] == _sha(b"numpy==2.0.0\n")
        assert entry["mismatches"] == ["numpy==2.0.0: installed 1.26.4"]

    def test_is_json_serialisable_and_written_to_disk(
        self, project: dict[str, Any], tmp_path: Path
    ) -> None:
        output = tmp_path / "out" / "manifest.json"

        write_manifest(_build(project), output)

        assert json.loads(output.read_text())["config"]["random_seed"] == 42


class TestWarnings:
    def _clean(self, project: dict[str, Any]) -> dict[str, Any]:
        (project["root"] / "requirements-lock.txt").write_text("numpy==1.26.4\n")
        return _build(project)

    def test_a_fully_pinned_run_has_no_warning(self, project: dict[str, Any]) -> None:
        assert manifest_warnings(self._clean(project)) == []

    def test_dirty_tree(self, project: dict[str, Any]) -> None:
        manifest = self._clean(project)
        manifest["environment"]["git_dirty"] = True

        assert any("uncommitted" in w for w in manifest_warnings(manifest))

    def test_unknown_commit(self, project: dict[str, Any]) -> None:
        manifest = self._clean(project)
        manifest["environment"]["git_commit"] = None

        assert any("no git commit" in w for w in manifest_warnings(manifest))

    def test_missing_lock_file(self, project: dict[str, Any]) -> None:
        assert any("no lock file" in w for w in manifest_warnings(_build(project)))

    def test_packages_that_differ_from_the_lock(self, project: dict[str, Any]) -> None:
        (project["root"] / "requirements-lock.txt").write_text("numpy==2.0.0\n")

        assert any("differ from the lock" in w for w in manifest_warnings(_build(project)))

    def test_hub_model_without_a_pinned_revision(self, project: dict[str, Any]) -> None:
        del project["config"]["strategy_params"]["nli"]["revision"]
        del project["config"]["strategy_params"]["nli_chunked"]["revision"]

        warnings = manifest_warnings(self._clean(project))

        assert any("cross-encoder/nli: no revision pinned" in w for w in warnings)

    def test_one_strategy_without_revision_is_enough_to_warn(
        self, project: dict[str, Any]
    ) -> None:
        del project["config"]["strategy_params"]["nli"]["revision"]

        warnings = manifest_warnings(self._clean(project))

        assert any("cross-encoder/nli: no revision pinned" in w for w in warnings)

    def test_pinned_revision_absent_from_the_cache(self, project: dict[str, Any]) -> None:
        for strategy in ("nli", "nli_chunked"):
            project["config"]["strategy_params"][strategy]["revision"] = "other"

        warnings = manifest_warnings(self._clean(project))

        assert any("revision other is not in the local cache" in w for w in warnings)

    def test_two_strategies_pinning_different_revisions(self, project: dict[str, Any]) -> None:
        project["config"]["strategy_params"]["nli"]["revision"] = "rev0"

        warnings = manifest_warnings(self._clean(project))

        assert any("pin different revisions: rev0, rev1" in w for w in warnings)

    def test_the_training_base_model_needs_no_pinned_revision(
        self, project: dict[str, Any]
    ) -> None:
        warnings = manifest_warnings(self._clean(project))

        assert not any("distilbert-base-uncased" in w for w in warnings)

    def test_missing_data_weights_and_protocol(self, project: dict[str, Any]) -> None:
        root = project["root"]
        (root / "MODEL_A" / "model.safetensors").unlink()
        (root / "MODEL_A").rmdir()
        (root / "protocol.md").unlink()
        manifest = _build(project, datasets_raw={"halueval": root / "missing.jsonl"})

        warnings = manifest_warnings(manifest)

        assert any("protocol file not found" in w for w in warnings)
        assert any("dataset halueval" in w for w in warnings)
        assert any(w.startswith("distilbert: no model files") for w in warnings)