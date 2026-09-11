import hashlib
import json
from pathlib import Path

import pytest

from marmoset_paper.provenance import recorded_run, validate_stage

ROOT = Path(__file__).resolve().parents[1]


def test_successful_run_hashes_outputs_and_survives_relocation(tmp_path):
    directory = tmp_path / "run"
    with recorded_run(directory, [ROOT / "pyproject.toml"], {"seed": 42}, ROOT) as record:
        (directory / "result.csv").write_text("value\n3\n")
        record["counts"] = {"rows": 1}
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"rows": 1}
    assert manifest["outputs"][0]["sha256"] == hashlib.sha256(b"value\n3\n").hexdigest()
    assert manifest["outputs"][0]["relative_path"] == "result.csv"
    moved = tmp_path / "moved"
    directory.rename(moved)
    assert validate_stage(moved, [moved / "result.csv"]) == moved / "manifest.json"


def test_changed_artifact_is_rejected(tmp_path):
    directory = tmp_path / "run"
    with recorded_run(directory, [], {}, ROOT):
        (directory / "result.csv").write_text("original")
    (directory / "result.csv").write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        validate_stage(directory, [directory / "result.csv"])


def test_failed_upstream_run_is_rejected_even_if_artifact_exists(tmp_path):
    directory = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with recorded_run(directory, [], {}, ROOT):
            (directory / "result.csv").write_text("partial")
            raise RuntimeError("failed")
    with pytest.raises(ValueError, match="not a completed"):
        validate_stage(directory, [directory / "result.csv"])


def test_unrecorded_artifact_is_rejected(tmp_path):
    directory = tmp_path / "run"
    with recorded_run(directory, [], {}, ROOT):
        pass
    (directory / "extra.csv").write_text("unrecorded")
    with pytest.raises(ValueError, match="does not match"):
        validate_stage(directory, [directory / "extra.csv"])
