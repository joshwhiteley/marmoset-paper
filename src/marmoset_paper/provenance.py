"""Run records for inputs, code, settings, and generated files."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def validate_stage(directory: Path, artifacts: list[Path]) -> Path:
    """Require a completed upstream run and verify every consumed artifact's hash."""
    manifest = directory / "manifest.json"
    record = json.loads(manifest.read_text())
    if record.get("manifest_version") != 1 or record.get("status") != "complete":
        raise ValueError(f"Upstream run is not a completed version-1 manifest: {manifest}")
    outputs = {entry["relative_path"]: entry for entry in record["outputs"]}
    for artifact in artifacts:
        relative = str(artifact.relative_to(directory))
        expected = outputs.get(relative)
        if expected is None or file_record(artifact)["sha256"] != expected["sha256"]:
            raise ValueError(f"Artifact does not match its upstream manifest: {artifact}")
    return manifest


@contextmanager
def recorded_run(output: Path, inputs: list[Path], config: dict, root: Path):
    """Create a new output directory; retain a failed manifest if a run raises."""
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir.")
    record = {
        "manifest_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_branch": git_output(root, "branch", "--show-current"),
        "git_status": git_output(root, "status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "config": config,
        "inputs": [file_record(path) for path in inputs],
        "lockfile": file_record(root / "uv.lock"),
        "source_files": [
            file_record(path)
            for directory in [root / "src", root / "scripts"]
            for path in sorted(directory.rglob("*.py"))
        ],
        "status": "running",
    }
    output.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    try:
        yield record
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    else:
        record["status"] = "complete"
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["outputs"] = [
            {**file_record(path), "relative_path": str(path.relative_to(output))}
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != manifest
        ]
        manifest.write_text(json.dumps(record, indent=2) + "\n")
