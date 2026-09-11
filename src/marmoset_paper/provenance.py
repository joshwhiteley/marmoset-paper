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


@contextmanager
def recorded_run(output: Path, inputs: list[Path], config: dict, root: Path):
    """Create a new output directory; retain a failed manifest if a run raises."""
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir.")
    record = {
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
            file_record(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != manifest
        ]
        manifest.write_text(json.dumps(record, indent=2) + "\n")
