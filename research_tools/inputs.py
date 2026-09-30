"""Locate and hash-check evidence inputs that may be absent from a fresh clone.

Manifests record absolute paths from the machine that produced them, and large
inputs (corpus, GGUF, mask banks, HF snapshots) are git-ignored.  Committed
files are always verified.  A missing external input is skipped with a printed
"SKIPPED" line, unless ARMSPARSE_STRICT_INPUTS=1, in which case it is an error.
A present file whose hash differs always fails.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT_ENV = "ARMSPARSE_STRICT_INPUTS"
SKIP_PREFIX = "SKIPPED"


class MissingInput(FileNotFoundError):
    pass


def strict() -> bool:
    return os.environ.get(STRICT_ENV, "").lower() not in ("", "0", "false", "no")


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path, root: Path = ROOT) -> Path:
    """Return path, or its repo-relative tail rebased onto root if the original is absent."""
    path = Path(path)
    if path.exists() or not path.is_absolute():
        return path
    top = {entry.name for entry in root.iterdir()}
    for index, part in enumerate(path.parts[1:], 1):
        if part in top and (root / Path(*path.parts[index:])).exists():
            return root / Path(*path.parts[index:])
    return path


def skip(message: str) -> bool:
    """Record a skipped check (fail in strict mode).  Always returns False."""
    if strict():
        raise MissingInput(f"missing input ({STRICT_ENV}=1): {message}")
    print(f"{SKIP_PREFIX} (not present in this checkout): {message}", flush=True)
    return False


def check_sha(path, expected: str, label: str, root: Path = ROOT) -> bool:
    """Verify a file's hash. True if verified, False if skipped; raises on mismatch."""
    found = resolve(path, root)
    if not found.is_file():
        return skip(f"{label} {Path(path).name} - hash recorded in manifest: {expected}")
    if sha(found) != expected:
        raise ValueError(f"{label} hash mismatch: {found}")
    return True


def git_show_sha(commit: str, relative: str, root: Path = ROOT) -> str:
    data = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                          capture_output=True, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def check_git_sources(manifest: dict, root: Path = ROOT) -> None:
    """Verify manifest source_sha256 against the archived commit's blobs."""
    for relative, expected in manifest["source_sha256"].items():
        if git_show_sha(manifest["git_commit"], relative, root) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
