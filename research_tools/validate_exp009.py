"""Validate EXP-009 provenance and recompute every retained timing aggregate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-009" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert sha(Path(manifest["mask_bank"])) == manifest["mask_bank_sha256"]
    for relative, expected in manifest["source_sha256"].items():
        archived = subprocess.run(["git", "show", f"{manifest['git_commit']}:{relative}"],
                                  cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(archived).hexdigest() == expected
    assert len(manifest["results"]) == manifest["runs"]
    expected_samples = manifest["repeats"] * 16
    for result in manifest["results"]:
        assert result["samples"] == expected_samples
        assert result["active_blocks"] == [922]
        assert len(result["raw_active_runs"]) == expected_samples
        assert result["active_runs"]["min"] == min(result["raw_active_runs"])
        assert result["active_runs"]["max"] == max(result["raw_active_runs"])
        assert result["active_runs"]["median"] == float(np.median(result["raw_active_runs"]))
        for key, values in result["raw_timings_ms"].items():
            assert len(values) == expected_samples and all(value >= 0 for value in values)
            aggregate = result["timings"][key]
            assert aggregate["median_ms"] == float(np.median(values))
            assert aggregate["p95_ms"] == float(np.percentile(values, 95))
    print("Validated EXP-009 provenance and all retained timing aggregates")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
