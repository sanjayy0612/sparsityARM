"""Validate EXP-010 provenance and recompute timing aggregates."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(directory: Path):
    m = json.loads((directory / "manifest.json").read_text())
    assert m["experiment_id"] == "EXP-010" and m["status"] == "completed"
    assert m["git_status"] == "" and sha(m["mask_bank"]) == m["mask_bank_sha256"]
    for relative, expected in m["source_sha256"].items():
        data = subprocess.run(["git", "show", f"{m['git_commit']}:{relative}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        assert hashlib.sha256(data).hexdigest() == expected
    assert len(m["results"]) == m["runs"]
    for result in m["results"]:
        assert result["samples"] == m["repeats"] * 16
        for key, values in result["raw_timings_ms"].items():
            assert len(values) == result["samples"] and all(v >= 0 for v in values)
            assert result["timings"][key]["median_ms"] == float(np.median(values))
            assert result["timings"][key]["p95_ms"] == float(np.percentile(values, 95))
    print("Validated EXP-010 provenance and all timing aggregates")

if __name__ == "__main__": validate(Path(sys.argv[1]))
