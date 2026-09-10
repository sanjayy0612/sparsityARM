"""Independently validate EXP-006 provenance, equivalence, and aggregates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-006" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert sha(Path(manifest["corpus"])) == manifest["corpus_sha256"]
    assert sha(Path(manifest["layout"])) == manifest["layout_sha256"]
    assert manifest["layout_sha256"] == manifest["layout_metadata"]["layout_sha256"]
    for relative, expected in manifest["source_sha256"].items():
        archived = subprocess.run(["git", "show", f"{manifest['git_commit']}:{relative}"],
                                  cwd=ROOT, capture_output=True, check=True).stdout
        assert hashlib.sha256(archived).hexdigest() == expected
    snapshot = Path(manifest["snapshot"])
    for name, expected in manifest["model_file_sha256"].items():
        assert sha(snapshot / name) == expected
    assert len(manifest["dense_equivalence"]) == 16
    assert max(item["relative_l2_error"] for item in manifest["dense_equivalence"]) < 2e-5
    results = manifest["results"]
    assert len(results) == 7 and results[0]["condition"] == "dense"
    baseline = results[0]["perplexity"]
    threshold = manifest["approved_max_relative_ppl_increase"]
    for result in results:
        count = sum(item["predicted_tokens"] for item in result["examples"])
        mean_nll = sum(item["nll"] for item in result["examples"]) / count
        perplexity = math.exp(mean_nll)
        assert count == result["predicted_tokens"] == 2032
        assert math.isclose(mean_nll, result["mean_nll"], rel_tol=1e-12)
        assert math.isclose(perplexity, result["perplexity"], rel_tol=1e-12)
        if result["condition"] != "dense":
            relative = perplexity / baseline - 1
            assert math.isclose(relative, result["relative_ppl_increase"], rel_tol=1e-12)
            assert (relative <= threshold) == result["within_approved_threshold"]
    print("Validated EXP-006: 7 conditions, dense equivalence, all hashes and aggregates")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
