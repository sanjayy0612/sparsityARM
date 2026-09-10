"""Independently validate EXP-008 provenance, equivalence and aggregates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["dense", "block_output_norm-B8-s20", "block_output_norm-B8-s30",
            "block_output_norm-B16-s20", "block_output_norm-B16-s30"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-008" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert manifest["layout_key"] == "hot_cold"
    assert manifest["conditions"] == EXPECTED
    assert manifest["approved_max_relative_ppl_increase"] == 0.05
    assert sha(Path(manifest["corpus"])) == manifest["corpus_sha256"]
    assert sha(Path(manifest["layout"])) == manifest["layout_sha256"]
    assert sha(Path(manifest["layout_metadata"])) == manifest["layout_metadata_sha256"]
    metadata = json.loads(Path(manifest["layout_metadata"]).read_text())
    assert metadata["layouts_sha256"] == manifest["layout_sha256"]
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
    assert [result["condition"] for result in results] == EXPECTED
    baseline = results[0]["perplexity"]
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
            assert (relative <= 0.05) == result["within_approved_threshold"]
    print("Validated EXP-008: hot/cold layout, five conditions, hashes and aggregates")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
