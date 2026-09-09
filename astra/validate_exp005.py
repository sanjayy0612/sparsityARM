"""Validate EXP-005 replay provenance, correctness, and raw timing summaries."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-005" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert len(manifest["case_artifacts"]) == manifest["runs"] == 3
    for source, expected in manifest["source_sha256"].items():
        assert sha(directory / "sources" / Path(source).name) == expected
    mask_path = Path(manifest["mask_metadata"]["mask_bank"])
    assert sha(mask_path) == manifest["mask_metadata"]["mask_bank_sha256"]
    with np.load(mask_path) as data:
        masks = data["block_masks"]
    assert masks.shape == (16, 1024)
    assert np.all(masks.sum(axis=1) == 922)
    weight_hashes = {}
    for artifact in manifest["case_artifacts"]:
        path = directory / artifact["path"]
        assert sha(path) == artifact["sha256"]
        case = json.loads(path.read_text())
        assert case["family"] == "replay" and case["block_size"] == 8
        assert case["target_sparsity"] == 0.10
        assert case["active_neurons_base"] == case["active_neurons_expanded"] == [7376] * 16
        assert case["correctness"] and all(
            value["max_relative_l2_error"] < 2e-4 for value in case["correctness"].values())
        for item in case["artifacts"].values():
            item_path = Path(item["path"])
            if not item_path.is_absolute():
                item_path = directory / item_path
            if item_path.name.startswith("weights-"):
                weight_hashes.setdefault(item_path.name, sha(item_path))
                actual = weight_hashes[item_path.name]
            else:
                actual = sha(item_path)
            assert actual == item["sha256"]
        for name, samples in case["raw_ms"].items():
            assert len(samples) == manifest["repeats"] == 200
            assert math.isclose(statistics.median(samples), case["summary"][name]["median_ms"])
            ordered = sorted(samples)
            position = .95 * (len(samples) - 1)
            lo, hi = math.floor(position), math.ceil(position)
            p95 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
            assert math.isclose(p95, case["summary"][name]["p95_ms"])
    print("Validated EXP-005: 3 runs, 3000 timings, replay budgets, correctness, summaries, and hashes")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
