"""Validate EXP-005 replay provenance, correctness, and raw timing summaries."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from research_tools.inputs import MissingInput, check_git_sources, check_sha, resolve, sha, strict


def validate(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-005" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert len(manifest["case_artifacts"]) == manifest["runs"] == 3
    # sources/ is git-ignored; the archived commit holds the same blobs.
    if (directory / "sources").is_dir():
        for source, expected in manifest["source_sha256"].items():
            assert sha(directory / "sources" / Path(source).name) == expected
    else:
        if strict():
            raise MissingInput(f"{directory / 'sources'} (strict mode)")
        print("NOTE: sources/ not present; verifying source hashes against the archived git commit")
        check_git_sources(manifest)
    mask_path = resolve(manifest["mask_metadata"]["mask_bank"])
    if check_sha(mask_path, manifest["mask_metadata"]["mask_bank_sha256"], "mask bank"):
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
            item_path = resolve(item_path)
            if not item_path.is_file():
                check_sha(item_path, item["sha256"], "case artifact")  # skip or strict failure
            elif item_path.name.startswith("weights-"):
                weight_hashes.setdefault(item_path.name, sha(item_path))
                assert weight_hashes[item_path.name] == item["sha256"]
            else:
                assert sha(item_path) == item["sha256"]
        for name, samples in case["raw_ms"].items():
            assert len(samples) == manifest["repeats"] == 200
            assert math.isclose(statistics.median(samples), case["summary"][name]["median_ms"])
            ordered = sorted(samples)
            position = .95 * (len(samples) - 1)
            lo, hi = math.floor(position), math.ceil(position)
            p95 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
            assert math.isclose(p95, case["summary"][name]["p95_ms"])
    print("Validated EXP-005: 3 runs, 3000 timings, replay budgets, correctness, summaries, and hashes of committed and present files")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
