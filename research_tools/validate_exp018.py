"""Validate EXP-018 cases, provenance, correctness, and retained medians."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from research_tools.inputs import check_git_sources

SPARSITIES = [0.2, 0.3, 0.4, 0.5, 0.6]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["experiment_id"] == "EXP-018" and manifest["status"] == "completed"
    assert manifest["git_status"] == ""
    assert manifest["shape"] == {"hidden": 2048, "intermediate": 5632, "tokens": 1}
    assert manifest["sparsities"] == SPARSITIES and manifest["block_size"] == 8
    assert len(manifest["cases"]) == manifest["runs"] * len(SPARSITIES)
    check_git_sources(manifest)
    seen = set()
    for item in manifest["cases"]:
        path = directory / item["path"]
        assert sha(path) == item["sha256"]
        case = json.loads(path.read_text())
        key = (case["seed"], case["target_sparsity"])
        assert key not in seen
        seen.add(key)
        assert case["shape"] == {"tokens": 1, "hidden": 2048, "intermediate": 5632}
        assert case["block_size"] == 8 and case["family"] == "clustered"
        assert case["active_neurons_base"] == case["active_neurons_expanded"]
        for result in case["correctness"].values():
            assert result["max_relative_l2_error"] < 3e-4
        for name, values in case["raw_ms"].items():
            assert len(values) == manifest["repeats"]
            assert math.isclose(statistics.median(values), case["summary"][name]["median_ms"])
    print("Validated EXP-018 cases, provenance, correctness, and retained medians")


if __name__ == "__main__":
    validate(sys.argv[1])
