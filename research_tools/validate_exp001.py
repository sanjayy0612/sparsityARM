"""Validate EXP-001 artifact integrity and measurement completeness (stdlib only)."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed", "incomplete experiment"
    config = manifest["configuration"]
    assert len(manifest["case_artifacts"]) == 24*config["runs"]
    seen_cases, cached_hashes = set(), {}
    for source, expected_hash in manifest["source_sha256"].items():
        assert sha(directory / "sources" / Path(source).name) == expected_hash
    for artifact in manifest["case_artifacts"]:
        path = directory / artifact["path"]
        assert sha(path) == artifact["sha256"], f"changed case: {path}"
        case = json.loads(path.read_text())
        assert case["experiment_id"] == "EXP-001"
        key = (case["seed"], case["block_size"], case["target_sparsity"], case["family"])
        assert key not in seen_cases, "duplicate case"
        seen_cases.add(key)
        assert case["shape"] == {"tokens": 1, "hidden": config["hidden"], "intermediate": config["intermediate"]}
        for item in case["artifacts"].values():
            if item["path"] not in cached_hashes:
                cached_hashes[item["path"]] = sha(directory / item["path"])
            assert cached_hashes[item["path"]] == item["sha256"], item["path"]
        assert set(case["raw_ms"]) == {"dense_accelerate", "irregular_native", "irregular_expanded_native", "block_native", "block_accelerate"}
        for name, values in case["raw_ms"].items():
            assert len(values) == config["repeats"]
            assert all(isinstance(x, (int, float)) and 0 <= x < float("inf") for x in values)
            assert math.isclose(statistics.median(values), case["summary"][name]["median_ms"])
            ordered = sorted(values)
            position = .95*(len(values)-1)
            lo, hi = math.floor(position), math.ceil(position)
            p95 = ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)
            assert math.isclose(p95, case["summary"][name]["p95_ms"])
        assert len(case["sample_schedule"]) == config["repeats"]
        for sample in case["sample_schedule"]:
            assert sorted(sample["order"]) == sorted(case["raw_ms"])
            assert 0 <= sample["mask_index"] < config["bank"]
        assert all(a <= b for a, b in zip(case["active_neurons_base"], case["active_neurons_expanded"]))
        if case["family"] == "clustered":
            assert case["active_neurons_base"] == case["active_neurons_expanded"]
    expected = {(config["seed"]+run, b, s, family) for run in range(config["runs"])
                for b in (8,16,32,64) for s in (.2,.3,.4) for family in ("scattered", "clustered")}
    assert seen_cases == expected, "missing or unexpected cases"
    print(f"Validated {len(manifest['case_artifacts'])} cases and all source/data hashes: {directory}")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
