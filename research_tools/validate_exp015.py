"""Validate EXP-015 provenance, correctness, and retained timing samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPARSITIES = [value / 10 for value in range(1, 9)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    assert data["experiment_id"] == "EXP-015" and data["status"] == "completed"
    assert data["block_size"] == 32 and data["sparsities"] == EXPECTED_SPARSITIES
    assert len(data["results"]) == data["runs"] * len(EXPECTED_SPARSITIES)
    for relative, expected in data["source_sha256"].items():
        assert sha(ROOT / relative) == expected, f"source changed: {relative}"
    for result in data["results"]:
        assert result["relative_l2_vs_masked_dense_control"] <= 3e-5
        assert abs(result["realized_sparsity"] - result["target_sparsity"]) <= 1 / 256
        for name in ("q8_0_dense_ms", "q8_0_b32_ms"):
            samples = result["raw_timings_ms"][name]
            assert len(samples) == data["repeats"] and min(samples) > 0
            assert result["timings"][name]["median_ms"] > 0
    print("Validated EXP-015 Q8_0 B32 replay provenance, correctness, and samples")


if __name__ == "__main__":
    main()
