"""Validate EXP-017 provenance, errors, sparsity, and timing aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    assert data["experiment_id"] == "EXP-017" and data["status"] == "completed"
    assert data["shape"] == {"hidden": 2048, "intermediate": 5632, "tokens": 1}
    assert data["block_size"] == 8 and data["target_sparsity"] == 0.2
    assert len(data["results"]) == data["runs"]
    for relative, expected in data["source_sha256"].items():
        assert sha(ROOT / relative) == expected, f"source changed: {relative}"
    for result in data["results"]:
        assert abs(result["realized_sparsity"] - 0.2) <= 1 / (5632 / 8)
        assert result["fp32_relative_l2_vs_masked_dense_reference"] < 3e-4
        assert result["int8_relative_l2_vs_dequantized_reference"] < 3e-4
        for family in ("fp32", "int8"):
            for name, samples in result[family]["raw_timings_ms"].items():
                assert len(samples) == data["repeats"] and min(samples) > 0
                summary = result[family]["timings"][name]
                assert summary["median_ms"] == float(np.median(samples))
                assert summary["p95_ms"] == float(np.percentile(samples, 95))
    print("Validated EXP-017 provenance, numerical errors, sparsity, and timings")


if __name__ == "__main__":
    main()
