"""Validate EXP-016 model, corpus, source, quality, and memory provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDITIONS = [
    "dense", "neuron-s30",
    "block_output_norm-B8-s10", "block_output_norm-B8-s20", "block_output_norm-B8-s30",
    "block_output_norm-B32-s10", "block_output_norm-B32-s20", "block_output_norm-B32-s30",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    assert data["experiment_id"] == "EXP-016" and data["status"] == "completed"
    assert data["conditions"] == EXPECTED_CONDITIONS
    assert data["geometry"] == {
        "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22}
    assert sha(Path(data["model_metadata"])) == data["model_metadata_sha256"]
    assert sha(Path(data["corpus"])) == data["corpus_sha256"]
    for relative, expected in data["source_sha256"].items():
        assert sha(ROOT / relative) == expected, f"source changed: {relative}"
    assert [result["condition"] for result in data["results"]] == EXPECTED_CONDITIONS
    dense = data["results"][0]["perplexity"]
    for result in data["results"]:
        assert result["predicted_tokens"] > 0 and result["perplexity"] > 0
        assert result["resident_memory_bytes_after_condition"] <= 5.5 * 2**30
        if result["condition"] != "dense":
            expected = result["perplexity"] / dense - 1
            assert abs(result["relative_ppl_increase"] - expected) < 1e-12
            assert result["within_approved_threshold"] == (
                expected <= data["approved_max_relative_ppl_increase"])
    print("Validated EXP-016 TinyLlama provenance, quality metrics, and memory ceiling")


if __name__ == "__main__":
    main()
