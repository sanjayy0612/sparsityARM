"""Independently validate EXP-004 provenance and aggregates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research_tools.inputs import check_git_sources, check_sha


def validate(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["experiment_id"] != "EXP-004" or manifest["status"] != "completed":
        raise ValueError("EXP-004 manifest is not completed")
    check_sha(manifest["corpus"], manifest["corpus_sha256"], "corpus")
    check_git_sources(manifest)
    snapshot = Path(manifest["snapshot"])
    for name, expected in manifest["model_file_sha256"].items():
        check_sha(snapshot / name, expected, "model file")
    results = manifest["results"]
    if not results or results[0]["condition"] != "dense":
        raise ValueError("missing dense baseline")
    baseline = results[0]["perplexity"]
    threshold = manifest["approved_max_relative_ppl_increase"]
    for result in results:
        count = sum(example["predicted_tokens"] for example in result["examples"])
        mean_nll = sum(example["nll"] for example in result["examples"]) / count
        perplexity = math.exp(mean_nll)
        if count != result["predicted_tokens"] or not math.isclose(
                mean_nll, result["mean_nll"], rel_tol=1e-12):
            raise ValueError(f"NLL mismatch: {result['condition']}")
        if not math.isclose(perplexity, result["perplexity"], rel_tol=1e-12):
            raise ValueError(f"perplexity mismatch: {result['condition']}")
        if result["condition"] != "dense":
            relative = perplexity / baseline - 1
            if not math.isclose(relative, result["relative_ppl_increase"], rel_tol=1e-12):
                raise ValueError(f"relative perplexity mismatch: {result['condition']}")
            if (relative <= threshold) != result["within_approved_threshold"]:
                raise ValueError(f"threshold mismatch: {result['condition']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = validate(args.directory)
    print(f"Validated EXP-004: {len(manifest['results'])} conditions, "
          f"{manifest['results'][0]['predicted_tokens']} predicted tokens each, all aggregates, and hashes of inputs present")


if __name__ == "__main__":
    main()
