"""Independently validate EXP-004 provenance and aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["experiment_id"] != "EXP-004" or manifest["status"] != "completed":
        raise ValueError("EXP-004 manifest is not completed")
    if sha(Path(manifest["corpus"])) != manifest["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    for relative, expected in manifest["source_sha256"].items():
        archived = subprocess.run(["git", "show", f"{manifest['git_commit']}:{relative}"],
                                  cwd=ROOT, capture_output=True, check=True).stdout
        if hashlib.sha256(archived).hexdigest() != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    snapshot = Path(manifest["snapshot"])
    for name, expected in manifest["model_file_sha256"].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"model hash mismatch: {name}")
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
          f"{manifest['results'][0]['predicted_tokens']} predicted tokens each, all hashes and aggregates")


if __name__ == "__main__":
    main()
