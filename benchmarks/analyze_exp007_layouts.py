"""Compare calibration-to-held-out block concentration before another PPL run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from armsparse.sparsity.reorder import hot_cold_permutation, lsh_permutation


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout.strip()


def top_mask(scores: np.ndarray, keep: int) -> np.ndarray:
    if scores.ndim != 2 or keep < 1 or keep > scores.shape[1]:
        raise ValueError("invalid score matrix or keep count")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :keep]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def metrics(activations: np.ndarray, permutation: np.ndarray,
            block_size: int, sparsity: float) -> dict:
    if activations.ndim != 2 or activations.shape[1] % block_size:
        raise ValueError("activations must be 2D and divisible by block size")
    if permutation.shape != (activations.shape[1],) or not np.array_equal(
            np.sort(permutation), np.arange(activations.shape[1])):
        raise ValueError("permutation must contain every neuron exactly once")
    scores = np.abs(activations.astype(np.float32, copy=False))[:, permutation]
    keep = round(scores.shape[1] * (1 - sparsity))
    neuron = top_mask(scores, keep)
    shaped = neuron.reshape(len(scores), -1, block_size)
    expanded = np.repeat(shaped.any(axis=2), block_size, axis=1)
    boundary = np.logical_and(shaped.any(axis=2), ~shaped.all(axis=2))
    block_scores = scores.reshape(len(scores), -1, block_size).sum(axis=2)
    keep_blocks = round(block_scores.shape[1] * (1 - sparsity))
    block_mask = np.repeat(top_mask(block_scores, keep_blocks), block_size, axis=1)
    mass = scores.sum(axis=1)
    return {"samples": len(scores),
            "mean_expanded_sparsity": float(np.mean(1 - expanded.mean(axis=1))),
            "mean_boundary_block_fraction": float(np.mean(boundary.mean(axis=1))),
            "mean_direct_block_retained_l1": float(np.mean(
                (scores * block_mask).sum(axis=1) / np.maximum(mass, 1e-12)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7100)
    args = parser.parse_args()
    if args.output.exists() or args.layouts.exists():
        parser.error("refusing to overwrite EXP-007 artifacts")
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != "EXP-002" or manifest.get("status") != "completed":
        parser.error("capture must be a completed EXP-002 artifact set")
    layers = int(manifest["config"]["num_hidden_layers"])
    width = int(manifest["config"]["intermediate_size"])
    data = {(prompt, layer): [] for prompt in range(1, 9) for layer in range(layers)}
    source_hashes = []
    for item in manifest["artifacts"]:
        path = args.capture / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"capture hash mismatch: {path}")
        prompt = int(item["prompt_id"][1:])
        if (prompt, item["layer"]) not in data:
            parser.error(f"unexpected capture key: P{prompt:02d}/layer {item['layer']}")
        with np.load(path) as artifact:
            activation = artifact["activation"].astype(np.float32)
        if activation.ndim != 2 or activation.shape[1] != width:
            parser.error(f"invalid activation shape in {path}")
        data[(prompt, item["layer"])].append(activation)
        source_hashes.append({"path": item["path"], "sha256": item["sha256"]})
    missing = [key for key, arrays in data.items() if len(arrays) != 1]
    if missing:
        parser.error(f"capture must contain exactly one artifact per prompt/layer: {missing[:3]}")
    layouts = {"original": [], "lsh": [], "hot_cold": []}
    rows = []
    for layer in range(layers):
        calibration = np.concatenate(
            [data[(prompt, layer)][0] for prompt in range(1, 5)], axis=0)
        held_out = np.concatenate(
            [data[(prompt, layer)][0] for prompt in range(5, 9)], axis=0)
        layer_layouts = {"original": np.arange(width, dtype=np.int32),
                         "lsh": lsh_permutation(calibration, 32, args.seed + layer),
                         "hot_cold": hot_cold_permutation(calibration, 0.5)}
        for name, permutation in layer_layouts.items():
            layouts[name].append(permutation)
            for split, activations in (("calibration", calibration), ("held_out", held_out)):
                for block_size in (8, 16, 32):
                    for sparsity in (0.2, 0.3):
                        rows.append({"layout": name, "split": split, "layer": layer,
                                     "block_size": block_size, "target_sparsity": sparsity,
                                     **metrics(activations, permutation, block_size, sparsity)})
    args.layouts.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.layouts, **{name: np.stack(value) for name, value in layouts.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "armsparse/sparsity/reorder.py",
                    Path(__file__).resolve().parents[1] / "research/experiments/EXP-007/protocol.md"]
    output = {"experiment_id": "EXP-007", "status": "completed",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "question": "Does calibration-learned ordering improve held-out block concentration?",
              "calibration_prompts": ["P01", "P02", "P03", "P04"],
              "held_out_prompts": ["P05", "P06", "P07", "P08"],
              "model_revision": manifest["model_revision"], "seed": args.seed,
              "lsh_projection_dim": 32, "hot_cold_active_fraction": 0.5,
              "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha(path)
                                for path in source_paths},
              "git_commit": command(["git", "rev-parse", "HEAD"]),
              "git_status": command(["git", "status", "--short"]),
              "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                           "os": platform.platform(), "arch": platform.machine()},
              "software": {"python": sys.version, "numpy": np.__version__},
              "capture_manifest_sha256": sha(manifest_path), "source_artifacts": source_hashes,
              "layouts_path": str(args.layouts.resolve()), "layouts_sha256": sha(args.layouts),
              "rows": rows}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} calibration and held-out layout metrics to {args.output}")


if __name__ == "__main__":
    main()
