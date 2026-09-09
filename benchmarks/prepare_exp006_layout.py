"""Learn fixed per-layer neuron layouts from the retained EXP-002 calibration set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from armsparse.sparsity.reorder import lsh_permutation


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6100)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        parser.error("refusing to overwrite layout artifacts")
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_layer = {layer: [] for layer in range(16)}
    sources = []
    for item in manifest["artifacts"]:
        path = args.capture / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"capture hash mismatch: {path}")
        with np.load(path) as data:
            by_layer[item["layer"]].append(data["activation"].astype(np.float32))
        sources.append({"path": item["path"], "sha256": item["sha256"],
                        "prompt_id": item["prompt_id"], "layer": item["layer"]})
    permutations, sample_counts = [], []
    for layer in range(16):
        calibration = np.concatenate(by_layer[layer], axis=0)
        if calibration.shape[1] != 8192:
            raise ValueError(f"unexpected layer {layer} calibration shape {calibration.shape}")
        permutations.append(lsh_permutation(calibration, args.projection_dim, args.seed + layer))
        sample_counts.append(calibration.shape[0])
    permutation = np.stack(permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, permutation=permutation)
    metadata = {"experiment_id": "EXP-006", "kind": "fixed_per_layer_neuron_layout",
                "method": "centered_absolute_activation_random_hyperplane_lsh_lexicographic",
                "projection_dim": args.projection_dim, "seed": args.seed,
                "shape": list(permutation.shape), "calibration_samples_per_layer": sample_counts,
                "compatible_block_sizes": [8, 16, 32],
                "layout_path": str(args.output.resolve()), "layout_sha256": sha(args.output),
                "capture_manifest_sha256": sha(manifest_path),
                "model_revision": manifest["model_revision"], "sources": sources}
    args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {permutation.shape} fixed layout to {args.output}")


if __name__ == "__main__":
    main()
