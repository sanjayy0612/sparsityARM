"""Reconstruct a deterministic B8/10% output-norm replay bank from EXP-002."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from armsparse.sparsity.quality import MaskCondition, activation_mask, block_statistics


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_file(snapshot: Path, key: str) -> Path:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return snapshot / index["weight_map"][key]
    single = snapshot / "model.safetensors"
    if single.exists():
        return single
    raise FileNotFoundError("checkpoint has neither indexed shards nor model.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        parser.error("refusing to overwrite replay artifacts")
    capture_manifest = json.loads((args.capture / "manifest.json").read_text())
    condition = MaskCondition("block_output_norm", 0.10, 8)
    statistics = {}
    masks, sources = [], []
    # One fixed token across all 16 layers of the first prompt gives a compact
    # real-model bank while keeping independent correctness checks practical.
    for item in capture_manifest["artifacts"][:16]:
        layer = item["layer"]
        if layer not in statistics:
            key = f"model.layers.{layer}.mlp.down_proj.weight"
            with safe_open(tensor_file(args.snapshot, key), framework="pt", device="cpu") as handle:
                statistics[layer] = block_statistics(handle.get_tensor(key), 8, condition.method)
        artifact = args.capture / item["path"]
        if sha(artifact) != item["sha256"]:
            raise ValueError(f"capture hash mismatch: {artifact}")
        with np.load(artifact) as data:
            activation = torch.from_numpy(data["activation"][0:1])
        mask = activation_mask(activation, condition, layer, statistics[layer])
        masks.append(mask.reshape(-1, 8).any(dim=1).numpy().astype(np.uint8))
        sources.append({"capture": item["path"], "capture_sha256": item["sha256"],
                        "prompt_id": item["prompt_id"], "layer": layer,
                        "token_offset_in_capture": 0})
    block_masks = np.stack(masks)
    if block_masks.shape != (16, 1024) or not np.all(block_masks.sum(axis=1) == 922):
        raise ValueError(f"unexpected replay mask shape or budget: {block_masks.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, block_masks=block_masks)
    metadata = {"experiment_id": "EXP-005", "kind": "offline_replay_mask_bank",
                "source_experiment": "EXP-002", "score": "block_output_norm",
                "block_size": 8, "target_sparsity": 0.10,
                "active_blocks_per_mask": 922, "mask_count": len(masks),
                "mask_bank": str(args.output.resolve()), "mask_bank_sha256": sha(args.output),
                "capture_manifest_sha256": sha(args.capture / "manifest.json"),
                "model_revision": capture_manifest["model_revision"], "sources": sources}
    args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {block_masks.shape} replay bank to {args.output}")


if __name__ == "__main__":
    main()
