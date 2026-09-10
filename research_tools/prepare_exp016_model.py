"""Download only the pinned TinyLlama files required by EXP-016."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
REVISION = "59f6f375b26bde864a6ca194a9a3044570490064"
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.metadata.exists():
        parser.error("refusing to overwrite model metadata")
    snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION,
                                      allow_patterns=ALLOW_PATTERNS))
    config = json.loads((snapshot / "config.json").read_text())
    expected = {"hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22}
    if any(config.get(key) != value for key, value in expected.items()):
        parser.error(f"unexpected model geometry: {config}")
    files = sorted(path for path in snapshot.iterdir() if path.is_file())
    metadata = {
        "experiment_id": "EXP-016", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "model_revision": REVISION, "snapshot": str(snapshot.resolve()),
        "geometry": expected,
        "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in files},
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(snapshot)


if __name__ == "__main__":
    main()
