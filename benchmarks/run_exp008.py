"""Screen hot/cold-reordered blocks for held-out Llama language quality."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from benchmarks.run_exp003 import MODEL, REVISION, ROOT, command, load_corpus, sha
    from benchmarks.run_exp006 import dense_equivalence, evaluate
except ModuleNotFoundError:
    from run_exp003 import MODEL, REVISION, ROOT, command, load_corpus, sha
    from run_exp006 import dense_equivalence, evaluate
from armsparse.sparsity.quality import MaskCondition


CONDITIONS = (
    MaskCondition("dense", 0),
    MaskCondition("block_output_norm", 0.2, 8),
    MaskCondition("block_output_norm", 0.3, 8),
    MaskCondition("block_output_norm", 0.2, 16),
    MaskCondition("block_output_norm", 0.3, 16),
)


def load_hot_cold_layout(layout_path: Path, metadata_path: Path) -> tuple[torch.Tensor, dict]:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("experiment_id") != "EXP-007" or metadata.get("status") != "completed":
        raise ValueError("layout metadata must be a completed EXP-007 analysis")
    if sha(layout_path) != metadata.get("layouts_sha256"):
        raise ValueError("layout hash does not match EXP-007 metadata")
    with np.load(layout_path) as archive:
        if "hot_cold" not in archive.files:
            raise ValueError("layout archive has no hot_cold permutation")
        values = archive["hot_cold"].astype(np.int64)
    if values.shape != (16, 8192):
        raise ValueError("expected one 8192-neuron permutation for each of 16 layers")
    expected = np.arange(8192)
    if any(not np.array_equal(np.sort(row), expected) for row in values):
        raise ValueError("every layout row must be a complete permutation")
    return torch.from_numpy(values), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--layout-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--approved-max-relative-ppl-increase", type=float, default=0.05)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned official revision")
    if args.approved_max_relative_ppl_increase != 0.05:
        parser.error("EXP-008 uses the preregistered 5% perplexity gate")
    permutations, layout_metadata = load_hot_cold_layout(args.layout, args.layout_metadata)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = load_corpus(args.corpus)
    source_paths = [Path(__file__).resolve(), ROOT / "benchmarks/run_exp006.py",
                    ROOT / "benchmarks/run_exp003.py", ROOT / "armsparse/sparsity/reorder.py",
                    ROOT / "armsparse/sparsity/quality.py",
                    ROOT / "research/experiments/EXP-008/protocol.md"]
    manifest = {
        "experiment_id": "EXP-008", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "model_revision": REVISION,
        "snapshot": str(args.snapshot.resolve()),
        "model_file_sha256": {path.name: sha(path) for path in sorted(args.snapshot.iterdir())
                              if path.is_file()},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "corpus": str(args.corpus.resolve()), "corpus_sha256": sha(args.corpus),
        "layout": str(args.layout.resolve()), "layout_sha256": sha(args.layout),
        "layout_key": "hot_cold", "layout_metadata": str(args.layout_metadata.resolve()),
        "layout_metadata_sha256": sha(args.layout_metadata),
        "exp007_capture_manifest_sha256": layout_metadata["capture_manifest_sha256"],
        "calibration_prompts": layout_metadata["calibration_prompts"],
        "record_ids": [record["id"] for record in records],
        "max_tokens_per_record": args.max_tokens,
        "conditions": [condition.name for condition in CONDITIONS],
        "approved_max_relative_ppl_increase": args.approved_max_relative_ppl_increase,
        "mask_semantics": "post-SwiGLU output-norm oracle over fixed EXP-007 hot/cold layout",
        "timing_scope": "operational evaluation duration; not inference latency",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": psutil.virtual_memory().total, "os": platform.platform(),
                     "arch": platform.machine()},
        "software": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__},
        "dtype": "bfloat16", "score_dtype": "float32", "device": "cpu", "threads": 1,
        "results": [],
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.snapshot, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False, attn_implementation="eager").eval()
        if psutil.Process().memory_info().rss > 5.5 * 2**30:
            raise RuntimeError("resident memory exceeds 5.5 GiB budget")
        manifest["dense_equivalence"] = dense_equivalence(model, permutations)
        for condition in CONDITIONS:
            result = evaluate(model, tokenizer, records, condition, permutations, args.max_tokens)
            if condition.method != "dense":
                baseline = manifest["results"][0]["perplexity"]
                result["relative_ppl_increase"] = result["perplexity"] / baseline - 1
                result["within_approved_threshold"] = (
                    result["relative_ppl_increase"] <= args.approved_max_relative_ppl_increase)
            manifest["results"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"{condition.name}: ppl={result['perplexity']:.6f}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
