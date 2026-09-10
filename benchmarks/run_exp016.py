"""Validate block-mask quality on the pinned TinyLlama 1.1B checkpoint."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from benchmarks.run_exp003 import ROOT, command, evaluate, load_corpus, parse_condition, sha
except ModuleNotFoundError:
    from run_exp003 import ROOT, command, evaluate, load_corpus, parse_condition, sha

MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
REVISION = "59f6f375b26bde864a6ca194a9a3044570490064"
EXPECTED_GEOMETRY = {"hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 22}
APPROVED_CONDITIONS = [
    "dense", "neuron-s30",
    "block_output_norm-B8-s10", "block_output_norm-B8-s20", "block_output_norm-B8-s30",
    "block_output_norm-B32-s10", "block_output_norm-B32-s20", "block_output_norm-B32-s30",
]


def validate_geometry(config) -> None:
    actual = {name: getattr(config, name, None) for name in EXPECTED_GEOMETRY}
    if actual != EXPECTED_GEOMETRY:
        raise ValueError(f"unexpected TinyLlama geometry: {actual}")
    if config.intermediate_size % 32:
        raise ValueError("TinyLlama intermediate width must be divisible by B32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=parse_condition, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--approved-max-relative-ppl-increase", type=float, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned TinyLlama revision")
    if args.max_tokens < 2 or args.approved_max_relative_ppl_increase < 0:
        parser.error("invalid token limit or acceptance threshold")
    names = [condition.name for condition in args.condition]
    if names != APPROVED_CONDITIONS:
        parser.error(f"conditions must exactly match the approved EXP-016 screen: {APPROVED_CONDITIONS}")
    metadata = json.loads(args.model_metadata.read_text())
    if metadata.get("model") != MODEL or metadata.get("model_revision") != REVISION:
        parser.error("model metadata does not identify the pinned TinyLlama snapshot")
    for filename, record in metadata.get("files", {}).items():
        path = args.snapshot / filename
        if not path.is_file() or sha(path) != record["sha256"]:
            parser.error(f"snapshot hash mismatch: {filename}")
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    validate_geometry(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = load_corpus(args.corpus)
    source_paths = [Path(__file__).resolve(), ROOT / "benchmarks/run_exp003.py",
                    ROOT / "armsparse/sparsity/quality.py",
                    ROOT / "research/experiments/EXP-016/protocol.md"]
    manifest = {
        "experiment_id": "EXP-016", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "model_revision": REVISION, "snapshot": str(args.snapshot.resolve()),
        "model_metadata": str(args.model_metadata.resolve()),
        "model_metadata_sha256": sha(args.model_metadata), "geometry": EXPECTED_GEOMETRY,
        "model_file_sha256": {name: record["sha256"] for name, record in metadata["files"].items()},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "corpus": str(args.corpus.resolve()), "corpus_sha256": sha(args.corpus),
        "record_ids": [record["id"] for record in records],
        "max_tokens_per_record": args.max_tokens, "conditions": names,
        "approved_max_relative_ppl_increase": args.approved_max_relative_ppl_increase,
        "mask_semantics": "post-SwiGLU oracle/reference; selector cost excluded",
        "score_semantics": "squared L2 norm of each block's isolated down-projection contribution",
        "timing_scope": "operational evaluation duration; must not be reported as inference latency",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": psutil.virtual_memory().total, "os": platform.platform(),
                     "arch": platform.machine()},
        "software": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "dtype": "bfloat16", "score_dtype": "float32", "device": "cpu", "threads": 1,
        "results": [],
    }
    manifest_path = output / "manifest.json"
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
        validate_geometry(model.config)
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.bfloat16
               for parameter in model.parameters()):
            raise RuntimeError("model must remain BF16 CPU")
        for condition in args.condition:
            result = evaluate(model, tokenizer, records, condition, args.max_tokens)
            result["resident_memory_bytes_after_condition"] = psutil.Process().memory_info().rss
            if result["resident_memory_bytes_after_condition"] > 5.5 * 2**30:
                raise RuntimeError("resident memory exceeds 5.5 GiB budget")
            if condition.method != "dense":
                baseline = manifest["results"][0]["perplexity"]
                result["relative_ppl_increase"] = result["perplexity"] / baseline - 1
                result["within_approved_threshold"] = (
                    result["relative_ppl_increase"] <= args.approved_max_relative_ppl_increase)
            manifest["results"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"{condition.name}: ppl={result['perplexity']:.6f}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
