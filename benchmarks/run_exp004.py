"""Compare contribution-aware block-mask quality on CPU (not a speed test)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from benchmarks.run_exp003 import (MODEL, REVISION, ROOT, command, evaluate, load_corpus,
                                       parse_condition, sha)
else:
    from run_exp003 import (MODEL, REVISION, ROOT, command, evaluate, load_corpus,
                            parse_condition, sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=parse_condition, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--approved-max-relative-ppl-increase", type=float, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned official revision")
    if args.max_tokens < 2 or args.approved_max_relative_ppl_increase < 0:
        parser.error("invalid token limit or acceptance threshold")
    names = [condition.name for condition in args.condition]
    if not names or names[0] != "dense" or len(names) != len(set(names)):
        parser.error("conditions must be unique and begin with dense")
    allowed = {"dense", "block", "block_weight_proxy", "block_output_norm"}
    if any(condition.method not in allowed for condition in args.condition):
        parser.error("EXP-004 accepts dense and the three registered block score methods only")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = load_corpus(args.corpus)
    source_paths = [Path(__file__).resolve(),
                    ROOT / "benchmarks/run_exp003.py", ROOT / "armsparse/sparsity/quality.py",
                    ROOT / "research/experiments/EXP-004/protocol.md"]
    manifest = {
        "experiment_id": "EXP-004", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "model_revision": REVISION, "snapshot": str(args.snapshot.resolve()),
        "model_file_sha256": {path.name: sha(path) for path in sorted(args.snapshot.iterdir())
                              if path.is_file()},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "corpus": str(args.corpus.resolve()), "corpus_sha256": sha(args.corpus),
        "record_ids": [record["id"] for record in records],
        "max_tokens_per_record": args.max_tokens, "conditions": names,
        "approved_max_relative_ppl_increase": args.approved_max_relative_ppl_increase,
        "score_semantics": {
            "block": "sum(abs(post-SwiGLU activation))",
            "block_weight_proxy": "L2(activation block) * Frobenius(down-weight block)",
            "block_output_norm": "squared L2 norm of each block's isolated down-projection contribution",
        },
        "caveat": "scores inspect post-SwiGLU activations; output_norm ranks isolated blocks and is not the combinatorial optimum",
        "timing_scope": "operational evaluation duration; must not be reported as inference latency",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": psutil.virtual_memory().total, "os": platform.platform(),
                     "arch": platform.machine()},
        "software": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "dtype": "bfloat16", "score_dtype": "float32", "device": "cpu", "threads": 1,
        "results": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.snapshot, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False, attn_implementation="eager").eval()
        if any(parameter.device.type != "cpu" or parameter.dtype != torch.bfloat16
               for parameter in model.parameters()):
            raise RuntimeError("model must remain BF16 CPU")
        if psutil.Process().memory_info().rss > 5.5 * 2**30:
            raise RuntimeError("resident memory exceeds 5.5 GiB budget")
        for condition in args.condition:
            result = evaluate(model, tokenizer, records, condition, args.max_tokens)
            if condition.method != "dense":
                baseline = manifest["results"][0]["perplexity"]
                result["relative_ppl_increase"] = result["perplexity"] / baseline - 1
                result["within_approved_threshold"] = (
                    result["relative_ppl_increase"] <= args.approved_max_relative_ppl_increase)
            manifest["results"].append(result)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"{condition.name}: ppl={result['perplexity']:.6f}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
