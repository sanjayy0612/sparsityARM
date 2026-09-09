"""Measure Llama quality under oracle/reference masks on CPU (not a speed test)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import psutil
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from armsparse.sparsity.quality import MaskCondition, apply_activation_mask

ROOT = Path(__file__).resolve().parents[1]
MODEL = "meta-llama/Llama-3.2-1B"
REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}


def load_corpus(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("id"), str) or not isinstance(record.get("text"), str):
            raise ValueError(f"invalid corpus record at line {line_number}")
        if not record["text"].strip():
            raise ValueError(f"empty corpus text at line {line_number}")
        records.append(record)
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("corpus must contain records with unique IDs")
    return records


@contextmanager
def masked_mlp(model, condition: MaskCondition):
    handles = []
    if condition.method != "dense":
        for layer, block in enumerate(model.model.layers):
            def hook(module, inputs, layer=layer):
                return (apply_activation_mask(inputs[0], condition, layer), *inputs[1:])
            handles.append(block.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate(model, tokenizer, records: list[dict], condition: MaskCondition, max_tokens: int):
    total_nll = 0.0
    predicted_tokens = 0
    examples = []
    started = time.perf_counter()
    with masked_mlp(model, condition), torch.inference_mode():
        for record in records:
            encoded = tokenizer(record["text"], return_tensors="pt", truncation=True,
                                max_length=max_tokens, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] < 2:
                raise ValueError(f"record {record['id']} has fewer than two tokens")
            logits = model(**encoded, use_cache=False).logits
            loss_sum = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       input_ids[:, 1:].reshape(-1), reduction="sum")
            count = input_ids.shape[1] - 1
            value = float(loss_sum)
            total_nll += value
            predicted_tokens += count
            examples.append({"id": record["id"], "predicted_tokens": count, "nll": value})
    mean_nll = total_nll / predicted_tokens
    return {"condition": condition.name, "method": condition.method,
            "sparsity": condition.sparsity, "block_size": condition.block_size,
            "predicted_tokens": predicted_tokens, "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll), "evaluation_seconds": time.perf_counter() - started,
            "examples": examples}


def parse_condition(value: str) -> MaskCondition:
    if value == "dense":
        return MaskCondition("dense", 0)
    parts = value.split(":")
    try:
        if parts[0] == "neuron" and len(parts) == 2:
            return MaskCondition("neuron", float(parts[1]))
        if parts[0] in {"block", "random_block"} and len(parts) == 3:
            return MaskCondition(parts[0], float(parts[2]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    raise argparse.ArgumentTypeError("use dense, neuron:0.2, block:8:0.2, or random_block:8:0.2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=parse_condition, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--approved-max-relative-ppl-increase", type=float, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned official revision")
    if args.max_tokens < 2 or args.approved_max_relative_ppl_increase < 0:
        parser.error("invalid token limit or acceptance threshold")
    names = [condition.name for condition in args.condition]
    if not names or names[0] != "dense" or len(names) != len(set(names)):
        parser.error("conditions must be unique and begin with dense")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = load_corpus(args.corpus)
    source_paths = [Path(__file__).resolve(), ROOT / "armsparse/sparsity/quality.py",
                    ROOT / "research/experiments/EXP-003/protocol.md"]
    manifest = {
        "experiment_id": "EXP-003", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "model_revision": REVISION, "snapshot": str(args.snapshot.resolve()),
        "model_file_sha256": {path.name: sha(path) for path in sorted(args.snapshot.iterdir())
                              if path.is_file()},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "corpus": str(args.corpus.resolve()), "corpus_sha256": sha(args.corpus),
        "record_ids": [record["id"] for record in records], "max_tokens_per_record": args.max_tokens,
        "conditions": names, "approved_max_relative_ppl_increase": args.approved_max_relative_ppl_increase,
        "mask_semantics": "post-SwiGLU activation-aware oracle/reference; selector cost excluded",
        "timing_scope": "operational evaluation duration; must not be reported as inference latency",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": psutil.virtual_memory().total, "os": platform.platform(),
                     "arch": platform.machine()},
        "software": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "dtype": "bfloat16", "device": "cpu", "threads": 1, "results": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, attn_implementation="eager").eval()
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
                    result["relative_ppl_increase"] <= args.approved_max_relative_ppl_increase
                )
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
