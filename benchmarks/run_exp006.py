"""Evaluate contribution-ranked blocks under fixed learned neuron layouts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from armsparse.sparsity.quality import (MaskCondition, activation_mask,
                                       block_statistics)
try:
    from benchmarks.run_exp003 import MODEL, REVISION, ROOT, command, load_corpus, parse_condition, sha
except ModuleNotFoundError:
    from run_exp003 import MODEL, REVISION, ROOT, command, load_corpus, parse_condition, sha


@contextmanager
def reordered_masked_mlp(model, condition: MaskCondition, permutations: torch.Tensor):
    handles = []
    if condition.method != "dense":
        for layer, block in enumerate(model.model.layers):
            permutation = permutations[layer]
            reordered_weight = block.mlp.down_proj.weight[:, permutation]
            statistics = block_statistics(reordered_weight, condition.block_size, condition.method)

            def hook(module, inputs, layer=layer, permutation=permutation, statistics=statistics):
                reordered = inputs[0].index_select(-1, permutation)
                mask = activation_mask(reordered, condition, layer, statistics)
                masked = torch.zeros_like(inputs[0])
                masked.index_copy_(-1, permutation, reordered * mask)
                return (masked, *inputs[1:])

            handles.append(block.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def dense_equivalence(model, permutations: torch.Tensor) -> list[dict]:
    """Check the algebraic layout transform independently in FP32."""
    generator = torch.Generator(device="cpu").manual_seed(6006)
    checks = []
    for layer, block in enumerate(model.model.layers):
        activation = torch.randn(block.mlp.intermediate_size, generator=generator)
        weight = block.mlp.down_proj.weight.float()
        permutation = permutations[layer]
        original = weight @ activation
        reordered = weight[:, permutation] @ activation[permutation]
        error = torch.linalg.vector_norm(original - reordered) / torch.linalg.vector_norm(original)
        checks.append({"layer": layer, "relative_l2_error": float(error)})
    if max(item["relative_l2_error"] for item in checks) > 2e-5:
        raise RuntimeError("dense permutation equivalence failed")
    return checks


def evaluate(model, tokenizer, records, condition, permutations, max_tokens):
    total_nll, predicted_tokens, examples = 0.0, 0, []
    started = time.perf_counter()
    with reordered_masked_mlp(model, condition, permutations), torch.inference_mode():
        for record in records:
            encoded = tokenizer(record["text"], return_tensors="pt", truncation=True,
                                max_length=max_tokens, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            logits = model(**encoded, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   input_ids[:, 1:].reshape(-1), reduction="sum")
            count, value = input_ids.shape[1] - 1, float(loss)
            total_nll += value
            predicted_tokens += count
            examples.append({"id": record["id"], "predicted_tokens": count, "nll": value})
    mean_nll = total_nll / predicted_tokens
    return {"condition": condition.name, "method": condition.method,
            "sparsity": condition.sparsity, "block_size": condition.block_size,
            "predicted_tokens": predicted_tokens, "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll), "evaluation_seconds": time.perf_counter() - started,
            "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--layout-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=parse_condition, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--approved-max-relative-ppl-increase", type=float, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned official revision")
    conditions = args.condition
    if not conditions or conditions[0].method != "dense" or any(
            condition.method not in {"dense", "block_output_norm"} for condition in conditions):
        parser.error("conditions must begin with dense and otherwise use block_output_norm")
    if len({condition.name for condition in conditions}) != len(conditions):
        parser.error("conditions must be unique")
    layout_metadata = json.loads(args.layout_metadata.read_text())
    if sha(args.layout) != layout_metadata["layout_sha256"]:
        parser.error("layout hash does not match metadata")
    with np.load(args.layout) as data:
        permutations = torch.from_numpy(data["permutation"].astype(np.int64))
    if permutations.shape != (16, 8192):
        parser.error("expected one 8192-neuron permutation for each of 16 layers")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = load_corpus(args.corpus)
    source_paths = [Path(__file__).resolve(), ROOT / "armsparse/sparsity/reorder.py",
                    ROOT / "armsparse/sparsity/quality.py",
                    ROOT / "research/experiments/EXP-006/protocol.md"]
    manifest = {"experiment_id": "EXP-006", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
                "model_revision": REVISION, "snapshot": str(args.snapshot.resolve()),
                "model_file_sha256": {path.name: sha(path) for path in sorted(args.snapshot.iterdir())
                                      if path.is_file()},
                "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
                "corpus": str(args.corpus.resolve()), "corpus_sha256": sha(args.corpus),
                "layout": str(args.layout.resolve()), "layout_sha256": sha(args.layout),
                "layout_metadata": layout_metadata,
                "record_ids": [record["id"] for record in records],
                "max_tokens_per_record": args.max_tokens,
                "conditions": [condition.name for condition in conditions],
                "approved_max_relative_ppl_increase": args.approved_max_relative_ppl_increase,
                "mask_semantics": "post-SwiGLU output-norm oracle over fixed calibration-only layout",
                "timing_scope": "operational evaluation duration; not inference latency",
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
        model = AutoModelForCausalLM.from_pretrained(args.snapshot, dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, attn_implementation="eager").eval()
        if psutil.Process().memory_info().rss > 5.5 * 2**30:
            raise RuntimeError("resident memory exceeds 5.5 GiB budget")
        manifest["dense_equivalence"] = dense_equivalence(model, permutations)
        for condition in conditions:
            result = evaluate(model, tokenizer, records, condition, permutations, args.max_tokens)
            if condition.method != "dense":
                baseline = manifest["results"][0]["perplexity"]
                result["relative_ppl_increase"] = result["perplexity"] / baseline - 1
                result["within_approved_threshold"] = result["relative_ppl_increase"] <= args.approved_max_relative_ppl_increase
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
