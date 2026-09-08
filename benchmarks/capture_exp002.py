"""Capture actual Llama FFN activations on CPU; never report this as a speed test."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import psutil
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "meta-llama/Llama-3.2-1B"
REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if not result.returncode else {"unavailable":result.stderr.strip()}


def swap_used():
    """Return system swap usage when psutil supports the current macOS build."""
    try:
        return psutil.swap_memory().used
    except (OSError, RuntimeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot.name != REVISION:
        parser.error("snapshot must be the pinned official revision")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = out / "sources"
    sources.mkdir()
    source_paths = [Path(__file__).resolve(), ROOT/"research/experiments/EXP-002/protocol.md",
                    ROOT/"research/experiments/EXP-002/prompts.json",
                    Path(transformers.__file__).parent/"models/llama/modeling_llama.py"]
    hashes = {}
    for source in source_paths:
        (sources/source.name).write_bytes(source.read_bytes())
        hashes[source.name] = sha(source)
    (out/"working-tree.patch").write_text(command(["git","diff","HEAD"]))
    model_files = {p.name: sha(p) for p in sorted(args.snapshot.iterdir()) if p.is_file()}
    config = json.loads((args.snapshot/"config.json").read_text())
    if (config["hidden_size"],config["intermediate_size"],config["num_hidden_layers"]) != (2048,8192,16):
        raise ValueError("unexpected model geometry")
    manifest = {"experiment_id":"EXP-002", "status":"running", "started_utc":datetime.now(timezone.utc).isoformat(),
                "model":MODEL, "model_revision":REVISION, "snapshot":str(args.snapshot.resolve()),
                "model_file_sha256":model_files, "source_sha256":hashes,
                "git_commit":command(["git","rev-parse","HEAD"]), "git_status":command(["git","status","--short"]),
                "hardware":{"cpu":command(["sysctl","-n","machdep.cpu.brand_string"]),
                            "memory_bytes":psutil.virtual_memory().total,"available_bytes_before_load":psutil.virtual_memory().available,
                            "system_swap_used_before_load":swap_used(),
                            "os":platform.platform(),"arch":platform.machine()},
                "software":{"python":sys.version,"torch":torch.__version__,"transformers":transformers.__version__,"numpy":np.__version__},
                "dtype":"bfloat16", "artifact_dtype":"float32", "device":"cpu", "threads":1,
                "capture_mode":"causal_teacher_forced_prefill", "max_tokens":64,"first_analyzed_position":8,
                "seed":17,"generated_tokens":0,"cache":False,"attention_backend":"eager",
                "timing_scope":"operational capture duration, not inference latency", "warmups":None,"repetitions":None,
                "config":config,"artifacts":[],"prompts":[]}
    write(out/"manifest.json",manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    handles = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True,trust_remote_code=False)
        model = AutoModel.from_pretrained(args.snapshot,dtype=torch.bfloat16,local_files_only=True,
                                         trust_remote_code=False,attn_implementation="eager").eval()
        if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 for p in model.parameters()):
            raise RuntimeError("model must remain BF16 CPU")
        manifest["resident_bytes_after_load"] = psutil.Process().memory_info().rss
        if manifest["resident_bytes_after_load"] > 5.5*2**30:
            raise RuntimeError("resident memory exceeds 5.5 GiB budget")
        captured = {}
        def input_hook(layer):
            def hook(module, inputs):
                captured.setdefault(layer,{})["x"] = inputs[0].detach().clone()
            return hook
        def activation_hook(layer):
            def hook(module, inputs):
                captured.setdefault(layer,{})["activation"] = inputs[0].detach().clone()
            return hook
        for layer, block in enumerate(model.layers):
            handles.append(block.mlp.register_forward_pre_hook(input_hook(layer)))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(activation_hook(layer)))
        prompts = json.loads((sources/"prompts.json").read_text())
        for prompt in prompts:
            captured.clear()
            tokens = tokenizer(prompt["text"],return_tensors="pt",truncation=True,max_length=64,add_special_tokens=True)
            ids = tokens["input_ids"][0].tolist()
            if len(ids) <= 8:
                raise ValueError("prompt too short")
            t = time.perf_counter()
            with torch.inference_mode():
                result = model(**tokens,use_cache=False)
                if not torch.isfinite(result.last_hidden_state).all():
                    raise RuntimeError("nonfinite model output")
                del result
                if set(captured) != set(range(16)):
                    raise RuntimeError("missing layer captures")
                for layer, values in captured.items():
                    mlp = model.layers[layer].mlp
                    # Same full sequence shape; separate functional expression.
                    recomputed = torch.nn.functional.silu(mlp.gate_proj(values["x"])) * mlp.up_proj(values["x"])
                    observed = values["activation"]
                    if not torch.equal(recomputed, observed):
                        raise RuntimeError(f"activation recomputation mismatch at layer {layer}")
                    if not torch.isfinite(observed).all():
                        raise RuntimeError("nonfinite activations")
                    path = out/f"{prompt['id']}-layer{layer:02d}.npz"
                    np.savez(path, x=values["x"][0,8:].float().numpy(),
                             activation=observed[0,8:].float().numpy(),
                             token_positions=np.arange(8,len(ids),dtype=np.int32),
                             token_ids=np.array(ids[8:],dtype=np.int32))
                    manifest["artifacts"].append({"path":path.name,"sha256":sha(path),"prompt_id":prompt["id"],
                                                  "layer":layer,"tokens":len(ids)-8,"recomputation":"bitwise_equal"})
            manifest["prompts"].append({**prompt,"input_ids":ids,"capture_seconds":time.perf_counter()-t})
            manifest["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            write(out/"manifest.json",manifest)
            print(f"Captured {prompt['id']}: {len(ids)-8} positions × 16 layers; peak RSS {manifest['peak_rss_bytes']/2**30:.2f} GiB",flush=True)
            if psutil.Process().memory_info().rss > 5.5*2**30:
                raise RuntimeError("resident memory exceeds 5.5 GiB budget")
        manifest.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat(),
                        capture_seconds=time.perf_counter()-started)
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for handle in handles:
            handle.remove()
        manifest["system_swap_used_at_end"] = swap_used()
        write(out/"manifest.json",manifest)


if __name__ == "__main__":
    main()
