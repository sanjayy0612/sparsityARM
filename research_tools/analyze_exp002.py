"""Verify capture and save every layer/prompt/token mask-coverage measurement."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from armsparse.sparsity.coverage import coverage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["status"] != "completed" or len(manifest["artifacts"]) != 128:
        raise ValueError("incomplete primary capture")
    for name, value in manifest["source_sha256"].items():
        if sha(directory/"sources"/name) != value:
            raise ValueError(f"source hash mismatch: {name}")
    out = directory/"analysis"
    out.mkdir(exist_ok=False)
    for source in [Path(__file__).resolve(),ROOT/"armsparse/sparsity/coverage.py"]:
        (out/source.name).write_bytes(source.read_bytes())
    rows, outputs = [], []
    seen = set()
    for item in manifest["artifacts"]:
        path = directory/item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"capture hash mismatch: {path}")
        key = (item["prompt_id"],item["layer"])
        if key in seen or item["recomputation"] != "bitwise_equal":
            raise ValueError("invalid capture")
        seen.add(key)
        with np.load(path,allow_pickle=False) as data:
            activations = data["activation"]
            positions = data["token_positions"]
            if activations.shape != (item["tokens"],8192) or data["x"].shape != (item["tokens"],2048):
                raise ValueError("invalid capture shapes")
        raw = {"token_positions":positions}
        for b in (8,16,32,64):
            for s in (.2,.3,.4):
                metrics = coverage(activations,b,s,seed=17+item["layer"])
                prefix = f"B{b}_s{round(s*100)}"
                for name, value in metrics.items():
                    raw[prefix+"_"+name] = value
                for i, position in enumerate(positions):
                    rows.append({"prompt_id":item["prompt_id"],"layer":item["layer"],"token_position":int(position),
                                 "block_size":b,"target_sparsity":s,
                                 **{name:float(metrics[name][i]) for name in ["base_sparsity","expanded_sparsity","budget_sparsity","neuron_retained_l1","budget_retained_l1"]},
                                 "permuted_expanded_sparsity_mean":float(metrics["permuted_expanded_sparsity"][:,i].mean()),
                                 "zero_mass":bool(metrics["zero_mass"][i])})
        target = out/(path.stem+"-masks.npz")
        np.savez_compressed(target,**raw)
        outputs.append({"path":target.name,"sha256":sha(target)})
        print(f"Analyzed {len(seen)}/128: {path.name}",flush=True)
    expected = {(p["id"],layer) for p in manifest["prompts"] for layer in range(16)}
    if seen != expected:
        raise ValueError("missing layer/prompt pairs")
    with (out/"metrics.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# EXP-002: Llama 3.2 1B magnitude-mask coverage", "",
             f"Model revision: `{manifest['model_revision']}`. Eight fixed prompts, all 16 layers, CPU BF16 teacher-forced prefill.","",
             f"Analyzed {sum(a['tokens'] for a in manifest['artifacts'])} layer/token activations. Token positions 8 onward, capped at 64 input tokens per prompt. Every captured activation was bitwise equal to independently recomputed SwiGLU activation.","",
             "Means below weight every analyzed layer/token equally. Layer min/max are ranges of layer means, not independent-run uncertainty. The corpus is a pilot convenience sample.","",
             "| Target skip | B | Expanded skip mean | Expanded skip layer range | Permuted skip mean | Budget-block actual skip | Neuron retained L1 | Budget-block retained L1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    aggregate = []
    for s in (.2,.3,.4):
        for b in (8,16,32,64):
            group = [r for r in rows if r["target_sparsity"]==s and r["block_size"]==b]
            mean = lambda name: float(np.mean([r[name] for r in group]))
            layer_means = [float(np.mean([r["expanded_sparsity"] for r in group if r["layer"]==layer])) for layer in range(16)]
            record = {"target_sparsity":s,"block_size":b,"samples":len(group),
                      **{name:mean(name) for name in ["base_sparsity","expanded_sparsity","budget_sparsity","permuted_expanded_sparsity_mean","neuron_retained_l1","budget_retained_l1"]},
                      "expanded_sparsity_layer_means":layer_means}
            aggregate.append(record)
            lines.append(f"| {s:.0%} | {b} | {mean('expanded_sparsity'):.6%} | {min(layer_means):.6%}–{max(layer_means):.6%} | {mean('permuted_expanded_sparsity_mean'):.6%} | {mean('budget_sparsity'):.4%} | {mean('neuron_retained_l1'):.4%} | {mean('budget_retained_l1'):.4%} |")
    lines += ["", "## Limits", "",
              "These are magnitude-reference masks, not optimal importance estimates. L1 retention is not language quality. Fixed-budget block selection changes which neurons survive; expansion preserves them but may erase sparsity.","",
              "This experiment does not measure sparse execution speed, selector cost, generation quality or end-to-end inference. BF16 captures and teacher-forced prefill must not be conflated with EXP-001 FP32 timings. No claim across ARM processors follows.","",
              f"Recorded peak capture RSS: {manifest['peak_rss_bytes']/2**30:.3f} GiB. Operational capture time: {manifest['capture_seconds']:.1f} seconds (not an inference benchmark).",""]
    (out/"summary.md").write_text("\n".join(lines))
    (out/"aggregate.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    for name in ["metrics.csv","summary.md","aggregate.json","analyze_exp002.py","coverage.py"]:
        outputs.append({"path":name,"sha256":sha(out/name)})
    (out/"manifest.json").write_text(json.dumps({"experiment_id":"EXP-002","status":"completed",
        "input_manifest_sha256":sha(directory/"manifest.json"),"tie_seed":"17+layer",
        "aggregation":"equal weight per analyzed layer/token; ranges of layer means",
        "metric_rows":len(rows),"artifacts":outputs},indent=2)+"\n")
    print(out/"summary.md")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    analyze(parser.parse_args().directory)
