"""Generate a transparent Markdown/CSV summary from validated EXP-001 data."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

from validate_exp001 import validate


def summarize(directory):
    validate(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = [(item["path"], json.loads((directory / item["path"]).read_text())) for item in manifest["case_artifacts"]]
    modes = ["dense_accelerate", "irregular_native", "irregular_expanded_native", "block_native", "block_accelerate"]
    rows = []
    for path, case in cases:
        for mode in modes:
            is_base = mode == "irregular_native"
            sparsities = [0.] if mode == "dense_accelerate" else case["realized_sparsity_base" if is_base else "realized_sparsity_block"]
            rows.append({"case": path, "seed": case["seed"], "family": case["family"],
                         "target_sparsity": case["target_sparsity"], "block_size": case["block_size"],
                         "executor": mode, "realized_sparsity_mean": statistics.mean(sparsities),
                         "median_ms": case["summary"][mode]["median_ms"],
                         "p95_ms": case["summary"][mode]["p95_ms"],
                         "speedup_vs_dense": case["summary"]["dense_accelerate"]["median_ms"]/case["summary"][mode]["median_ms"]})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    config = manifest["configuration"]
    lines = ["# EXP-001 measured results", "",
             f"Shape: {config['hidden']} → {config['intermediate']} → {config['hidden']}; CPU FP32, one thread.", "",
             f"{len(cases)} cases; {config['runs']} independent seeds; {config['warmups']} warmups and {config['repeats']} samples per executor/case. All raw samples retained.", "",
             "Synthetic executor measurements only. No real model, selector, language-quality evaluation, cache counters or end-to-end generation was measured.", "",
             "## Latency and equal-work controls", "",
             "Cells are the median of independent-run medians (ms). Per-run p95 values are in summary.csv and case JSON; no pooled p95 is reported. I = irregular base mask; IE = irregular expanded mask; BN = native block; BA = Accelerate block.", "",
             "| Mask family | Target skip | B | Realized block skip | Dense | I | IE | BN | BA | BN speedup vs dense range | BN speedup vs I range |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for family in ("scattered", "clustered"):
        for s in (.2,.3,.4):
            for b in (8,16,32,64):
                group = [c for _, c in cases if (c["family"], c["target_sparsity"], c["block_size"]) == (family,s,b)]
                vals = [statistics.median(c["summary"][mode]["median_ms"] for c in group) for mode in modes]
                realized = statistics.mean(statistics.mean(c["realized_sparsity_block"]) for c in group)
                ratios = [c["summary"]["dense_accelerate"]["median_ms"]/c["summary"]["block_native"]["median_ms"] for c in group]
                relative = [c["summary"]["irregular_native"]["median_ms"]/c["summary"]["block_native"]["median_ms"] for c in group]
                lines.append(f"| {family} | {s:.0%} | {b} | {realized:.4%} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {min(ratios):.3f}–{max(ratios):.3f}× | {min(relative):.3f}–{max(relative):.3f}× |")
    peaks = [c["process_peak_rss_bytes_including_validation"]/2**20 for _,c in cases]
    worst_error = max(e["max_relative_l2_error"] for _,c in cases for e in c["correctness"].values())
    stable = []
    expansion_stable = []
    for family in ("scattered", "clustered"):
        for s in (.2,.3,.4):
            for b in (8,16,32,64):
                group = [c for _,c in cases if (c["family"],c["target_sparsity"],c["block_size"]) == (family,s,b)]
                ratios = [c["summary"]["irregular_native"]["median_ms"]/c["summary"]["block_native"]["median_ms"] for c in group]
                if all(r > 1 for r in ratios):
                    item = f"{family}, target {s:.0%}, B={b}: {min(ratios):.3f}–{max(ratios):.3f}× versus base irregular"
                    stable.append(item)
                    if all(sum(c["active_neurons_expanded"]) > sum(c["active_neurons_base"]) for c in group):
                        expansion_stable.append(item)
    lines += ["", "## Repeatability screen", "",
              "The following configurations have a lower native-block median than base-irregular median in every independent run. This is a descriptive screen of the complete grid, not a significance test or proof of an optimum.", ""]
    lines += [f"- {item}" for item in stable] or ["None."]
    lines += ["", "Configurations passing the same screen while computing additional neurons:", ""]
    lines += [f"- {item}" for item in expansion_stable] or ["None. H2's extra-computation benefit is not established by this sweep."]
    lines += ["", "## Memory, correctness and overhead", "",
              f"Peak process RSS including all layouts and FP64 validation: {min(peaks):.1f}–{max(peaks):.1f} MiB. This is not per-executor deployment memory.", "",
              f"Maximum recorded relative L2 error against FP64 reference: {worst_error:.3g}. Every saved input/mask passed rtol=2e-4, atol=2e-5 before timing.", "",
              "Packing, synthetic-mask construction and standalone scan/compaction timings are retained per case. The scan diagnostic is not subtracted from executor time. Deployable selector cost is unknown, represented as null.", "",
              "## Interpretation boundaries", "",
              "- Scattered-to-block expansion may erase almost all sparsity; compare realized active counts before interpreting latency.",
              "- Clustered masks deliberately favor block execution and use equal neuron sets across executors. They are not representative model masks until validated against real activations.",
              "- IE versus BN compares the same expanded neurons. BN versus BA also changes the arithmetic backend. No isolated cache/SIMD causal claim follows from these timings.",
              "- A configuration faster than dense here is not a deployable LLM speedup; selector, attention, normalization, runtime integration and quality remain unmeasured.",
              "- Single-thread macOS scheduling and thermal/background activity are uncontrolled; results do not establish optimality on other ARM CPUs.", "",
              "Sources: manifest.json, its hashed case_artifacts, raw case JSON, and saved weights/input/mask NPZ files. summary.csv contains one row per executor/case.", ""]
    (directory / "summary.md").write_text("\n".join(lines))
    analysis_sources = directory / "analysis-sources"
    analysis_sources.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).with_name("validate_exp001.py")):
        (analysis_sources / source.name).write_bytes(source.read_bytes())
    derived = {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "validator_sha256": hashlib.sha256(Path(__file__).with_name("validate_exp001.py").read_bytes()).hexdigest(),
               "input_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
               "outputs": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("summary.md", "summary.csv")}}
    (directory / "summary-provenance.json").write_text(json.dumps(derived, indent=2)+"\n")
    print(directory / "summary.md")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
