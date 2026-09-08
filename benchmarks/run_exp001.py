"""Run EXP-001 in isolated case processes and retain every raw timing.

macOS 15+, CPU only. See research/experiments/EXP-001/protocol.md.
"""
from __future__ import annotations

import argparse
import ctypes
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

# Set before importing NumPy or loading Accelerate. Native code also calls
# BLASSetThreading(SINGLE_THREADED), checked for success.
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from native_ffn import BUILD_COMMAND, ROOT, NativeFFN, build, mask_pair, reference


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(samples):
    return {"median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95, method="linear")),
            "min_ms": float(min(samples)), "max_ms": float(max(samples))}


def run_case(args):
    native = NativeFFN(ROOT / "build/libffn.dylib")
    h, m, b = args.hidden, args.intermediate, args.block
    rng = np.random.default_rng(args.seed)
    # Seed and all inputs/weights saved, avoiding dependence on RNG implementation.
    gate = rng.standard_normal((m, h), dtype=np.float32) / np.float32(np.sqrt(h))
    up = rng.standard_normal((m, h), dtype=np.float32) / np.float32(np.sqrt(h))
    down = rng.standard_normal((h, m), dtype=np.float32) / np.float32(np.sqrt(m))
    xs = rng.standard_normal((args.bank, h), dtype=np.float32)
    start = time.perf_counter_ns()
    neuron_down = down.T.copy()
    neuron_pack_ms = (time.perf_counter_ns() - start)/1e6
    start = time.perf_counter_ns()
    block_down = down.reshape(h, m//b, b).transpose(1, 0, 2).copy()
    block_pack_ms = (time.perf_counter_ns() - start)/1e6
    base_masks, block_masks, construction_ms = [], [], []
    for _ in range(args.bank):
        start = time.perf_counter_ns()
        base, blocks = mask_pair(rng, m, b, args.sparsity, args.family)
        construction_ms.append((time.perf_counter_ns() - start)/1e6)
        base_masks.append(base)
        block_masks.append(blocks)
    base_masks, block_masks = np.array(base_masks), np.array(block_masks)
    expanded = np.repeat(block_masks, b, axis=1)
    artifact = Path(args.case_output).with_suffix(".npz")
    # Full weights are shared per run; inputs and masks are case artifacts.
    weights_path = artifact.parent / f"weights-seed-{args.seed}.npz"
    if not weights_path.exists():
        np.savez(weights_path, gate=gate, up=up, down=down)
    np.savez(artifact, x=xs, base_masks=base_masks, block_masks=block_masks)
    scratch, output = np.empty(2*m, np.float32), np.empty(h, np.float32)
    dense_masks = np.ones((args.bank, m), np.uint8)
    variants = {
        "dense_accelerate": (0, down, dense_masks),
        "irregular_native": (1, neuron_down, base_masks),
        "irregular_expanded_native": (1, neuron_down, expanded),
        "block_native": (2, block_down, block_masks),
        "block_accelerate": (3, block_down, block_masks),
    }
    errors = {}
    for name, (mode, weights, masks) in variants.items():
        max_abs, max_relative_l2 = 0., 0.
        for k in range(args.bank):
            native.execute(mode, xs[k], gate, up, weights, masks[k], b, scratch, output)
            mask = np.repeat(masks[k], b) if mode >= 2 else masks[k]
            expected = reference(xs[k], gate, up, down, mask)
            max_abs = max(max_abs, float(np.max(np.abs(output - expected))))
            max_relative_l2 = max(max_relative_l2, float(np.linalg.norm(output-expected)/max(np.linalg.norm(expected), 1e-12)))
            if not np.allclose(output, expected, rtol=2e-4, atol=2e-5):
                raise RuntimeError(f"correctness failed: {name}, mask {k}, abs {max_abs}")
        errors[name] = {"max_absolute_error": max_abs, "max_relative_l2_error": max_relative_l2}
    # CPU process peak RSS includes correctness reference temporaries; label it.
    samples = {name: [] for name in variants}
    orders = []
    names = list(variants)
    order_rng = np.random.default_rng(args.seed + 999)
    for iteration in range(args.warmups + args.repeats):
        order = order_rng.permutation(names).tolist()
        k = iteration % args.bank
        for name in order:
            mode, weights, masks = variants[name]
            elapsed = native.execute(mode, xs[k], gate, up, weights, masks[k], b, scratch, output)
            if iteration >= args.warmups:
                samples[name].append(elapsed)
        if iteration >= args.warmups:
            orders.append({"mask_index": k, "order": order})
    scan = {}
    index_buffer = np.empty(m, np.int32)
    count = ctypes.c_int()
    for name, masks in (("neuron", base_masks), ("block", block_masks)):
        scan[name] = [native.lib.scan_mask(masks[k % args.bank], masks.shape[1], index_buffer,
                                         ctypes.byref(count)) for k in range(args.repeats)]
    active_base = base_masks.sum(axis=1).astype(int)
    active_block = expanded.sum(axis=1).astype(int)
    record = {
        "experiment_id": "EXP-001", "scope": "synthetic_executor_only",
        "shape": {"tokens": 1, "hidden": h, "intermediate": m},
        "dtype": "float32", "threads": 1, "seed": args.seed,
        "block_size": b, "target_sparsity": args.sparsity, "family": args.family,
        "warmups_per_variant": args.warmups, "repetitions_per_variant": args.repeats,
        "mask_bank_size": args.bank, "sample_schedule": orders,
        "active_neurons_base": active_base.tolist(), "active_neurons_expanded": active_block.tolist(),
        "realized_sparsity_base": (1-active_base/m).tolist(),
        "realized_sparsity_block": (1-active_block/m).tolist(),
        "dense_projection_macs": 3*h*m,
        "base_projection_macs": (3*h*active_base).tolist(),
        "block_projection_macs": (3*h*active_block).tolist(),
        "operation_note": "MACs cover projections only; FLOPs=2*MACs; not selection/SiLU/indexing",
        "raw_ms": samples, "summary": {name: stats(values) for name, values in samples.items()},
        "correctness": errors,
        "packing_ms": {"neuron_down": neuron_pack_ms, "block_down": block_pack_ms},
        "mask_construction_ms": construction_ms,
        "mask_scan_diagnostic_ms": scan,
        "selector": {"kind": "none_synthetic_masks", "deployable_selector_ms": None},
        "weight_storage_bytes": {"canonical": gate.nbytes+up.nbytes+down.nbytes,
                                 "neuron_extra": neuron_down.nbytes, "block_extra": block_down.nbytes},
        "process_peak_rss_bytes_including_validation": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": {"inputs_masks": {"path": artifact.name, "sha256": digest(artifact)},
                      "weights": {"path": weights_path.name, "sha256": digest(weights_path)}},
    }
    write_json(Path(args.case_output), record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--bank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-output")
    parser.add_argument("--block", type=int, choices=[8, 16, 32, 64], default=8)
    parser.add_argument("--sparsity", type=float, default=.2)
    parser.add_argument("--family", choices=["scattered", "clustered"], default="scattered")
    args = parser.parse_args()
    if min(args.hidden, args.intermediate, args.repeats, args.runs, args.bank) <= 0 or args.warmups < 0:
        parser.error("dimensions/repeats/runs/bank must be positive and warmups nonnegative")
    if args.intermediate % 64 or not 0 <= args.sparsity <= 1:
        parser.error("intermediate must be divisible by 64; sparsity must be in [0,1]")
    if args.case_output:
        run_case(args)
        return
    out = args.output or ROOT / "research/results/EXP-001" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    library = build()
    source_paths = [ROOT / "cpp/ffn.cpp", ROOT / "benchmarks/native_ffn.py", Path(__file__).resolve(),
                    ROOT / "research/experiments/EXP-001/protocol.md"]
    source_dir = out / "sources"
    source_dir.mkdir()
    source_hashes = {}
    for path in source_paths:
        snapshot = source_dir / path.name
        snapshot.write_bytes(path.read_bytes())
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    (out / "working-tree.patch").write_text(command(["git", "diff", "HEAD"]))
    metadata = {
        "experiment_id": "EXP-001", "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]), "source_sha256": source_hashes,
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
                     "cpu_count": os.cpu_count(), "os": platform.platform(), "arch": platform.machine()},
        "compiler": command(["xcrun", "clang++", "--version"]), "build_command": BUILD_COMMAND,
        "binary_sha256": digest(library), "python": sys.version, "numpy": np.__version__,
        "thread_policy": "BLASSetThreading SINGLE_THREADED; custom kernels serial; no core pinning",
        "invocation": sys.argv, "configuration": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "aggregation": "per-case median and linear-interpolated p95; retain independent runs separately",
        "model_revision": None, "prompts": None, "generated_tokens": None,
        "limitations": ["synthetic masks, no language quality or selector", "no PMU/cache measurements",
                        "no controlled core affinity or thermal instrumentation", "FP32 only, single-thread only",
                        "packed layouts and loop/backend differences are combined interventions"],
    }
    write_json(out / "manifest.json", metadata)
    case_paths = []
    for run in range(args.runs):
        cases = [(b, s, family) for b in (8,16,32,64) for s in (.2,.3,.4) for family in ("scattered","clustered")]
        np.random.default_rng(args.seed+run).shuffle(cases)
        for b, sparsity, family in cases:
            path = out / f"run{run+1}-B{b}-s{round(sparsity*100)}-{family}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--case-output", str(path),
                   "--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
                   "--warmups", str(args.warmups), "--repeats", str(args.repeats), "--bank", str(args.bank),
                   "--seed", str(args.seed+run), "--block", str(b), "--sparsity", str(sparsity), "--family", family]
            result = subprocess.run(cmd, text=True, capture_output=True)
            (out / (path.stem + ".log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                metadata.update(status="failed", failed_case=path.name)
                write_json(out / "manifest.json", metadata)
                raise RuntimeError(f"case failed; inspect {path.with_suffix('.log')}")
            case_paths.append(path)
            print(f"{len(case_paths)}/{24*args.runs} {path.name}", flush=True)
    metadata.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                    case_artifacts=[{"path": p.name, "sha256": digest(p)} for p in case_paths])
    write_json(out / "manifest.json", metadata)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
