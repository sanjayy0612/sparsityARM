"""Measure B8 execution at TinyLlama's quality-compatible 20% sparsity."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from native_ffn import (BUILD_COMMAND, ROOT, NativeFFN, build, mask_pair,
                        quantize_symmetric, reference)
from run_exp009 import aggregate, command, sha

HIDDEN = 2048
INTERMEDIATE = 5632
BLOCK_SIZE = 8
SPARSITY = 0.2


def measure_pair(native, mode_dense, mode_sparse, dense_args, sparse_args,
                 warmups, repeats):
    raw = {"dense_ms": [], "b8_ms": []}
    for index in range(warmups + repeats):
        calls = [("dense_ms", mode_dense, dense_args), ("b8_ms", mode_sparse, sparse_args)]
        if index % 2:
            calls.reverse()
        measured = {}
        for name, mode, arguments in calls:
            function = native.execute_quantized if arguments[0] == "quantized" else native.execute
            measured[name] = function(mode, *arguments[1:])
        if index >= warmups:
            for name in raw:
                raw[name].append(measured[name])
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    library = build()
    native = NativeFFN(library)
    sources = [ROOT / "cpp/ffn.cpp", ROOT / "benchmarks/native_ffn.py",
               Path(__file__).resolve(), ROOT / "research/experiments/EXP-017/protocol.md"]
    manifest = {
        "experiment_id": "EXP-017", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does B8 execution win at TinyLlama's quality-compatible 20% sparsity?",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "binary_sha256": sha(library), "build_command": BUILD_COMMAND,
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "os": platform.platform(), "arch": platform.machine()},
        "python": sys.version, "numpy": np.__version__,
        "shape": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "tokens": 1},
        "threads": 1, "block_size": BLOCK_SIZE, "target_sparsity": SPARSITY,
        "mask_family": "synthetic clustered equal-work",
        "executors": {
            "fp32": "Accelerate dense versus native packed B8",
            "int8": "matched custom symmetric per-tensor weight-only INT8; FP32 activation and accumulation",
        },
        "warmups": args.warmups, "repeats": args.repeats, "runs": args.runs,
        "results": [],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    try:
        for run in range(args.runs):
            rng = np.random.default_rng(17001 + run)
            x = rng.standard_normal(HIDDEN, dtype=np.float32)
            gate, up, down = [rng.standard_normal(shape, dtype=np.float32) * 0.02
                              for shape in ((INTERMEDIATE, HIDDEN),
                                            (INTERMEDIATE, HIDDEN),
                                            (HIDDEN, INTERMEDIATE))]
            masks = np.stack([mask_pair(rng, INTERMEDIATE, BLOCK_SIZE, SPARSITY, "clustered")[1]
                              for _ in range(8)])
            mask = masks[0]
            neuron_mask = np.repeat(mask, BLOCK_SIZE)
            dense_mask = np.ones(INTERMEDIATE, np.uint8)
            packed = down.reshape(HIDDEN, INTERMEDIATE // BLOCK_SIZE, BLOCK_SIZE).transpose(1, 0, 2).copy()
            scratch = np.empty(2 * INTERMEDIATE, np.float32)
            result = np.empty(HIDDEN, np.float32)
            expected = reference(x, gate, up, down, neuron_mask)
            native.execute(2, x, gate, up, packed, mask, BLOCK_SIZE, scratch, result)
            fp32_error = float(np.linalg.norm(result - expected) / max(np.linalg.norm(expected), 1e-12))
            if fp32_error > 3e-4:
                raise RuntimeError(f"FP32 relative L2 error {fp32_error}")

            fp32_dense = ("fp32", x, gate, up, down, dense_mask, BLOCK_SIZE, scratch, result)
            fp32_sparse = ("fp32", x, gate, up, packed, mask, BLOCK_SIZE, scratch, result)
            fp32_raw = measure_pair(native, 0, 2, fp32_dense, fp32_sparse,
                                    args.warmups, args.repeats)

            (qgate, gate_scale), (qup, up_scale), (qdown, down_scale) = [
                quantize_symmetric(weight) for weight in (gate, up, down)]
            qpacked = qdown.reshape(HIDDEN, INTERMEDIATE // BLOCK_SIZE, BLOCK_SIZE).transpose(1, 0, 2).copy()
            qexpected = reference(x, (qgate * gate_scale).astype(np.float32),
                                  (qup * up_scale).astype(np.float32),
                                  (qdown * down_scale).astype(np.float32), neuron_mask)
            scales = (gate_scale, up_scale, down_scale)
            native.execute_quantized(1, x, qgate, qup, qpacked, scales, mask,
                                     BLOCK_SIZE, scratch, result)
            int8_error = float(np.linalg.norm(result - qexpected) / max(np.linalg.norm(qexpected), 1e-12))
            if int8_error > 3e-4:
                raise RuntimeError(f"INT8 relative L2 error {int8_error}")
            int8_dense = ("quantized", x, qgate, qup, qdown, scales, dense_mask,
                          BLOCK_SIZE, scratch, result)
            int8_sparse = ("quantized", x, qgate, qup, qpacked, scales, mask,
                           BLOCK_SIZE, scratch, result)
            int8_raw = measure_pair(native, 0, 1, int8_dense, int8_sparse,
                                    args.warmups, args.repeats)

            manifest["results"].append({
                "run": run + 1, "realized_sparsity": float(1 - masks.mean()),
                "fp32_relative_l2_vs_masked_dense_reference": fp32_error,
                "int8_relative_l2_vs_dequantized_reference": int8_error,
                "fp32": {"timings": {name: aggregate(values) for name, values in fp32_raw.items()},
                         "raw_timings_ms": fp32_raw},
                "int8": {"timings": {name: aggregate(values) for name, values in int8_raw.items()},
                         "raw_timings_ms": int8_raw},
            })
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"Completed {run + 1}/{args.runs}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
