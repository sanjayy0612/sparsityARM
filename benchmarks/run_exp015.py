"""Measure fixed-mask B32 execution with llama.cpp Q8_0 primitives."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_exp009 import aggregate, command, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/llama_cpp"))
from q8_0_ffn import Q80FFN, build

SPARSITIES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    library, build_command = build(args.llama_build.resolve())
    kernel = Q80FFN(library)
    sources = [ROOT / "cpp/q8_0_ffn.cpp", ROOT / "integrations/llama_cpp/q8_0_ffn.py",
               Path(__file__).resolve(), ROOT / "research/experiments/EXP-015/protocol.md"]
    manifest = {
        "experiment_id": "EXP-015", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-token synthetic Llama 3.2 1B-shaped FFN replay; not end-to-end inference",
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "library_sha256": sha(library), "build_command": build_command,
        "llama_cpp_commit": command(["git", "-C", str(ROOT / "third_party/llama.cpp"), "rev-parse", "HEAD"]),
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "os": platform.platform(), "arch": platform.machine()},
        "python": sys.version, "numpy": np.__version__,
        "shape": {"hidden": 2048, "intermediate": 8192, "tokens": 1},
        "quantization": "llama.cpp Q8_0 weights and dynamically quantized Q8_0 activations",
        "block_size": 32, "sparsities": SPARSITIES, "threads": 1,
        "warmups": args.warmups, "repeats": args.repeats, "runs": args.runs, "results": [],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    try:
        hidden, intermediate = 2048, 8192
        for run in range(args.runs):
            rng = np.random.default_rng(15001 + run)
            input_ = rng.standard_normal(hidden, dtype=np.float32)
            weights = [rng.standard_normal(shape, dtype=np.float32) * 0.02
                       for shape in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))]
            gate, up, down = [kernel.quantize(weight) for weight in weights]
            activation = np.empty(intermediate, np.float32)
            qinput = np.empty(kernel.row_bytes(hidden), np.uint8)
            qactivation = np.empty(kernel.row_bytes(intermediate), np.uint8)
            result = np.empty(hidden, np.float32)
            control = np.empty(hidden, np.float32)
            dense_mask = np.ones(intermediate // 32, np.uint8)
            for sparsity in SPARSITIES:
                masks = []
                active = round(len(dense_mask) * (1 - sparsity))
                for _ in range(8):
                    mask = np.zeros_like(dense_mask)
                    mask[rng.permutation(len(mask))[:active]] = 1
                    masks.append(mask)
                kernel.execute(1, input_, gate, up, down, masks[0], activation, qinput, qactivation, result)
                kernel.execute(2, input_, gate, up, down, masks[0], activation, qinput, qactivation, control)
                error = float(np.linalg.norm(result - control) / max(np.linalg.norm(control), 1e-12))
                if error > 3e-5:
                    raise RuntimeError(f"sparse/control relative L2 error {error}")
                raw = {"q8_0_dense_ms": [], "q8_0_b32_ms": []}
                for index in range(args.warmups + args.repeats):
                    mask = masks[index % len(masks)]
                    calls = [("q8_0_dense_ms", 0, dense_mask), ("q8_0_b32_ms", 1, mask)]
                    if index % 2:
                        calls.reverse()
                    measured = {}
                    for name, mode, selected in calls:
                        measured[name] = kernel.execute(mode, input_, gate, up, down, selected,
                                                        activation, qinput, qactivation, result)
                    if index >= args.warmups:
                        for name in raw:
                            raw[name].append(measured[name])
                manifest["results"].append({
                    "run": run + 1, "target_sparsity": sparsity,
                    "realized_sparsity": float(1 - np.mean(masks)),
                    "relative_l2_vs_masked_dense_control": error,
                    "timings": {name: aggregate(values) for name, values in raw.items()},
                    "raw_timings_ms": raw,
                })
                path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
                print(f"Completed {len(manifest['results'])}/{args.runs * len(SPARSITIES)}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
