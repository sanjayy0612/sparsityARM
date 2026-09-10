"""Profile staged B8 executor phases with the retained EXP-005 replay masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from native_ffn import BUILD_COMMAND, ROOT, NativeFFN, build, reference


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}


def aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--mask-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    metadata = json.loads(args.mask_metadata.read_text())
    if sha(args.mask_bank) != metadata["mask_bank_sha256"]:
        parser.error("mask bank hash mismatch")
    with np.load(args.mask_bank) as archive:
        masks = archive["block_masks"].astype(np.uint8)
    if masks.shape != (16, 1024) or metadata["block_size"] != 8:
        parser.error("EXP-009 requires the retained 16x1024 B8 mask bank")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    library = build()
    native = NativeFFN(library)
    sources = [ROOT / "cpp/ffn.cpp", ROOT / "benchmarks/native_ffn.py",
               Path(__file__).resolve(), ROOT / "research/experiments/EXP-009/protocol.md"]
    manifest = {"experiment_id": "EXP-009", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": command(["git", "rev-parse", "HEAD"]),
                "git_status": command(["git", "status", "--short"]),
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "binary_sha256": sha(library), "build_command": BUILD_COMMAND,
                "mask_bank": str(args.mask_bank.resolve()), "mask_bank_sha256": sha(args.mask_bank),
                "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                             "os": platform.platform(), "arch": platform.machine()},
                "python": sys.version, "numpy": np.__version__, "threads": 1, "dtype": "float32",
                "shape": {"hidden": 2048, "intermediate": 8192, "tokens": 1},
                "block_size": 8, "warmups": args.warmups, "repeats": args.repeats,
                "runs": args.runs, "timing_scope": "standalone staged diagnostic; selector excluded",
                "results": []}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    try:
        h, m, b = 2048, 8192, 8
        for run in range(args.runs):
            rng = np.random.default_rng(9001 + run)
            x = rng.standard_normal(h, dtype=np.float32)
            gate = (rng.standard_normal((m, h), dtype=np.float32) * .02).astype(np.float32)
            up = (rng.standard_normal((m, h), dtype=np.float32) * .02).astype(np.float32)
            dense_down = (rng.standard_normal((h, m), dtype=np.float32) * .02).astype(np.float32)
            packed = dense_down.reshape(h, m//b, b).transpose(1, 0, 2).copy()
            scratch, output = np.empty(2*m, np.float32), np.empty(h, np.float32)
            indices = np.empty(m//b, np.int32)
            expected = reference(x, gate, up, dense_down, np.repeat(masks[0], b))
            native.profile_block_staged(x, gate, up, packed, masks[0], b, scratch, output, indices)
            np.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-5)
            for _ in range(args.warmups):
                for mask in masks:
                    native.execute(2, x, gate, up, packed, mask, b, scratch, output)
                    native.profile_block_staged(x, gate, up, packed, mask, b, scratch, output, indices)
            recorded = {key: [] for key in ("interleaved_total_ms", "output_init_ms",
                        "mask_scan_ms", "gate_up_activation_ms", "down_projection_ms",
                        "staged_total_ms")}
            active_blocks, active_runs = [], []
            for _ in range(args.repeats):
                for mask in masks:
                    recorded["interleaved_total_ms"].append(
                        native.execute(2, x, gate, up, packed, mask, b, scratch, output))
                    profile = native.profile_block_staged(
                        x, gate, up, packed, mask, b, scratch, output, indices)
                    for key in recorded:
                        if key != "interleaved_total_ms":
                            recorded[key].append(profile["total_ms" if key == "staged_total_ms" else key])
                    active_blocks.append(profile["active_blocks"])
                    active_runs.append(profile["active_runs"])
            manifest["results"].append({"run": run + 1,
                "samples": args.repeats * len(masks),
                "active_blocks": sorted(set(active_blocks)),
                "active_runs": {"min": min(active_runs), "max": max(active_runs),
                                "median": float(np.median(active_runs))},
                "timings": {key: aggregate(value) for key, value in recorded.items()},
                "raw_timings_ms": recorded, "raw_active_runs": active_runs})
            path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            print(f"Completed run {run + 1}/{args.runs}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
