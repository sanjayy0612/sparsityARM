"""Run the B8/10% real-mask replay executor gate on Apple M2."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from native_ffn import BUILD_COMMAND, ROOT, build


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--mask-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    metadata = json.loads(args.mask_metadata.read_text())
    if sha(args.mask_bank) != metadata["mask_bank_sha256"] or metadata["mask_count"] != 16:
        parser.error("mask bank does not match its metadata")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    library = build()
    source_paths = [ROOT / "cpp/ffn.cpp", ROOT / "benchmarks/native_ffn.py",
                    ROOT / "benchmarks/run_exp001.py", Path(__file__).resolve(),
                    ROOT / "benchmarks/prepare_exp005_masks.py",
                    ROOT / "research/experiments/EXP-005/protocol.md"]
    source_dir = out / "sources"
    source_dir.mkdir()
    source_hashes = {}
    for source in source_paths:
        (source_dir / source.name).write_bytes(source.read_bytes())
        source_hashes[str(source.relative_to(ROOT))] = sha(source)
    manifest = {"experiment_id": "EXP-005", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": command(["git", "rev-parse", "HEAD"]),
                "git_status": command(["git", "status", "--short"]),
                "source_sha256": source_hashes, "binary_sha256": sha(library),
                "build_command": BUILD_COMMAND,
                "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                             "memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
                             "os": platform.platform(), "arch": platform.machine()},
                "python": sys.version, "threads": 1, "dtype": "float32",
                "shape": {"hidden": 2048, "intermediate": 8192, "tokens": 1},
                "block_size": 8, "target_sparsity": 0.10,
                "warmups": args.warmups, "repeats": args.repeats, "runs": args.runs,
                "mask_metadata": metadata, "case_artifacts": [],
                "timing_scope": "standalone FFN executor including mask scan/branch; excludes mask selection and packing"}
    write(out / "manifest.json", manifest)
    try:
        for run in range(args.runs):
            case = out / f"run{run+1}-B8-s10-replay.json"
            cmd = [sys.executable, str(ROOT / "benchmarks/run_exp001.py"),
                   "--case-output", str(case), "--hidden", "2048", "--intermediate", "8192",
                   "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                   "--runs", "1", "--bank", "16", "--seed", str(5101 + run),
                   "--block", "8", "--sparsity", "0.10", "--family", "replay",
                   "--mask-bank", str(args.mask_bank.resolve())]
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            (out / f"run{run+1}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"run {run+1} failed; inspect {out / f'run{run+1}.log'}")
            manifest["case_artifacts"].append({"path": case.name, "sha256": sha(case)})
            write(out / "manifest.json", manifest)
            print(f"Completed run {run+1}/{args.runs}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
