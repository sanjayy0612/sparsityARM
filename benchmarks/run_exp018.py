"""Map the B8 FP32 break-even frontier at TinyLlama FFN geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from native_ffn import BUILD_COMMAND, ROOT, build

HIDDEN = 2048
INTERMEDIATE = 5632
BLOCK_SIZE = 8
SPARSITIES = (0.2, 0.3, 0.4, 0.5, 0.6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else {"unavailable": result.stderr.strip()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = build()
    sources = [ROOT / "cpp/ffn.cpp", ROOT / "benchmarks/native_ffn.py",
               ROOT / "benchmarks/run_exp001.py", Path(__file__).resolve(),
               ROOT / "research/experiments/EXP-018/protocol.md"]
    manifest = {
        "experiment_id": "EXP-018", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "binary_sha256": sha(library), "build_command": BUILD_COMMAND,
        "hardware": {"cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                     "os": platform.platform(), "arch": platform.machine()},
        "python": sys.version,
        "shape": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "tokens": 1},
        "dtype": "float32", "threads": 1, "block_size": BLOCK_SIZE,
        "mask_family": "clustered_equal_work", "sparsities": SPARSITIES,
        "warmups": args.warmups, "repeats": args.repeats, "runs": args.runs,
        "cases": [],
    }
    manifest_path = output / "manifest.json"
    write(manifest_path, manifest)
    try:
        for run in range(args.runs):
            for sparsity in SPARSITIES:
                case = output / f"run{run + 1}-B8-s{round(sparsity * 100)}-clustered.json"
                invocation = [
                    sys.executable, str(ROOT / "benchmarks/run_exp001.py"),
                    "--case-output", str(case), "--hidden", str(HIDDEN),
                    "--intermediate", str(INTERMEDIATE), "--warmups", str(args.warmups),
                    "--repeats", str(args.repeats), "--bank", "8",
                    "--seed", str(18001 + run), "--block", str(BLOCK_SIZE),
                    "--sparsity", str(sparsity), "--family", "clustered",
                ]
                result = subprocess.run(invocation, cwd=ROOT, text=True, capture_output=True)
                (output / f"{case.stem}.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"failed: {case.name}")
                manifest["cases"].append({"path": case.name, "sha256": sha(case)})
                write(manifest_path, manifest)
                print(f"Completed {len(manifest['cases'])}/{args.runs * len(SPARSITIES)}", flush=True)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()
