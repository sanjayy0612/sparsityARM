"""Benchmark contiguous-run B8 against native B8 and Accelerate dense."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
from native_ffn import BUILD_COMMAND, ROOT, NativeFFN, build, reference
from run_exp009 import aggregate, command, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--mask-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    metadata = json.loads(args.mask_metadata.read_text())
    if sha(args.mask_bank) != metadata["mask_bank_sha256"]:
        parser.error("mask bank hash mismatch")
    with np.load(args.mask_bank) as archive:
        masks = archive["block_masks"].astype(np.uint8)
    if masks.shape != (16, 1024) or not np.all(masks.sum(axis=1) == 922):
        parser.error("EXP-011 requires the retained B8/10% replay bank")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    library = build()
    native = NativeFFN(library)
    sources = [ROOT/"cpp/ffn.cpp", ROOT/"benchmarks/native_ffn.py",
               ROOT/"benchmarks/run_exp009.py", Path(__file__).resolve(),
               ROOT/"research/experiments/EXP-011/protocol.md"]
    manifest = {"experiment_id": "EXP-011", "status": "running",
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
        "block_size": 8, "realized_sparsity": 1-922/1024,
        "warmups": args.warmups, "repeats": args.repeats, "runs": args.runs,
        "timing_scope": "native FFN only; selection, packing and Python dispatch excluded",
        "results": []}
    path = out/"manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    try:
        h, m, b = 2048, 8192, 8
        for run in range(args.runs):
            rng = np.random.default_rng(11001+run)
            x = rng.standard_normal(h, dtype=np.float32)
            gate = (rng.standard_normal((m,h), dtype=np.float32)*.02).astype(np.float32)
            up = (rng.standard_normal((m,h), dtype=np.float32)*.02).astype(np.float32)
            down = (rng.standard_normal((h,m), dtype=np.float32)*.02).astype(np.float32)
            packed = down.reshape(h,m//b,b).transpose(1,0,2).copy()
            scratch, output = np.empty(2*m,np.float32), np.empty(h,np.float32)
            expected = reference(x,gate,up,down,np.repeat(masks[0],b))
            for mode, weights in ((2,packed),(5,down)):
                native.execute(mode,x,gate,up,weights,masks[0],b,scratch,output)
                np.testing.assert_allclose(output,expected,rtol=3e-4,atol=3e-5)
            dense_mask = np.ones(m,np.uint8)
            timings = {"accelerate_dense_ms":[],"native_b8_ms":[],"coalesced_b8_ms":[]}
            for _ in range(args.warmups):
                for mask in masks:
                    native.execute(0,x,gate,up,down,dense_mask,b,scratch,output)
                    native.execute(2,x,gate,up,packed,mask,b,scratch,output)
                    native.execute(5,x,gate,up,down,mask,b,scratch,output)
            for _ in range(args.repeats):
                for mask in masks:
                    timings["accelerate_dense_ms"].append(native.execute(0,x,gate,up,down,dense_mask,b,scratch,output))
                    timings["native_b8_ms"].append(native.execute(2,x,gate,up,packed,mask,b,scratch,output))
                    timings["coalesced_b8_ms"].append(native.execute(5,x,gate,up,down,mask,b,scratch,output))
            manifest["results"].append({"run":run+1,"samples":args.repeats*len(masks),
                "timings":{k:aggregate(v) for k,v in timings.items()},"raw_timings_ms":timings})
            path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
            print(f"Completed run {run+1}/{args.runs}",flush=True)
        manifest.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")

if __name__ == "__main__": main()
