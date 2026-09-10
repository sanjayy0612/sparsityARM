"""Map the native B8 sparsity break-even frontier against Accelerate dense."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from native_ffn import BUILD_COMMAND, ROOT, build

SPARSITIES=(.1,.2,.3,.4,.5,.6,.7,.8)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args):
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    return r.stdout.strip() if r.returncode==0 else {"unavailable":r.stderr.strip()}
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--warmups",type=int,default=20)
    p.add_argument("--repeats",type=int,default=200); p.add_argument("--runs",type=int,default=3)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    library=build(); sources=[ROOT/"cpp/ffn.cpp",ROOT/"benchmarks/native_ffn.py",
        ROOT/"benchmarks/run_exp001.py",Path(__file__).resolve(),ROOT/"research/experiments/EXP-012/protocol.md"]
    m={"experiment_id":"EXP-012","status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
       "git_commit":command(["git","rev-parse","HEAD"]),"git_status":command(["git","status","--short"]),
       "source_sha256":{str(x.relative_to(ROOT)):sha(x) for x in sources},"binary_sha256":sha(library),
       "build_command":BUILD_COMMAND,"hardware":{"cpu":command(["sysctl","-n","machdep.cpu.brand_string"]),
       "os":platform.platform(),"arch":platform.machine()},"python":sys.version,
       "shape":{"hidden":2048,"intermediate":8192,"tokens":1},"dtype":"float32","threads":1,
       "block_size":8,"mask_family":"clustered_equal_work","sparsities":SPARSITIES,
       "warmups":a.warmups,"repeats":a.repeats,"runs":a.runs,"cases":[]}
    manifest=out/"manifest.json"; write(manifest,m)
    try:
        for run in range(a.runs):
            for sparsity in SPARSITIES:
                case=out/f"run{run+1}-B8-s{round(sparsity*100)}-clustered.json"
                cmd=[sys.executable,str(ROOT/"benchmarks/run_exp001.py"),"--case-output",str(case),
                    "--hidden","2048","--intermediate","8192","--warmups",str(a.warmups),
                    "--repeats",str(a.repeats),"--bank","8","--seed",str(12001+run),
                    "--block","8","--sparsity",str(sparsity),"--family","clustered"]
                r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
                (out/f"{case.stem}.log").write_text(r.stdout+r.stderr)
                if r.returncode: raise RuntimeError(f"failed: {case.name}")
                m["cases"].append({"path":case.name,"sha256":sha(case)})
                write(manifest,m); print(f"Completed {len(m['cases'])}/{a.runs*len(SPARSITIES)}",flush=True)
        m.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        m.update(status="failed",error=f"{type(exc).__name__}: {exc}"); raise
    finally: write(manifest,m)
if __name__=="__main__": main()
