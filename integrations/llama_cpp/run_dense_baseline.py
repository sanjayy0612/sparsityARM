"""Run and retain the pinned llama.cpp Q8_0 CPU-only dense baseline."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from datetime import datetime,timezone
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ROOT=Path(__file__).resolve().parents[2]
def command(args):
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    return result.stdout.strip() if result.returncode==0 else {"unavailable":result.stderr.strip()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--llama-bench',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--model-metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('refusing to overwrite baseline artifact')
    metadata=json.loads(a.model_metadata.read_text())
    if sha(a.model)!=metadata['gguf_sha256']:p.error('GGUF hash mismatch')
    cmd=[str(a.llama_bench),'-m',str(a.model),'-p','128','-n','32','-t','1','-ngl','0',
         '-r','5','--delay','1','-o','json']
    result=subprocess.run(cmd,text=True,capture_output=True,check=True)
    rows=json.loads(result.stdout)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),ROOT/'integrations/llama_cpp/prepare_q8_model.py',
             ROOT/'integrations/llama_cpp/build_cpu.py',ROOT/'research/experiments/EXP-014/protocol.md']
    artifact={'experiment_id':'EXP-014','status':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'dense Q8_0 llama.cpp CPU baseline; no sparsity','command':cmd,
              'git_commit':command(['git','rev-parse','HEAD']),
              'git_status':command(['git','status','--short']),
              'source_sha256':{str(path.relative_to(ROOT)):sha(path) for path in sources},
              'llama_bench_sha256':sha(a.llama_bench),'model_metadata':metadata,
              'hardware':{'os':platform.platform(),'arch':platform.machine()},
              'stdout_rows':rows,'stderr':result.stderr}
    a.output.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
