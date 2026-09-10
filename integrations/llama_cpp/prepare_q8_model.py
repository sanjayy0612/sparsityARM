"""Convert the pinned local Hugging Face snapshot directly to Q8_0 GGUF."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; LLAMA=ROOT/'third_party/llama.cpp'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.metadata.exists():p.error('refusing to overwrite model artifacts')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','-C',str(LLAMA),'rev-parse','HEAD'],text=True).strip()
    cmd=[sys.executable,str(LLAMA/'convert_hf_to_gguf.py'),str(a.snapshot),'--outfile',str(a.output),'--outtype','q8_0']
    subprocess.run(cmd,check=True,cwd=ROOT)
    metadata={'experiment_id':'EXP-014','created_utc':datetime.now(timezone.utc).isoformat(),
              'model':'meta-llama/Llama-3.2-1B','model_revision':a.snapshot.name,
              'llama_cpp_commit':commit,'command':cmd,'gguf':str(a.output.resolve()),
              'gguf_sha256':sha(a.output),'gguf_bytes':a.output.stat().st_size,
              'converter_sha256':sha(LLAMA/'convert_hf_to_gguf.py')}
    a.metadata.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
