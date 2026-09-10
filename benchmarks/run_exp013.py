"""Measure the matched weight-only INT8 dense/B8 break-even frontier."""
from __future__ import annotations
import argparse,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from native_ffn import ROOT,BUILD_COMMAND,NativeFFN,build,mask_pair,quantize_symmetric,reference
from run_exp009 import aggregate,command,sha
SPARSITIES=(.1,.2,.3,.4,.5,.6,.7,.8)

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
 p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=200);p.add_argument('--runs',type=int,default=3)
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib=build();native=NativeFFN(lib)
 sources=[ROOT/'cpp/ffn.cpp',ROOT/'benchmarks/native_ffn.py',Path(__file__).resolve(),ROOT/'research/experiments/EXP-013/protocol.md']
 m={'experiment_id':'EXP-013','status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
 'git_commit':command(['git','rev-parse','HEAD']),'git_status':command(['git','status','--short']),
 'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in sources},'binary_sha256':sha(lib),'build_command':BUILD_COMMAND,
 'hardware':{'cpu':command(['sysctl','-n','machdep.cpu.brand_string']),'os':platform.platform(),'arch':platform.machine()},
 'python':sys.version,'numpy':np.__version__,'shape':{'hidden':2048,'intermediate':8192,'tokens':1},'threads':1,
 'quantization':'symmetric per-tensor INT8 weights; FP32 activations and accumulation','block_size':8,
 'sparsities':SPARSITIES,'warmups':a.warmups,'repeats':a.repeats,'runs':a.runs,'results':[]}
 path=out/'manifest.json';path.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
 try:
  h,mid,b=2048,8192,8
  for run in range(a.runs):
   rng=np.random.default_rng(13001+run);x=rng.standard_normal(h,dtype=np.float32)
   ws=[rng.standard_normal(shape,dtype=np.float32)*.02 for shape in ((mid,h),(mid,h),(h,mid))]
   (qg,sg),(qu,su),(qd,sd)=[quantize_symmetric(w) for w in ws]
   packed=qd.reshape(h,mid//b,b).transpose(1,0,2).copy();scratch=np.empty(2*mid,np.float32);output=np.empty(h,np.float32)
   dense_mask=np.ones(mid,np.uint8)
   for sparsity in SPARSITIES:
    masks=np.stack([mask_pair(rng,mid,b,sparsity,'clustered')[1] for _ in range(8)])
    expected=reference(x,(qg*sg).astype(np.float32),(qu*su).astype(np.float32),(qd*sd).astype(np.float32),np.repeat(masks[0],b))
    native.execute_quantized(1,x,qg,qu,packed,(sg,su,sd),masks[0],b,scratch,output)
    error=float(np.linalg.norm(output-expected)/max(np.linalg.norm(expected),1e-12));assert error<3e-4
    raw={'int8_dense_ms':[],'int8_b8_ms':[]}
    for i in range(a.warmups+a.repeats):
     mask=masks[i%len(masks)]
     d=native.execute_quantized(0,x,qg,qu,qd,(sg,su,sd),dense_mask,b,scratch,output)
     s=native.execute_quantized(1,x,qg,qu,packed,(sg,su,sd),mask,b,scratch,output)
     if i>=a.warmups:raw['int8_dense_ms'].append(d);raw['int8_b8_ms'].append(s)
    m['results'].append({'run':run+1,'target_sparsity':sparsity,'realized_sparsity':float(1-masks.mean()),
      'relative_l2_vs_dequantized_reference':error,'timings':{k:aggregate(v) for k,v in raw.items()},'raw_timings_ms':raw})
    path.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');print(f"Completed {len(m['results'])}/{a.runs*len(SPARSITIES)}",flush=True)
  m.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
 except Exception as e:m.update(status='failed',error=f'{type(e).__name__}: {e}');raise
 finally:path.write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
