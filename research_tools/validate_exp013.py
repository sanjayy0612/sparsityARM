"""Validate EXP-013 provenance, errors and timing aggregates."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def validate(directory):
 m=json.loads((Path(directory)/'manifest.json').read_text());assert m['experiment_id']=='EXP-013' and m['status']=='completed' and m['git_status']==''
 assert len(m['results'])==m['runs']*len(m['sparsities'])
 for rel,expected in m['source_sha256'].items():
  data=subprocess.run(['git','show',f"{m['git_commit']}:{rel}"],cwd=ROOT,capture_output=True,check=True).stdout;assert hashlib.sha256(data).hexdigest()==expected
 for r in m['results']:
  assert r['relative_l2_vs_dequantized_reference']<3e-4
  for k,v in r['raw_timings_ms'].items():
   assert len(v)==m['repeats'];assert r['timings'][k]['median_ms']==float(np.median(v));assert r['timings'][k]['p95_ms']==float(np.percentile(v,95))
 print('Validated EXP-013 provenance, numerical errors and timing aggregates')
if __name__=='__main__':validate(sys.argv[1])
