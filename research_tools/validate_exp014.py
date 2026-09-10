"""Validate EXP-014 model, source and benchmark provenance."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(path):
 d=json.loads(Path(path).read_text());assert d['experiment_id']=='EXP-014' and d['status']=='completed'
 assert d['git_status']=='' and sha(d['model_metadata']['gguf'])==d['model_metadata']['gguf_sha256']
 assert len(d['stdout_rows'])==2
 for rel,expected in d['source_sha256'].items():
  data=subprocess.run(['git','show',f"{d['git_commit']}:{rel}"],cwd=ROOT,capture_output=True,check=True).stdout
  assert hashlib.sha256(data).hexdigest()==expected
 for row in d['stdout_rows']:
  assert row['build_commit']=='fa67698' and row['n_threads']==1 and row['n_gpu_layers']==0
  assert len(row['samples_ts'])==5 and row['avg_ts']>0
 print('Validated EXP-014 Q8_0 dense baseline provenance and samples')
if __name__=='__main__':validate(sys.argv[1])
