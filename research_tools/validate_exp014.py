"""Validate EXP-014 model, source and benchmark provenance."""
import json,sys
from pathlib import Path
from research_tools.inputs import check_git_sources,check_sha
def validate(path):
 d=json.loads(Path(path).read_text());assert d['experiment_id']=='EXP-014' and d['status']=='completed'
 assert d['git_status']==''
 check_sha(d['model_metadata']['gguf'],d['model_metadata']['gguf_sha256'],'GGUF model')
 assert len(d['stdout_rows'])==2
 check_git_sources(d)
 for row in d['stdout_rows']:
  assert row['build_commit']=='fa67698' and row['n_threads']==1 and row['n_gpu_layers']==0
  assert len(row['samples_ts'])==5 and row['avg_ts']>0
 print('Validated EXP-014 Q8_0 dense baseline provenance and samples')
if __name__=='__main__':validate(sys.argv[1])
