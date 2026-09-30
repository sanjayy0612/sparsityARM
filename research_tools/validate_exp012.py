"""Validate EXP-012 cases, hashes and retained timing aggregates."""
from __future__ import annotations
import hashlib,json,math,statistics,sys
from pathlib import Path
from research_tools.inputs import check_git_sources
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(directory):
    d=Path(directory); m=json.loads((d/"manifest.json").read_text())
    assert m["experiment_id"]=="EXP-012" and m["status"]=="completed" and m["git_status"]==""
    assert len(m["cases"])==m["runs"]*len(m["sparsities"])
    check_git_sources(m)
    seen=set()
    for item in m["cases"]:
        p=d/item["path"]; assert sha(p)==item["sha256"]; c=json.loads(p.read_text())
        key=(c["seed"],c["target_sparsity"]); assert key not in seen; seen.add(key)
        assert c["block_size"]==8 and c["family"]=="clustered"
        assert c["active_neurons_base"]==c["active_neurons_expanded"]
        for name,values in c["raw_ms"].items():
            assert len(values)==m["repeats"] and math.isclose(statistics.median(values),c["summary"][name]["median_ms"])
    print("Validated EXP-012 cases, provenance and retained medians")
if __name__=="__main__": validate(sys.argv[1])
