"""Check EXP-002 hashes, masks, counts and reported aggregation independently."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(directory):
    capture = json.loads((directory/"manifest.json").read_text())
    analysis = json.loads((directory/"analysis/manifest.json").read_text())
    require(capture["status"] == analysis["status"] == "completed","incomplete experiment")
    require(sha(directory/"manifest.json") == analysis["input_manifest_sha256"],"changed capture manifest")
    for name, digest in capture["source_sha256"].items():
        require(sha(directory/"sources"/name) == digest,f"changed source {name}")
    for item in analysis["artifacts"]:
        require(sha(directory/"analysis"/item["path"]) == item["sha256"],f"changed analysis {item['path']}")
    require(len(capture["artifacts"]) == 128,"missing capture files")
    token_count = 0
    for item in capture["artifacts"]:
        path = directory/item["path"]
        require(sha(path) == item["sha256"],f"changed capture {path}")
        with np.load(path,allow_pickle=False) as data:
            require(np.isfinite(data["activation"]).all(),"nonfinite capture")
            positions = data["token_positions"]
            require(len(positions) == item["tokens"],"incorrect token count")
        token_count += item["tokens"]
        with np.load(directory/"analysis"/(path.stem+"-masks.npz"),allow_pickle=False) as masks:
            require(np.array_equal(positions,masks["token_positions"]),"changed token positions")
            for b in (8,16,32,64):
                for s in (.2,.3,.4):
                    prefix=f"B{b}_s{round(s*100)}_"
                    base=masks[prefix+"base_mask"]
                    expanded=masks[prefix+"expanded_block_mask"]
                    budget=masks[prefix+"budget_block_mask"]
                    require(base.shape==(item["tokens"],8192),"invalid mask shape")
                    require(np.all(base.sum(axis=1)==round(8192*(1-s))),"wrong base budget")
                    require(np.array_equal(expanded,base.reshape(len(base),-1,b).any(axis=2)),"incorrect expansion")
                    require(np.all(budget.sum(axis=1)==round((8192//b)*(1-s))),"wrong block budget")
                    require(np.array_equal(masks[prefix+"expanded_sparsity"],1-expanded.mean(axis=1)),"incorrect coverage metric")
    with (directory/"analysis/metrics.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    require(len(rows)==analysis["metric_rows"]==token_count*12,"missing metric rows")
    aggregates=json.loads((directory/"analysis/aggregate.json").read_text())
    require(len(aggregates)==12,"missing aggregate configurations")
    for group in aggregates:
        selected=[r for r in rows if int(r["block_size"])==group["block_size"] and float(r["target_sparsity"])==group["target_sparsity"]]
        require(len(selected)==group["samples"]==token_count,"wrong aggregate count")
        for field in ["base_sparsity","expanded_sparsity","budget_sparsity","permuted_expanded_sparsity_mean","neuron_retained_l1","budget_retained_l1"]:
            mean=sum(float(row[field]) for row in selected)/len(selected)
            require(math.isclose(mean,group[field],rel_tol=1e-10,abs_tol=1e-12),f"incorrect mean: {field}")
    print(f"Validated EXP-002: 128 capture files, {token_count} layer/tokens, {len(rows)} metric rows, all artifact hashes and mask budgets")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    validate(parser.parse_args().directory)
