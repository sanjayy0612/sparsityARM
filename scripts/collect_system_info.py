#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armsparse.utils.hardware import system_info

info = system_info()
path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "system_info.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2)); print(f"Saved {path}")
