"""Run the decisive evidence validators and package consistency check."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ["research_tools/validate_exp004.py", "research/results/EXP-004/m2-oracle-screening-20260909"],
    ["research_tools/validate_exp005.py", "research/results/EXP-005/m2-replay-20260909"],
    ["research_tools/validate_exp012.py", "research/results/EXP-012/m2-b8-break-even-20260910"],
    ["research_tools/validate_exp014.py", "research/results/EXP-014/m2-q8-dense-20260910-r1.json"],
    ["research_tools/validate_exp016.py", "research/results/EXP-016/m2-tinyllama-screening-20260910-r1/manifest.json"],
    ["research_tools/validate_exp018.py", "research/results/EXP-018/m2-tinyllama-b8-break-even-20260911-r1"],
]


def main():
    for script, artifact in COMMANDS:
        subprocess.run([sys.executable, script, artifact], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "research_tools/package_results.py", "--check"], cwd=ROOT, check=True)
    print("Verified decisive ARM-Sparse evidence package")


if __name__ == "__main__":
    main()
