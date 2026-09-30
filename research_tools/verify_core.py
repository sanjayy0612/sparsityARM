"""Run the decisive evidence validators and package consistency check."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_tools.inputs import SKIP_PREFIX, STRICT_ENV  # noqa: E402

COMMANDS = [
    ["research_tools/validate_exp004.py", "research/results/EXP-004/m2-oracle-screening-20260909"],
    ["research_tools/validate_exp005.py", "research/results/EXP-005/m2-replay-20260909"],
    ["research_tools/validate_exp012.py", "research/results/EXP-012/m2-b8-break-even-20260910"],
    ["research_tools/validate_exp014.py", "research/results/EXP-014/m2-q8-dense-20260910-r1.json"],
    ["research_tools/validate_exp016.py", "research/results/EXP-016/m2-tinyllama-screening-20260910-r1/manifest.json"],
    ["research_tools/validate_exp018.py", "research/results/EXP-018/m2-tinyllama-b8-break-even-20260911-r1"],
]


def run(args, env, skipped):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(f"FAILED: {' '.join(args)}")
    skipped.extend(line for line in result.stdout.splitlines() if line.startswith(SKIP_PREFIX))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help=f"fail on missing ignored/external inputs (same as {STRICT_ENV}=1)")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if args.strict:
        env[STRICT_ENV] = "1"
    skipped = []
    for script, artifact in COMMANDS:
        run([script, artifact], env, skipped)
    run(["research_tools/package_results.py", "--check"], env, skipped)
    if skipped:
        print(f"WARNING: {len(skipped)} input check(s) skipped because files are not in this checkout; "
              f"rerun with {STRICT_ENV}=1 where the full inputs exist")
    print(f"Verified decisive ARM-Sparse evidence package ({len(skipped)} input checks skipped)")


if __name__ == "__main__":
    main()
