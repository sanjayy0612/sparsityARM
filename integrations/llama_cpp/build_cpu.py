"""Build the pinned llama.cpp CPU benchmark and quantization tools."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LLAMA=ROOT/'third_party/llama.cpp'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,default=LLAMA/'build-arm-sparse')
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    subprocess.run(['cmake','-S',str(LLAMA),'-B',str(a.build_dir),'-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_METAL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF'],check=True)
    subprocess.run(['cmake','--build',str(a.build_dir),'--config','Release','--target',
                    'llama-bench','llama-quantize','-j',str(a.jobs)],check=True)
if __name__=='__main__': main()
