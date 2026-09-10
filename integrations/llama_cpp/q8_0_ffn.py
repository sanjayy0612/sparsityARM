"""ctypes bridge for the fixed-mask llama.cpp Q8_0 FFN replay kernel."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "third_party/llama.cpp"


def build(build_dir: Path) -> tuple[Path, list[str]]:
    output = ROOT / "build/libq8_0_ffn.dylib"
    output.parent.mkdir(exist_ok=True)
    command = [
        "xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-dynamiclib",
        f"-I{LLAMA / 'ggml/include'}",
        f"-L{build_dir / 'bin'}", "-lggml-cpu", "-lggml-base",
        f"-Wl,-rpath,{build_dir / 'bin'}", str(ROOT / "cpp/q8_0_ffn.cpp"), "-o", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return output, command


class Q80FFN:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.armsparse_q8_0_row_bytes.argtypes = [ctypes.c_int]
        self.lib.armsparse_q8_0_row_bytes.restype = ctypes.c_size_t
        floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        bytes_ = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.lib.armsparse_q8_0_quantize_rows.argtypes = [floats, ctypes.c_int, ctypes.c_int, bytes_]
        self.lib.armsparse_q8_0_quantize_rows.restype = ctypes.c_int
        self.lib.armsparse_q8_0_ffn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, floats,
            bytes_, bytes_, bytes_, bytes_, floats, bytes_, bytes_, floats]
        self.lib.armsparse_q8_0_ffn.restype = ctypes.c_double

    def row_bytes(self, columns: int) -> int:
        result = self.lib.armsparse_q8_0_row_bytes(columns)
        if not result:
            raise ValueError("Q8_0 columns must be a positive multiple of 32")
        return result

    def quantize(self, values: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("expected a matrix")
        result = np.empty(values.shape[0] * self.row_bytes(values.shape[1]), dtype=np.uint8)
        if self.lib.armsparse_q8_0_quantize_rows(values, *values.shape, result) != 0:
            raise ValueError("Q8_0 quantization failed")
        return result

    def execute(self, mode: int, input_: np.ndarray, gate: np.ndarray, up: np.ndarray,
                down: np.ndarray, mask: np.ndarray, activation: np.ndarray,
                quantized_input: np.ndarray, quantized_activation: np.ndarray,
                output: np.ndarray) -> float:
        intermediate = activation.size
        hidden = input_.size
        if mode not in (0, 1, 2) or mask.shape != (intermediate // 32,):
            raise ValueError("invalid mode or B32 mask")
        return self.lib.armsparse_q8_0_ffn(mode, hidden, intermediate, input_, gate, up, down,
            mask, activation, quantized_input, quantized_activation, output)
