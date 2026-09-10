"""Small ctypes bridge to the EXP-001 kernels; no model or torch dependency."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD_COMMAND = ["xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG",
                 "-DACCELERATE_NEW_LAPACK", "-mmacosx-version-min=15.0",
                 "-dynamiclib", "-framework", "Accelerate", str(ROOT / "cpp/ffn.cpp"),
                 "-o", str(ROOT / "build/libffn.dylib")]


def build() -> Path:
    (ROOT / "build").mkdir(exist_ok=True)
    subprocess.run(BUILD_COMMAND, check=True, cwd=ROOT)
    return ROOT / "build/libffn.dylib"


class NativeFFN:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.configure_threads.restype = ctypes.c_int
        if self.lib.configure_threads() != 0:
            raise RuntimeError("Accelerate single-thread configuration failed")
        float_array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        byte_array = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.ffn.argtypes = [ctypes.c_int]*4 + [float_array]*4 + [byte_array] + [float_array]*2
        self.lib.ffn.restype = ctypes.c_double
        self.lib.scan_mask.argtypes = [byte_array, ctypes.c_int, int_array, ctypes.POINTER(ctypes.c_int)]
        self.lib.scan_mask.restype = ctypes.c_double
        double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.profile_block_staged.argtypes = ([ctypes.c_int] * 3 + [float_array] * 4
            + [byte_array] + [float_array] * 2 + [int_array]
            + [ctypes.POINTER(ctypes.c_int)] * 2 + [double_array])
        self.lib.profile_block_staged.restype = None
        int8_array = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        self.lib.qffn.argtypes = [ctypes.c_int]*4 + [float_array] + [int8_array]*3 \
            + [ctypes.c_float]*3 + [byte_array] + [float_array]*2
        self.lib.qffn.restype = ctypes.c_double

    def execute(self, mode, x, gate, up, down, mask, block_size, scratch, output):
        m, h = gate.shape
        if (x.shape != (h,) or up.shape != gate.shape or output.shape != (h,)
                or scratch.size != 2*m or block_size <= 0 or m % block_size
                or mode not in (0, 1, 2, 3, 4, 5) or (mode == 4 and block_size != 8)):
            raise ValueError("invalid FFN dimensions or mode")
        expected_down = {0: (h, m), 1: (m, h), 2: (m//block_size, h, block_size),
                         3: (m//block_size, h, block_size),
                         4: (m//block_size, h, block_size), 5: (h, m)}[mode]
        if down.shape != expected_down or mask.shape != ((m,) if mode < 2 else (m//block_size,)):
            raise ValueError("invalid packed weights or mask")
        return self.lib.ffn(mode, h, m, block_size, x, gate, up, down, mask, scratch, output)

    def profile_block_staged(self, x, gate, up, down, mask, block_size, scratch, output,
                             indices):
        m, h = gate.shape
        if (up.shape != gate.shape or down.shape != (m//block_size, h, block_size)
                or mask.shape != (m//block_size,) or indices.shape != (m//block_size,)):
            raise ValueError("invalid staged profile inputs")
        active, runs = ctypes.c_int(), ctypes.c_int()
        timings = np.empty(5, dtype=np.float64)
        self.lib.profile_block_staged(h, m, block_size, x, gate, up, down, mask, scratch,
                                      output, indices, ctypes.byref(active),
                                      ctypes.byref(runs), timings)
        return {"output_init_ms": timings[0], "mask_scan_ms": timings[1],
                "gate_up_activation_ms": timings[2], "down_projection_ms": timings[3],
                "total_ms": timings[4], "active_blocks": active.value,
                "active_runs": runs.value}

    def execute_quantized(self, mode, x, gate, up, down, scales, mask, block_size,
                          scratch, output):
        m, h = gate.shape
        expected = (h, m) if mode == 0 else (m//block_size, h, block_size)
        if (mode not in (0, 1) or up.shape != gate.shape or down.shape != expected
                or gate.dtype != np.int8 or up.dtype != np.int8 or down.dtype != np.int8):
            raise ValueError("invalid quantized FFN inputs")
        return self.lib.qffn(mode, h, m, block_size, x, gate, up, down,
            *map(float, scales), mask, scratch, output)


def quantize_symmetric(values):
    """Per-tensor symmetric INT8 quantization for the EXP-013 feasibility test."""
    scale = float(np.max(np.abs(values))) / 127
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("quantization requires finite nonzero weights")
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8), scale


def reference(x, gate, up, down, mask):
    """Independent FP64 NumPy masked dense reference, outside timed regions."""
    g = gate.astype(np.float64) @ x.astype(np.float64)
    u = up.astype(np.float64) @ x.astype(np.float64)
    return down.astype(np.float64) @ ((g / (1 + np.exp(-g))) * u * mask)


def mask_pair(rng, width, block_size, sparsity, family):
    """Return base neuron mask and its exact block expansion.

    Scattered masks expose expansion inflation. Clustered masks have B-wide
    active runs, so both executors perform exactly the same neuron work.
    These are synthetic locality controls, not activation-derived masks.
    """
    base = np.zeros(width, dtype=np.uint8)
    if family == "scattered":
        base[rng.permutation(width)[:round(width*(1-sparsity))]] = 1
    elif family == "clustered":
        blocks = np.zeros(width//block_size, dtype=np.uint8)
        blocks[rng.permutation(len(blocks))[:round(len(blocks)*(1-sparsity))]] = 1
        base = np.repeat(blocks, block_size)
    else:
        raise ValueError(f"unknown family: {family}")
    blocks = base.reshape(-1, block_size).any(axis=1).astype(np.uint8)
    return base, blocks
