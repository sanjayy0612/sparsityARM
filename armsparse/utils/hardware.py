import json, os, platform, shutil, subprocess, sys
try:
    import psutil
except ImportError:  # Keep hardware discovery usable before optional deps are installed.
    psutil = None


def _command_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=2).splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return None


def system_info() -> dict:
    arch = platform.machine().lower()
    info = {"os": platform.platform(), "arch": arch, "cpu": platform.processor() or "unknown",
            "logical_cores": psutil.cpu_count() if psutil else os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False) if psutil else None,
            "python": sys.version.split()[0], "compiler": _command_version(["clang", "--version"]),
            "neon_available": arch in {"arm64", "aarch64"}, "sve_available": False}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    if arch == "aarch64" and os.path.exists("/proc/cpuinfo"):
        flags = open("/proc/cpuinfo", encoding="utf-8").read().lower()
        info["sve_available"] = " sve" in flags
    return info


def system_info_json() -> str:
    return json.dumps(system_info(), indent=2, sort_keys=True)
