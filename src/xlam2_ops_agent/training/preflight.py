from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_PARAMETERS = 32_763_900_000
MIN_A100_CLASS_GIB = 38.0
RECOMMENDED_FREE_DISK_GIB = 110.0
RECOMMENDED_HOST_RAM_GIB = 64.0
REQUIRED_TRAIN_PACKAGES = {
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "bitsandbytes": "0.50.2",
}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host_memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib / 1024**2
    return None


def _existing_ancestor(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing ancestor for filesystem path: {path}")
    return candidate


def collect_preflight(*, path: str | Path = ".") -> dict[str, Any]:
    checked_path = _existing_ancestor(path)
    disk = shutil.disk_usage(checked_path)
    host_ram_gib = _host_memory_gib()
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "filesystem_checked": str(checked_path),
        "host_ram_gib": host_ram_gib,
        "disk_free_gib": disk.free / 1024**3,
        "ideal_bf16_weight_gib": MODEL_PARAMETERS * 2 / 1024**3,
        "ideal_raw_4bit_weight_gib": MODEL_PARAMETERS * 0.5 / 1024**3,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "datasets",
                "bitsandbytes",
                "vllm",
                "vllm-bnb-plugin",
            )
        },
        "gpu": None,
        "nvidia_smi": None,
        "warnings": [],
        "errors": [],
    }

    if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
        result["errors"].append(
            f"Python 3.10-3.13 is required by this repository; found {sys.version.split()[0]}"
        )

    for package, expected in REQUIRED_TRAIN_PACKAGES.items():
        actual = result["packages"].get(package)
        if actual is None:
            result["errors"].append(f"Required package is not installed: {package}=={expected}")
        elif actual != expected and not actual.startswith(expected + "+"):
            result["errors"].append(
                f"Package version mismatch: {package}=={actual}; expected {expected}"
            )

    try:
        result["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["warnings"].append(f"nvidia-smi unavailable: {exc}")

    try:
        import torch
    except ImportError:
        result["errors"].append("PyTorch is not installed in this environment")
        return result

    if not torch.cuda.is_available():
        result["errors"].append("torch.cuda.is_available() is false")
        return result

    visible_devices = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    result["gpu"] = {
        "visible_devices": visible_devices,
        "name": props.name,
        "total_memory_gib": total_gib,
        "compute_capability": f"{props.major}.{props.minor}",
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cuda_runtime": torch.version.cuda,
    }
    if total_gib < MIN_A100_CLASS_GIB:
        result["errors"].append(
            f"GPU reports only {total_gib:.1f} GiB; this recipe targets an A100 40 GB class GPU"
        )
    if not torch.cuda.is_bf16_supported():
        result["errors"].append("bfloat16 is not supported by the active CUDA device/runtime")
    if disk.free < RECOMMENDED_FREE_DISK_GIB * 1024**3:
        result["warnings"].append(
            f"Less than {RECOMMENDED_FREE_DISK_GIB:.0f} GiB of free disk is available; "
            "the original checkpoint cache plus outputs may exhaust it"
        )
    if host_ram_gib is not None and host_ram_gib < RECOMMENDED_HOST_RAM_GIB:
        result["warnings"].append(
            f"Host RAM is {host_ram_gib:.1f} GiB; at least "
            f"{RECOMMENDED_HOST_RAM_GIB:.0f} GiB is recommended for loading and preprocessing"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check an A100 host before xLAM-2 QLoRA")
    parser.add_argument("--path", default=".", help="Filesystem whose free space should be checked")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on preflight errors")
    args = parser.parse_args(argv)
    result = collect_preflight(path=args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
