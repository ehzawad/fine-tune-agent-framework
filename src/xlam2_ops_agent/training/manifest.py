from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import QLoRAConfig


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_info() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def write_training_manifest(
    *,
    config: QLoRAConfig,
    output_path: Path,
    data_stats: dict[str, Any],
    model: Any,
    packages: Iterable[str] = (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
    ),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "model_id": config.model.model_id,
        "requested_revision": config.model.revision,
        "resolved_model_commit": commit_hash,
        "config": config.model_dump(mode="json"),
        "data": {
            "train_sha256": sha256_file(config.data.train_file),
            "eval_sha256": sha256_file(config.data.eval_file),
            "template_sha256": sha256_file(config.data.template_file),
            "stats": data_stats,
        },
        "gpu": _gpu_info(),
        "packages": {name: _version(name) for name in packages},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
