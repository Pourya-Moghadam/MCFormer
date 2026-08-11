"""Seed control, provenance capture, hashing, and run initialization."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, bool]:
    """Seed available RNG libraries and configure deterministic PyTorch behavior."""

    random.seed(seed)
    seeded = {"python": True, "numpy": False, "torch": False}
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded["numpy"] = True

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
        seeded["torch"] = True
    return seeded


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect stable runtime and dependency provenance."""

    packages = {
        name: version
        for name in ("numpy", "PyYAML", "torch", "torchvision")
        if (version := _package_version(name)) is not None
    }
    environment: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        environment["torch_available"] = False
    else:
        environment.update(
            {
                "torch_available": True,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
                "gpu_names": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        )
    return environment


def collect_git_state(repository: str | Path) -> dict[str, Any]:
    """Return commit and dirty status without mutating the repository."""

    root = Path(repository).resolve()

    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON mapping, creating parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def initialize_run(
    *,
    output_dir: str | Path,
    resolved_config: Mapping[str, Any],
    repository: str | Path,
) -> dict[str, Path]:
    """Create an auditable run directory and write its immutable starting metadata."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    files = {
        "config": destination / "resolved_config.json",
        "environment": destination / "environment.json",
        "git": destination / "git_state.json",
    }
    write_json_atomic(files["config"], resolved_config)
    write_json_atomic(files["environment"], collect_environment())
    write_json_atomic(files["git"], collect_git_state(repository))
    return files
