from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_path: Path
    output_dir: Path


def find_project_root(start: str | os.PathLike | None = None) -> Path:
    """Find project root by walking parents until we see requirements.txt or src/.

    This keeps notebooks runnable even if executed from different working dirs.
    """

    if start is None:
        here = Path.cwd()
    else:
        here = Path(start).resolve()

    candidates = [here] + list(here.parents)
    for p in candidates:
        if (p / "requirements.txt").exists() and (p / "src").exists():
            return p

    # Fallback: assume current working directory is project root
    return here


def resolve_data_path(project_root: Path) -> Path:
    """Pick the first existing CSV path from known locations."""

    candidates = [
        project_root / "data" / "Walmart_Sales.csv",
        project_root / "Walmart_Sales.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Default to data/ even if missing, so the notebook shows a clear error.
    return candidates[0]


def get_project_paths(
    *,
    project_root: str | os.PathLike | None = None,
    output_dir: str | os.PathLike = "outputs/E0_ablation",
) -> ProjectPaths:
    root = find_project_root(project_root)
    data_path = resolve_data_path(root)
    out_dir = (root / Path(output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(project_root=root, data_path=data_path, output_dir=out_dir)


def set_global_seed(seed: int = 42, deterministic: bool = False) -> Dict[str, Any]:
    """Set random seeds across random/numpy/torch.

    If deterministic=True, enables deterministic algorithms where possible.
    This may reduce throughput on GPU.
    """

    info: Dict[str, Any] = {"seed": int(seed), "deterministic": bool(deterministic)}

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        info["numpy"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["numpy"] = f"unavailable: {exc}"

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older torch versions may not support this.
                pass
        info["torch"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable: {exc}"

    return info


def get_torch_device(prefer_cuda: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Return device string ("cuda"/"cpu") and a dict with GPU details."""

    details: Dict[str, Any] = {}
    try:
        import torch

        has_cuda = bool(prefer_cuda and torch.cuda.is_available())
        device = "cuda" if has_cuda else "cpu"
        details["torch_cuda_available"] = bool(torch.cuda.is_available())
        if has_cuda:
            details["cuda_device_name"] = torch.cuda.get_device_name(0)
            details["cuda_device_count"] = int(torch.cuda.device_count())
            details["cuda_capability"] = tuple(torch.cuda.get_device_capability(0))
        return device, details
    except Exception as exc:  # noqa: BLE001
        return "cpu", {"torch_unavailable": str(exc)}


def collect_versions() -> Dict[str, Optional[str]]:
    """Collect versions for core libs used in notebooks."""

    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import numpy as np

        versions["numpy"] = np.__version__
    except Exception:
        versions["numpy"] = None

    try:
        import pandas as pd

        versions["pandas"] = pd.__version__
    except Exception:
        versions["pandas"] = None

    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda
    except Exception:
        versions["torch"] = None
        versions["torch_cuda"] = None

    try:
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        versions["scikit_learn"] = None

    try:
        import statsmodels

        versions["statsmodels"] = statsmodels.__version__
    except Exception:
        versions["statsmodels"] = None

    try:
        import prophet

        versions["prophet"] = prophet.__version__
    except Exception:
        versions["prophet"] = None

    return versions
