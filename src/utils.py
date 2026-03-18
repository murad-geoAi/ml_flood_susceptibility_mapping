from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_score(metric_fn: Any, *args: Any, default: float = float("nan"), **kwargs: Any) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return default


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        ensure_directory(path)


def write_json(payload: dict[str, Any], file_path: Path) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
