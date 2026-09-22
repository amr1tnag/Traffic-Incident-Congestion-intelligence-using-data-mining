"""Loads config.yaml and exposes it as a plain dictionary with path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and resolve every configured path to an absolute one."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["data"]["raw_dir"] = _abs(cfg["data"]["raw_dir"])
    cfg["data"]["processed_dir"] = _abs(cfg["data"]["processed_dir"])
    cfg["warehouse"]["path"] = _abs(cfg["warehouse"]["path"])
    cfg["reports"]["dir"] = _abs(cfg["reports"]["dir"])
    cfg["reports"]["figures_dir"] = _abs(cfg["reports"]["figures_dir"])
    return cfg


def _abs(rel: str) -> Path:
    path = PROJECT_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
    return path
