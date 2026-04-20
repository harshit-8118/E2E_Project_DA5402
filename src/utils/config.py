from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_repo_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", REPO_ROOT)).resolve()


def load_params(section: str | None = None) -> dict[str, Any]:
    with open(get_repo_root() / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params if section is None else params[section]


def resolve_path(path_value: str | os.PathLike[str]) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((get_repo_root() / path).resolve())


def ensure_parent(path_value: str | os.PathLike[str]) -> str:
    path = Path(resolve_path(path_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def ensure_dir(path_value: str | os.PathLike[str]) -> str:
    path = Path(resolve_path(path_value))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)