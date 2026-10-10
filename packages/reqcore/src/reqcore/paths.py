"""Locate the repository root and its well-known directories."""

import os
from functools import cache
from pathlib import Path

_ROOT_MARKER = "uv.lock"


@cache
def repo_root() -> Path:
    """Return the workspace root: `REQ_ROOT` if set, else the nearest ancestor holding `uv.lock`."""
    override = os.environ.get("REQ_ROOT")
    if override:
        return Path(override).resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"No {_ROOT_MARKER} found above {Path.cwd()}; set REQ_ROOT.")


def questions_dir() -> Path:
    return repo_root() / "questions"


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def cache_db_path() -> Path:
    return repo_root() / "cache" / "inference.sqlite"
