"""Shared import-path setup for the agency-detect monorepo."""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGES = (
    "agency_detect",
    "learn_agents",
    "agent_spotlight",
    "hierarchical_spotlight",
    "amortized_agency",
    "intention_detect",
    "data_collect",
    "uad_handles",
    "uad_worm",
)


def repo_root(start: Path | None = None) -> Path:
    here = start or Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "agency_detect").is_dir():
            return candidate
    raise RuntimeError("agency-detect repo root not found")


def install(start: Path | None = None) -> Path:
    root = repo_root(start)
    for name in _PACKAGES:
        src = root / name / "src"
        if src.is_dir():
            path = str(src)
            if path not in sys.path:
                sys.path.insert(0, path)
    return root
