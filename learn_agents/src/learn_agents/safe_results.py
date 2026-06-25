"""Avoid clobbering existing experiment artifacts on re-run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_path(
    path: Path,
    *,
    force: bool = False,
    archive_existing: bool = True,
) -> Path:
    """Return ``path`` if safe to write; otherwise a timestamped sibling path.

    When ``archive_existing`` and ``path`` exists, copy the current file to
    ``path.parent / archive / <stem>_<ts><suffix>`` before returning a new name
    if ``force`` is false.
    """
    path = path.resolve()
    if not path.exists():
        return path
    if force:
        return path

    parent = path.parent
    archive_dir = parent / "archive"
    if archive_existing:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / f"{path.stem}_{timestamp_slug()}{path.suffix}"
        archived.write_bytes(path.read_bytes())

    stamped = parent / f"{path.stem}_{timestamp_slug()}{path.suffix}"
    return stamped


def write_json(payload: Any, path: Path, *, force: bool = False, archive_existing: bool = True) -> Path:
    out = resolve_output_path(path, force=force, archive_existing=archive_existing)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
