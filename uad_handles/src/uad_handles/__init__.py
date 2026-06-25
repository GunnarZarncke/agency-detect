"""Handle-aware UAD toy benchmarks for the access-uad paper.

Synthetic worlds with embedded observation/operation handles, passive alias decoys,
and active handle tests. See README.md and docs/papers/access-uad/access-uad.tex.
"""

from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "agency_detect").is_dir():
            return candidate
    raise RuntimeError("agency-detect repo root not found")


REPO_ROOT = _repo_root()
RESULTS_DIR = REPO_ROOT / "results" / "handles"


def default_outdir(name: str) -> Path:
    return RESULTS_DIR / name
