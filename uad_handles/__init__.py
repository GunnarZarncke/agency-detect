"""Handle-aware UAD toy benchmarks for the access-uad paper.

Synthetic worlds with embedded observation/operation handles, passive alias decoys,
and active handle tests. See README.md and docs/papers/access-uad/access-uad.tex.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "handles"


def default_outdir(name: str) -> Path:
    return RESULTS_DIR / name
