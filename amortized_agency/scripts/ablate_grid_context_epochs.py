#!/usr/bin/env python3
"""Simplest grid-transfer ablation: train longer on EXTENDED_TRAIN_KINDS.

Hypothesis: grid context failure is primarily **under-training** (fixed 40/25 epochs
on a 5-kind pool), not missing model capacity. Compares default vs 2× epochs on
grid_pomdp_3x3 context ARI only (full eval still runs).

See ``run_dataset_vs_baseline.py`` for flags and pool definition.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _bootstrap_repo() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "repo_bootstrap.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError("agency-detect repo root not found")
    import repo_bootstrap

    return repo_bootstrap.install(here)


REPO_ROOT = _bootstrap_repo()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    base = [
        sys.executable,
        str(REPO_ROOT / "amortized_agency/scripts/run_dataset_vs_baseline.py"),
        "--extended-pool",
        "--run-mi",
    ]
    if args.device:
        base += ["--device", args.device]
    base += ["--eval-seeds", str(args.eval_seeds)]
    if args.force:
        base.append("--force")

    runs = [
        ("default", 40, 25, REPO_ROOT / "results/amortized/ablate_grid_epochs_default.json"),
        ("2x_epochs", 80, 50, REPO_ROOT / "results/amortized/ablate_grid_epochs_2x.json"),
    ]
    for label, ctx_ep, siam_ep, out in runs:
        cmd = base + [
            "--context-epochs",
            str(ctx_ep),
            "--siamese-epochs",
            str(siam_ep),
            "--out",
            str(out),
        ]
        print(f"\n=== {label}: context_epochs={ctx_ep} siamese_epochs={siam_ep} ===")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    import json

    rows = {}
    for label, _, _, out in runs:
        data = json.loads(out.read_text())
        for r in data["rows"]:
            if r["kind"] == "grid_pomdp_3x3":
                rows[label] = r
                break

    print("\n--- grid_pomdp_3x3 context ARI ---")
    for label, r in rows.items():
        print(
            f"  {label:12}  MI={r.get('mi_ari', 0):.3f}  "
            f"context={r.get('context_ari', 0):.3f}"
        )


if __name__ == "__main__":
    main()
