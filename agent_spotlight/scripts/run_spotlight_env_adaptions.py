#!/usr/bin/env python3
"""
E9c: sequential realism adaptions (1→2→3) with E9b spotlight (mi_cluster).

Adaptions (cumulative):
  1. Rebalance coupling — weak A-A ring, stronger private env drive
  2. Shared exogenous world (replaces passive decoys)
  3. Optional weak local exogenous patches (world.local{k}.*, read-only)

Example:
  .venv/bin/python scripts/spotlight/run_spotlight_env_adaptions.py
"""

from __future__ import annotations

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

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import run_spotlight_peel

# E9b method defaults + realism knobs (adaptions stack).
BASE = SpotlightConfig(
    candidate_mode="mi_cluster",
    decoy_vars=12,
    decoy_mode="noise",
    verbose=False,
)

ADAPTION_1 = dict(
    local_env_strength=1.8,
    interaction_strength=0.10,
    mixing_strength=0.02,
    leakage_strength=0.01,
)

ADAPTION_2 = dict(
    env_vars_per_agent=0,
    env_action_coupling=0.0,
    world_vars=12,
    world_to_sensor_strength=0.08,
    decoy_vars=0,
)

ADAPTION_3 = dict(
    env_vars_per_agent=2,
    env_to_sensor_strength=0.05,
)

CONDITIONS = [
    ("adapt1_rebalance", ADAPTION_1),
    ("adapt2_per_agent_env", {**ADAPTION_1, **ADAPTION_2}),
    ("adapt3_shared_world", {**ADAPTION_1, **ADAPTION_2, **ADAPTION_3}),
]

BASELINE_RECALL = 0.625  # E9b ring-heavy setting


def main() -> None:
    out_dir = REPO_ROOT / "results" / "spotlight" / "e9"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print("E9c: spotlight env-realism adaptions (E9b / mi_cluster)\n")
    print(f"{'condition':<28} {'recall':>8} {'pass1_J':>8} {'admitted':>12} {'n_vars':>8}")
    print("-" * 70)

    for name, kwargs in CONDITIONS:
        cfg = replace(BASE, **kwargs)
        report = run_spotlight_peel(cfg)
        s = report["summary"]
        n_vars = report["sim_metadata"]["num_vars"]
        row = {
            "condition": name,
            "cumulative_recall": s["cumulative_recall"],
            "pass1_jaccard": s["pass1_jaccard"],
            "n_admitted": s["n_admitted"],
            "admitted_agent_ids": s["admitted_agent_ids"],
            "num_vars": n_vars,
            "config": cfg.to_dict(),
        }
        summary_rows.append(row)

        out_path = out_dir / f"spotlight_{name}.json"
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        report["adaptation_summary"] = row
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            f"{name:<28} {s['cumulative_recall']:8.3f} {s['pass1_jaccard']:8.3f} "
            f"{s['n_admitted']:>4}/8      {n_vars:8d}"
        )

    manifest = {
        "baseline_e9b_recall": BASELINE_RECALL,
        "conditions": summary_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "spotlight_env_adaptions_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("-" * 70)
    print(f"baseline E9b (ring-heavy): recall={BASELINE_RECALL:.3f}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
