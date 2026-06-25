#!/usr/bin/env python3
"""
Small exploratory sweep: sim noise/coupling + proposal K tweaks for 8/8 recovery.

Hypothesis: partial-peel orphans come from cross-agent MI merge (coupling) or
K=16 clusters that span >1 agent. Lower A-A coupling, higher K, or weaker world
readout may yield purer single-agent clusters.

Example:
  .venv/bin/python scripts/spotlight/run_spotlight_recovery_sweep.py --fast
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

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import run_spotlight_peel

# Keep the sweep baseline at the original E10 setting so the K=24 recovery
# remains visible even if SpotlightConfig defaults move forward.
BASE = SpotlightConfig(verbose=False, num_agents=8, max_passes=8, proposal_mi_k=16)
FAST = dict(pretrain_epochs=25, refine_epochs=20)


def _run(label: str, cfg: SpotlightConfig) -> Dict[str, Any]:
    report = run_spotlight_peel(cfg)
    s = report["summary"]
    missed = s.get("missed_agent_ids", [])
    return {
        "label": label,
        "recall": s["cumulative_recall"],
        "pass1_j": s["pass1_jaccard"],
        "admitted": s["admitted_agent_ids"],
        "missed": missed,
        "n_admitted": len(s["admitted_agent_ids"]),
        "interaction_strength": cfg.interaction_strength,
        "mixing_strength": cfg.mixing_strength,
        "leakage_strength": cfg.leakage_strength,
        "local_env_strength": cfg.local_env_strength,
        "process_noise": cfg.process_noise,
        "observation_noise": cfg.observation_noise,
        "world_vars": cfg.world_vars,
        "world_to_sensor_strength": cfg.world_to_sensor_strength,
        "proposal_mi_k": cfg.proposal_mi_k,
        "seed": cfg.seed,
    }


def conditions() -> List[Tuple[str, Dict[str, Any]]]:
    """One-at-a-time perturbations from exogenous benchmark defaults."""
    rows: List[Tuple[str, Dict[str, Any]]] = [("baseline", {})]

    for v in (0.05, 0.03, 0.0):
        rows.append((f"interaction_{v}", {"interaction_strength": v}))
    for v in (0.0, 0.01):
        rows.append((f"mixing_{v}", {"mixing_strength": v}))
    for v in (0.0,):
        rows.append((f"leakage_{v}", {"leakage_strength": v}))
    for v in (2.0, 2.4):
        rows.append((f"local_env_{v}", {"local_env_strength": v}))
    for v in (0.01, 0.005):
        rows.append((f"process_noise_{v}", {"process_noise": v}))
    for v in (0.005,):
        rows.append((f"obs_noise_{v}", {"observation_noise": v}))
    for v in (6, 8):
        rows.append((f"world_vars_{v}", {"world_vars": v}))
    for v in (0.04, 0.05):
        rows.append((f"world_to_sensor_{v}", {"world_to_sensor_strength": v}))
    for k in (20, 24, 32):
        rows.append((f"mi_k_{k}", {"proposal_mi_k": k}))

    # Promising combos (weaker cross-agent + purer partition)
    rows.append(("combo_low_coupling_k24", {
        "interaction_strength": 0.03,
        "mixing_strength": 0.0,
        "leakage_strength": 0.0,
        "proposal_mi_k": 24,
    }))
    rows.append(("combo_low_coupling_k32", {
        "interaction_strength": 0.03,
        "mixing_strength": 0.0,
        "local_env_strength": 2.2,
        "proposal_mi_k": 32,
    }))
    rows.append(("combo_weak_world_k24", {
        "world_vars": 8,
        "world_to_sensor_strength": 0.04,
        "proposal_mi_k": 24,
    }))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exploratory recovery sweep (8 agents).")
    p.add_argument("--fast", action="store_true", default=True)
    p.add_argument("--full-epochs", action="store_true", help="Use default 50/40 train epochs.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--output-json", type=str, default="results/spotlight/e10/spotlight_recovery_sweep.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    epoch_kw = {} if args.full_epochs else FAST
    all_rows: List[Dict[str, Any]] = []

    print("Recovery sweep (8 agents, data-only peel)\n")
    print(f"{'label':<28} {'recall':>7} {'pass1J':>7} {'missed'}")
    print("-" * 60)

    for seed in args.seeds:
        for label, kw in conditions():
            cfg = replace(BASE, **epoch_kw, seed=seed, **kw)
            if seed != 1:
                label = f"{label}_s{seed}"
            row = _run(label, cfg)
            all_rows.append(row)
            mark = " ***" if row["recall"] >= 1.0 else ""
            print(
                f"{row['label']:<28} {row['recall']:7.3f} {row['pass1_j']:7.3f} "
                f"{row['missed']}{mark}"
            )

    perfect = [r for r in all_rows if r["recall"] >= 1.0]
    best = sorted(all_rows, key=lambda r: (-r["recall"], -r["pass1_j"]))[:5]

    print("-" * 60)
    print(f"Perfect (8/8): {len(perfect)} / {len(all_rows)}")
    if perfect:
        for r in perfect:
            print(f"  {r['label']}")
    print("\nTop 5 by recall:")
    for r in best:
        print(f"  {r['label']}: recall={r['recall']:.3f} missed={r['missed']}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rows": all_rows,
        "perfect": perfect,
        "top5": best,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
