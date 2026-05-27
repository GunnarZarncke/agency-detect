#!/usr/bin/env python3
"""
E10 sweeps: miss diagnosis, agent count, decoys, world vars, agency gate modes.

Example:
  .venv/bin/python scripts/run_spotlight_sweeps.py --fast
  .venv/bin/python scripts/run_spotlight_sweeps.py --diagnose-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import run_spotlight_peel

BASE = SpotlightConfig(verbose=False)

FAST = dict(pretrain_epochs=25, refine_epochs=20)
FULL = dict(pretrain_epochs=50, refine_epochs=40)


def _run(label: str, cfg: SpotlightConfig) -> Dict[str, Any]:
    report = run_spotlight_peel(cfg)
    s = report["summary"]
    diag = report.get("agent_diagnostics", {})
    row = {
        "label": label,
        "cumulative_recall": s["cumulative_recall"],
        "pass1_jaccard": s["pass1_jaccard"],
        "n_admitted": s["n_admitted"],
        "num_agents": cfg.num_agents,
        "admitted_agent_ids": s["admitted_agent_ids"],
        "missed_agent_ids": s.get("missed_agent_ids", diag.get("missed_agent_ids", [])),
        "agency_gate_mode": s.get("agency_gate_mode", cfg.effective_agency_gate_mode()),
        "decoy_vars": cfg.decoy_vars,
        "world_vars": cfg.world_vars,
        "num_vars": report["sim_metadata"]["num_vars"],
        "agent_diagnostics": diag.get("by_agent", {}),
        "mi_residual": report.get("mi_residual", {}),
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spotlight E10 sweeps.")
    p.add_argument("--fast", action="store_true", help="Use reduced train epochs for grid sweeps.")
    p.add_argument("--diagnose-only", action="store_true", help="Only run 8-agent miss diagnosis.")
    p.add_argument("--output-json", type=str, default="results/spotlight_e10_sweeps.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    epoch_kw = FAST if args.fast else FULL
    rows: List[Dict[str, Any]] = []

    print("E10 spotlight sweeps\n")

    # --- Miss diagnosis (8 agents, full/default benchmark) ---
    diag_cfg = replace(BASE, **epoch_kw, max_passes=8, agency_gate_mode="off")
    print("=== Miss diagnosis (8 agents, gate=off) ===")
    diag_row = _run("diagnose_8agent_baseline", diag_cfg)
    rows.append(diag_row)
    for aid in diag_row["missed_agent_ids"]:
        d = diag_row["agent_diagnostics"].get(str(aid), {})
        residual = diag_row["mi_residual"].get("by_agent", {}).get(str(aid), {})
        print(
            f"  missed agent {aid}: reason={d.get('miss_reason')} "
            f"proposal_J={d.get('best_proposal_jaccard', 0):.2f} "
            f"residual_J={residual.get('best_jaccard', 0):.2f}"
        )
    print(
        f"  recall={diag_row['cumulative_recall']:.3f} admitted={diag_row['admitted_agent_ids']}\n"
    )

    if args.diagnose_only:
        _write(args.output_json, rows)
        return

    # --- Agent count ---
    print("=== Agent count sweep ===")
    for n in (8, 12, 16):
        cfg = replace(
            BASE,
            **epoch_kw,
            num_agents=n,
            max_passes=n,
            proposal_mi_k=max(16, n * 2),
            agency_gate_mode="off",
        )
        row = _run(f"agents_{n}", cfg)
        rows.append(row)
        print(
            f"  n={n:2d} recall={row['cumulative_recall']:.3f} "
            f"admitted={len(row['admitted_agent_ids'])}/{n} missed={row['missed_agent_ids']}"
        )
    print()

    # --- Decoy sweep ---
    print("=== Decoy sweep (8 agents) ===")
    for d in (0, 6, 12, 18):
        cfg = replace(BASE, **epoch_kw, decoy_vars=d, max_passes=8, agency_gate_mode="off")
        row = _run(f"decoys_{d}", cfg)
        rows.append(row)
        print(
            f"  decoys={d:2d} recall={row['cumulative_recall']:.3f} "
            f"n_vars={row['num_vars']} missed={row['missed_agent_ids']}"
        )
    print()

    # --- World sweep ---
    print("=== World sweep (8 agents) ===")
    for w in (0, 6, 12, 18):
        cfg = replace(BASE, **epoch_kw, world_vars=w, max_passes=8, agency_gate_mode="off")
        row = _run(f"world_{w}", cfg)
        rows.append(row)
        print(
            f"  world={w:2d} recall={row['cumulative_recall']:.3f} "
            f"n_vars={row['num_vars']} missed={row['missed_agent_ids']}"
        )
    print()

    # --- Agency gate modes ---
    print("=== Agency gate sweep (8 agents, world=12, decoys=0) ===")
    for mode in ("off", "actions_only", "score_penalty", "soft", "strict"):
        cfg = replace(
            BASE,
            **epoch_kw,
            agency_gate_mode=mode,
            require_agency_signature=False,
            max_passes=8,
        )
        row = _run(f"gate_{mode}", cfg)
        rows.append(row)
        print(
            f"  gate={mode:<14} recall={row['cumulative_recall']:.3f} "
            f"missed={row['missed_agent_ids']}"
        )

    _write(args.output_json, rows)


def _write(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rows": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
