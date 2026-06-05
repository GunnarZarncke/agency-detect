#!/usr/bin/env python3
"""E19 — collect real CPU/RAM traces with confounded shell agents, then score."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_collect.config import MachineRunConfig  # noqa: E402
from data_collect.pack_run import pack_machine_run  # noqa: E402
from data_collect.run import collect_machine_run  # noqa: E402
from intention_detect.evaluate import auroc, score_simulation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="E19 real-machine outcome-influence dataset")
    parser.add_argument("--duration", type=float, default=1800.0, help="seconds (default 30 min)")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--stressor-cores", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "intention" / "machine_runs",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "intention" / "e19_machine_dataset.json",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect traces only; skip scoring",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="score an existing run directory instead of collecting",
    )
    args = parser.parse_args()

    if args.score_only:
        run_dir = args.score_only
        print(f"packing {run_dir} ...", flush=True)
        result = pack_machine_run(run_dir, seed=args.seed)
    else:
        cfg = MachineRunConfig(
            duration_s=args.duration,
            dt=args.dt,
            max_cores=args.max_cores,
            stressor_cores=args.stressor_cores,
            output_dir=args.output_dir,
        )
        print(
            f"collecting {cfg.n_ticks} ticks ({cfg.duration_s:.0f}s, dt={cfg.dt}s, "
            f"max_cores={cfg.max_cores}) ...",
            flush=True,
        )
        run_dir = collect_machine_run(cfg, seed=args.seed)
        print(f"collection finished: {run_dir}", flush=True)
        if args.collect_only:
            return
        result = pack_machine_run(run_dir, seed=args.seed)

    summary = score_simulation(result, seed=args.seed)
    per_agent = []
    gt = summary.get("ground_truth") or {}
    agents = summary.get("agents") or {}
    for aid, info in agents.items():
        name = result.metadata.get("agent_labels", {}).get(aid, aid)
        per_agent.append(
            {
                "agent": int(aid),
                "name": name,
                "gt": bool(gt.get(aid, False)),
                "flagged": bool(info.get("flagged", False)),
                "max_combined": float(info.get("max_combined", 0.0)),
                "best_outcome": info.get("best_outcome"),
            }
        )

    scores = [r["max_combined"] for r in per_agent]
    labels = [r["gt"] for r in per_agent]
    pooled = auroc(scores, labels)
    correct = sum(1 for r in per_agent if r["flagged"] == r["gt"])

    payload = {
        "experiment": "E19_machine_dataset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "duration_s": args.duration,
        "T": summary.get("T"),
        "pooled_auroc": pooled,
        "agent_accuracy": f"{correct}/{len(per_agent)}",
        "summary": summary,
        "per_agent": per_agent,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output_json}")
    print(f"AUROC={pooled:.3f}  agent acc={correct}/{len(per_agent)}")
    for row in per_agent:
        mark = "OK" if row["flagged"] == row["gt"] else "MISS"
        print(
            f"  [{mark}] {row['name']:18s} gt={row['gt']} flagged={row['flagged']} "
            f"score={row['max_combined']:.3f}"
        )


if __name__ == "__main__":
    main()
