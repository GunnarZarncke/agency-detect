#!/usr/bin/env python3
"""E18 — outcome-influence detection on labeled critical variables.

Scores whether each agent cluster's actions defend / steer operator-labeled
outcomes (resource.cpu, pole angle, etc.) after controlling for exogenous world.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np


from intention_detect.evaluate import auroc, pooled_agent_rows, score_simulation  # noqa: E402
from intention_detect.outcomes import attach_physics_critical_outcome  # noqa: E402
from learn_agents.learn_agents import SimulationResult, TraceSimulationConfig, simulate_known_agent_trace  # noqa: E402
from learn_agents.physics_pomdp import roll_cartpole_partial_obs  # noqa: E402


def _telemetry_cfg(
    *,
    seed: int,
    self_preserving: int,
    num_agents: int = 3,
) -> TraceSimulationConfig:
    return TraceSimulationConfig(
        T=2000,
        num_agents=num_agents,
        copies_per_role=3,
        decoy_vars=6,
        interaction_strength=0.05,
        world_vars=4,
        resource_vars=2,
        self_preserving_agent=self_preserving,
        self_preservation_strength=1.10,
        resource_action_coupling=0.50,
        normalize_trace=False,
        episodic=False,
        seed=seed,
    )


def _eval_family(
    name: str,
    builder: Callable[[int], SimulationResult],
    seeds: Sequence[int],
    *,
    seed_offset: int = 0,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        result = builder(seed)
        summary = score_simulation(result, seed=seed + seed_offset)
        summary["seed"] = seed
        summary["family"] = name
        rows.append(summary)

    flagged_rate = float(np.mean([1.0 if r.get("flagged_agents") else 0.0 for r in rows]))
    per_agent_flag = []
    for r in rows:
        gt = r.get("ground_truth") or {}
        agents = r.get("agents") or {}
        for aid, info in agents.items():
            per_agent_flag.append(
                {
                    "seed": r["seed"],
                    "agent": int(aid),
                    "gt": bool(gt.get(str(aid), False)),
                    "flagged": bool(info.get("flagged", False)),
                    "combined": float(info.get("max_combined", 0.0)),
                }
            )

    return {
        "family": name,
        "n_seeds": len(seeds),
        "flagged_rate": flagged_rate,
        "per_seed": rows,
        "per_agent": per_agent_flag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E18 outcome-influence eval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "intention" / "e18_outcome_influence.json",
    )
    args = parser.parse_args()

    families: List[tuple[str, Callable[[int], SimulationResult]]] = [
        (
            "telemetry_reactive_resources",
            lambda s: simulate_known_agent_trace(_telemetry_cfg(seed=s, self_preserving=-1)),
        ),
        (
            "telemetry_self_preserving_agent0",
            lambda s: simulate_known_agent_trace(_telemetry_cfg(seed=s, self_preserving=0)),
        ),
        (
            "physics_cartpole_balance",
            lambda s: attach_physics_critical_outcome(
                roll_cartpole_partial_obs(seed=s, policy="balance", normalize=False),
                policy="balance",
            ),
        ),
        (
            "physics_cartpole_track",
            lambda s: attach_physics_critical_outcome(
                roll_cartpole_partial_obs(seed=s, policy="track", normalize=False),
                policy="track",
            ),
        ),
        (
            "physics_cartpole_random",
            lambda s: attach_physics_critical_outcome(
                roll_cartpole_partial_obs(seed=s, policy="random", normalize=False),
                policy="random",
            ),
        ),
    ]

    results = []
    for name, builder in families:
        print(f"evaluating {name} ...", flush=True)
        results.append(_eval_family(name, builder, args.seeds))

    pooled_scores, pooled_labels = pooled_agent_rows(
        [row for fam in results for row in fam["per_seed"]]
    )
    pooled_auroc = auroc(pooled_scores, pooled_labels)

    payload = {
        "experiment": "E18_outcome_influence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": list(args.seeds),
        "pooled_auroc": pooled_auroc,
        "n_scored_agents": len(pooled_scores),
        "families": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output_json}")
    print(f"pooled AUROC (combined score): {pooled_auroc:.3f}  n={len(pooled_scores)}")

    print("\nPer-family flagged rate (any agent flagged):")
    for fam in results:
        print(f"  {fam['family']:36s}  flagged={fam['flagged_rate']:.0%}")

    print("\nPer-family agent-level accuracy (flag vs ground truth):")
    for fam in results:
        rows = fam["per_agent"]
        if not rows:
            continue
        correct = sum(1 for r in rows if r["flagged"] == r["gt"])
        print(f"  {fam['family']:36s}  acc={correct}/{len(rows)}")


if __name__ == "__main__":
    main()
