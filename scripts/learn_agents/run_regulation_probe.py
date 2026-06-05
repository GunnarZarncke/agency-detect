#!/usr/bin/env python3
"""E17 — Option D regulation probe across agent families.

Runs the data-only homeostatic/setpoint probe (flatness × compensation) on
telemetry sim agents, CartPole balance vs track, and external baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from amortized_agency.kinds import ALL_KINDS  # noqa: E402
from learn_agents.grid_pomdp import GridPomdpConfig, roll_grid_pomdp  # noqa: E402
from learn_agents.learn_agents import SimulationResult, TraceSimulationConfig, simulate_known_agent_trace  # noqa: E402
from learn_agents.physics_pomdp import roll_cartpole_partial_obs  # noqa: E402
from learn_agents.regulation_probe import score_simulation  # noqa: E402
from learn_agents.rock_sample import RockSampleConfig, roll_rock_sample  # noqa: E402


def _mean_agent_regulation(summary: Dict[str, object]) -> float:
    agents = summary.get("agents", {})
    if not agents:
        return 0.0
    vals = [float(v.get("max_regulation", 0.0)) for v in agents.values()]
    return float(np.mean(vals))


def _eval_family(
    name: str,
    builder: Callable[[int], SimulationResult],
    seeds: Sequence[int],
    *,
    threshold: float,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        result = builder(seed)
        summary = score_simulation(result, threshold=threshold)
        rows.append(
            {
                "seed": seed,
                "T": summary["T"],
                "flagged_agents": summary["flagged_agents"],
                "mean_max_regulation": _mean_agent_regulation(summary),
                "agents": summary["agents"],
            }
        )
    regs = [r["mean_max_regulation"] for r in rows]
    flagged_rate = float(np.mean([1.0 if r["flagged_agents"] else 0.0 for r in rows]))
    return {
        "family": name,
        "n_seeds": len(seeds),
        "mean_max_regulation": float(np.mean(regs)),
        "std_max_regulation": float(np.std(regs)),
        "flagged_rate": flagged_rate,
        "per_seed": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Option D regulation probe (E17)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results" / "intention" / "e17_regulation_probe.json",
    )
    args = parser.parse_args()

    families: List[tuple[str, Callable[[int], SimulationResult]]] = []

    for kind in ALL_KINDS:
        name = kind.name

        def _sim_builder(seed: int, k=kind) -> SimulationResult:
            cfg = TraceSimulationConfig(
                T=2000,
                num_agents=k.num_agents,
                copies_per_role=k.copies_per_role,
                decoy_vars=k.decoy_vars,
                interaction_strength=k.interaction_strength,
                agent_variant_mode=k.variant_mode,
                episodic=False,
                seed=seed,
            )
            return simulate_known_agent_trace(cfg)

        families.append((f"telemetry_{name}", _sim_builder))

    families.extend(
        [
            (
                "physics_cartpole_balance",
                lambda s: roll_cartpole_partial_obs(seed=s, policy="balance", normalize=False),
            ),
            (
                "physics_cartpole_track",
                lambda s: roll_cartpole_partial_obs(seed=s, policy="track", theta_ref=0.12, normalize=False),
            ),
            (
                "physics_cartpole_random",
                lambda s: roll_cartpole_partial_obs(seed=s, policy="random", normalize=False),
            ),
            (
                "rock_sample_5x5",
                lambda s: roll_rock_sample(RockSampleConfig(seed=s, max_steps=100)),
            ),
            (
                "grid_pomdp_3x3",
                lambda s: roll_grid_pomdp(
                    GridPomdpConfig(grid=3, view=3, num_agents=2, max_steps=250, seed=s)
                ),
            ),
        ]
    )

    results = []
    for name, builder in families:
        print(f"evaluating {name} ...", flush=True)
        results.append(_eval_family(name, builder, args.seeds, threshold=args.threshold))

    payload = {
        "experiment": "E17_regulation_probe",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "seeds": list(args.seeds),
        "families": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output_json}")

    print("\nSummary (mean max regulation R per family):")
    for row in sorted(results, key=lambda r: -r["mean_max_regulation"]):
        print(
            f"  {row['family']:32s}  R={row['mean_max_regulation']:.3f} "
            f"(±{row['std_max_regulation']:.3f})  flagged={row['flagged_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
