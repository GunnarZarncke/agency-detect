#!/usr/bin/env python3
"""MI + oracle UAD detectability for external POMDP trace families (E15).

Sources:
  physics  — Gymnasium CartPole-v1 partial obs (short episode, dynamical baseline)
  rock     — 5×5 RockSample, K=3 (~100 steps)
  grid3    — 3×3 grid, 2 agents, 250 steps (Melting-Pot stepping stone)
  grid5    — 5×5 grid, 2 agents, 250 steps
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
from sklearn.metrics import adjusted_rand_score


from learn_agents.grid_pomdp import GridPomdpConfig, roll_grid_pomdp  # noqa: E402
from learn_agents.learn_agents import SimulationResult, mi_cluster_variable_labels, oracle_uad_scores  # noqa: E402
from learn_agents.physics_pomdp import roll_cartpole_multi, roll_cartpole_partial_obs  # noqa: E402
from learn_agents.rock_sample import RockSampleConfig, roll_rock_sample  # noqa: E402

WINDOWS_DEFAULT = [50, 100, 150, 250]


def _mi_ari(result: SimulationResult, window: int) -> float:
    trace = result.trace
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    num_agents = int(result.metadata["config"].num_agents)
    agent_cols = np.where(var_agent >= 0)[0]
    sub = trace[:window, agent_cols]
    true_ids = var_agent[agent_cols]
    labels = mi_cluster_variable_labels(sub, num_clusters=num_agents)
    active = labels >= 0
    if active.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(true_ids[active], labels[active]))


def _oracle_summary(result: SimulationResult) -> Dict[str, float]:
    scores = oracle_uad_scores(result.trace, result.metadata)
    ratios = [v.get("separation_ratio", 0.0) for v in scores.values() if v]
    return {
        "separation_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "separation_ratio_min": float(np.min(ratios)) if ratios else 0.0,
    }


BUILDERS: Dict[str, Callable[[int], SimulationResult]] = {
    "physics": lambda seed: roll_cartpole_partial_obs(seed=seed),
    "physics3": lambda seed: roll_cartpole_multi(seed=seed, num_agents=3, n_decoy_env=8),
    "rock": lambda seed: roll_rock_sample(RockSampleConfig(seed=seed)),
    "grid3": lambda seed: roll_grid_pomdp(
        GridPomdpConfig(grid=3, view=3, num_agents=2, max_steps=250, seed=seed)
    ),
    "grid5": lambda seed: roll_grid_pomdp(
        GridPomdpConfig(grid=5, view=3, num_agents=2, max_steps=250, seed=seed)
    ),
}


def run(sources: Sequence[str], seeds: Sequence[int], windows: Sequence[int]) -> Dict:
    rows: List[Dict] = []
    for name in sources:
        build = BUILDERS[name]
        for seed in seeds:
            result = build(seed)
            T = result.trace.shape[0]
            oracle = _oracle_summary(result)
            for window in windows:
                w = min(window, T)
                ari = _mi_ari(result, w)
                rows.append(
                    {
                        "source": name,
                        "seed": seed,
                        "T": T,
                        "window": w,
                        "mi_ari": ari,
                        **oracle,
                    }
                )
            print(
                f"{name:8s} seed={seed} T={T:4d}  "
                f"oracle_sep_mean={oracle['separation_ratio_mean']:.3f}  "
                f"W={windows[-1]} ARI={_mi_ari(result, min(windows[-1], T)):.3f}"
            )
    by_source: Dict[str, List[float]] = {}
    for r in rows:
        if r["window"] == max(windows):
            by_source.setdefault(r["source"], []).append(r["mi_ari"])
    summary = {
        name: {"ari_mean": float(np.mean(v)), "ari_std": float(np.std(v)), "n": len(v)}
        for name, v in by_source.items()
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(sources),
        "seeds": list(seeds),
        "windows": list(windows),
        "rows": rows,
        "summary_max_window": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="physics,rock,grid3,grid5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/learn_agents/e15_external_pomdp_detectability.json",
    )
    p.add_argument("--force", action="store_true", help="Overwrite --out in place")
    args = p.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    payload = run(sources, args.seeds, args.windows)
    from learn_agents.safe_results import write_json

    written = write_json(payload, args.out, force=args.force)
    print("wrote", written)
    for name, s in payload["summary_max_window"].items():
        print(f"  {name}: ARI={s['ari_mean']:.3f}+-{s['ari_std']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
