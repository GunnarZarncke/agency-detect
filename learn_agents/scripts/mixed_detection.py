#!/usr/bin/env python3
"""Mixed heterogeneous detection: CartPole + RockSample + grid in one trace.

Tests whether lagged-MI clustering separates **different** agent/plant types coexisting
in one window (not three identical CartPoles). Oracle ε-blanket scores use ground-truth
agent ids from the merged metadata.
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
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score


from learn_agents.external_traces import merge_agent_traces  # noqa: E402
from learn_agents.grid_pomdp import GridPomdpConfig, roll_grid_pomdp  # noqa: E402
from learn_agents.learn_agents import SimulationResult, mi_cluster_variable_labels, oracle_uad_scores  # noqa: E402
from learn_agents.physics_pomdp import roll_cartpole_partial_obs  # noqa: E402
from learn_agents.rock_sample import RockSampleConfig, roll_rock_sample  # noqa: E402

WINDOWS_DEFAULT = [50, 100, 150, 250]


def build_mixed_physics_rock_grid3(
    seed: int, *, grid_steps: int = 250, cartpole_policy: str = "balance"
) -> SimulationResult:
    parts = [
        roll_cartpole_partial_obs(seed=seed, max_steps=grid_steps, policy=cartpole_policy),
        roll_rock_sample(RockSampleConfig(seed=seed, max_steps=grid_steps)),
        roll_grid_pomdp(
            GridPomdpConfig(grid=3, view=3, num_agents=2, max_steps=grid_steps, seed=seed)
        ),
    ]
    return merge_agent_traces(
        parts,
        seed=seed,
        source="mixed_physics_rock_grid3",
        n_decoy_env=8,
        max_T=grid_steps,
    )


def mi_ari(result: SimulationResult, window: int) -> float:
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    num_agents = int(result.metadata["config"].num_agents)
    agent_cols = np.where(var_agent >= 0)[0]
    w = min(window, result.trace.shape[0])
    sub = result.trace[:w, agent_cols]
    true_ids = var_agent[agent_cols]
    labels = mi_cluster_variable_labels(sub, num_clusters=num_agents)
    active = labels >= 0
    if active.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(true_ids[active], labels[active]))


def oracle_summary(result: SimulationResult) -> Dict[str, float]:
    scores = oracle_uad_scores(result.trace, result.metadata)
    ratios = [v.get("separation_ratio", 0.0) for v in scores.values() if v]
    return {
        "separation_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "separation_ratio_min": float(np.min(ratios)) if ratios else 0.0,
    }


def per_agent_ari(result: SimulationResult, window: int) -> Dict[int, float]:
    """ARI if we only score columns belonging to one ground-truth agent id."""
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    agent_cols = np.where(var_agent >= 0)[0]
    w = min(window, result.trace.shape[0])
    sub = result.trace[:w, agent_cols]
    labels = mi_cluster_variable_labels(sub, num_clusters=int(result.metadata["config"].num_agents))
    out: Dict[int, float] = {}
    for agent in range(int(result.metadata["config"].num_agents)):
        mask = var_agent[agent_cols] == agent
        if mask.sum() < 2:
            out[agent] = float("nan")
            continue
        true = var_agent[agent_cols][mask]
        pred = labels[mask]
        act = pred >= 0
        out[agent] = float(adjusted_rand_score(true[act], pred[act])) if act.sum() >= 2 else 0.0
    return out


def run(seeds: Sequence[int], windows: Sequence[int], cartpole_policy: str = "balance") -> Dict:
    rows: List[Dict] = []
    for seed in seeds:
        result = build_mixed_physics_rock_grid3(seed, cartpole_policy=cartpole_policy)
        T = int(result.trace.shape[0])
        n_agents = int(result.metadata["config"].num_agents)
        n_agent_cols = int(np.sum(np.asarray(result.metadata["var_agent"]) >= 0))
        oracle = oracle_summary(result)
        for window in windows:
            w = min(window, T)
            ari = mi_ari(result, w)
            rows.append(
                {
                    "source": "mixed_physics_rock_grid3",
                    "seed": seed,
                    "T": T,
                    "window": w,
                    "num_agents": n_agents,
                    "num_agent_cols": n_agent_cols,
                    "mi_ari": ari,
                    **oracle,
                }
            )
        pa = per_agent_ari(result, min(max(windows), T))
        print(
            f"mixed seed={seed} T={T} agents={n_agents} cols={n_agent_cols}  "
            f"oracle_sep_mean={oracle['separation_ratio_mean']:.3f}  "
            f"W={min(max(windows), T)} MI_ARI={mi_ari(result, min(max(windows), T)):.3f}  "
            f"per_agent_ari={{{', '.join(f'{k}:{v:.2f}' for k, v in pa.items())}}}"
        )
    max_w = max(windows)
    at_max = [r for r in rows if r["window"] == min(max_w, r["T"])]
    aris = [r["mi_ari"] for r in at_max]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "CartPole(1) + RockSample(1) + grid3(2) merged; MI + oracle on agent cols",
        "cartpole_policy": cartpole_policy,
        "seeds": list(seeds),
        "windows": list(windows),
        "rows": rows,
        "summary": {
            "mi_ari_mean": float(np.mean(aris)),
            "mi_ari_std": float(np.std(aris)),
            "n": len(aris),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--windows", type=int, nargs="+", default=WINDOWS_DEFAULT)
    p.add_argument("--cartpole-policy", choices=["random", "balance"], default="balance")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/learn_agents/mixed_detection.json",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    payload = run(args.seeds, args.windows, cartpole_policy=args.cartpole_policy)
    from learn_agents.safe_results import write_json

    written = write_json(payload, args.out, force=args.force)
    print("wrote", written)
    s = payload["summary"]
    print(f"  mixed @ max window: ARI={s['mi_ari_mean']:.3f} +- {s['mi_ari_std']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
