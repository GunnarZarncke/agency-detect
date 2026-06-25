#!/usr/bin/env python3
"""Detectability matrix for telemetry-style agent extensions.

Verifies that each new realism extension keeps agents recoverable by the repo's
parameter-free MI proposal step (`mi_cluster_variable_labels`), both individually
and all combined.

Extensions (orthogonal toggles on a fixed base kind):
  - periodic   : shared diurnal driver  (shared_period / shared_periodic_strength)
  - heavytail  : bursty per-agent sensor (innovation_dist="t")
  - regime     : on/off episodes        (episodic=True, short episodes)
  - saturate   : saturating coupling     (agent_variant_mode="complex")
  - all         : every extension combined

Metric is restricted to agent variables (var_agent >= 0) so it isolates the
extension's effect on agent separability from decoy/world rejection.
Same protocol as scripts/amortized/baseline_window_breaking_point.py
(simulate one T-step trace per seed, slice the first W steps).
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
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score


from learn_agents.learn_agents import (  # noqa: E402
    TraceSimulationConfig,
    mi_cluster_variable_labels,
    simulate_known_agent_trace,
)

EVAL_T_STEPS = 2000  # fixed long trace; slice W from the front (matches E13 baseline)

# Base kind: a moderate world where MI is reliable at W>=250, so any ARI drop is
# attributable to the extension rather than to baseline difficulty.
BASE = dict(
    num_agents=5,
    copies_per_role=3,
    decoy_vars=8,
    interaction_strength=0.05,
    agent_variant_mode="rich",
    episodic=False,
)

# Each extension as a set of config overrides on BASE.
EXTENSIONS: Dict[str, Dict] = {
    "periodic": dict(shared_period=200, shared_periodic_strength=0.5),
    "heavytail": dict(innovation_dist="t", innovation_df=3.0, innovation_strength=0.6),
    # short episodes so every agent toggles at least once inside a 250-step window
    "regime": dict(episodic=True, episode_len=120, episode_gap=60),
    "saturate": dict(agent_variant_mode="complex"),
}

# Which combinations to evaluate: baseline, each extension alone, and all combined.
COMBOS: List[List[str]] = [
    [],
    ["periodic"],
    ["heavytail"],
    ["regime"],
    ["saturate"],
    ["periodic", "heavytail", "regime", "saturate"],
]


def combo_overrides(names: Sequence[str]) -> Dict:
    overrides: Dict = {}
    for name in names:
        overrides.update(EXTENSIONS[name])
    return overrides


def simulate(overrides: Dict, t_steps: int, seed: int):
    params = {**BASE, **overrides}  # overrides win on shared keys (e.g. episodic)
    cfg = TraceSimulationConfig(T=t_steps, seed=seed, **params)
    result = simulate_known_agent_trace(cfg)
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    return result.trace, var_agent, cfg.num_agents


def evaluate_window(trace: np.ndarray, var_agent: np.ndarray, num_agents: int, window: int) -> float:
    agent_cols = np.where(var_agent >= 0)[0]
    sub = trace[:window, agent_cols]
    true_ids = var_agent[agent_cols]
    labels = mi_cluster_variable_labels(sub, num_clusters=num_agents)
    active = labels >= 0
    if active.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(true_ids[active], labels[active]))


def run(seeds: Sequence[int], windows: Sequence[int]) -> Dict:
    rows: List[Dict] = []
    for names in COMBOS:
        label = "none" if not names else "+".join(names) if len(names) < len(EXTENSIONS) else "all"
        overrides = combo_overrides(names)
        traces = [simulate(overrides, EVAL_T_STEPS, s) for s in seeds]
        for window in windows:
            aris = [evaluate_window(tr, va, na, window) for tr, va, na in traces]
            rows.append(
                {
                    "combo": label,
                    "extensions": list(names),
                    "window": window,
                    "ari_mean": float(np.mean(aris)),
                    "ari_std": float(np.std(aris)),
                    "n_seeds": len(seeds),
                }
            )
            print(f"{label:32s} W={window:4d}  ARI={np.mean(aris):.3f}+-{np.std(aris):.3f}")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base": BASE,
        "extensions": EXTENSIONS,
        "seeds": list(seeds),
        "windows": list(windows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--windows", type=int, nargs="+", default=[500, 250])
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "learn_agents" / "telemetry_extensions"))
    parser.add_argument("--force", action="store_true", help="Overwrite detectability JSON in place")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run(seeds, args.windows)
    from learn_agents.safe_results import write_json

    out_json = out_dir / "telemetry_extension_detectability.json"
    written = write_json(summary, out_json, force=args.force)
    print(f"\nWrote {written}")


if __name__ == "__main__":
    main()
