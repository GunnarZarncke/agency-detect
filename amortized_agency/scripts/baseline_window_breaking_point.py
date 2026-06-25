#!/usr/bin/env python3
"""MI-affinity short-window breaking-point baseline.

Measures where the repo's existing per-trace MI proposal step
(`mi_cluster_variable_labels`: discretize -> lagged MI -> agglomerative)
loses the ability to separate agents as the observation window W shrinks.

This is the baseline the amortized (learned, pooled-across-worlds) detector
must beat at short W. No learning happens here; it characterizes the current
method's statistical-power limit per agent kind.

Metric is restricted to agent variables (var_agent >= 0) so it isolates the
duration/sample-count effect from decoy/world rejection (a separate axis).
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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Kind:
    """An agent 'kind' for the held-out-of-kinds spectrum."""

    name: str
    num_agents: int
    variant_mode: str
    copies_per_role: int = 3
    interaction_strength: float = 0.05
    decoy_vars: int = 8


KINDS: List[Kind] = [
    Kind("easy3_redundant", num_agents=3, variant_mode="redundant", decoy_vars=6),
    Kind("med5_rich", num_agents=5, variant_mode="rich", decoy_vars=8),
    Kind("hard8_complex", num_agents=8, variant_mode="complex", decoy_vars=8),
]

WINDOWS: List[int] = [2000, 1000, 500, 250, 125, 60]


def _best_mean_jaccard(labels: np.ndarray, true_ids: np.ndarray) -> float:
    """Mean over predicted clusters of best Jaccard against any true agent."""
    pred_clusters = [np.where(labels == c)[0] for c in sorted(set(labels.tolist())) if c >= 0]
    true_clusters = [np.where(true_ids == a)[0] for a in sorted(set(true_ids.tolist()))]
    if not pred_clusters or not true_clusters:
        return 0.0
    jacc = []
    for pc in pred_clusters:
        pcs = set(pc.tolist())
        best = 0.0
        for tc in true_clusters:
            tcs = set(tc.tolist())
            inter = len(pcs & tcs)
            union = len(pcs | tcs)
            if union:
                best = max(best, inter / union)
        jacc.append(best)
    return float(np.mean(jacc))


def _simulate(kind: Kind, t_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = TraceSimulationConfig(
        T=t_steps,
        num_agents=kind.num_agents,
        copies_per_role=kind.copies_per_role,
        decoy_vars=kind.decoy_vars,
        interaction_strength=kind.interaction_strength,
        agent_variant_mode=kind.variant_mode,
        episodic=False,
        seed=seed,
    )
    result = simulate_known_agent_trace(cfg)
    var_agent = np.asarray(result.metadata["var_agent"], dtype=np.int64)
    return result.trace, var_agent


def evaluate_window(
    trace: np.ndarray,
    var_agent: np.ndarray,
    num_agents: int,
    window: int,
) -> Dict[str, float]:
    """Cluster agent variables from the first `window` steps; score vs truth."""
    agent_cols = np.where(var_agent >= 0)[0]
    sub = trace[:window, agent_cols]
    true_ids = var_agent[agent_cols]

    labels = mi_cluster_variable_labels(sub, num_clusters=num_agents)
    active = labels >= 0
    if active.sum() < 2:
        return {"ari": 0.0, "mean_jaccard": 0.0}

    ari = float(adjusted_rand_score(true_ids[active], labels[active]))
    mean_jacc = _best_mean_jaccard(labels[active], true_ids[active])
    return {"ari": ari, "mean_jaccard": mean_jacc}


def run(seeds: Sequence[int], windows: Sequence[int], kinds: Sequence[Kind]) -> Dict:
    t_max = max(windows)
    rows: List[Dict] = []
    for kind in kinds:
        # Simulate one long trace per seed, then slice each window from it.
        traces = [_simulate(kind, t_max, s) for s in seeds]
        for window in windows:
            aris, jaccs = [], []
            for trace, var_agent in traces:
                m = evaluate_window(trace, var_agent, kind.num_agents, window)
                aris.append(m["ari"])
                jaccs.append(m["mean_jaccard"])
            rows.append(
                {
                    "kind": kind.name,
                    "num_agents": kind.num_agents,
                    "variant_mode": kind.variant_mode,
                    "window": window,
                    "ari_mean": float(np.mean(aris)),
                    "ari_std": float(np.std(aris)),
                    "jaccard_mean": float(np.mean(jaccs)),
                    "jaccard_std": float(np.std(jaccs)),
                    "n_seeds": len(seeds),
                }
            )
            print(
                f"{kind.name:18s} W={window:5d}  "
                f"ARI={np.mean(aris):.3f}+-{np.std(aris):.3f}  "
                f"Jacc={np.mean(jaccs):.3f}+-{np.std(jaccs):.3f}"
            )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "windows": list(windows),
        "rows": rows,
    }


def maybe_plot(summary: Dict, out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    kinds = sorted({r["kind"] for r in summary["rows"]})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in kinds:
        rs = sorted((r for r in summary["rows"] if r["kind"] == kind), key=lambda r: r["window"])
        ws = [r["window"] for r in rs]
        ari = [r["ari_mean"] for r in rs]
        err = [r["ari_std"] for r in rs]
        ax.errorbar(ws, ari, yerr=err, marker="o", capsize=3, label=kind)
    ax.set_xscale("log")
    ax.set_xlabel("observation window W (timesteps)")
    ax.set_ylabel("agent-separation ARI")
    ax.set_title("MI-affinity baseline: agent recovery vs window length")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds per kind")
    parser.add_argument("--fast", action="store_true", help="3 seeds, fewer windows")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "amortized"),
    )
    args = parser.parse_args()

    seeds = list(range(3 if args.fast else args.seeds))
    windows = [1000, 500, 250, 125, 60] if args.fast else WINDOWS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run(seeds, windows, KINDS)

    out_json = out_dir / "baseline_window_breaking_point.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    out_png = out_dir / "baseline_window_breaking_point.png"
    if maybe_plot(summary, out_png):
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
