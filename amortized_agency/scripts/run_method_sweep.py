#!/usr/bin/env python3
"""Sweep test-time parameters to see which direction each method trends.

Primary amortized target is the **reference regime** (E13: MI reliable at W>=250).
Use `run_reference_benchmark.py` for the canonical gap-to-MI table on all kinds.

This script is a **secondary** diagnostic: train once, vary one test parameter on
held-out complex agents. Modes:

  reference  — all kinds x W in {250, 500} only (gap-to-MI focus)
  trends     — window / noise / num_agents axes on complex8 (default)
  both       — reference grid then trend axes

Slot attention is excluded: E13d confirmed it sits at chance for this readout.
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
import torch


from amortized_agency.cluster import labels_from_affinity  # noqa: E402
from amortized_agency.context_model import ContextTrainConfig, ContextualAffinityModel, train_context  # noqa: E402
from amortized_agency.benchmark import (  # noqa: E402
    EVAL_T_STEPS,
    LEARNED_METHODS,
    REFERENCE_WINDOWS,
    reference_row_metrics,
)
from amortized_agency.kinds import ALL_KINDS, TRAIN_KINDS, Kind  # noqa: E402
from amortized_agency.metrics import score_clustering  # noqa: E402
from amortized_agency.siamese import SiameseAffinityModel, train_siamese  # noqa: E402
from amortized_agency.worlds import generate_pool, simulate_episode  # noqa: E402



def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_point(
    kind: Kind,
    window: int,
    overrides: Dict[str, float] | None,
    seeds: Sequence[int],
    siamese: SiameseAffinityModel,
    context: ContextualAffinityModel,
    device: torch.device,
    *,
    run_mi: bool,
) -> Dict[str, Dict[str, float]]:
    num_clusters = int(overrides.get("num_agents", kind.num_agents)) if overrides else kind.num_agents
    methods = (["mi"] if run_mi else []) + list(LEARNED_METHODS)
    per_method: Dict[str, List[float]] = {m: [] for m in methods}
    for seed in seeds:
        ep = simulate_episode(kind, window, seed, t_steps=EVAL_T_STEPS, overrides=overrides)
        labels = {
            "siamese": labels_from_affinity(siamese.affinity_matrix(ep.window, device), num_clusters),
            "context": labels_from_affinity(context.affinity_matrix(ep.window, device), num_clusters),
        }
        if run_mi:
            from amortized_agency.cluster import mi_affinity_labels

            labels["mi"] = mi_affinity_labels(ep.window, num_clusters)
        for m in methods:
            per_method[m].append(score_clustering(labels[m], ep.agent_ids)["ari"])
    return {m: {"ari_mean": float(np.mean(v)), "ari_std": float(np.std(v))} for m, v in per_method.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-worlds", type=int, default=40)
    parser.add_argument("--context-epochs", type=int, default=40)
    parser.add_argument("--siamese-epochs", type=int, default=25)
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["reference", "trends", "both"],
        default="trends",
        help="reference=MI-ceiling grid; trends=parameter axes; both=both",
    )
    parser.add_argument(
        "--run-mi",
        action="store_true",
        help="Include per-trace MI (slow). Default: learned-only; use frozen ref for gaps.",
    )
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "amortized"))
    args = parser.parse_args()

    if args.fast:
        args.train_worlds = 20
        args.context_epochs = 15
        args.siamese_epochs = 12
        args.eval_seeds = 3

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.eval_seeds))

    complex8 = Kind("hard8_complex", num_agents=8, variant_mode="complex", decoy_vars=8)

    ref_windows = [250, 500] if args.fast else REFERENCE_WINDOWS
    reference_points = [(kind, w, None) for kind in ALL_KINDS for w in ref_windows]

    if args.fast:
        window_vals = [60, 125, 250]
        noise_vals = [0.04, 0.16]
        agent_vals = [3, 8]
    else:
        window_vals = [30, 45, 60, 90, 125, 175, 250, 400]
        noise_vals = [0.02, 0.04, 0.08, 0.16, 0.32]
        agent_vals = [3, 5, 8, 12]

    trend_axes = {
        "window": [(complex8, w, None) for w in window_vals],
        "obs_noise": [
            (complex8, 125, {"observation_noise": nv, "process_noise": nv}) for nv in noise_vals
        ],
        "num_agents": [
            (Kind(f"complex{n}", num_agents=n, variant_mode="complex", decoy_vars=8), 125, None)
            for n in agent_vals
        ],
    }
    trend_axis_x = {"window": window_vals, "obs_noise": noise_vals, "num_agents": agent_vals}

    print(f"Device: {device}  train_worlds={args.train_worlds}  seeds={seeds}")
    train_episodes = generate_pool(TRAIN_KINDS, args.train_worlds, 1000, window_choices=[500, 1000])
    print(f"Training pool: {len(train_episodes)} episodes (kinds={[k.name for k in TRAIN_KINDS]})")

    siamese = SiameseAffinityModel(window=1000).to(device)
    train_siamese(siamese, train_episodes, epochs=args.siamese_epochs,
                  pairs_per_episode=96, lr=args.lr, device=device)
    siamese.eval()

    context = ContextualAffinityModel(window=1000).to(device)
    train_context(context, train_episodes, cfg=ContextTrainConfig(epochs=args.context_epochs, lr=args.lr), device=device)
    context.eval()

    rows: List[Dict] = []
    axis_x: Dict[str, List] = {}

    if args.mode in ("reference", "both"):
        print("\n=== reference regime (MI ceiling, all kinds) ===")
        for kind, window, overrides in reference_points:
            res = eval_point(kind, window, overrides, seeds, siamese, context, device, run_mi=args.run_mi)
            row = {
                "axis": "reference",
                "x": window,
                "kind": kind.name,
                "held_out": kind.name == "hard8_complex",
                "window": window,
            }
            for m in res:
                row[f"{m}_ari"] = res[m]["ari_mean"]
                row[f"{m}_std"] = res[m]["ari_std"]
            row.update(reference_row_metrics(row, kind=kind.name, window=window))
            rows.append(row)
            gaps = "  ".join(f"gap_{m}={row.get(f'gap_{m}', float('nan')):.3f}" for m in LEARNED_METHODS)
            parts = [f"{m}={row[f'{m}_ari']:.3f}" for m in res]
            print(f"  {kind.name:18} W={window:4}  " + "  ".join(parts) + f"  {gaps}")
        axis_x["reference"] = ref_windows

    if args.mode in ("trends", "both"):
        for axis_name, points in trend_axes.items():
            print(f"\n=== axis: {axis_name} ===")
            for (kind, window, overrides), xval in zip(points, trend_axis_x[axis_name]):
                res = eval_point(kind, window, overrides, seeds, siamese, context, device, run_mi=args.run_mi)
                row = {"axis": axis_name, "x": xval, "window": window, "kind": kind.name}
                for m in res:
                    row[f"{m}_ari"] = res[m]["ari_mean"]
                    row[f"{m}_std"] = res[m]["ari_std"]
                rows.append(row)
                print(
                    f"  {axis_name}={xval!s:>6}  "
                    + "  ".join(f"{m}={res[m]['ari_mean']:.3f}" for m in res)
                )
        axis_x.update(trend_axis_x)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "mode": args.mode,
        "reference_windows": REFERENCE_WINDOWS,
        "train_kinds": [k.name for k in TRAIN_KINDS],
        "train_worlds_per_kind": args.train_worlds,
        "eval_seeds": seeds,
        "run_mi": args.run_mi,
        "learned_methods": list(LEARNED_METHODS),
        "axis_x": axis_x,
        "rows": rows,
    }
    out_json = out_dir / "method_sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    if args.mode in ("trends", "both"):
        maybe_plot(summary, out_dir / "method_sweep.png")


def maybe_plot(summary: Dict, out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    axes_names = list(summary["axis_x"].keys())
    fig, axs = plt.subplots(1, len(axes_names), figsize=(5 * len(axes_names), 4.2))
    if len(axes_names) == 1:
        axs = [axs]
    for ax, axis_name in zip(axs, axes_names):
        xs = summary["axis_x"][axis_name]
        rows = [r for r in summary["rows"] if r["axis"] == axis_name]
        rows = sorted(rows, key=lambda r: xs.index(r["x"]))
        plot_methods = (["mi"] if summary.get("run_mi") else []) + list(LEARNED_METHODS)
        for m in plot_methods:
            ax.errorbar(
                [r["x"] for r in rows],
                [r[f"{m}_ari"] for r in rows],
                yerr=[r[f"{m}_std"] for r in rows],
                marker="o", capsize=3, label=m,
            )
        ax.set_title(axis_name)
        ax.set_xlabel(axis_name)
        ax.set_ylabel("ARI")
        ax.grid(True, alpha=0.25)
        if axis_name in {"window", "obs_noise"}:
            ax.set_xscale("log")
        ax.legend()
    fig.suptitle("Method trends across test-time parameters (complex agents)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
