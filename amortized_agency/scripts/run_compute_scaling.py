#!/usr/bin/env python3
"""Measure inference compute scaling of each method (trace -> labels wall time).

Accuracy benchmarks should use the **reference regime** first (E13: MI reliable at
W>=250; see `run_reference_benchmark.py`). This script answers the complementary
question: given comparable inference, how does **compute** scale?

MI is the most accurate per-trace estimator but recomputes pairwise lagged MI on
every trace; amortized models pay training once then cheap forward passes. Times
the full trace->labels pipeline for MI / Siamese / context vs:

  * N  (variables, via agent count) at W=REFERENCE_WINDOW (250)
  * W  (window length) at the E13 reference agent count (8)

The reference anchor point (N~72, W=250) is where MI is both trusted and still
expensive enough to matter vs learned ~20ms inference.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch


from amortized_agency.cluster import labels_from_affinity, mi_affinity_labels  # noqa: E402
from amortized_agency.context_model import ContextualAffinityModel  # noqa: E402
from amortized_agency.benchmark import REFERENCE_WINDOWS  # noqa: E402
from amortized_agency.kinds import Kind  # noqa: E402

REFERENCE_WINDOW = REFERENCE_WINDOWS[0]  # 250: E13 MI ceiling start
REFERENCE_AGENTS = 8  # hard8_complex held-out kind
from amortized_agency.siamese import SiameseAffinityModel  # noqa: E402
from amortized_agency.worlds import simulate_episode  # noqa: E402

METHODS = ["mi", "siamese", "context"]


def median_time(fn: Callable[[], object], repeats: int) -> float:
    fn()  # warmup
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def time_point(window: np.ndarray, n_clusters: int, device: torch.device, repeats: int) -> Dict[str, float]:
    w_len, n = window.shape
    siamese = SiameseAffinityModel(window=w_len).to(device).eval()
    context = ContextualAffinityModel(window=w_len).to(device).eval()
    fns = {
        "mi": lambda: mi_affinity_labels(window, n_clusters),
        "siamese": lambda: labels_from_affinity(siamese.affinity_matrix(window, device), n_clusters),
        "context": lambda: labels_from_affinity(context.affinity_matrix(window, device), n_clusters),
    }
    return {m: median_time(fns[m], repeats) for m in METHODS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "amortized"))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        agent_vals = [3, 8, 16]
        window_vals = [125, 500]
    else:
        agent_vals = [3, 5, 8, 12, 16, 24]
        window_vals = [60, 125, 250, 500, 1000]

    rows: List[Dict] = []

    # Axis N: vary agent count at reference window (MI-trusted band).
    print(f"=== axis: num_vars (window={REFERENCE_WINDOW}, MI reference band) ===")
    for n_ag in agent_vals:
        kind = Kind(f"complex{n_ag}", num_agents=n_ag, variant_mode="complex", decoy_vars=8)
        ep = simulate_episode(kind, REFERENCE_WINDOW, seed=0, t_steps=500)
        times = time_point(ep.window, n_ag, device, args.repeats)
        n_vars = ep.window.shape[1]
        rows.append({
            "axis": "num_vars", "x": n_vars, "agents": n_ag,
            "window": REFERENCE_WINDOW, **{f"{m}_s": times[m] for m in METHODS},
        })
        print(f"  N={n_vars:>3} (agents={n_ag:>2})  " + "  ".join(f"{m}={times[m]*1e3:8.1f}ms" for m in METHODS))

    print(f"\n=== axis: window (agents={REFERENCE_AGENTS}) ===")
    for w in window_vals:
        kind = Kind("complex8", num_agents=REFERENCE_AGENTS, variant_mode="complex", decoy_vars=8)
        ep = simulate_episode(kind, w, seed=0, t_steps=max(w, 500))
        times = time_point(ep.window, REFERENCE_AGENTS, device, args.repeats)
        rows.append({"axis": "window", "x": w, "agents": 8, "window": w, "n_vars": ep.window.shape[1], **{f"{m}_s": times[m] for m in METHODS}})
        print(f"  W={w:>4}  " + "  ".join(f"{m}={times[m]*1e3:8.1f}ms" for m in METHODS))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "repeats": args.repeats,
        "methods": METHODS,
        "rows": rows,
        "reference_window": REFERENCE_WINDOW,
        "reference_agents": REFERENCE_AGENTS,
        "note": (
            "Accuracy: use run_reference_benchmark.py at W>=250 first. "
            "Times here are inference-only (random weights); training is one-time."
        ),
    }
    out_json = out_dir / "compute_scaling_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")
    maybe_plot(summary, out_dir / "compute_scaling.png")


def maybe_plot(summary: Dict, out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    axes_names = ["num_vars", "window"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, axis_name in zip(axs, axes_names):
        rows = sorted([r for r in summary["rows"] if r["axis"] == axis_name], key=lambda r: r["x"])
        for m in summary["methods"]:
            ax.plot([r["x"] for r in rows], [r[f"{m}_s"] * 1e3 for r in rows], marker="o", label=m)
        ax.set_title(axis_name)
        ax.set_xlabel("N variables" if axis_name == "num_vars" else "window length W")
        ax.set_ylabel("inference time per trace (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("Per-trace inference compute scaling (CPU)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
