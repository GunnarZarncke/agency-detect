#!/usr/bin/env python3
"""Sweep learned model variants (no per-trace MI).

Routine experiments should not pay the O(N²) MI cost (~5s+ per trace at N≈72).
Gap-to-MI uses frozen E13 reference ARIs (hard8 W=250 → 0.964).

Grid over context model scale, training pool size, and epochs; evaluates on the
reference regime (all kinds, W in {250, 500}).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from amortized_agency.benchmark import (  # noqa: E402
    EVAL_T_STEPS,
    LEARNED_METHODS,
    REFERENCE_WINDOWS,
    reference_row_metrics,
)
from amortized_agency.cluster import labels_from_affinity  # noqa: E402
from amortized_agency.context_model import ContextTrainConfig, train_context  # noqa: E402
from amortized_agency.kinds import ALL_KINDS, TRAIN_KINDS, Kind  # noqa: E402
from amortized_agency.metrics import score_clustering  # noqa: E402
from amortized_agency.model_presets import CONTEXT_SCALES, build_context_model  # noqa: E402
from amortized_agency.worlds import generate_pool, simulate_episode  # noqa: E402


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_context(
    model,
    kind: Kind,
    window: int,
    seeds: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    aris: List[float] = []
    infer_s: List[float] = []
    for seed in seeds:
        ep = simulate_episode(kind, window, seed, t_steps=EVAL_T_STEPS)
        t0 = time.perf_counter()
        labels = labels_from_affinity(
            model.affinity_matrix(ep.window, device), kind.num_agents
        )
        infer_s.append(time.perf_counter() - t0)
        aris.append(score_clustering(labels, ep.agent_ids)["ari"])
    row = {
        "context_ari": float(sum(aris) / len(aris)),
        "infer_seconds_median": float(sorted(infer_s)[len(infer_s) // 2]),
    }
    row.update(reference_row_metrics(row, kind=kind.name, window=window))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--scales",
        type=str,
        default="base,large",
        help="Comma-separated context scales: base, large, xl",
    )
    parser.add_argument(
        "--train-worlds",
        type=str,
        default="40",
        help="Comma-separated worlds per train kind",
    )
    parser.add_argument(
        "--context-epochs",
        type=str,
        default="40",
        help="Comma-separated epoch counts",
    )
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "amortized"))
    args = parser.parse_args()

    if args.fast:
        args.scales = "base"
        args.train_worlds = "20"
        args.context_epochs = "15"
        args.eval_seeds = 3

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.eval_seeds))

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    worlds_list = [int(x) for x in args.train_worlds.split(",")]
    epochs_list = [int(x) for x in args.context_epochs.split(",")]

    sweep_rows: List[Dict] = []
    grid = list(itertools.product(scales, worlds_list, epochs_list))
    print(f"Device: {device}  grid size: {len(grid)}  (no MI — frozen reference gaps)")

    for scale, n_worlds, epochs in grid:
        tag = f"scale={scale}_worlds={n_worlds}_ep={epochs}"
        print(f"\n=== {tag} ===")
        t0 = time.perf_counter()
        train_episodes = generate_pool(
            TRAIN_KINDS, n_worlds, 1000, window_choices=[500, 1000]
        )
        pool_s = time.perf_counter() - t0
        model = build_context_model(1000, scale=scale).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        t1 = time.perf_counter()
        train_context(
            model, train_episodes,
            cfg=ContextTrainConfig(epochs=epochs, lr=args.lr),
            device=device,
        )
        model.eval()
        train_s = time.perf_counter() - t1

        eval_rows: List[Dict] = []
        t_eval = time.perf_counter()
        for kind in ALL_KINDS:
            for w in REFERENCE_WINDOWS:
                m = eval_context(model, kind, w, seeds, device)
                row = {
                    "scale": scale,
                    "train_worlds": n_worlds,
                    "context_epochs": epochs,
                    "n_params": n_params,
                    "kind": kind.name,
                    "held_out": kind.name == "hard8_complex",
                    "window": w,
                    **m,
                }
                eval_rows.append(row)
                gap = m.get("gap_context", float("nan"))
                infer_ms = m.get("infer_seconds_median", 0.0) * 1e3
                print(
                    f"  {kind.name:18} W={w:4}  context={m['context_ari']:.3f}  "
                    f"gap_to_mi_ref={gap:.3f}  infer={infer_ms:.1f}ms"
                )

        eval_s = time.perf_counter() - t_eval
        held = [r for r in eval_rows if r["held_out"] and r["window"] == 250]
        mean_gap = sum(r["gap_context"] for r in held) / max(len(held), 1)
        held_infer_ms = (
            held[0].get("infer_seconds_median", 0.0) * 1e3 if held else None
        )
        sweep_rows.append({
            "tag": tag,
            "scale": scale,
            "train_worlds": n_worlds,
            "context_epochs": epochs,
            "n_params": n_params,
            "pool_seconds": pool_s,
            "train_seconds": train_s,
            "eval_seconds": eval_s,
            "held_out_w250_context_ari": held[0]["context_ari"] if held else None,
            "held_out_w250_gap_context": mean_gap,
            "held_out_w250_infer_ms_median": held_infer_ms,
            "eval_rows": eval_rows,
        })
        print(
            f"  pool {pool_s:.1f}s  train {train_s:.1f}s  eval {eval_s:.1f}s  "
            f"held-out W=250 gap={mean_gap:.3f} infer={held_infer_ms:.1f}ms"
        )

    best = min(sweep_rows, key=lambda r: r["held_out_w250_gap_context"])
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "skip_mi": True,
        "mi_reference": "frozen E13 (hard8_complex W=250 → 0.964)",
        "scales_available": list(CONTEXT_SCALES.keys()),
        "eval_seeds": seeds,
        "best_by_held_out_gap": {
            "tag": best["tag"],
            "gap_context": best["held_out_w250_gap_context"],
            "context_ari": best["held_out_w250_context_ari"],
        },
        "sweep_rows": sweep_rows,
    }
    out_json = out_dir / "learned_sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nBest held-out W=250: {best['tag']}  gap={best['held_out_w250_gap_context']:.3f}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
