#!/usr/bin/env python3
"""Primary amortized benchmark: close the gap to MI in the reference regime.

E13 established MI as a trusted per-trace reference at W >= 250 (ARI ~1.0 on
train kinds, ~0.96 on held-out hard8_complex). This script trains learned models
once, evaluates all kinds at those windows, and reports gap_to_mi so sweeps can
target matching MI where it is reliable — not only beating MI in the short-W band
where MI is already weak.

Secondary short-window eval is optional (--also-breaking).
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


from amortized_agency.benchmark import (  # noqa: E402
    BREAKING_WINDOWS,
    EVAL_T_STEPS,
    LEARNED_METHODS,
    REFERENCE_WINDOWS,
    reference_row_metrics,
)
from amortized_agency.cluster import labels_from_affinity  # noqa: E402
from amortized_agency.context_model import ContextTrainConfig, ContextualAffinityModel, train_context  # noqa: E402
from amortized_agency.kinds import (  # noqa: E402
    ALL_KINDS,
    EXTENDED_ALL_KINDS,
    EXTENDED_TRAIN_KINDS,
    TRAIN_KINDS,
    Kind,
)
from amortized_agency.metrics import score_clustering  # noqa: E402
from amortized_agency.siamese import SiameseAffinityModel, train_siamese  # noqa: E402
from amortized_agency.worlds import generate_pool, simulate_episode  # noqa: E402

ALL_METHODS = ["mi", *LEARNED_METHODS]


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_cell(
    kind: Kind,
    window: int,
    seeds: Sequence[int],
    siamese: SiameseAffinityModel,
    context: ContextualAffinityModel,
    device: torch.device,
    *,
    run_mi: bool,
) -> Dict[str, float]:
    methods = list(LEARNED_METHODS)
    if run_mi:
        methods = ["mi", *methods]
    per_method: Dict[str, List[float]] = {m: [] for m in methods}
    for seed in seeds:
        ep = simulate_episode(kind, window, seed, t_steps=EVAL_T_STEPS)
        labels = {
            "siamese": labels_from_affinity(
                siamese.affinity_matrix(ep.window, device), kind.num_agents
            ),
            "context": labels_from_affinity(
                context.affinity_matrix(ep.window, device), kind.num_agents
            ),
        }
        if run_mi:
            from amortized_agency.cluster import mi_affinity_labels

            labels["mi"] = mi_affinity_labels(ep.window, kind.num_agents)
        for m in methods:
            per_method[m].append(score_clustering(labels[m], ep.agent_ids)["ari"])
    row: Dict[str, float] = {}
    for m in methods:
        row[f"{m}_ari"] = float(np.mean(per_method[m]))
        row[f"{m}_std"] = float(np.std(per_method[m]))
    row.update(reference_row_metrics(row, kind=kind.name, window=window))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-worlds", type=int, default=40)
    parser.add_argument("--context-epochs", type=int, default=40)
    parser.add_argument("--siamese-epochs", type=int, default=25)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--also-breaking",
        action="store_true",
        help="Also eval at W in {125, 60} (secondary short-window band)",
    )
    parser.add_argument(
        "--run-mi",
        action="store_true",
        help="Run per-trace MI (slow). Default: frozen E13 reference for gaps only.",
    )
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "amortized"))
    parser.add_argument(
        "--extended-pool",
        action="store_true",
        help="Train/eval on E16 pool (sim + E15 externals + melting_pot). Not run by default.",
    )
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

    eval_windows = list(REFERENCE_WINDOWS)
    if args.also_breaking:
        for w in BREAKING_WINDOWS:
            if w not in eval_windows:
                eval_windows.append(w)

    print(f"Device: {device}  reference_windows={REFERENCE_WINDOWS}")
    print(f"Eval windows: {eval_windows}  seeds={seeds}")
    train_kinds = EXTENDED_TRAIN_KINDS if args.extended_pool else TRAIN_KINDS
    eval_kinds = EXTENDED_ALL_KINDS if args.extended_pool else ALL_KINDS
    train_episodes = generate_pool(train_kinds, args.train_worlds, 1000, window_choices=[500, 1000])
    print(f"Training pool: {len(train_episodes)} episodes ({'extended' if args.extended_pool else 'default'})")

    siamese = SiameseAffinityModel(window=1000).to(device)
    train_siamese(
        siamese, train_episodes, epochs=args.siamese_epochs,
        pairs_per_episode=96, lr=args.lr, device=device,
    )
    siamese.eval()

    context = ContextualAffinityModel(window=1000).to(device)
    train_context(
        context, train_episodes,
        cfg=ContextTrainConfig(epochs=args.context_epochs, lr=args.lr),
        device=device,
    )
    context.eval()

    rows: List[Dict] = []
    if not args.run_mi:
        print("MI skipped — using frozen E13 reference ARIs for gap_to_mi")
    print("\n=== reference benchmark (gap_to_mi = MI_ref − learned) ===")
    for kind in eval_kinds:
        held = kind.name in ("hard8_complex", "grid_pomdp_5x5", "melting_pot_cooking_ring")
        for w in eval_windows:
            row = eval_cell(kind, w, seeds, siamese, context, device, run_mi=args.run_mi)
            row.update({
                "regime": "reference" if w in REFERENCE_WINDOWS else "breaking",
                "kind": kind.name,
                "held_out": held,
                "window": w,
                "num_agents": kind.num_agents,
            })
            rows.append(row)
            gaps = "  ".join(f"gap_{m}={row.get(f'gap_{m}', float('nan')):.3f}" for m in LEARNED_METHODS)
            mi_s = f"mi={row['mi_ari']:.3f}  " if "mi_ari" in row else f"mi_ref={row.get('mi_reference_ari', 0):.3f}  "
            print(
                f"  {kind.name:18} W={w:4}  {mi_s}"
                f"context={row['context_ari']:.3f}  siamese={row['siamese_ari']:.3f}  {gaps}"
            )

    ref_rows = [r for r in rows if r["regime"] == "reference"]
    held_ref = [r for r in ref_rows if r["held_out"]]
    if held_ref:
        best_ctx = min(held_ref, key=lambda r: r["gap_context"])
        print(
            f"\nHeld-out reference best context: {best_ctx['kind']} W={best_ctx['window']} "
            f"gap_context={best_ctx['gap_context']:.3f} (MI_ref={best_ctx.get('mi_reference_ari', best_ctx.get('mi_ari')):.3f}, "
            f"context={best_ctx['context_ari']:.3f})"
        )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "benchmark": "reference",
        "reference_windows": REFERENCE_WINDOWS,
        "mi_reference_note": "E13: MI ARI ~1.0 at W>=250 train kinds, ~0.96 held-out hard8",
        "extended_pool": args.extended_pool,
        "train_kinds": [k.name for k in train_kinds],
        "eval_kinds": [k.name for k in eval_kinds],
        "eval_windows": eval_windows,
        "eval_seeds": seeds,
        "run_mi": args.run_mi,
        "learned_methods": LEARNED_METHODS,
        "rows": rows,
    }
    out_json = out_dir / "reference_benchmark_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
