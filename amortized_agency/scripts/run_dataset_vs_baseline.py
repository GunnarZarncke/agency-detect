#!/usr/bin/env python3
"""Benchmark E15 external datasets against the E13 sim baseline (amortized context).

Default: train on sim ``TRAIN_KINDS`` (easy3 + med5). With ``--extended-pool``,
train on ``EXTENDED_TRAIN_KINDS`` (+ physics, rock, grid3). Then evaluate:
  - Sim reference: ``ALL_KINDS`` @ W=250 (MI ceiling + transfer)
  - External datasets: physics, rock, grid3, grid5 (zero-shot transfer)

Skips Melting Pot unless ``--include-melting-pot`` and deps are installed.
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
from typing import Dict, List, Sequence

import numpy as np
import torch


from amortized_agency.benchmark import (  # noqa: E402
    EVAL_T_STEPS,
    LEARNED_METHODS,
    REFERENCE_WINDOWS,
    mi_reference_ari,
    reference_row_metrics,
)
from amortized_agency.cluster import labels_from_affinity, mi_affinity_labels  # noqa: E402
from amortized_agency.context_model import ContextTrainConfig, ContextualAffinityModel, train_context  # noqa: E402
from amortized_agency.kinds import (  # noqa: E402
    ALL_KINDS,
    EXTENDED_TRAIN_KINDS,
    EXTERNAL_KINDS,
    TRAIN_KINDS,
    Kind,
)
from amortized_agency.metrics import score_clustering  # noqa: E402
from amortized_agency.siamese import SiameseAffinityModel, train_siamese  # noqa: E402
from amortized_agency.worlds import generate_pool, simulate_episode  # noqa: E402

EVAL_WINDOW = 250
DATASET_KINDS = [k for k in EXTERNAL_KINDS if k.name != "melting_pot_cooking_ring"]


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_kind(
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
    mi_seconds: List[float] = []
    ctx_seconds: List[float] = []
    trace_lengths: List[int] = []

    for seed in seeds:
        t0 = time.perf_counter()
        ep = simulate_episode(kind, window, seed, t_steps=EVAL_T_STEPS)
        sim_s = time.perf_counter() - t0
        trace_lengths.append(ep.trace_T)

        labels = {}
        if run_mi:
            t1 = time.perf_counter()
            labels["mi"] = mi_affinity_labels(ep.window, kind.num_agents)
            mi_seconds.append(sim_s + (time.perf_counter() - t1))
        t2 = time.perf_counter()
        labels["siamese"] = labels_from_affinity(
            siamese.affinity_matrix(ep.window, device), kind.num_agents
        )
        labels["context"] = labels_from_affinity(
            context.affinity_matrix(ep.window, device), kind.num_agents
        )
        ctx_seconds.append(sim_s + (time.perf_counter() - t2))

        for m in methods:
            per_method[m].append(score_clustering(labels[m], ep.agent_ids)["ari"])

    row: Dict[str, float] = {
        "trace_T_median": float(np.median(trace_lengths)),
    }
    for m in methods:
        row[f"{m}_ari"] = float(np.mean(per_method[m]))
        row[f"{m}_std"] = float(np.std(per_method[m]))
    if run_mi:
        row["mi_sec_mean"] = float(np.mean(mi_seconds))
    row["context_infer_sec_mean"] = float(np.mean(ctx_seconds))
    row.update(reference_row_metrics(row, kind=kind.name, window=window))
    return row


def markdown_table(rows: List[Dict]) -> str:
    lines = [
        "| Kind | group | n | T_med | MI ARI | Context ARI | gap_ctx | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        mi = r.get("mi_ari")
        ctx = r.get("context_ari")
        gap = r.get("gap_context")
        mi_s = f"{mi:.3f}" if mi is not None else (f"{r.get('mi_reference_ari', 0):.3f}*" if "mi_reference_ari" in r else "—")
        ctx_s = f"{ctx:.3f}" if ctx is not None else "—"
        gap_s = f"{gap:.3f}" if gap is not None else "—"
        note = "external zero-shot" if r["group"] == "dataset" else ("held-out sim" if r.get("held_out") else "train sim")
        if r.get("mi_reference_ari") and "mi_ari" not in r:
            note += " (*frozen MI)"
        lines.append(
            f"| {r['kind']} | {r['group']} | {r['num_agents']} | {int(r.get('trace_T_median', 0))} | "
            f"{mi_s} | {ctx_s} | {gap_s} | {note} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--train-worlds", type=int, default=40)
    p.add_argument("--context-epochs", type=int, default=40)
    p.add_argument("--siamese-epochs", type=int, default=25)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--run-mi", action="store_true", help="Live MI on all kinds (slow)")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--include-melting-pot", action="store_true")
    p.add_argument(
        "--extended-pool",
        action="store_true",
        help="Train on EXTENDED_TRAIN_KINDS (sim + physics + rock + grid3)",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/amortized/dataset_vs_baseline.json",
    )
    args = p.parse_args()

    if args.fast:
        args.train_worlds = 20
        args.context_epochs = 15
        args.siamese_epochs = 12

    device = pick_device(args.device)
    seeds = list(range(args.eval_seeds))
    eval_kinds = list(ALL_KINDS)
    datasets = list(DATASET_KINDS)
    if args.include_melting_pot:
        mp = next(k for k in EXTERNAL_KINDS if k.name == "melting_pot_cooking_ring")
        datasets.append(mp)

    train_kinds = EXTENDED_TRAIN_KINDS if args.extended_pool else TRAIN_KINDS
    print(f"Train: {[k.name for k in train_kinds]}")
    print(f"Eval: sim {len(ALL_KINDS)} + datasets {len(datasets)} @ W={EVAL_WINDOW}")

    train_eps = generate_pool(train_kinds, args.train_worlds, 1000, window_choices=[500, 1000])
    print(f"Training pool: {len(train_eps)} episodes")

    siamese = SiameseAffinityModel(window=1000).to(device)
    train_siamese(
        siamese,
        train_eps,
        epochs=args.siamese_epochs,
        pairs_per_episode=96,
        lr=3e-4,
        device=device,
    )
    siamese.eval()

    context = ContextualAffinityModel(window=1000).to(device)
    train_context(
        context,
        train_eps,
        cfg=ContextTrainConfig(epochs=args.context_epochs, lr=3e-4),
        device=device,
    )
    context.eval()

    rows: List[Dict] = []
    for kind in eval_kinds:
        row = eval_kind(kind, EVAL_WINDOW, seeds, siamese, context, device, run_mi=args.run_mi)
        row.update(
            {
                "kind": kind.name,
                "group": "sim_baseline",
                "num_agents": kind.num_agents,
                "held_out": kind.name == "hard8_complex",
                "backend": kind.backend,
            }
        )
        if not args.run_mi:
            ref = mi_reference_ari(kind.name, EVAL_WINDOW)
            if ref is not None:
                row["mi_reference_ari"] = ref
                if "context_ari" in row:
                    row["gap_context"] = float(ref - row["context_ari"])
        rows.append(row)
        print(
            f"  {kind.name:20} ctx={row.get('context_ari', 0):.3f}  "
            f"mi={'live' if args.run_mi else row.get('mi_reference_ari', '—')}"
        )

    for kind in datasets:
        try:
            row = eval_kind(kind, EVAL_WINDOW, seeds, siamese, context, device, run_mi=True)
        except Exception as e:
            print(f"  SKIP {kind.name}: {e}")
            continue
        row.update(
            {
                "kind": kind.name,
                "group": "dataset",
                "num_agents": kind.num_agents,
                "held_out": kind.name in ("grid_pomdp_5x5", "melting_pot_cooking_ring"),
                "backend": kind.backend,
            }
        )
        rows.append(row)
        gap = row.get("gap_context", row.get("mi_ari", 0) - row.get("context_ari", 0))
        print(
            f"  {kind.name:20} MI={row.get('mi_ari', 0):.3f}  ctx={row.get('context_ari', 0):.3f}  "
            f"T_med={int(row.get('trace_T_median', 0))}"
        )

    md = markdown_table(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Train EXTENDED_TRAIN_KINDS"
            if args.extended_pool
            else "Train sim TRAIN_KINDS"
        )
        + "; eval sim ALL_KINDS + external at W=250",
        "extended_pool": args.extended_pool,
        "train_kinds": [k.name for k in train_kinds],
        "eval_window": EVAL_WINDOW,
        "eval_seeds": seeds,
        "run_mi": args.run_mi,
        "markdown_table": md,
        "rows": rows,
    }

    from learn_agents.safe_results import write_json

    out = args.out
    if args.extended_pool and out == REPO_ROOT / "results/amortized/dataset_vs_baseline.json":
        out = REPO_ROOT / "results/amortized/dataset_vs_baseline_extended.json"
    written = write_json(payload, out, force=args.force)
    print(f"\nWrote {written}\n{md}")


if __name__ == "__main__":
    main()
