#!/usr/bin/env python3
"""Train pooled Siamese + slot-affinity models and evaluate vs MI on held-out kinds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from amortized_agency.evaluate import evaluate_kind  # noqa: E402
from amortized_agency.kinds import (  # noqa: E402
    ALL_KINDS,
    EVAL_WINDOWS,
    HELDOUT_KINDS,
    TRAIN_KINDS,
)
from amortized_agency.context_model import ContextTrainConfig, ContextualAffinityModel, train_context  # noqa: E402
from amortized_agency.siamese import SiameseAffinityModel, train_siamese  # noqa: E402
from amortized_agency.slot_model import SlotAttentionAffinity, SlotTrainConfig, train_slot  # noqa: E402
from amortized_agency.worlds import generate_pool  # noqa: E402


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_rows(rows: Sequence[Dict], heldout_only: bool = False) -> None:
    for r in rows:
        if heldout_only and r["kind"] not in {k.name for k in HELDOUT_KINDS}:
            continue
        print(
            f"{r['kind']:18s} W={r['window']:5d} {r['method']:8s}  "
            f"ARI={r['ari_mean']:.3f}+-{r['ari_std']:.3f}  "
            f"Jacc={r['jaccard_mean']:.3f}+-{r['jaccard_std']:.3f}"
        )


def maybe_plot(summary: Dict, out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    heldout = HELDOUT_KINDS[0].name
    methods = ["mi", "siamese", "slot", "context"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in methods:
        rs = sorted(
            (r for r in summary["rows"] if r["kind"] == heldout and r["method"] == method),
            key=lambda r: r["window"],
        )
        if not rs:
            continue
        ws = [r["window"] for r in rs]
        ari = [r["ari_mean"] for r in rs]
        err = [r["ari_std"] for r in rs]
        ax.errorbar(ws, ari, yerr=err, marker="o", capsize=3, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("observation window W")
    ax.set_ylabel("agent-separation ARI")
    ax.set_title(f"Held-out kind ({heldout}): MI vs pooled models")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-windows",
        type=str,
        default="500,1000",
        help="comma-separated window lengths sampled during training (train long, detect short)",
    )
    parser.add_argument("--train-worlds", type=int, default=40, help="worlds per train kind")
    parser.add_argument("--siamese-epochs", type=int, default=20)
    parser.add_argument("--slot-epochs", type=int, default=25)
    parser.add_argument("--slot-temp", type=float, default=0.2, help="supervised-contrastive temperature")
    parser.add_argument("--slot-no-sample", action="store_true", help="disable shared-Gaussian sampled slots (#5)")
    parser.add_argument("--context-epochs", type=int, default=40)
    parser.add_argument("--pairs-per-episode", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--run-mi",
        action="store_true",
        help="Evaluate per-trace MI (slow). Default: learned methods only.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "amortized"),
    )
    args = parser.parse_args()

    if args.fast:
        args.train_worlds = 20
        args.siamese_epochs = 8
        args.slot_epochs = 6
        args.pairs_per_episode = 32
        args.eval_seeds = 3
        windows = [250, 125, 60]
    else:
        windows = EVAL_WINDOWS

    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_window_choices = [int(x) for x in args.train_windows.split(",")]
    max_train_window = max(train_window_choices)

    print(f"Device: {device}")
    print(f"Train kinds: {[k.name for k in TRAIN_KINDS]}")
    print(f"Held-out kinds: {[k.name for k in HELDOUT_KINDS]}")
    print(f"Training pool: {args.train_worlds} worlds/kind, W in {train_window_choices}")

    train_episodes = generate_pool(
        TRAIN_KINDS,
        args.train_worlds,
        max_train_window,
        window_choices=train_window_choices,
    )
    print(f"Generated {len(train_episodes)} training episodes")

    siamese = SiameseAffinityModel(window=max_train_window).to(device)
    siamese_losses = train_siamese(
        siamese,
        train_episodes,
        epochs=args.siamese_epochs,
        pairs_per_episode=args.pairs_per_episode,
        lr=args.lr,
        device=device,
    )

    slot_cfg = SlotTrainConfig(
        epochs=args.slot_epochs,
        lr=args.lr,
        contrast_temp=args.slot_temp,
    )
    slot = SlotAttentionAffinity(
        window=max_train_window,
        num_slots=16,
        sample_slots=not args.slot_no_sample,
    ).to(device)
    slot_losses = train_slot(slot, train_episodes, cfg=slot_cfg, device=device)

    context = ContextualAffinityModel(window=max_train_window).to(device)
    context_losses = train_context(
        context,
        train_episodes,
        cfg=ContextTrainConfig(epochs=args.context_epochs, lr=args.lr),
        device=device,
    )

    seeds = list(range(args.eval_seeds))
    methods = ["mi", "siamese", "slot", "context"] if args.run_mi else ["siamese", "slot", "context"]
    rows: List[Dict] = []
    for kind in ALL_KINDS:
        kind_rows = evaluate_kind(
            kind,
            windows,
            seeds,
            methods,
            siamese=siamese,
            slot=slot,
            context=context,
            device=device,
            t_steps=None,
        )
        rows.extend(kind_rows)

    print("\n=== Evaluation (all kinds) ===")
    print_rows(rows)

    print("\n=== Held-out kind only ===")
    print_rows(rows, heldout_only=True)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "train_kinds": [k.name for k in TRAIN_KINDS],
        "heldout_kinds": [k.name for k in HELDOUT_KINDS],
        "train_windows": train_window_choices,
        "train_worlds_per_kind": args.train_worlds,
        "siamese_epochs": args.siamese_epochs,
        "slot_epochs": args.slot_epochs,
        "eval_windows": windows,
        "eval_seeds": seeds,
        "siamese_final_loss": siamese_losses[-1] if siamese_losses else None,
        "slot_final_loss": slot_losses[-1] if slot_losses else None,
        "context_final_loss": context_losses[-1] if context_losses else None,
        "rows": rows,
    }

    out_json = out_dir / "pooled_experiment_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    out_png = out_dir / "pooled_heldout_comparison.png"
    if maybe_plot(summary, out_png):
        print(f"Wrote {out_png}")

    torch.save(siamese.state_dict(), out_dir / "siamese_model.pt")
    torch.save(slot.state_dict(), out_dir / "slot_model.pt")
    torch.save(context.state_dict(), out_dir / "context_model.pt")


if __name__ == "__main__":
    main()
