#!/usr/bin/env python3
"""
Run a go/no-go sweep for learn_agents latent recovery.

This script targets the "clean but scaled" regime discussed in experiments:
- higher number of agents,
- low noise,
- no confounds/decoys/episodes.

It sweeps slots and epoch checkpoints, reports recovery metrics, and summarizes
the best epoch per (seed, slot) by variable-agent accuracy.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repo root is importable when run as a script from /scripts.

from learn_agents.learn_agents import (
    ModelConfig,
    TraceSimulationConfig,
    TrainConfig,
    encode_trace,
    score_against_ground_truth,
    simulate_known_agent_trace,
    train_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learn_agents go/no-go sweep.")

    parser.add_argument("--slots", nargs="+", type=int, default=[12, 16, 20, 24, 25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epoch-checkpoints", nargs="+", type=int, default=[25, 50, 75, 100])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--slot-dim", type=int, default=16)

    # Simulation defaults tuned to requested sweep scenario.
    parser.add_argument("--T", type=int, default=6000)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--copies-per-role", type=int, default=2)
    parser.add_argument("--decoy-vars", type=int, default=0)
    parser.add_argument("--process-noise", type=float, default=0.02)
    parser.add_argument("--observation-noise", type=float, default=0.01)
    parser.add_argument("--redundancy-noise", type=float, default=0.0)
    parser.add_argument("--interaction-strength", type=float, default=0.45)
    parser.add_argument("--confound-strength", type=float, default=0.0)
    parser.add_argument("--leakage-strength", type=float, default=0.0)
    parser.add_argument("--mixing-strength", type=float, default=0.0)
    parser.add_argument("--episodic", action="store_true", default=False)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)
    return parser.parse_args()


def _build_sim_config(args: argparse.Namespace, seed: int) -> TraceSimulationConfig:
    return TraceSimulationConfig(
        T=args.T,
        num_agents=args.num_agents,
        copies_per_role=args.copies_per_role,
        decoy_vars=args.decoy_vars,
        process_noise=args.process_noise,
        observation_noise=args.observation_noise,
        redundancy_noise=args.redundancy_noise,
        interaction_strength=args.interaction_strength,
        confound_strength=args.confound_strength,
        leakage_strength=args.leakage_strength,
        mixing_strength=args.mixing_strength,
        episodic=args.episodic,
        seed=seed,
    )


def _run_single(seed: int, slots: int, epochs: int, args: argparse.Namespace) -> Dict[str, Any]:
    sim_cfg = _build_sim_config(args, seed=seed)
    sim = simulate_known_agent_trace(sim_cfg)
    trace = sim.trace

    model_cfg = ModelConfig(
        num_vars=trace.shape[1],
        window=args.window,
        num_slots=slots,
        slot_dim=args.slot_dim,
    )
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        use_agency_regularizer=False,
        device=args.device,
    )

    model, history = train_model(trace, model_cfg, train_cfg)
    latent = encode_trace(model, trace)
    metrics = score_against_ground_truth(
        clusters=None,
        metadata=sim.metadata,
        assign=latent["assign"],
        learned_adjacency=latent["adjacency"],
    )
    edge = metrics.get("edge_recovery", {})

    return {
        "seed": seed,
        "slots": slots,
        "epochs": epochs,
        "final_loss": float(history["loss"][-1]),
        "var_acc": float(metrics["variable_agent_accuracy"]),
        "slot_purity_weighted": float(metrics["slot_purity_weighted"]),
        "slot_agent_matching_accuracy": float(metrics["slot_agent_matching_accuracy"]),
        "agent_concentration_mean": float(metrics["agent_concentration_mean"]),
        "edge_separation": float(edge.get("edge_separation", 0.0)),
    }


def _summarize_best_by_seed_slot(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row["seed"], row["slots"])
        current = best.get(key)
        if current is None or row["var_acc"] > current["var_acc"]:
            best[key] = row
    return list(best.values())


def _print_slot_summary(best_rows: List[Dict[str, Any]]) -> None:
    slots = sorted({r["slots"] for r in best_rows})
    print("\n=== Best-epoch summary by slot (aggregated over seeds) ===")
    for slot in slots:
        subset = [r for r in best_rows if r["slots"] == slot]
        var_acc = np.array([r["var_acc"] for r in subset], dtype=float)
        purity = np.array([r["slot_purity_weighted"] for r in subset], dtype=float)
        match = np.array([r["slot_agent_matching_accuracy"] for r in subset], dtype=float)
        best_epoch = np.array([r["epochs"] for r in subset], dtype=float)
        print(
            f"slots={slot:2d} | "
            f"var_acc={var_acc.mean():.3f}±{var_acc.std():.3f} | "
            f"purity={purity.mean():.3f} | "
            f"match={match.mean():.3f} | "
            f"best_epoch_mean={best_epoch.mean():.1f}"
        )


def _write_jsonl(path_str: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        meta = {
            "type": "metadata",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "sim_config_template": asdict(_build_sim_config(args, seed=-1)),
        }
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            payload = {"type": "result", **row}
            f.write(json.dumps(payload) + "\n")
    print(f"\nWrote JSONL results to {path}")


def main() -> None:
    args = _parse_args()
    epoch_checkpoints = sorted(set(args.epoch_checkpoints))
    slots_list = sorted(set(args.slots))
    seeds_list = sorted(set(args.seeds))

    chance = 1.0 / max(args.num_agents, 1)
    print("=== Learn Agents Go/No-Go Sweep ===")
    print(f"Chance var_acc baseline: {chance:.3f}")
    print(f"slots={slots_list}")
    print(f"seeds={seeds_list}")
    print(f"epoch_checkpoints={epoch_checkpoints}")

    all_rows: List[Dict[str, Any]] = []
    total = len(seeds_list) * len(slots_list) * len(epoch_checkpoints)
    idx = 0

    for seed in seeds_list:
        for slots in slots_list:
            for epochs in epoch_checkpoints:
                idx += 1
                row = _run_single(seed=seed, slots=slots, epochs=epochs, args=args)
                all_rows.append(row)
                print(
                    f"[{idx:03d}/{total:03d}] "
                    f"seed={seed} slots={slots} epochs={epochs} "
                    f"loss={row['final_loss']:.4f} "
                    f"var_acc={row['var_acc']:.3f} "
                    f"match={row['slot_agent_matching_accuracy']:.3f} "
                    f"purity={row['slot_purity_weighted']:.3f} "
                    f"edge_sep={row['edge_separation']:.4f}"
                )

    best_rows = _summarize_best_by_seed_slot(all_rows)
    _print_slot_summary(best_rows)

    top = sorted(best_rows, key=lambda r: r["var_acc"], reverse=True)[:10]
    print("\n=== Top best-epoch runs (by var_acc) ===")
    for r in top:
        print(r)

    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, all_rows, args)


if __name__ == "__main__":
    main()
