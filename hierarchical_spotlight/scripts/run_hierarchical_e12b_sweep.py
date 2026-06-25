#!/usr/bin/env python3
"""Run a six-case E12b sample over complex fixed-coordinate agents."""

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
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import run_spotlight_peel
from hierarchical_spotlight.config import HierarchicalConfig
from hierarchical_spotlight.fusion import run_hierarchical_fusion, write_graphviz


@dataclass(frozen=True)
class E12bRun:
    run_id: str
    proposal_mi_k: int
    max_passes: int
    interaction_strength: float
    mixing_strength: float


@dataclass(frozen=True)
class CaseArgs:
    run: E12bRun
    t_steps: int
    pretrain_epochs: int
    refine_epochs: int
    max_passes_cap: Optional[int]
    skip_existing: bool
    verbose: bool
    device: Optional[str]
    torch_threads: int


RUNS: List[E12bRun] = [
    E12bRun("complex_k24_p16_i002_m000", 24, 16, 0.02, 0.00),
    E12bRun("complex_k24_p16_i005_m000", 24, 16, 0.05, 0.00),
    E12bRun("complex_k24_p16_i010_m000", 24, 16, 0.10, 0.00),
    E12bRun("complex_k32_p16_i005_m000", 32, 16, 0.05, 0.00),
    E12bRun("complex_k24_p20_i005_m002", 24, 20, 0.05, 0.02),
    E12bRun("complex_k32_p20_i010_m002", 32, 20, 0.10, 0.02),
]
NUM_AGENTS = 8
MIN_PASSES = 16


def _spotlight_config(case: CaseArgs) -> SpotlightConfig:
    run = case.run
    return SpotlightConfig(
        seed=1,
        T=case.t_steps,
        num_agents=8,
        copies_per_role=3,
        decoy_vars=0,
        agent_variant_mode="complex",
        agent_variant_delay=2,
        process_noise=0.02,
        observation_noise=0.01,
        interaction_strength=run.interaction_strength,
        confound_strength=0.0,
        leakage_strength=0.01,
        mixing_strength=run.mixing_strength,
        local_env_strength=1.8,
        world_vars=12,
        world_to_sensor_strength=0.08,
        proposal_mi_k=run.proposal_mi_k,
        max_passes=_effective_max_passes(run.max_passes, case.max_passes_cap),
        pretrain_epochs=case.pretrain_epochs,
        refine_epochs=case.refine_epochs,
        candidate_mode="mi_cluster",
        device=case.device,
        verbose=case.verbose,
    )


def _effective_max_passes(run_max_passes: int, cap: Optional[int]) -> int:
    requested = min(run_max_passes, cap) if cap else run_max_passes
    if requested < MIN_PASSES:
        print(
            f"warning: raising max_passes from {requested} to {MIN_PASSES} "
            f"(need >= agents={NUM_AGENTS} and chunk passes for hierarchy test)",
            flush=True,
        )
        requested = MIN_PASSES
    return requested


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_timing(timing: Dict[str, float]) -> str:
    return (
        f"total={timing.get('total_sec', 0.0):.1f}s "
        f"spotlight={timing.get('spotlight_sec', 0.0):.1f}s "
        f"fusion={timing.get('fusion_sec', 0.0):.1f}s "
        f"graphviz={timing.get('graphviz_sec', 0.0):.1f}s"
    )


def _configure_torch_threads(torch_threads: int) -> None:
    if torch_threads <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except ImportError:
        pass


def run_case(case: CaseArgs) -> Dict[str, Any]:
    _configure_torch_threads(case.torch_threads)
    run = case.run
    case_started = time.perf_counter()
    timing: Dict[str, float] = {
        "spotlight_sec": 0.0,
        "fusion_sec": 0.0,
        "graphviz_sec": 0.0,
        "total_sec": 0.0,
    }
    spotlight_stages: Dict[str, float] = {}

    spotlight_path = Path("results/spotlight/e12b") / f"spotlight_{run.run_id}.json"
    hierarchy_path = Path("results/hierarchical/e12b") / f"hierarchical_{run.run_id}.json"
    dot_path = hierarchy_path.with_suffix(".dot")
    png_path = hierarchy_path.with_suffix(".png")

    if case.skip_existing and spotlight_path.exists():
        spotlight_report = json.loads(spotlight_path.read_text(encoding="utf-8"))
        spotlight_stages = spotlight_report.get("timing", {})
    else:
        spotlight_cfg = _spotlight_config(case)
        print(
            f"spotlight {run.run_id}: T={spotlight_cfg.T} passes={spotlight_cfg.max_passes} "
            f"pretrain={spotlight_cfg.pretrain_epochs} refine={spotlight_cfg.refine_epochs} "
            f"device={spotlight_cfg.device or 'auto'}",
            flush=True,
        )
        t0 = time.perf_counter()
        spotlight_report = run_spotlight_peel(spotlight_cfg)
        timing["spotlight_sec"] = time.perf_counter() - t0
        spotlight_stages = spotlight_report.get("timing", {})
        spotlight_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(spotlight_path, spotlight_report)
        print(
            f"wrote {spotlight_path} ({_format_timing({**timing, 'total_sec': timing['spotlight_sec']})})",
            flush=True,
        )
        if spotlight_stages:
            print(
                f"  spotlight stages {run.run_id}: "
                f"sim={spotlight_stages.get('simulate_sec', 0.0):.1f}s "
                f"proposal={spotlight_stages.get('proposal_sec', 0.0):.1f}s "
                f"pretrain={spotlight_stages.get('pretrain_sec', 0.0):.1f}s "
                f"refine={spotlight_stages.get('refine_sec', 0.0):.1f}s "
                f"uad={spotlight_stages.get('uad_sec', 0.0):.1f}s "
                f"diag={spotlight_stages.get('diagnostics_sec', 0.0):.1f}s",
                flush=True,
            )

    graph_paths: Dict[str, str] = {}
    if case.skip_existing and hierarchy_path.exists():
        hierarchy_report = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    else:
        hierarchy_cfg = HierarchicalConfig(
            input_json=str(spotlight_path),
            output_json=str(hierarchy_path),
            output_dot=str(dot_path),
            output_png=str(png_path),
            timestamp_outputs=False,
        )
        print(f"fusion {run.run_id}: reading {spotlight_path}", flush=True)
        t0 = time.perf_counter()
        hierarchy_report = run_hierarchical_fusion(hierarchy_cfg)
        timing["fusion_sec"] = time.perf_counter() - t0
        hierarchy_report["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(hierarchy_path, hierarchy_report)
        t0 = time.perf_counter()
        graph_paths = write_graphviz(hierarchy_report, str(dot_path), str(png_path))
        timing["graphviz_sec"] = time.perf_counter() - t0
        print(f"wrote {hierarchy_path}", flush=True)

    timing["total_sec"] = time.perf_counter() - case_started
    spotlight_summary = spotlight_report["summary"]
    hierarchy_summary = hierarchy_report["summary"]
    return {
        "run": asdict(run),
        "spotlight_json": str(spotlight_path),
        "hierarchical_json": str(hierarchy_path),
        "graph_paths": graph_paths,
        "timing": {**timing, "spotlight_stages": spotlight_stages},
        "spotlight": {
            "cumulative_recall": spotlight_summary["cumulative_recall"],
            "pass1_jaccard": spotlight_summary["pass1_jaccard"],
            "n_passes": spotlight_summary["n_passes"],
            "n_admitted": spotlight_summary["n_admitted"],
            "admitted_agent_ids": spotlight_summary["admitted_agent_ids"],
        },
        "hierarchical": {
            "n_nodes": hierarchy_summary["n_nodes"],
            "n_edges": hierarchy_summary["n_edges"],
            "n_components": hierarchy_summary["n_components"],
            "agent_graph_recall": hierarchy_summary["agent_graph_recall"],
            "clean_graph_recall": hierarchy_summary["clean_graph_recall"],
            "covered_agent_ids": hierarchy_summary["covered_agent_ids"],
            "clean_graph_agent_ids": hierarchy_summary["clean_graph_agent_ids"],
        },
    }


def _run_case_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    case = CaseArgs(
        run=E12bRun(**payload["run"]),
        t_steps=payload["t_steps"],
        pretrain_epochs=payload["pretrain_epochs"],
        refine_epochs=payload["refine_epochs"],
        max_passes_cap=payload["max_passes_cap"],
        skip_existing=payload["skip_existing"],
        verbose=payload["verbose"],
        device=payload["device"],
        torch_threads=payload["torch_threads"],
    )
    return run_case(case)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E12b complex-agent hierarchy sample.")
    parser.add_argument("--limit", type=int, default=len(RUNS), help="Run only the first N cases.")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run only these case ids (repeatable).",
    )
    parser.add_argument("--fast", action="store_true", help="Shorter trace and fewer train epochs only.")
    parser.add_argument("--t-steps", type=int, default=4000, help="Simulation length per run.")
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="Pretrain epochs per pass.")
    parser.add_argument("--refine-epochs", type=int, default=40, help="Refine epochs per pass.")
    parser.add_argument(
        "--max-passes-cap",
        type=int,
        default=0,
        help=f"Optional cap on spotlight passes (never below {MIN_PASSES}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for spotlight training (default: auto mps/cuda/cpu).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes for independent sweep cases.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch/BLAS threads per worker (use 1 when --jobs > 1).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-run artifacts.")
    parser.add_argument("--quiet", action="store_true", help="Suppress spotlight pass logs.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="results/hierarchical/e12b/e12b_complex_sample_summary.json",
    )
    return parser.parse_args()


def _select_runs(args: argparse.Namespace) -> List[E12bRun]:
    if args.run_id:
        by_id = {run.run_id: run for run in RUNS}
        missing = [run_id for run_id in args.run_id if run_id not in by_id]
        if missing:
            raise SystemExit(f"Unknown run id(s): {', '.join(missing)}")
        return [by_id[run_id] for run_id in args.run_id]
    return RUNS[: max(0, min(args.limit, len(RUNS)))]


def _case_payload(args: argparse.Namespace, run: E12bRun) -> Dict[str, Any]:
    return {
        "run": asdict(run),
        "t_steps": args.t_steps,
        "pretrain_epochs": args.pretrain_epochs,
        "refine_epochs": args.refine_epochs,
        "max_passes_cap": args.max_passes_cap or None,
        "skip_existing": args.skip_existing,
        "verbose": not args.quiet,
        "device": args.device,
        "torch_threads": args.torch_threads,
    }


def main() -> None:
    args = parse_args()
    if args.fast:
        args.t_steps = min(args.t_steps, 1600)
        args.pretrain_epochs = min(args.pretrain_epochs, 10)
        args.refine_epochs = min(args.refine_epochs, 8)
    if args.jobs > 1 and args.torch_threads == 1:
        pass
    elif args.jobs > 1 and args.torch_threads > 1:
        print(
            f"warning: --jobs={args.jobs} with --torch-threads={args.torch_threads} "
            "may oversubscribe CPU; prefer --torch-threads 1",
            flush=True,
        )

    selected = _select_runs(args)
    payloads = [_case_payload(args, run) for run in selected]
    results: List[Dict[str, Any]] = []

    if args.jobs <= 1:
        for index, payload in enumerate(payloads, start=1):
            run_id = payload["run"]["run_id"]
            print(f"=== E12b run {index}/{len(payloads)}: {run_id} ===", flush=True)
            result = _run_case_worker(payload)
            results.append(result)
            h = result["hierarchical"]
            print(
                f"done {run_id}: graph={h['agent_graph_recall']:.3f} "
                f"clean={h['clean_graph_recall']:.3f} components={h['n_components']} "
                f"{_format_timing(result['timing'])}",
                flush=True,
            )
    else:
        print(f"=== E12b parallel sweep: {len(payloads)} cases, jobs={args.jobs} ===", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run_case_worker, payload): payload for payload in payloads}
            for future in as_completed(futures):
                payload = futures[future]
                run_id = payload["run"]["run_id"]
                result = future.result()
                results.append(result)
                h = result["hierarchical"]
                print(
                    f"done {run_id}: graph={h['agent_graph_recall']:.3f} "
                    f"clean={h['clean_graph_recall']:.3f} components={h['n_components']} "
                    f"{_format_timing(result['timing'])}",
                    flush=True,
                )
        results.sort(key=lambda item: item["run"]["run_id"])

    summary = {
        "experiment": "E12b complex fixed-coordinate hierarchy sample",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "n_runs": len(results),
        "runs": results,
    }
    _write_json(Path(args.summary_json), summary)

    print("=== E12b complex hierarchy sample ===")
    for result in results:
        run_id = result["run"]["run_id"]
        spotlight = result["spotlight"]
        hierarchical = result["hierarchical"]
        print(
            f"{run_id}: spot={spotlight['cumulative_recall']:.3f} "
            f"graph={hierarchical['agent_graph_recall']:.3f} "
            f"clean={hierarchical['clean_graph_recall']:.3f} "
            f"components={hierarchical['n_components']} "
            f"{_format_timing(result['timing'])}"
        )
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
