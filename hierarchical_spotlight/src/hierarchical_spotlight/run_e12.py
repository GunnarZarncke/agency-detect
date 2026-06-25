#!/usr/bin/env python3
"""Run E12 hierarchical chunk-fusion on a spotlight artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
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

from hierarchical_spotlight.config import HierarchicalConfig
from hierarchical_spotlight.fusion import run_hierarchical_fusion, write_graphviz


def _timestamped(path: str, stamp: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{stamp}{p.suffix}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E12 hierarchical spotlight fusion.")
    for f in fields(HierarchicalConfig):
        name = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(name, type=type(f.default), default=f.default)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HierarchicalConfig(**{f.name: getattr(args, f.name) for f in fields(HierarchicalConfig)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if cfg.timestamp_outputs:
        cfg.output_json = _timestamped(cfg.output_json, stamp)
        if cfg.output_dot:
            cfg.output_dot = _timestamped(cfg.output_dot, stamp)
        if cfg.output_png:
            cfg.output_png = _timestamped(cfg.output_png, stamp)

    result = run_hierarchical_fusion(cfg)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    summary = result["summary"]
    print("=== E12 hierarchical fusion ===")
    print(
        f"nodes={summary['n_nodes']} edges={summary['n_edges']} "
        f"components={summary['n_components']}"
    )
    print(
        f"agent_graph_recall={summary['agent_graph_recall']:.3f} "
        f"covered={summary['covered_agent_ids']}"
    )
    print(
        f"clean_graph_recall={summary['clean_graph_recall']:.3f} "
        f"clean={summary['clean_graph_agent_ids']}"
    )

    out = Path(cfg.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    dot_path = cfg.output_dot or str(out.with_suffix(".dot"))
    png_path = cfg.output_png or str(out.with_suffix(".png"))
    graph_paths = write_graphviz(result, dot_path, png_path)
    print(f"Wrote {graph_paths['dot']}")
    if "png" in graph_paths:
        print(f"Wrote {graph_paths['png']}")
    elif "png_error" in graph_paths:
        print(f"Skipped PNG render: {graph_paths['png_error']}")


if __name__ == "__main__":
    main()

