#!/usr/bin/env python3
"""
E9a: serial spotlight peel-off on the E8 setting (8 agents, 20% decoys).

Example:
  .venv/bin/python scripts/spotlight/run_spotlight_e9a.py \\
    --output-json results/spotlight/e9/spotlight_peel_e8_decoy20.json
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
import typing
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args, get_origin


from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import run_spotlight_peel


def _add_config_args(p: argparse.ArgumentParser) -> None:
    hints = typing.get_type_hints(SpotlightConfig)
    for f in fields(SpotlightConfig):
        name = f"--{f.name.replace('_', '-')}"
        hinted = hints[f.name]
        origin = get_origin(hinted)
        if hinted is bool:
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        elif origin is typing.Literal:
            p.add_argument(name, type=str, default=f.default)
        elif origin is typing.Union or str(origin) == "types.UnionType":
            non_none = [a for a in get_args(hinted) if a is not type(None)]
            inner = non_none[0] if non_none else str
            p.add_argument(name, type=inner, default=f.default)
        elif hinted in (int, float, str):
            p.add_argument(name, type=hinted, default=f.default)
        else:
            p.add_argument(name, type=str, default=f.default)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E9a spotlight peel experiment.")
    _add_config_args(p)
    p.add_argument("--output-json", type=str, default="results/spotlight/e10/spotlight_exogenous_baseline.json")
    return p.parse_args()


def args_to_config(args: argparse.Namespace) -> SpotlightConfig:
    kwargs = {}
    for f in fields(SpotlightConfig):
        kwargs[f.name] = getattr(args, f.name)
    return SpotlightConfig(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = args_to_config(args)
    report = run_spotlight_peel(cfg)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    print("\n=== E9a summary ===")
    s = report["summary"]
    print(f"cumulative_recall={s['cumulative_recall']:.3f} pass1_jaccard={s['pass1_jaccard']:.3f}")
    print(f"n_admitted={s['n_admitted']} agents={s['admitted_agent_ids']}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
