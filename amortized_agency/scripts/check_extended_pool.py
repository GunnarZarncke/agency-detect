#!/usr/bin/env python3
"""Smoke-check extended amortized pool wiring (no training run).

Validates registry keys, builds one sim + one external episode per train kind.
Use --melting-pot only when meltingpot is installed.
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
import sys
from pathlib import Path


from amortized_agency.kinds import EXTENDED_ALL_KINDS, EXTENDED_TRAIN_KINDS  # noqa: E402
from amortized_agency.worlds import simulate_episode  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--melting-pot", action="store_true", help="Include melting_pot_cooking_ring")
    args = p.parse_args()

    kinds = list(EXTENDED_TRAIN_KINDS)
    if args.melting_pot:
        kinds = [k for k in EXTENDED_ALL_KINDS if k.name == "melting_pot_cooking_ring"] + kinds

    print("Extended pool kinds to check:")
    for k in kinds:
        print(f"  {k.name:28} backend={k.backend} n={k.num_agents} key={k.external_key}")

    for kind in kinds:
        ep = simulate_episode(kind, args.window, seed=0, t_steps=500)
        print(
            f"OK {kind.name:28} window={ep.window.shape} trace_T={ep.trace_T} "
            f"agents={len(set(ep.agent_ids.tolist()))}"
        )
    print("\nAll checks passed (no model training).")


if __name__ == "__main__":
    main()
