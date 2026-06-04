#!/usr/bin/env python3
"""Install-check for E15/E16 external deps and optional Melting Pot assets.

Does not overwrite result artifacts. Melting Pot resources use repo-local
``data/melting_pot/`` when MELTINGPOT_RESOURCES_PATH is unset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / "data" / "melting_pot"


def _check_import(name: str, import_fn) -> bool:
    try:
        import_fn()
        print(f"  OK   {name}")
        return True
    except ImportError as e:
        print(f"  MISS {name}: {e}")
        return False


def check_core() -> bool:
    print("Core / E15:")
    ok = True
    ok &= _check_import("numpy", lambda: __import__("numpy"))
    ok &= _check_import("sklearn", lambda: __import__("sklearn"))
    ok &= _check_import("gymnasium", lambda: __import__("gymnasium"))
    from learn_agents.external_registry import EXTERNAL_BUILDERS

    for key in ("physics_cartpole", "rock_sample_5x5", "grid_pomdp_3x3", "grid_pomdp_5x5"):
        if key not in EXTERNAL_BUILDERS:
            print(f"  MISS registry[{key}]")
            ok = False
        else:
            print(f"  OK   registry[{key}]")
    return ok


def check_melting_pot(*, smoke_rollout: bool) -> bool:
    print("\nMelting Pot (E16):")
    if not _check_import("dm_env", lambda: __import__("dm_env")):
        return False
    if not _check_import("meltingpot", lambda: __import__("meltingpot")):
        return False

    if not os.environ.get("MELTINGPOT_RESOURCES_PATH"):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ["MELTINGPOT_RESOURCES_PATH"] = str(DATA_ROOT)
        print(f"  SET  MELTINGPOT_RESOURCES_PATH={DATA_ROOT}")

    from meltingpot import substrate

    name = "collaborative_cooking__ring"
    print(f"  ... building substrate {name!r} (may download assets on first use)")
    env = substrate.build(name, roles=["default", "default"])
    try:
        n = len(env.observation_spec())
        print(f"  OK   build({name!r}) players={n}")
        if smoke_rollout:
            from learn_agents.melting_pot import MeltingPotConfig, roll_melting_pot

            result = roll_melting_pot(
                MeltingPotConfig(substrate_name=name, max_steps=20, seed=0)
            )
            print(f"  OK   roll_melting_pot trace shape {result.trace.shape}")
    finally:
        env.close()
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--melting-pot", action="store_true", help="Also verify meltingpot build + short rollout")
    args = p.parse_args()

    core_ok = check_core()
    if args.melting_pot:
        mp_ok = check_melting_pot(smoke_rollout=True)
    else:
        mp_ok = True
        print("\n(Skip Melting Pot; use --melting-pot after: pip install dm-env meltingpot)")

    if core_ok and mp_ok:
        print("\nAll requested checks passed.")
        sys.exit(0)
    print("\nSome checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
