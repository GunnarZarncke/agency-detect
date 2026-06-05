"""E20 probe: why does the command-circuit anchor score 0/8 on the within-animal contrast?

Uses the cached cohort to isolate the cause without progressing past M6:
  P1 null reference  — random partitions drawn from ALL neurons vs LABELED-only.
  P2 representation   — whitened vs raw.
  P3 external rank    — ext_dim sweep.

Run: PYTHONPATH=. .venv/bin/python scripts/worm/probe.py
"""

from __future__ import annotations

import numpy as np

from uad_worm.candidates import ANCHOR_CLASSES, classes_to_indices
from uad_worm.data import is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.preprocess import preprocess
from uad_worm.score import score_members

MAX_ANIMALS = 8


def load_cohort():
    cohort = []
    for ds_id in neuropal_labeled_ids():
        if len(cohort) >= MAX_ANIMALS:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            cohort.append(preprocess(ds))
    return cohort


def pass_rate(cohort, *, representation, ext_dim, labeled_only, n_perm=200):
    ps, losses = [], []
    for proc in cohort:
        trace = proc.representation(representation)
        members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
        if len(members) < 3:
            continue
        pool = proc.labeled_index if labeled_only else None
        res = score_members(
            trace, members, ext_dim=ext_dim, n_perm=n_perm, seed=0, pool=pool
        )
        if res is None:
            continue
        ps.append(res.pvalue)
        losses.append(res.loss)
    if not ps:
        return None
    return {
        "n": len(ps),
        "pass_rate": float(np.mean([p < 0.05 for p in ps])),
        "mean_loss": float(np.mean(losses)),
        "median_p": float(np.median(ps)),
    }


def main():
    # labeled_index lives on WormDataset; expose via Processed by re-deriving from classes.
    cohort = load_cohort()
    # Attach labeled_index to Processed objects (indices with a class label).
    for proc in cohort:
        object.__setattr__(
            proc, "labeled_index", [i for i, c in enumerate(proc.neuron_class) if c]
        )
    print(f"cohort: {len(cohort)} animals\n")

    print("P1 — null reference (whitened, ext_dim=6):")
    for labeled_only in (False, True):
        r = pass_rate(cohort, representation="whitened", ext_dim=6, labeled_only=labeled_only)
        tag = "labeled-only" if labeled_only else "all-neurons"
        print(f"  {tag:12s}: pass={r['pass_rate']:.2f} mean_loss={r['mean_loss']:.4f} "
              f"median_p={r['median_p']:.3f}")

    print("\nP2 — representation (all-neuron null, ext_dim=6):")
    for rep in ("whitened", "raw"):
        r = pass_rate(cohort, representation=rep, ext_dim=6, labeled_only=False)
        print(f"  {rep:9s}: pass={r['pass_rate']:.2f} mean_loss={r['mean_loss']:.4f} "
              f"median_p={r['median_p']:.3f}")

    print("\nP3 — external rank ext_dim (whitened, all-neuron null):")
    for ed in (4, 8, 12, 20):
        r = pass_rate(cohort, representation="whitened", ext_dim=ed, labeled_only=False)
        print(f"  ext_dim={ed:2d}: pass={r['pass_rate']:.2f} mean_loss={r['mean_loss']:.4f} "
              f"median_p={r['median_p']:.3f}")

    print("\nP1b — labeled-only null across representations (ext_dim=6):")
    for rep in ("whitened", "raw"):
        r = pass_rate(cohort, representation=rep, ext_dim=6, labeled_only=True)
        print(f"  {rep:9s}: pass={r['pass_rate']:.2f} mean_loss={r['mean_loss']:.4f} "
              f"median_p={r['median_p']:.3f}")


if __name__ == "__main__":
    main()
