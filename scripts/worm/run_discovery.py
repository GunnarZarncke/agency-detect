"""E20 discovery runner: pooled blanket discovery on a NeuroPAL-Baseline cohort.

Runs the v1 bet end-to-end on real WormWideWeb data:
  1. load + preprocess (whiten) a cohort of NeuroPAL-Baseline animals,
  2. score the command-circuit anchor candidate (pooled + leave-one-animal-out),
  3. M5 random-class-set null, M6 behavior-prediction gain,
  4. also report a data-driven recurrent (unsupervised) candidate.

Usage:
  .venv/bin/python scripts/worm/run_discovery.py --max-animals 8 --n-perm 100
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from uad_worm.candidates import (
    ANCHOR_CLASSES,
    agglomerative_communities,
    community_class_sets,
)
from uad_worm.data import is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.evaluate import pooled_behavior_gain, random_class_set_null
from uad_worm.preprocess import preprocess
from uad_worm.score import leave_one_animal_out, pooled_score

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"


def recurrent_class_set(train, *, k=8, representation="whitened"):
    """Top-k most frequent labeled classes among per-animal lagged-corr communities."""
    counter: collections.Counter = collections.Counter()
    for proc in train:
        comms = agglomerative_communities(proc.representation(representation))
        for cs in community_class_sets(comms, proc.neuron_class, min_classes=3):
            counter.update(cs)
    return frozenset(c for c, _ in counter.most_common(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-animals", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--ext-dim", type=int, default=6)
    ap.add_argument("--representation", default="whitened")
    args = ap.parse_args()

    print("Fetching NeuroPAL-labeled dataset index ...")
    ids = neuropal_labeled_ids()
    cohort = []
    for ds_id in ids:
        if len(cohort) >= args.max_animals:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            cohort.append(preprocess(ds))
            print(f"  + {ds_id}: {ds.activity.shape}, {len(ds.labeled_index)} labeled")
    print(f"Cohort size: {len(cohort)} NeuroPAL-Baseline animals\n")

    kw = dict(representation=args.representation, ext_dim=args.ext_dim, min_members=3)

    print("=== Anchor candidate: locomotor command circuit ===")
    print(f"classes: {sorted(ANCHOR_CLASSES)}")
    anchor_pool = pooled_score(cohort, ANCHOR_CLASSES, n_perm=args.n_perm, **kw)
    anchor_loao = leave_one_animal_out(
        cohort, lambda _train: ANCHOR_CLASSES, n_perm=args.n_perm, **kw
    )
    anchor_null = random_class_set_null(cohort, ANCHOR_CLASSES, n_sets=40, **kw)
    anchor_gain = pooled_behavior_gain(cohort, ANCHOR_CLASSES, feature="velocity", representation="raw")

    print(f"  pooled: n={anchor_pool.n_animals} mean_loss={anchor_pool.mean_loss:.4f} "
          f"pass_rate={anchor_pool.pass_rate:.2f} median_z={anchor_pool.median_z:.2f} "
          f"combined_p={anchor_pool.combined_p:.2e}")
    print(f"  leave-one-animal-out: held={anchor_loao['n_held']} "
          f"holdout_pass_rate={anchor_loao['holdout_pass_rate']:.2f} "
          f"mean_holdout_loss={anchor_loao['mean_holdout_loss']:.4f}")
    print(f"  random-class-set null: obs_loss={anchor_null.observed_loss:.4f} "
          f"null_mean={anchor_null.null_losses.mean():.4f} z={anchor_null.z:.2f} "
          f"p={anchor_null.pvalue:.3f}")
    print(f"  behavior-prediction gain (velocity): {anchor_gain:.4f}\n")

    print("=== Unsupervised recurrent candidate (lagged-corr communities) ===")
    rec = recurrent_class_set(cohort, k=len(ANCHOR_CLASSES), representation=args.representation)
    print(f"classes: {sorted(rec)}")
    rec_pool = pooled_score(cohort, rec, n_perm=args.n_perm, **kw)
    rec_null = random_class_set_null(cohort, rec, n_sets=40, **kw)
    rec_gain = pooled_behavior_gain(cohort, rec, feature="velocity", representation="raw")
    print(f"  pooled: n={rec_pool.n_animals} mean_loss={rec_pool.mean_loss:.4f} "
          f"pass_rate={rec_pool.pass_rate:.2f} combined_p={rec_pool.combined_p:.2e}")
    print(f"  random-class-set null: z={rec_null.z:.2f} p={rec_null.pvalue:.3f}")
    print(f"  behavior-prediction gain (velocity): {rec_gain:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "cohort": [p.dataset_id for p in cohort],
        "config": vars(args),
        "anchor": {
            "classes": sorted(ANCHOR_CLASSES),
            "pooled": anchor_pool.__dict__,
            "leave_one_animal_out": anchor_loao,
            "random_class_set_null": {
                "observed_loss": anchor_null.observed_loss,
                "null_mean": float(anchor_null.null_losses.mean()),
                "z": anchor_null.z,
                "pvalue": anchor_null.pvalue,
            },
            "behavior_gain_velocity": anchor_gain,
        },
        "recurrent": {
            "classes": sorted(rec),
            "pooled_mean_loss": rec_pool.mean_loss,
            "pooled_combined_p": rec_pool.combined_p,
            "random_class_set_null_p": rec_null.pvalue,
            "behavior_gain_velocity": rec_gain,
        },
    }
    out = RESULTS_DIR / "discovery_report.json"
    out.write_text(json.dumps(report, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
