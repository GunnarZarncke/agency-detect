"""E20 unexplored directions: nonparametric kNN (KSG) CMI + larger cohort.

Two asks in one run on an enlarged NeuroPAL-Baseline cohort:
  - larger cohort: does the 0/N leave-one-animal-out negative hold with more animals?
  - kNN CMI: does a nonparametric estimator (captures non-monotone dependence) change it?

Both estimators score the command-circuit anchor (pooled + leave-one-animal-out) on the SAME
cohort. kNN keeps dimensionality low (ext_dim/int_dim small) since KSG degrades in high-dim.
Memory (M7) untouched.

Run: PYTHONPATH=. .venv/bin/python scripts/worm/explore_knn.py --max-animals 20 --n-perm 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uad_worm.candidates import ANCHOR_CLASSES
from uad_worm.data import is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.preprocess import preprocess
from uad_worm.score import leave_one_animal_out, pooled_score

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"


def load_cohort(max_animals: int):
    out = []
    for ds_id in neuropal_labeled_ids():
        if len(out) >= max_animals:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            out.append(preprocess(ds))
            print(f"  + {ds_id}")
    return out


def run(cohort, *, estimator, ext_dim, int_dim, n_perm):
    kw = dict(representation="whitened", ext_dim=ext_dim, int_dim=int_dim,
              estimator=estimator, n_perm=n_perm, min_members=3)
    pool = pooled_score(cohort, ANCHOR_CLASSES, **kw)
    loao = leave_one_animal_out(cohort, lambda _t: ANCHOR_CLASSES, **kw)
    return {
        "estimator": estimator, "ext_dim": ext_dim, "int_dim": int_dim,
        "n_animals": pool.n_animals, "pass_rate": pool.pass_rate,
        "median_z": pool.median_z, "combined_p": pool.combined_p,
        "holdout_pass_rate": loao["holdout_pass_rate"],
        "mean_holdout_loss": loao["mean_holdout_loss"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-animals", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=80)
    args = ap.parse_args()

    print(f"Loading up to {args.max_animals} NeuroPAL-Baseline animals ...")
    cohort = load_cohort(args.max_animals)
    print(f"cohort: {len(cohort)} animals\n")

    rows = []
    print("Gaussian (ext_dim=6):")
    rows.append(run(cohort, estimator="gaussian", ext_dim=6, int_dim=3, n_perm=args.n_perm))
    print(f"  n={rows[-1]['n_animals']} pass={rows[-1]['pass_rate']:.2f} "
          f"median_z={rows[-1]['median_z']:.2f} combined_p={rows[-1]['combined_p']:.2e} "
          f"LOAO={rows[-1]['holdout_pass_rate']:.2f}")

    print("kNN/KSG (ext_dim=3, int_dim=3):")
    rows.append(run(cohort, estimator="knn", ext_dim=3, int_dim=3, n_perm=args.n_perm))
    print(f"  n={rows[-1]['n_animals']} pass={rows[-1]['pass_rate']:.2f} "
          f"median_z={rows[-1]['median_z']:.2f} combined_p={rows[-1]['combined_p']:.2e} "
          f"LOAO={rows[-1]['holdout_pass_rate']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "exploration_knn.json"
    out.write_text(json.dumps({
        "cohort": [p.dataset_id for p in cohort],
        "n_perm": args.n_perm,
        "runs": rows,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
