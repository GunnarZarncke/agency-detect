"""E20 exploration (post-M6): why no blanket, and is there one anywhere?

Three probes, do-one-thing-first:
  X1 timescale  — does a slower lag move the anchor off median p≈0.5?
  X2 joint axes — anchor in the (internal-autonomy, blanket-loss) plane vs random sets:
                  is it a coupled-but-leaky subsystem or nothing?
  X3 per-animal — unsupervised: score every lagged-corr community per animal on both axes;
                  does ANY land in the agent corner (low loss + high autonomy) beyond random?

Run: PYTHONPATH=. .venv/bin/python scripts/worm/explore.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uad_worm.candidates import (
    ANCHOR_CLASSES,
    agglomerative_communities,
    classes_of,
    classes_to_indices,
)
from uad_worm.data import is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.evaluate import joint_null
from uad_worm.preprocess import preprocess
from uad_worm.score import score_members

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"
MAX_ANIMALS = 8
REP = "whitened"
EXT_DIM = 6


def load_cohort():
    out = []
    for ds_id in neuropal_labeled_ids():
        if len(out) >= MAX_ANIMALS:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            out.append((ds, preprocess(ds)))
    return out


def x1_lag_sweep(cohort):
    print("X1 — timescale (anchor, whitened, ext_dim=6):")
    rows = []
    for lag in (1, 2, 3, 5):
        ps, losses = [], []
        for _, proc in cohort:
            trace = proc.representation(REP)
            members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
            if len(members) < 3:
                continue
            res = score_members(trace, members, ext_dim=EXT_DIM, lag=lag, n_perm=120, seed=0)
            if res:
                ps.append(res.pvalue)
                losses.append(res.loss)
        row = {"lag": lag, "median_p": float(np.median(ps)), "mean_loss": float(np.mean(losses))}
        rows.append(row)
        print(f"  lag={lag} ({lag*0.6:.1f}s): median_p={row['median_p']:.3f} "
              f"mean_loss={row['mean_loss']:.4f}")
    return rows


def x2_joint_anchor(cohort):
    print("\nX2 — anchor in (autonomy, loss) plane (n_perm=120):")
    rows = []
    for ds, proc in cohort:
        trace = proc.representation(REP)
        members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
        if len(members) < 3:
            continue
        jn = joint_null(trace, members, ext_dim=EXT_DIM, n_perm=120, seed=0)
        if jn is None:
            continue
        rows.append({"dataset_id": ds.dataset_id, "loss_p": jn.loss_p,
                     "autonomy_p": jn.autonomy_p, "agent_corner": jn.agent_corner})
        print(f"  {ds.dataset_id}: loss_p={jn.loss_p:.2f} autonomy_p={jn.autonomy_p:.2f} "
              f"{'AGENT-CORNER' if jn.agent_corner else ''}")
    med_lp = float(np.median([r['loss_p'] for r in rows]))
    med_ap = float(np.median([r['autonomy_p'] for r in rows]))
    print(f"  median loss_p={med_lp:.2f} median autonomy_p={med_ap:.2f}  "
          f"(agent corner = low loss_p + high autonomy_p)")
    return rows, med_lp, med_ap


def x3_per_animal(cohort):
    print("\nX3 — per-animal unsupervised communities in (coupling, loss) plane:")
    rows = []
    hits = 0
    for ds, proc in cohort:
        trace = proc.representation(REP)
        comms = agglomerative_communities(trace, min_size=3, max_size=30)
        best = None
        for members in comms:
            jn = joint_null(trace, members, ext_dim=EXT_DIM, n_perm=80, seed=0)
            if jn is None:
                continue
            # rank by agent-ness: low loss_p and high autonomy_p
            score = jn.autonomy_p - jn.loss_p
            if best is None or score > best[0]:
                cls = sorted(classes_of(members, proc.neuron_class))
                best = (score, jn, cls, len(members))
        if best is None:
            continue
        _, jn, cls, n = best
        if jn.agent_corner:
            hits += 1
        rows.append({"dataset_id": ds.dataset_id, "n_communities": len(comms),
                     "best_n_members": n, "best_loss_p": jn.loss_p,
                     "best_autonomy_p": jn.autonomy_p, "agent_corner": jn.agent_corner,
                     "best_classes": cls})
        print(f"  {ds.dataset_id}: {len(comms)} comms, best(size={n}) "
              f"loss_p={jn.loss_p:.2f} autonomy_p={jn.autonomy_p:.2f} "
              f"{'AGENT-CORNER' if jn.agent_corner else ''} classes={cls[:6]}")
    print(f"  animals with an agent-corner community: {hits}/{len(rows)}")
    return rows, hits


def main():
    cohort = load_cohort()
    print(f"cohort: {len(cohort)} NeuroPAL-Baseline animals\n")
    x1 = x1_lag_sweep(cohort)
    x2_rows, med_lp, med_ap = x2_joint_anchor(cohort)
    x3_rows, hits = x3_per_animal(cohort)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "exploration.json"
    out.write_text(json.dumps({
        "cohort": [ds.dataset_id for ds, _ in cohort],
        "x1_lag_sweep": x1,
        "x2_anchor_joint": {"per_animal": x2_rows, "median_loss_p": med_lp,
                            "median_autonomy_p": med_ap},
        "x3_per_animal_communities": {"per_animal": x3_rows, "agent_corner_hits": hits},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
