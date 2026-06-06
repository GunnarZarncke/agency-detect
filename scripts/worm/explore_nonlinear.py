"""E20 unexplored direction: nonlinear-robustness via Gaussian-copula CMI.

Tests whether the robust negative is an artifact of the linear/Gaussian assumption by
re-running the headline scorers on the normal-scores (copula) representation, which captures
monotone nonlinear dependence. Compares Gaussian ("whitened") vs copula ("whitened_copula"):
  N1 anchor pooled + leave-one-animal-out
  N2 anchor per-animal (autonomy, loss) plane — does the agent-corner set change?
  N3 X3 community AIM/ASE/RMD on its animal

Memory (M7) intentionally NOT touched. Run:
  PYTHONPATH=. .venv/bin/python scripts/worm/explore_nonlinear.py
"""

from __future__ import annotations

import json
from pathlib import Path

from uad_worm.candidates import ANCHOR_CLASSES, classes_to_indices
from uad_worm.data import is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.evaluate import joint_null
from uad_worm.preprocess import preprocess
from uad_worm.score import leave_one_animal_out, pooled_score, score_members

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"
MAX_ANIMALS = 8
EXT_DIM = 6
N_PERM = 120
REPS = ["whitened", "whitened_copula"]
X3_ANIMAL = "atanas_kim_2023-2023-01-23-08"
X3_CLASSES = frozenset({"AIM", "ASE", "RMD"})


def load_cohort():
    out = []
    for ds_id in neuropal_labeled_ids():
        if len(out) >= MAX_ANIMALS:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            out.append((ds, preprocess(ds)))
    return out


def main():
    cohort = load_cohort()
    procs = [p for _, p in cohort]
    print(f"cohort: {len(cohort)} animals\n")
    report = {"cohort": [ds.dataset_id for ds, _ in cohort], "ext_dim": EXT_DIM, "n_perm": N_PERM}

    print("N1 — anchor pooled + leave-one-animal-out:")
    report["n1_anchor"] = {}
    for rep in REPS:
        pool = pooled_score(procs, ANCHOR_CLASSES, representation=rep, ext_dim=EXT_DIM,
                            n_perm=N_PERM, min_members=3)
        loao = leave_one_animal_out(procs, lambda _t: ANCHOR_CLASSES, representation=rep,
                                    ext_dim=EXT_DIM, n_perm=N_PERM, min_members=3)
        report["n1_anchor"][rep] = {
            "pass_rate": pool.pass_rate, "median_z": pool.median_z,
            "combined_p": pool.combined_p, "holdout_pass_rate": loao["holdout_pass_rate"],
        }
        print(f"  {rep:16s}: pass={pool.pass_rate:.2f} median_z={pool.median_z:.2f} "
              f"combined_p={pool.combined_p:.2e} LOAO={loao['holdout_pass_rate']:.2f}")

    print("\nN2 — anchor per-animal agent corner (loss_p<0.5 & autonomy_p>0.5):")
    report["n2_agent_corner"] = {}
    for rep in REPS:
        hits = []
        for ds, proc in cohort:
            members = classes_to_indices(ANCHOR_CLASSES, proc.neuron_class)
            if len(members) < 3:
                continue
            jn = joint_null(proc.representation(rep), members, ext_dim=EXT_DIM,
                            n_perm=N_PERM, seed=0)
            if jn and jn.agent_corner:
                hits.append(ds.dataset_id)
        report["n2_agent_corner"][rep] = {"n_hits": len(hits), "animals": hits}
        print(f"  {rep:16s}: {len(hits)}/{len(cohort)} agent-corner  {hits}")

    print(f"\nN3 — X3 community {sorted(X3_CLASSES)} on {X3_ANIMAL}:")
    report["n3_x3_community"] = {}
    ds = load_dataset(X3_ANIMAL)
    proc = preprocess(ds)
    members = classes_to_indices(X3_CLASSES, proc.neuron_class)
    for rep in REPS:
        res = score_members(proc.representation(rep), members, ext_dim=EXT_DIM,
                            n_perm=N_PERM, seed=0)
        jn = joint_null(proc.representation(rep), members, ext_dim=EXT_DIM, n_perm=N_PERM, seed=0)
        report["n3_x3_community"][rep] = {
            "loss": res.loss, "pvalue": res.pvalue,
            "loss_p": jn.loss_p, "autonomy_p": jn.autonomy_p, "agent_corner": jn.agent_corner,
        }
        print(f"  {rep:16s}: loss={res.loss:.4f} p={res.pvalue:.2f} "
              f"loss_p={jn.loss_p:.2f} autonomy_p={jn.autonomy_p:.2f} "
              f"{'AGENT-CORNER' if jn.agent_corner else ''}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "exploration_nonlinear.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
