"""E20 export: discovered S/A/I role assignments per animal → JSON with provenance.

For each candidate (command-circuit anchor + unsupervised recurrent set) and each
NeuroPAL-Baseline animal, records the operational role assignment (sensor / action /
internal) of the candidate's neurons, the blanket score, and full source/provenance meta.

Roles are *operational*, not anatomical: assigned by lagged influence to/from the rest of
the brain (README §3), so they can differ across animals. NOTE: the v1 result is a robust
negative (EXPERIMENTS.md §E20) — these assignments are exported for inspection, not as a
validated agent boundary.

Run: PYTHONPATH=. .venv/bin/python scripts/worm/export_assignments.py --max-animals 8
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from uad_worm.candidates import (
    ANCHOR_CLASSES,
    agglomerative_communities,
    classes_to_indices,
    community_class_sets,
)
from uad_worm.data import BASE_URL, is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.preprocess import preprocess
from uad_worm.score import score_candidate_per_animal

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "worm"


def neuron_record(ds, idx: int) -> dict:
    """One neuron with cross-dataset linking keys.

    `label` (canonical NeuroPAL identity, e.g. AVDL) joins to other datasets of the *same
    animal* on (uid, label); `roi_id` joins to that animal's NeuroPAL segmentation/positions.
    `trace_index` is recording-internal (the column in this bundle's trace array) and is
    NOT a stable cross-dataset key.
    """
    info = (ds.neuron_label_info[idx] if ds.neuron_label_info else None) or {}
    return {
        "label": ds.neuron_label[idx] if ds.neuron_label else info.get("label"),
        "neuron_class": ds.neuron_class[idx],
        "roi_id": ds.roi_id[idx],
        "LR": info.get("LR"),
        "DV": info.get("DV"),
        "confidence": info.get("confidence"),
        "trace_index": int(idx),
    }


def recurrent_class_set(procs, *, k: int, representation: str):
    counter: collections.Counter = collections.Counter()
    for proc in procs:
        comms = agglomerative_communities(proc.representation(representation))
        for cs in community_class_sets(comms, proc.neuron_class, min_classes=3):
            counter.update(cs)
    return frozenset(c for c, _ in counter.most_common(k))


def animal_assignment(ds, proc, class_set, *, representation, ext_dim, n_perm, min_members):
    res = score_candidate_per_animal(
        proc, class_set, representation=representation, ext_dim=ext_dim,
        n_perm=n_perm, min_members=min_members,
    )
    present = classes_to_indices(class_set, ds.neuron_class)
    upstream = ds.provenance.get("upstream_checksums") or {}
    meta = {
        "dataset_id": ds.dataset_id,
        "uid": ds.animal_id,
        "animal_id": ds.animal_id,
        "paper_id": ds.provenance.get("paper_id"),
        "source_url": f"{BASE_URL}/download/{ds.dataset_id}/",
        "source_cache": f"data/worm/{ds.dataset_id}.json.bz2",
        "source_filename": ds.provenance.get("source_filename"),
        "archive_sha256": ds.provenance.get("archive_sha256"),
        "upstream_checksums": upstream,
        "neuropal_reference": {"blake3_neuropal_dict": upstream.get("blake3_neuropal_dict")},
        "n_neuron": ds.n_neurons,
        "n_labeled": ds.provenance.get("n_labeled"),
        "mean_timestep_s": ds.provenance.get("mean_timestep"),
        "candidate_classes_present": sorted({ds.neuron_class[i] for i in present}),
        "n_candidate_neurons": len(present),
    }
    if res is None:
        meta["status"] = "skipped (too few candidate neurons present)"
        meta["roles"] = None
        return meta
    meta["status"] = "scored"
    meta["representation"] = representation
    meta["blanket_loss"] = res.loss
    meta["pvalue_vs_random_partition"] = res.pvalue
    meta["z_vs_random_partition"] = res.z
    meta["passes_at_0.05"] = bool(res.pvalue < 0.05)
    meta["roles"] = {
        "sensors": [neuron_record(ds, i) for i in res.roles.sensors],
        "actions": [neuron_record(ds, i) for i in res.roles.actions],
        "internal": [neuron_record(ds, i) for i in res.roles.internal],
    }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-animals", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--ext-dim", type=int, default=6)
    ap.add_argument("--representation", default="whitened")
    ap.add_argument("--out", default=str(RESULTS_DIR / "role_assignments.json"))
    args = ap.parse_args()

    cohort = []  # list of (WormDataset, Processed)
    for ds_id in neuropal_labeled_ids():
        if len(cohort) >= args.max_animals:
            break
        ds = load_dataset(ds_id)
        if is_neuropal_baseline(ds):
            cohort.append((ds, preprocess(ds)))
    procs = [p for _, p in cohort]
    print(f"cohort: {len(cohort)} NeuroPAL-Baseline animals")

    recurrent = recurrent_class_set(procs, k=len(ANCHOR_CLASSES), representation=args.representation)
    candidates = [
        ("command_circuit_anchor", sorted(ANCHOR_CLASSES), ANCHOR_CLASSES),
        ("unsupervised_recurrent", sorted(recurrent), recurrent),
    ]

    kw = dict(
        representation=args.representation, ext_dim=args.ext_dim,
        n_perm=args.n_perm, min_members=3,
    )
    export = {
        "experiment": "E20",
        "title": "Unsupervised Agent Discovery on C. elegans whole-brain calcium (WormWideWeb)",
        "organism": "Caenorhabditis elegans",
        "data_source": {
            "name": "WormWideWeb",
            "paper": "Atanas & Kim et al., 2023",
            "api_base": BASE_URL,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "candidate_level": "neuron_class (pooled across animals)",
            "representation": args.representation,
            "blanket_loss": "I(internal_{t+1}; external_{t+1} | sensors_t, actions_t), lag=1",
            "external_reduction": f"PCA to {args.ext_dim} components",
            "role_rule": {
                "sensor": "candidate neuron most driven by external state (E_t -> member_{t+1})",
                "action": "candidate neuron most driving external future (member_t -> E_{t+1})",
                "internal": "remaining candidate neurons",
                "external": "all non-candidate neurons (reduced to PCs; not listed per-neuron)",
            },
            "null": "random same-size neuron partition contrast (n_perm)",
            "n_perm": args.n_perm,
        },
        "linking": {
            "organism_exemplar_key": "uid (== animal_id); dataset_id = paper_id-uid identifies the individual recorded animal",
            "neuron_identity_key": "roles[*].label — canonical NeuroPAL identity (e.g. AVDL); join across datasets of the SAME animal on (uid, label)",
            "position_key": "roles[*].roi_id — segmentation ROI id; join to that animal's NeuroPAL segmentation/positions (see per-animal neuropal_reference.blake3_neuropal_dict)",
            "trace_index_note": "roles[*].trace_index is the column in THIS bundle's trace array (recording-internal); not a stable cross-dataset key",
        },
        "result_note": (
            "v1 result is a robust negative: command-circuit anchor scores 0/8 on "
            "leave-one-animal-out; assignments below are for inspection, not a validated boundary."
        ),
        "cohort": [ds.dataset_id for ds, _ in cohort],
        "candidates": [],
    }
    for name, classes, class_set in candidates:
        animals = [
            animal_assignment(ds, proc, class_set, **kw) for ds, proc in cohort
        ]
        export["candidates"].append(
            {"name": name, "classes": classes, "animals": animals}
        )
        n_pass = sum(1 for a in animals if a.get("passes_at_0.05"))
        print(f"  {name}: {len(classes)} classes, scored "
              f"{sum(1 for a in animals if a['status']=='scored')}/{len(animals)} animals, "
              f"{n_pass} pass@0.05")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
