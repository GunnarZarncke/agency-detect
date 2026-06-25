"""Export S/A/I assignments for each agent-corner hit from exploration.json.

Reads existing artifacts only (no discovery rerun):
  - X2 command-circuit anchor hits → roles from role_assignments.json
  - X3 per-animal community hits → roles from cached bundle (score_members only)

Run: PYTHONPATH=. .venv/bin/python scripts/worm/export_recovered.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from uad_worm.candidates import ANCHOR_CLASSES, agglomerative_communities, classes_of, classes_to_indices
from uad_worm.data import BASE_URL, load_dataset
from uad_worm.preprocess import preprocess
from uad_worm.score import score_members

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_assignments import RESULTS_DIR, neuron_record

EXT_DIM = 6
REP = "whitened"
N_PERM = 100


def _provenance_block(ds, present_indices, *, classes):
    upstream = ds.provenance.get("upstream_checksums") or {}
    return {
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
        "candidate_classes": classes,
        "candidate_classes_present": sorted({ds.neuron_class[i] for i in present_indices}),
        "n_candidate_neurons": len(present_indices),
    }


def _roles_block(ds, res) -> dict:
    return {
        "sensors": [neuron_record(ds, i) for i in res.roles.sensors],
        "actions": [neuron_record(ds, i) for i in res.roles.actions],
        "internal": [neuron_record(ds, i) for i in res.roles.internal],
    }


def main() -> None:
    explore_path = RESULTS_DIR / "exploration.json"
    roles_path = RESULTS_DIR / "role_assignments.json"
    explore = json.loads(explore_path.read_text())
    roles_export = json.loads(roles_path.read_text())

    anchor_animals = {
        a["dataset_id"]: a
        for a in roles_export["candidates"][0]["animals"]
        if a.get("roles")
    }

    recovered = []
    for row in explore["x2_anchor_joint"]["per_animal"]:
        if not row.get("agent_corner"):
            continue
        ds_id = row["dataset_id"]
        base = anchor_animals.get(ds_id)
        if base is None:
            raise RuntimeError(f"missing anchor roles for {ds_id}")
        recovered.append({
            "recovery_source": "x2_command_circuit_anchor",
            "agent_corner_criterion": "loss_p < 0.5 and autonomy_p > 0.5",
            "loss_p": row["loss_p"],
            "autonomy_p": row["autonomy_p"],
            "candidate_classes": sorted(ANCHOR_CLASSES),
            **{k: base[k] for k in base if k != "roles"},
            "roles": base["roles"],
        })

    for row in explore["x3_per_animal_communities"]["per_animal"]:
        if not row.get("agent_corner"):
            continue
        ds_id = row["dataset_id"]
        class_set = frozenset(row["best_classes"])
        ds = load_dataset(ds_id, write_provenance=False)
        proc = preprocess(ds)
        # Match the exact index community from explore (not all neurons of those classes).
        members = None
        for comm in agglomerative_communities(
            proc.representation(REP), min_size=3, max_size=30
        ):
            if (
                len(comm) == row["best_n_members"]
                and classes_of(comm, proc.neuron_class) == class_set
            ):
                members = comm
                break
        if members is None:
            members = classes_to_indices(class_set, ds.neuron_class)
        res = score_members(
            proc.representation(REP), members, ext_dim=EXT_DIM, n_perm=N_PERM, seed=0
        )
        if res is None:
            raise RuntimeError(f"could not score X3 community for {ds_id}")
        entry = _provenance_block(ds, members, classes=sorted(class_set))
        entry.update({
            "recovery_source": "x3_per_animal_lagged_corr_community",
            "agent_corner_criterion": "loss_p < 0.5 and autonomy_p > 0.5",
            "loss_p": row["best_loss_p"],
            "autonomy_p": row["best_autonomy_p"],
            "status": "scored",
            "representation": REP,
            "blanket_loss": res.loss,
            "pvalue_vs_random_partition": res.pvalue,
            "z_vs_random_partition": res.z,
            "passes_at_0.05": bool(res.pvalue < 0.05),
            "roles": _roles_block(ds, res),
        })
        recovered.append(entry)

    out = {
        "experiment": "E20",
        "organism": "Caenorhabditis elegans",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "exploration": str(explore_path.relative_to(RESULTS_DIR.parent.parent)),
            "role_assignments": str(roles_path.relative_to(RESULTS_DIR.parent.parent)),
        },
        "linking": roles_export.get("linking"),
        "note": (
            "Questionably recovered agents (agent-corner hits from explore.py). "
            "Not validated by leave-one-animal-out; for inspection and cross-dataset linking."
        ),
        "n_recovered": len(recovered),
        "recovered_agents": recovered,
    }

    out_path = RESULTS_DIR / "recovered_assignments.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(recovered)} recovered agents to {out_path}")
    for r in recovered:
        print(f"  {r['recovery_source']}: {r['dataset_id']} "
              f"loss_p={r['loss_p']:.2f} autonomy_p={r['autonomy_p']:.2f}")


if __name__ == "__main__":
    main()
