"""Load a NeuroPAL-labeled worm cohort for E20 runners."""

from __future__ import annotations

from typing import List, Optional, Tuple

from uad_worm.candidates import ANCHOR_CLASSES
from uad_worm.data import WormDataset, is_neuropal_baseline, load_dataset, neuropal_labeled_ids
from uad_worm.preprocess import Processed, preprocess

QUALITY_MIN_T = 1500
QUALITY_MIN_LABELED = 70
QUALITY_MIN_ANCHOR = 5


def cohort_quality(ds: WormDataset) -> dict:
    """Per-animal quality proxies used for cohort filtering."""
    t, _ = ds.activity.shape
    n_labeled = int(ds.provenance.get("n_labeled", sum(1 for c in ds.neuron_class if c)))
    labeled = {c for c in ds.neuron_class if c}
    anchor_present = len(ANCHOR_CLASSES & labeled)
    return {"T": t, "n_labeled": n_labeled, "anchor_present": anchor_present}


def passes_quality_filter(
    ds: WormDataset,
    *,
    min_t: int = QUALITY_MIN_T,
    min_labeled: int = QUALITY_MIN_LABELED,
    min_anchor: int = QUALITY_MIN_ANCHOR,
) -> bool:
    q = cohort_quality(ds)
    return q["T"] >= min_t and q["n_labeled"] >= min_labeled and q["anchor_present"] >= min_anchor


def load_neuropal_cohort(
    *,
    max_animals: Optional[int] = None,
    baseline_only: bool = False,
    quality_filter: bool = False,
    min_t: int = QUALITY_MIN_T,
    min_labeled: int = QUALITY_MIN_LABELED,
    min_anchor: int = QUALITY_MIN_ANCHOR,
    write_provenance: bool = False,
) -> Tuple[List[Tuple[WormDataset, Processed]], List[str], List[str]]:
    """Return (cohort, load_skipped, quality_filtered_ids)."""
    out: List[Tuple[WormDataset, Processed]] = []
    skipped: List[str] = []
    quality_filtered: List[str] = []
    for ds_id in neuropal_labeled_ids():
        if max_animals is not None and len(out) >= max_animals:
            break
        try:
            ds = load_dataset(ds_id, write_provenance=write_provenance)
        except Exception as exc:
            skipped.append(ds_id)
            print(f"  ! skip {ds_id}: {exc}")
            continue
        if baseline_only and not is_neuropal_baseline(ds):
            continue
        if quality_filter and not passes_quality_filter(
            ds, min_t=min_t, min_labeled=min_labeled, min_anchor=min_anchor
        ):
            q = cohort_quality(ds)
            quality_filtered.append(ds_id)
            print(
                f"  - {ds.animal_id} ({ds_id}) [quality: T={q['T']} "
                f"n_labeled={q['n_labeled']} anchor={q['anchor_present']}]"
            )
            continue
        out.append((ds, preprocess(ds)))
        tag = "baseline" if is_neuropal_baseline(ds) else "other"
        print(f"  + {ds.animal_id} ({ds_id}) [{tag}]")
    return out, skipped, quality_filtered
