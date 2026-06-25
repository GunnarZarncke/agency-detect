"""M1 — WormWideWeb dataset ingestion.

Fetches a per-dataset bundle from the public WormWideWeb API, caches the raw archive,
records provenance (upstream checksums + our own sha256), and normalizes to one internal
schema (`WormDataset`). See README §1 (verified endpoints) and §4 (bundle schema).

Layout convention:
- raw cache  → ``data/worm/<dataset_id>.json.bz2`` (gitignored)
- provenance → ``uad_worm/manifests/<dataset_id>.json`` (tracked)
"""

from __future__ import annotations

import bz2
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BASE_URL = "https://wormwideweb.org/activity/api/data"
# The API returns 403 to bare urllib; a browser User-Agent is required (README §1).
USER_AGENT = "Mozilla/5.0 (uad_worm research client)"

def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "agency_detect").is_dir():
            return candidate
    raise RuntimeError("agency-detect repo root not found")


_REPO_ROOT = _repo_root()
CACHE_DIR = _REPO_ROOT / "data" / "worm"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"

BEHAVIOR_FEATURES = ("velocity", "angular_velocity", "head_angle", "pumping")


@dataclass(frozen=True)
class WormDataset:
    """Normalized single-animal recording.

    activity / activity_original are (T, N): T frames × N neurons. ``neuron_class[i]`` is
    the NeuroPAL class for neuron i (None if unlabeled). The label-dict key in the raw
    bundle is the 0-based neuron index into the trace/encoding arrays (verified: keys all
    lie in 0..N-1); ``roi_id[i]`` keeps the upstream segmentation ROI id(s).
    """

    dataset_id: str
    animal_id: str
    time: np.ndarray                  # (T,)
    activity: np.ndarray              # (T, N) provided z-scored traces
    activity_original: np.ndarray     # (T, N) original F/F0
    neuron_class: List[Optional[str]] # length N (e.g. "AVD")
    roi_id: List[object]              # length N — segmentation ROI id(s); links to positions
    behavior: Dict[str, np.ndarray]   # feature -> (T,)
    reversal_events: List[object]
    encoding: Dict[str, object]       # CePNEM summaries (eval only)
    provenance: Dict[str, object]     # checksums + sha256 + fetch info
    # Cross-dataset linking keys (per neuron index, length N):
    # neuron_label is the canonical NeuroPAL identity (e.g. "AVDL", L/R resolved) — the key
    # to join to other datasets of the *same animal* (positions, connectome) on (uid, label).
    # neuron_label_info keeps the full raw label entry (label/neuron_class/LR/DV/confidence/roi_id).
    neuron_label: List[Optional[str]] = field(default_factory=list)
    neuron_label_info: List[Optional[dict]] = field(default_factory=list)

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[1]

    @property
    def n_frames(self) -> int:
        return self.activity.shape[0]

    @property
    def labeled_index(self) -> List[int]:
        return [i for i, c in enumerate(self.neuron_class) if c]


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (trusted host)
        return resp.read()


def fetch_bundle(
    dataset_id: str,
    *,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> tuple[dict, str, Path]:
    """Return (bundle dict, sha256 of raw archive, cache path). Caches the raw bz2."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{dataset_id}.json.bz2"
    if force or not raw_path.exists():
        raw = _download(f"{BASE_URL}/download/{dataset_id}/")
        raw_path.write_bytes(raw)
    raw = raw_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    bundle = json.loads(bz2.decompress(raw))
    return bundle, sha256, raw_path


def bundle_to_dataset(bundle: dict, *, sha256: str = "", fetched_at: str = "") -> WormDataset:
    """Pure normalization of a raw bundle dict → WormDataset (no network; testable)."""
    meta = bundle["metadata"]
    gcamp = bundle["gcamp"]
    n_neuron = int(meta["n_neuron"])

    activity = np.asarray(gcamp["trace_array"], dtype=np.float64).T          # (T, N)
    activity_original = np.asarray(gcamp["trace_array_original"], dtype=np.float64).T
    if activity.shape[1] != n_neuron:
        raise ValueError(f"trace count {activity.shape[1]} != n_neuron {n_neuron}")

    labels = bundle.get("label", {})
    neuron_class: List[Optional[str]] = [None] * n_neuron
    roi_id: List[object] = [None] * n_neuron
    neuron_label: List[Optional[str]] = [None] * n_neuron
    neuron_label_info: List[Optional[dict]] = [None] * n_neuron
    for key, entry in labels.items():
        idx = int(key)
        if 0 <= idx < n_neuron:
            neuron_class[idx] = entry.get("neuron_class")
            roi_id[idx] = entry.get("roi_id")
            neuron_label[idx] = entry.get("label")
            neuron_label_info[idx] = entry

    behavior_raw = bundle.get("behavior", {})
    behavior = {
        feat: np.asarray(behavior_raw[feat], dtype=np.float64)
        for feat in BEHAVIOR_FEATURES
        if feat in behavior_raw
    }

    timing = bundle.get("timing", {})
    time = np.asarray(timing.get("timestamp_confocal", []), dtype=np.float64)

    provenance = {
        "dataset_id": f"{meta.get('paper_id')}-{meta.get('uid')}",
        "uid": meta.get("uid"),
        "paper_id": meta.get("paper_id"),
        "n_neuron": n_neuron,
        "n_labeled": sum(1 for c in neuron_class if c),
        "dataset_type": meta.get("dataset_type"),
        "source_filename": meta.get("source_filename"),
        "upstream_checksums": {k: v for k, v in meta.items() if k.startswith(("checksum_", "blake3_"))},
        "archive_sha256": sha256,
        "fetched_at": fetched_at,
        "mean_timestep": timing.get("mean_timestep"),
        "max_t": timing.get("max_t"),
    }

    return WormDataset(
        dataset_id=f"{meta.get('paper_id')}-{meta.get('uid')}",
        animal_id=str(meta.get("uid")),
        time=time,
        activity=activity,
        activity_original=activity_original,
        neuron_class=neuron_class,
        roi_id=roi_id,
        behavior=behavior,
        reversal_events=list(behavior_raw.get("reversal_events", [])),
        encoding=bundle.get("encoding", {}),
        provenance=provenance,
        neuron_label=neuron_label,
        neuron_label_info=neuron_label_info,
    )


def validate_dataset(ds: WormDataset) -> None:
    """Raise if the normalized dataset is internally inconsistent."""
    T, N = ds.activity.shape
    if ds.activity_original.shape != (T, N):
        raise ValueError("activity_original shape mismatch")
    if len(ds.neuron_class) != N or len(ds.roi_id) != N:
        raise ValueError("neuron_class / roi_id length must equal N")
    for name, lst in (("neuron_label", ds.neuron_label), ("neuron_label_info", ds.neuron_label_info)):
        if lst and len(lst) != N:
            raise ValueError(f"{name} length must equal N")
    if ds.time.size and ds.time.size != T:
        raise ValueError(f"time length {ds.time.size} != T {T}")
    for feat, arr in ds.behavior.items():
        if arr.shape[0] != T:
            raise ValueError(f"behavior[{feat}] length {arr.shape[0]} != T {T}")
    if not np.all(np.isfinite(ds.activity)):
        raise ValueError("activity contains non-finite values")


def write_manifest(ds: WormDataset, *, manifest_dir: Path = MANIFEST_DIR) -> Path:
    """Write a small tracked provenance manifest (json under uad_worm/, not gitignored)."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{ds.dataset_id}.json"
    path.write_text(json.dumps(ds.provenance, indent=2, sort_keys=True))
    return path


def load_dataset(
    dataset_id: str,
    *,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    write_provenance: bool = True,
) -> WormDataset:
    """Fetch (or load from cache), normalize, validate, and record provenance."""
    bundle, sha256, _ = fetch_bundle(dataset_id, cache_dir=cache_dir, force=force)
    ds = bundle_to_dataset(
        bundle, sha256=sha256, fetched_at=datetime.now(timezone.utc).isoformat()
    )
    validate_dataset(ds)
    if write_provenance:
        write_manifest(ds)
    return ds


def list_datasets(*, cache_dir: Path = CACHE_DIR, force: bool = False) -> List[dict]:
    """Dataset index from the API (deduplicated by dataset_id), cached locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "datasets_index.json"
    if force or not path.exists():
        path.write_bytes(_download(f"{BASE_URL}/datasets/"))
    records = json.loads(path.read_text())
    seen: Dict[str, dict] = {}
    for r in records:
        seen.setdefault(r["dataset_id"], r)
    return list(seen.values())


def is_neuropal_baseline(ds: WormDataset) -> bool:
    """True if the bundle's metadata.dataset_type marks it NeuroPAL + baseline.

    Verified shape: ``dataset_type`` is a list like ``["baseline", "neuropal"]``; Heat
    runs carry ``"heat"`` instead. This is the v1 cohort filter (README §5 M1).
    """
    types = ds.provenance.get("dataset_type") or []
    types = [str(t).lower() for t in (types if isinstance(types, list) else [types])]
    return "baseline" in types and "neuropal" in types and ds.provenance.get("n_labeled", 0) > 0


def neuropal_labeled_ids(*, min_labeled: int = 1, **kwargs) -> List[str]:
    """Dataset ids with NeuroPAL labels (n_labeled >= min_labeled) — the v1 cohort seed.

    Baseline-vs-Heat refinement requires the bundle's metadata.dataset_type and is applied
    after load (README §5 M1: start with the NeuroPAL Baseline cohort).
    """
    return [r["dataset_id"] for r in list_datasets(**kwargs) if r.get("n_labeled", 0) >= min_labeled]
