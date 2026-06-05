"""M1: ingestion schema + provenance, validated offline on a synthetic bundle."""

import bz2
import hashlib
import json

import numpy as np
import pytest

from uad_worm.data import bundle_to_dataset, validate_dataset, write_manifest


def _synthetic_bundle(T: int = 50):
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((3, T))  # [neuron][time]
    return {
        "metadata": {
            "n_neuron": 3,
            "paper_id": "test_paper",
            "uid": "2099-01-01-01",
            "dataset_type": ["baseline"],
            "source_filename": "x.h5",
            "checksum_h5": "deadbeef",
            "blake3_fit_results": "cafef00d",
        },
        "gcamp": {
            "trace_array": traces.tolist(),
            "trace_array_original": (traces + 1.0).tolist(),
        },
        "label": {
            "0": {"neuron_class": "AVA", "roi_id": [10]},
            "2": {"neuron_class": "RIM", "roi_id": [12]},
        },
        "behavior": {
            "velocity": rng.standard_normal(T).tolist(),
            "angular_velocity": rng.standard_normal(T).tolist(),
            "head_angle": rng.standard_normal(T).tolist(),
            "pumping": rng.standard_normal(T).tolist(),
            "reversal_events": [[1, 2]],
        },
        "timing": {
            "timestamp_confocal": np.linspace(0, T * 0.6, T).tolist(),
            "mean_timestep": 0.6,
            "max_t": T,
        },
        "encoding": {"forwardness": [0.1, 0.2, 0.3]},
    }


def test_normalization_schema():
    ds = bundle_to_dataset(_synthetic_bundle(), sha256="abc123", fetched_at="t0")
    assert ds.activity.shape == (50, 3)         # (T, N), transposed from [N][T]
    assert ds.activity_original.shape == (50, 3)
    assert ds.neuron_class == ["AVA", None, "RIM"]
    assert ds.labeled_index == [0, 2]
    assert ds.dataset_id == "test_paper-2099-01-01-01"
    assert ds.behavior["velocity"].shape == (50,)
    validate_dataset(ds)  # must not raise


def test_provenance_preserves_upstream_checksums():
    ds = bundle_to_dataset(_synthetic_bundle(), sha256="abc123", fetched_at="t0")
    up = ds.provenance["upstream_checksums"]
    assert up["checksum_h5"] == "deadbeef"
    assert up["blake3_fit_results"] == "cafef00d"
    assert ds.provenance["archive_sha256"] == "abc123"
    assert ds.provenance["n_labeled"] == 2


def test_manifest_round_trip(tmp_path):
    ds = bundle_to_dataset(_synthetic_bundle(), sha256="abc123", fetched_at="t0")
    path = write_manifest(ds, manifest_dir=tmp_path)
    assert json.loads(path.read_text()) == ds.provenance


def test_archive_sha256_is_stable():
    raw = bz2.compress(json.dumps(_synthetic_bundle()).encode())
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(raw).hexdigest()


def test_validate_catches_bad_length():
    ds = bundle_to_dataset(_synthetic_bundle(), sha256="x", fetched_at="t0")
    bad = ds.__class__(**{**ds.__dict__, "neuron_class": ["AVA"]})  # wrong length
    with pytest.raises(ValueError):
        validate_dataset(bad)
