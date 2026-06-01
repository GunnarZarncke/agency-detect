# amortized_agency

Pooled (amortized) agency detection: train a same-agent affinity model across many
simulated worlds, then apply it to new traces without per-agent relearning.

All methods emit a permutation-invariant `N×N` affinity matrix → identical
downstream agglomerative clustering → comparable ARI / Jaccard.

## Layout

| Module | Role |
|--------|------|
| `kinds.py` | Agent kind definitions; train vs held-out split |
| `worlds.py` | Simulator episodes (agent variables only) |
| `siamese.py` | Pairwise Siamese baseline (per-channel encoder) |
| `slot_model.py` | Slot attention + `CrossChannelEncoder` (documented negative result for the slot readout) |
| `context_model.py` | Cross-channel encoder → direct pairwise affinity (best learned method) |
| `cluster.py` | Affinity → labels; MI baseline wrapper |
| `metrics.py` | ARI, mean best-Jaccard |
| `evaluate.py` | Cross-method evaluation on kinds / windows |

## Scripts

```bash
# MI breaking-point baseline (no learning)
.venv/bin/python scripts/amortized/baseline_window_breaking_point.py

# Train Siamese + slot on pool; evaluate vs MI (held-out kinds)
.venv/bin/python scripts/amortized/run_pooled_experiment.py --device cpu
```

# Train long (500–1000 steps), detect short (250/125/60) — default after E13c
.venv/bin/python scripts/amortized/run_pooled_experiment.py --device cpu \
  --train-windows 500,1000 --slot-epochs 20
```

Results: `results/amortized/`

## Design notes

- **Train kinds:** `easy3_redundant`, `med5_rich`
- **Held-out kind:** `hard8_complex` (transfer test)
- **Train long / detect short:** pool uses long windows; encoders accept variable W at inference.
- **Same-agent membership is relational** (cross-channel correlation). A per-channel,
  time-pooled encoder is blind to it; a cross-channel encoder is required.
- **Slot readout is a dead end here:** routing N variables through K competing slots
  cannot express membership (BCE floors at ln 2 / chance). The winning model
  (`context_model.py`) uses a direct pairwise Gram affinity instead of slots.
- **Current standing:** context model is the best *learned* method and generalizes
  across kinds/windows; MI still leads the short-window band on held-out complex agents.
