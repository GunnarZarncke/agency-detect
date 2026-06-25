# amortized_agency

Pooled (amortized) agency detection: train a same-agent affinity model across many
simulated worlds, then apply it to new traces without per-agent relearning.

All methods emit a permutation-invariant `N×N` affinity matrix → identical
downstream agglomerative clustering → comparable ARI / Jaccard.

## Layout

| Path | Role |
|------|------|
| `src/amortized_agency/` | Package source |
| `tests/` | Pytest suite |
| `scripts/` | Command-line runners |

## Scripts

```bash
# Main loop: context scale × train worlds × epochs (no MI)
.venv/bin/python amortized_agency/scripts/run_learned_sweep.py --device cpu \
  --scales base,large,xl --train-worlds 40,80 --context-epochs 40,60

# Reference eval (learned only; frozen MI gaps) — add --run-mi to re-check live MI
.venv/bin/python amortized_agency/scripts/run_reference_benchmark.py --device cpu

# Extended pool wiring check (builds episodes only; no training)
.venv/bin/python amortized_agency/scripts/check_extended_pool.py

# Extended pool benchmark (when ready; not part of default E13 protocol yet)
# .venv/bin/python amortized_agency/scripts/run_reference_benchmark.py --device cpu --extended-pool

# One-off MI ceiling curve (slow)
.venv/bin/python amortized_agency/scripts/baseline_window_breaking_point.py

# Pooled train + eval (learned only by default)
.venv/bin/python amortized_agency/scripts/run_pooled_experiment.py --device cpu \
  --train-windows 500,1000 --context-epochs 40
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
