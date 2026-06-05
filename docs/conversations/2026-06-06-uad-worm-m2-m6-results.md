# Conversation Summary: uad_worm (E20) — M2–M6 + first real-data result

Date: 2026-06-06

Continues [E20 planning + M0/M1](2026-06-05-06-uad-worm-celegans.md). Built the discovery
pipeline M2–M6, ran it on a real NeuroPAL-Baseline cohort, and probed the (negative) result.

## What shipped

- **M2 `preprocess.py`** — resample to uniform clock → per-animal z-score → temporal-derivative
  whitening; behavior resampled to the same clock. Test: whitening cuts AR(0.9) lag-1
  autocorrelation ~0.9 → <0.3.
- **M3 `candidates.py`** — primary generator = lagged-correlation agglomerative communities
  (mapped to `neuron_class` sets); plus command-circuit anchor seed and random class-set
  controls. Other clusterings remain on the deferred checklist.
- **M4 `score.py`** — operational S/A/I roles (by lagged influence to/from the rest of the
  brain), external set reduced to PCs (otherwise the CMI is singular at T≈1600), blanket loss
  vs a per-animal random-partition contrast, pooled across animals with leave-one-animal-out.
- **M5/M6 `evaluate.py`** — random-class-set null (the valid additional blanket null) and
  behavior-prediction gain (CV-R²).
- **Runner/probe** — `scripts/worm/run_discovery.py`, `scripts/worm/probe.py`. 25 worm tests green.

## Two corrections made along the way

1. **Plan M5 was wrong.** "Circular-shift → blanket null" is invalid: destroying cross-neuron
   timing drives blanket loss toward 0 and would reject true blankets. Circular/phase nulls are
   only valid for the *higher-is-better* memory/seed stats (M7). The correct additional blanket
   null is the **random-class-set** null. README updated.
2. **Synthetic validation insight (now in `FINDINGS.md`).** The random-partition contrast only
   discriminates when there is a **densely-coupled background**: an isolated noise variable
   scores as blanket-like as a true agent (low CMI ≠ agency). The M4 test was rebuilt to embed
   the agent in coupled background, where it separates cleanly (agent p=0.007 vs coupled-decoy
   p=0.95). Implication: blanket score needs a predictive-coupling companion term.

## Headline result (8 NeuroPAL-Baseline animals, whitened, n_perm=100)

Command-circuit anchor (`AIB AVA AVB AVD AVE PVC RIB RIM`):
- random-partition contrast **pass_rate 0/8**, **leave-one-animal-out 0/8**, combined p≈0.36
- random-class-set null z=−1.35, p≈0.10 (marginally below other *class sets*)
- behavior (velocity) gain +0.013

Unsupervised recurrent candidate: worse on the blanket (z=+2.56), gain +0.065.

## Probe — why 0/8 (`scripts/worm/probe.py`)

Ruled out the obvious knobs; the negative is robust:

| Probe | Result | Verdict |
|-------|--------|---------|
| P1 null reference (all-neuron vs labeled-only) | median p 0.41 vs 0.37 | not a null-reference artifact |
| P2 representation (whitened vs raw) | raw median p 0.58 | whitened better, still 0/8 |
| P3 external rank (ext_dim 4→20) | pass 0→1/8, median p ≈0.5 | sharpens slightly, not decisive |

The command circuit's blanket loss sits **in the middle** of the random same-size partition
distribution (median p≈0.5) everywhere ⇒ genuine negative: at the class level, lag-1,
Gaussian-CMI, PC-reduced, the command circuit is not a distinguishable Markov blanket in this
cohort. (Note: this dataset is **not** added to the general agent-detection pool.)

## Next directions (not started)

Multi-lag / slower timescale conditioning · nonlinear CMI · per-animal (not class-pooled)
discovery · add a predictive-coupling term · M7 memory localization.
