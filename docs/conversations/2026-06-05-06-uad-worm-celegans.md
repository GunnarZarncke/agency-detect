# Conversation Summary: UAD on C. elegans (uad_worm, E20) — planning + M0/M1

Date: 2026-06-05 / 2026-06-06

New experiment line **E20**: run Unsupervised Agent Discovery on real *C. elegans*
whole-brain calcium imaging from **WormWideWeb** (Atanas & Kim 2023). Scoped plan +
working M0/M1 in `uad_worm/`. Detailed plan: [`uad_worm/README.md`](../../uad_worm/README.md).

## Initial problem

An ambitious brief asked for a full `uad_worm` package (ingest → discover → eval → report)
on worm data. Open questions: is the data actually available, what's realistic in this
repo's scope, and what's scientifically defensible given no ground-truth agents.

## Key decisions

- **Reuse, don't rebuild.** Worm ingestion targets the existing `SimulationResult` /
  Markov-blanket machinery rather than the brief's standalone `src/uad_worm/` tree. Folder
  layout reconciled with repo conventions (flat package, `scripts/`, `results/`, top-level
  `tests/`).
- **Ground truth is not a blocker.** This is an *application*: M0 synthetic benchmarks
  anchor estimator correctness; the data carries enough structure to interpret S/A/I roles.
- **The v1 bet:** single-animal `T≈1600` + slow calcium gives tiny `N_eff`, so per-animal
  CMI is underpowered. Pull three levers at once — (1) pool across NeuroPAL-Baseline
  animals at the `neuron_class` level, (2) whiten (derivative/deconvolution), (3) score
  relative to nulls — with **leave-one-animal-out generalization** as the headline.
- **Do-one-thing-first** (per AGENTS.md): M3 (clustering) and M5 (extra nulls) each start
  with a single most-likely option + a checklist of alternatives to add only on evidence.
  **Memory localization deferred to M7**; M6 eval starts with the simplest plausibility
  checks.
- **Autocorrelation correction (important reasoning):** the blanket check is *not* fooled
  by autocorrelation — a shared slow drive still injects a shared innovation each step, so
  the residual CMI stays positive and random cuts are rejected (synthetic system 2 proves
  this). Autocorrelation actually bites via **estimator power** (`N_eff`) and the
  **auxiliary** memory/Granger-seed scores, not the criterion. Fix: null-relative scoring
  (random-partition for the blanket, AR-matched for memory/seeds) + whitening, not a fixed
  ε.
- **`.gitignore` fixed:** `uad_worm/**/*.{json,yaml}` tracked (manifests/configs/
  provenance); `data/worm/` + `results/worm/` stay ignored.

## Data availability (verified live)

WormWideWeb public API is live and downloadable. 91 datasets, **56 NeuroPAL-labeled**,
median ~140 neurons, T mostly 1600 @ ~0.6 s (~16 min). Per-dataset download is a
**bzip2-JSON** bundle (`gcamp`/`behavior`/`encoding`/`label`/`metadata`/`timing`) with
upstream checksums present. Label-dict key = 0-based neuron index into trace/encoding
arrays. `dataset_type` is machine-readable (e.g. `["baseline","neuropal"]`). Timestamps
are monotonic with only ~2.5% jitter → linear interpolation suffices.

## Progression

**M0 — core estimator + synthetic validation (offline, green).** `uad_worm/{cmi,blanket,
nulls,synth}.py`: Gaussian partial-correlation CMI; blanket loss with the **random-
partition contrast** (the right arbiter for the lower-is-better blanket); circular-shift +
spectrum-exact phase nulls; three synthetic systems with **named, commented constants**
(spectral radii 0.755 / 0.954 cited). Validated: CMI detects conditional independence; the
true blanket beats random partitions (p≈0.015); strong-correlation-no-blanket is rejected
(p>0.1). A first unstable controller (loop gain >1) exploded into collinearity — retuned to
stable gains.

**M1 — ingestion (tests + live smoke).** `uad_worm/data.py`: fetch (browser UA) → cache
raw bz2 under `data/worm/` (ignored) → normalize to `WormDataset` → validate → tracked
provenance manifest under `uad_worm/manifests/`. Live smoke on `atanas_kim_2023-2023-01-23-01`:
1600×151, 91 labeled, command-circuit classes (AVA/AVE/AVD/AIB) present, upstream
checksums + our `archive_sha256` recorded.

## Current state

- `uad_worm/` package with M0 + M1; **12 worm tests green**, no lint errors.
- Plan in `uad_worm/README.md` (milestones M0–M8, v1 bet, blockers §7).
- E20 not yet logged in `EXPERIMENTS.md` (no real-data results yet — only synthetic + one
  ingestion smoke).

## Follow-up ideas

1. **M2 — preprocess + whiten:** resample to uniform 0.6 s grid, per-animal z-score,
   temporal-derivative whitened primary (keep raw), behavior align; test that whitening
   cuts lag-1 autocorrelation.
2. **M3/M4 — the bet:** lagged-corr class-level candidates + command-circuit anchor seed +
   random class controls; per-animal blanket loss → pool → leave-one-animal-out.
3. Log E20 in `EXPERIMENTS.md` once the first pooled real-data result exists.
