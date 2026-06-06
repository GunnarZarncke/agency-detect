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

## Post-M6 exploration (same day)

Continued the exploration to explain/escape the negative (`scripts/worm/explore.py`):

- **Timescale ruled out** — lag sweep 1→5 (0.6→3.0 s) leaves the anchor at median p≈0.36–0.43.
- **Added an internal-autonomy axis** `I(C_{t+1};C_t | E_t)` (self-prediction beyond the
  environment) to tell a real coupled subsystem from disconnected low-loss noise. **Correction:**
  a naive self-prediction R² is invalid — it rewards redundancy (a shared-latent block beats a
  true controller on synthetic); conditioning on the environment fixes it (agent ≈0.70 vs block
  ≈0.01). Now in `FINDINGS.md` #4.
- **(autonomy, loss) plane:** command circuit lands in the agent corner in only 3/8 animals
  (median autonomy below chance); per-animal lagged-corr communities are coupled but leaky
  (only 1/4 borderline). **Coupling and encapsulation are anti-located** in this cohort, so the
  agent corner is sparse/weak ⇒ structural negative, not a tuning/timescale artifact.
- Also added the **S/A/I assignment exporter** with cross-dataset linking keys (canonical
  NeuroPAL `label`, `roi_id`→positions, `uid`).

27 worm tests green. Remaining directions: nonlinear CMI, stricter agent-corner thresholds +
per-animal stability across seeds, larger / Heat-vs-Baseline cohort, M7 memory.

## Connectome plausibility + nonlinear robustness (same day)

- **Connectome plausibility (manual).** Checked the 4 recovered S/A/I hits against canonical
  command-circuit wiring (White 1986 / Cook 2019 — qualitative, not a programmatic edge-join;
  the WWW `get-edges` endpoint refuses plain requests). `2023-01-09-28` (AVD→AVA/AVE→RIM) is
  strongly consistent; output (RIM→action) and sensor (ASE/AVD) roles recover reliably;
  AVA/AVE land in *internal*; fails without a true input neuron (AVE-as-sensor) or with a
  motor neuron tagged internal (RMD). Documented in `EXPERIMENTS.md` §E20; re-check after each
  new experiment.
- **Nonlinear robustness (copula CMI).** Added a normal-scores (Gaussian-copula) representation
  and re-ran the headline scorers. Negative is robust to monotone nonlinearity (LOAO 0/8;
  pooled combined_p 0.37→0.88). Copula is a stricter filter (agent-corner 3/8→1/8), keeping
  only `2023-01-09-28` — the same strong + connectome-plausible hit. Non-monotone (kNN/kernel)
  CMI still open. Memory deferred per request. 29 worm tests green.

## kNN/KSG CMI + larger cohort (same day)

- **User direction:** drop the interim copula path to keep it simple; try kNN and a larger
  cohort instead. Removed `normal_scores`/`whitened_copula`/`explore_nonlinear.py`.
- **Added `uad_worm.cmi.knn_cmi`** — KSG / Frenzel–Pompe nonparametric CMI (captures
  non-monotone dependence; verified it catches Y=X² where Gaussian CMI is ~0). Threaded an
  `estimator="knn"` path through `blanket_loss_for_members`/`score_members` with low-dim PC
  reduction (ext_dim=int_dim=3) since KSG degrades in high dimension.
- **Larger cohort = 20 NeuroPAL-Baseline animals** (`scripts/worm/explore_knn.py`). Result is a
  **robust negative on both axes**: Gaussian anchor pass/LOAO at chance (1/20 ≈ α), kNN 0/20
  (slightly stronger negative). No new agent-corner hits ⇒ manual connectome recheck vacuous.
  The earlier `2023-01-09-28` command-circuit recovery stays the only non-generalizing positive.
- 30 worm tests green. (Unrelated pre-existing failure: `test_spotlight_agency_gate`.)
