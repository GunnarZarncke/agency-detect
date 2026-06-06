# uad_worm — Unsupervised Agent Discovery on C. elegans whole-brain data

**Status:** M0–M6 implemented, run, explored, and **wrapped up** (E20). Accepted result is a
**mostly negative with one consistent positive** — see below. Full writeup:
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) §E20.

## Results (v1, 2026-06-06) — final

8 NeuroPAL-Baseline animals, whitened, class-level pooling. **Command-circuit anchor
(`AIB AVA AVB AVD AVE PVC RIB RIM`): leave-one-animal-out 0/8** (combined p≈0.36). Marginally
below random *class sets* (z=−1.35, p≈0.10) but middle-of-the-pack vs random same-size *neuron*
partitions (median p≈0.5). Unsupervised recurrent candidate worse (z=+2.56). Behavior gain small.

`scripts/worm/probe.py` rules out the obvious causes: null reference (labeled-only ≡ all-neuron),
representation (whitened beats raw, still 0/8), external rank (ext_dim 4→20 barely moves it). The
exploration (`scripts/worm/explore.py`) further shows coupling and encapsulation are
*anti-located* ⇒ a **structural** negative, not a tuning/timescale artifact.

**The one positive:** `2023-01-09-28` (AVD→AVA/AVE→RIM) is the statistically strongest hit *and*
connectome-plausible (input→recurrent-core→output matches canonical wiring; see
`results/worm/recovered_assignments.json`).

**Interpretation (accepted):** the broad negative is a **power / temporal-resolution /
signal-quality** limit — short, slow-GCaMP recordings collected for sensory/behavior *encoding*
studies, not for blanket discovery; the dataset was never intended for this algorithm. That a
faithful pipeline still recovers one coherent, biologically sensible agent boundary is a modest
but encouraging signal. (This dataset is not added to the general agent-detection pool.)

**Out-of-scope future work** (explored briefly during development, then removed to keep the folder
simple): nonparametric kNN/KSG CMI and a larger / Heat-vs-Baseline cohort — both reproduced the
negative; per-animal (not pooled) discovery; and M7 memory localization.

This folder is a **variant** of the existing UAD work (`agency_detect/`,
`learn_agents/`, `agent_spotlight/`) pointed at **real neural data**: freely-moving
whole-brain calcium imaging of *C. elegans* from **WormWideWeb** (Atanas & Kim et
al., 2023). Every prior E-line ran on simulators or a synthetic machine dataset
(E19). E20 is the first attempt to run the discovery criterion on a biological system
where there is **no ground-truth agent**, so the bar is operational: *beat strong
nulls and simple baselines*, not "prove the worm brain is an agent".

The brief this folder implements (verbatim user request) is archived as the design
target; this README is the **scoped, project-realistic** version of it.

---

## 1. Data-source availability (verified live, 2026-06-05)

All checks below were run against the public WormWideWeb API. **Data is available and
downloadable** — this is *not* a blocker.

| Resource | Endpoint | Status |
|----------|----------|--------|
| Dataset index | `GET /activity/api/data/datasets/` | ✅ 200, JSON list |
| Per-dataset bundle | `GET /activity/api/data/download/<dataset_id>/` | ✅ 200, bzip2-JSON (~3.9 MB) |
| Behavior only | `GET /activity/api/data/<dataset_id>/behavior/` | ✅ 200, JSON |
| CePNEM encoding (bulk) | `GET /activity/api/data/atanas_kim_2023_encoding/` | ✅ 200, JSON (~530 KB) |
| Connectome edges | `POST /connectome/api/get-edges/`, `GET /connectome/api/available-neurons/?datasets=<id>` | ✅ live (needs `datasets` param + browser UA) |

**Quantities (Atanas & Kim 2023 paper, the canonical set):**

- **91** dataset records, **56 NeuroPAL-labeled** (identified neurons) — the brief's
  preferred substrate for cross-animal comparison.
- Median **~140 neurons/animal**, **~93 labeled** in labeled sets.
- `max_t` mostly **1600 frames** (range 200–1615), **~0.6 s/frame** ⇒ ~16 min/recording.

**Access gotchas (confirmed):**
- API rejects bare `urllib` (HTTP 403) — **send a browser `User-Agent`** (curl works).
- The "download" is a **single bzip2-compressed JSON**, *not* a tar of HDF5. Top-level
  keys: `behavior`, `encoding`, `gcamp`, `label`, `metadata`, `timing` (schema in §4).
- Provenance checksums **are present** in `metadata` (`checksum_h5`, `blake3_*`), so the
  brief's "preserve checksums" requirement is satisfiable directly from the bundle.
- Connectome is keyed by **neuron_class** (e.g. `AVAL`), not by per-recording ROI id;
  joining requires the NeuroPAL `label` map and only covers the ~60% labeled neurons.

Raw curl recipe (for the ingestion module):

```bash
curl -sL -A "Mozilla/5.0" \
  "https://wormwideweb.org/activity/api/data/download/atanas_kim_2023-2023-01-23-01/" \
  | bunzip2 > 2023-01-23-01.json
```

---

## 2. What "works in scope of this project"

The single most important design decision: **reuse the existing UAD machinery instead
of building the brief's full `src/uad_worm/` mega-package.** Two reuse hooks already
exist and define the integration surface:

1. **`learn_agents.external_traces.pack_trace` / `SimulationResult`** — the canonical
   in-repo trace container (`trace[T, V]` + `metadata` with `role_indices`,
   `agent_clusters`, `var_names`). E14–E16 already feed *external* simulators through it.
   **Worm ingestion should emit this same object**, so MI clustering, oracle UAD
   scoring, and `agent_spotlight` run unchanged.
2. **`agency_detect.markov_blanket`** — already defines the blanket loss
   `I(I_{t+1}; E_{t+1} | S_t, A_t)` and S/A/I classification. We keep the *structure*
   but **must add a Gaussian/partial-correlation CMI estimator** (the existing discrete
   Laplace plug-in cannot handle continuous calcium at T≈1600; see §7 blockers).

So E20 is a **thin worm-specific front end** (fetch → schema → preprocess → candidate
seeds → blanket scoring → nulls → eval overlays) on top of the existing core, **not** a
parallel reimplementation.

### Folder layout (reconciled with repo conventions)

The brief asks for `src/uad_worm/{data,preprocess,discovery,eval,viz,cli}` +
`configs/notebooks/reports/tests`. This repo uses **flat top-level packages**
(`intention_detect/`, `amortized_agency/`), tests in top-level `tests/`, runnable
scripts in `scripts/<family>/`, and artifacts in `results/<family>/` (gitignored).
Mapped:

| Brief | This repo |
|-------|-----------|
| `src/uad_worm/data/` | `uad_worm/data.py` (fetch + schema + checksums) |
| `src/uad_worm/preprocess/` | `uad_worm/preprocess.py` |
| `src/uad_worm/discovery/` | `uad_worm/candidates.py`, `uad_worm/blanket.py` (Gaussian CMI), `uad_worm/memory.py` |
| `src/uad_worm/eval/` | `uad_worm/nulls.py`, `uad_worm/evaluate.py` (connectome/CePNEM/baselines) |
| `src/uad_worm/viz/` | `uad_worm/viz.py` |
| `src/uad_worm/cli/` | `uad_worm/cli.py` (or `scripts/worm/*.py`) |
| `configs/` | `uad_worm/configs/*.yaml` |
| `notebooks/` | `notebooks/worm_uad_e20.ipynb` |
| `reports/` + `tests/` | `results/worm/` (gitignored) + top-level `tests/test_worm_*.py` |

> Provenance / .gitignore (resolved): the root `.gitignore` globally ignores `*.json`,
> `*.h5`, `*.csv`, and `results/`. It now carries `uad_worm`-scoped negations so
> **`*.json`/`*.yaml` under `uad_worm/` are tracked** (manifests, configs, provenance),
> while **raw dataset caches (`data/worm/`) and run artifacts (`results/worm/`) stay
> ignored**. So: small tracked provenance manifests live in `uad_worm/`; bulk downloads
> and outputs do not get committed.

---

## 3. Scientific framing (operational, scoped)

Discovery objective (unchanged from the project's core), per candidate neuron subset C:

```
L_blanket(C) = I(I^C_{t+1}; E^C_{t+1} | S^C_t, A^C_t)      # minimize
```

with `E^C` = all neurons **not** in C (optionally + behavior channels as extra external
observables, never as part of C). Roles within C assigned operationally:

- **S** = C-neurons most driven by `E_t` (outside→inside lagged prediction),
- **A** = C-neurons most driving `E_{t+1}` (inside→outside lagged prediction),
- **I** = remaining C-neurons retaining within-C state across time.

Memory localization (a primary output, not optional):

```
Δ_m(k) = I(m_{t-k}; I^C_{t+1} | S^C_t, A^C_t, I^C_t \ {m_t})
```

B-IQ-style predictive/control information is a **descriptive score on accepted
subsystems**, never the discovery objective. Proto-goal / inter-agent modeling are
**v1.5 / v2**, explicitly deferred.

Selection score (not blanket loss alone — avoids the trivial giant-cluster solution):

```
Score(C) = -Î(I;E|S,A) + λ_stab·Stability(C) + λ_pred·PredGain(C) − λ_size·Penalty(C)
```

---

## 4. Bundle schema → internal schema

Per-dataset bundle (confirmed by inspection of `atanas_kim_2023-2023-01-23-01`):

| Key | Contents |
|-----|----------|
| `gcamp.trace_array` | `[n_neuron][T]` z-scored GCaMP traces |
| `gcamp.trace_array_original` | `[n_neuron][T]` original F/F0 (keep as canonical continuous signal) |
| `behavior` | `velocity`, `angular_velocity`, `head_angle`, `pumping` (`[T]`), `reversal_events` |
| `label` | per ROI id → `{label, neuron_class, confidence, DV, LR, region, roi_id}` (NeuroPAL) |
| `encoding` | CePNEM per-neuron: `forwardness`, `dorsalness`, `feedingness`, `rel_enc_str_{v,θh,P}`, `tau_vals`, `neuron_categorization`, `encoding_changing_neurons` |
| `metadata` | `uid`, `paper_id`, `n_neuron`, `dataset_type`, `source_filename`, `checksum_h5`, `blake3_*` |
| `timing` | `max_t`, `mean_timestep`, `timestamp_confocal[T]` |

Internal schema (the brief's minimum, as an `xarray.Dataset` + sidecar provenance yaml):
`dataset_id`, `animal_id`, `time`, `neuron_ids`, `neuron_class` (label), `activity[t,neuron]`
(3 representations, §5), `behavior[t,feature]`, `connectome_edges[src,tgt,weight,type]`,
`masks` (label confidence / NaN), `metadata` (incl. upstream checksums + our sha256).

---

## 5. Pipeline stages & milestones (goal-driven)

### Recommended v1 bet

Single-animal `T≈1600` + slow calcium gives a tiny effective sample size, so the (sound)
blanket test is underpowered per animal. The v1 bet pulls the only three levers that buy
power back **at once**: **(1) pool across the NeuroPAL-Baseline animals at the
`neuron_class` level**, **(2) whiten** (derivative/deconvolution), **(3) score relative
to nulls** instead of an absolute ε. The headline result is then **leave-one-animal-out
generalization** (a class-level subsystem that stays below its own null in held-out
animals), which is far more defensible than one low CMI on one worm. The milestones below
are sequenced to reach that bet with the *minimum* machinery.

> **Do-one-thing-first discipline (per project AGENTS.md "Simplicity First"):** M3
> (clustering) and M5 (extra nulls) each have many options. **Start with the single most
> likely option, prove M4 works on it, and only add alternatives if there is evidence
> the first is insufficient.** Each keeps a checklist of deferred options.

Each stage has a **verify** gate; do not advance until it passes.

**M0 — Synthetic validation + core estimator (MANDATORY FIRST, no worm data, offline).**
`uad_worm/{cmi,blanket,nulls,synth}.py`: Gaussian partial-correlation CMI, blanket loss
`I(I_{t+1};E_{t+1}|S_t,A_t)`, the **random-partition contrast** (does *this* cut mediate
better than chance cuts of the same vars?), and the **circular-shift** autocorrelation
null. Toy systems:
1. hidden-state controller with an explicit Markov blanket (true cut is low-loss),
2. strong-correlation-but-no-blanket system (method must *reject* it),
3. two coupled subsystems with one memory node (generator now; memory assertion deferred
   with M7).
*Verify:* CMI sanity (independence→0, conditional independence detected); system 1 true
partition has low loss **and** beats random partitions; system 2 best cut does **not**
beat random partitions. → `tests/test_worm_cmi.py`, `tests/test_worm_blanket.py`.

**M1 — Ingestion.** `uad_worm/data.py`: fetch by id (browser UA), cache bzip2-JSON under
`data/worm/`, record upstream checksums + our `sha256` into a tracked
`uad_worm/manifest_*.json`, normalize to internal schema. Start with the **NeuroPAL
Baseline** cohort. *Verify:* schema validation + checksum round-trip on 1 dataset.

**M2 — Preprocess + whiten.** Resample to a uniform clock (see §7.3 — mild jitter,
**linear interpolation onto median `dt`≈0.6 s**), z-score per animal, build the **whitened
primary** (temporal derivative; deconvolved optional) and keep raw for reporting, align
behavior to the same grid. *Verify:* shapes/clock tests; whitened representation reduces
lag-1 autocorrelation vs raw.

**M3 — Candidate subsystem (ONE primary first).** `candidates.py`. Define candidates at
the **`neuron_class`** level. **Primary:** lagged-correlation graph communities (cheap,
captures directed temporal structure) **+** a biologically grounded **anchor seed** (the
locomotor command circuit, e.g. reverse AVA/AVE/AVD/RIM/AIB, forward AVB/PVC/RIB — *seed
only*, accepted/rejected by UAD) **+** random matched class sets as controls. *Verify:*
the primary yields coherent class-level candidates + the anchor + controls exist.
*Checklist — add only if M4 shows no signal on the primary:* correlation-only
communities · sparse-VAR/Granger communities · spectral on lagged graph ·
connectome-constrained communities.

**M4 — Blanket scoring + pooled holdout (the bet).** `blanket.py`: per-animal low-dim
S/A/I assignment + Gaussian CMI blanket loss with the M0 random-partition contrast; then
**pool the same class-defined candidate across animals** and run **leave-one-animal-out**
(motif stays below its null in the held-out animal?). *Verify:* on the primary candidate,
pooled loss beats random class sets and generalizes to ≥1 held-out animal; reported
result always names the estimator. *(kNN-CMI confirmation on top-k only — deferred.)*

**M5 — Extra null (ONE primary first).** **Primary: random-class-set null** — is the
candidate's pooled blanket loss below that of biologically-matched random class sets? This
is the right *additional* arbiter for the blanket (lower-is-better) beyond M4's
random-partition contrast. NOTE (correction to earlier draft): circular-shift / phase
nulls are **not** valid for the blanket loss — destroying cross-neuron timing drives the
loss toward 0, so they would reject true blankets. They are the correct arbiter only for
the *higher-is-better* memory/seed statistics, so they live in M7. *Checklist — add only
to harden a headline result:* connectome degree-preserving rewire · label permutation ·
behavior-misalignment. *Verify:* headline candidate's pooled loss beats the random-class-set
null.

**M6 — Evaluation (simplest plausibility checks first).** `evaluate.py`. **First:** does
the candidate beat **random class sets**, and is its **behavior-prediction gain**
(predict `velocity`/`reversal_events` from candidate vs from a same-size correlation
cluster) positive? Only if those pass, add richer baselines (PCA/ICA, spectral,
connectome-only, CePNEM grouping), connectome enrichment, and CePNEM overlap
(evaluation only, never supervision). *Verify:* the two simple checks emit a table first.

**M7 — Memory localization (DEFERRED — do after M4–M6 hold).** `memory.py`: `Δ_m(k)` over
lags, self-history conditioned out, **circular/block nulls as arbiter**; concentrated vs
distributed, short vs long timescale. *Verify:* significant lagged contributor on ≥1
dataset beyond the autocorrelation null.

**M8 — Report + viz.** `viz.py` + report: activity heatmap with subsystem, connectome
plot colored by S/A/I role, blanket-loss-vs-null histogram, behavior-prediction
comparison, (`Δ_m`-vs-lag once M7 lands). Replay animation = nice-to-have.

### CLI

```
uad-worm fetch     --dataset <id>
uad-worm preprocess --dataset <id>
uad-worm discover  --dataset <id> --config uad_worm/configs/default.yaml
uad-worm evaluate  --run <run_id>
uad-worm report    --run <run_id>
```

---

## 6. Cross-animal combination & holdout

**Yes, combine animals — but at the right level.** The 56 NeuroPAL-labeled recordings
are **independent dynamical systems** with a **shared identity namespace** (`neuron_class`
from NeuroPAL). That makes two kinds of combination valid, and one invalid:

- ❌ **Do not** concatenate raw time across animals and run lagged models across the
  seam — there is no dynamical continuity between animals, so cross-animal lagged pairs
  are meaningless. (If you concatenate for convenience, carry a **seam mask** and forbid
  lagged windows that straddle it.)
- ✅ **Statistic/affinity pooling.** Pool *per-animal* estimates (correlation, lagged
  dependence, blanket loss) indexed by `neuron_class` into an `animal × class × class`
  tensor with missing entries (not every class is labeled in every animal). Aggregate
  across animals to find **recurrent low-blanket subsystems** (class-level motifs).
- ✅ **Motif-level meta-analysis.** Run the full per-animal pipeline independently, then
  ask which `neuron_class` sets recur as candidate subsystems / memory-bearers across
  animals.

**Train-on-all-but-holdout (the E13 amortized protocol, reused).** This is exactly the
structure of `amortized_agency/` (E13): train an affinity/discovery model on a **pool**
and test transfer to a **held-out** instance. Worm analogue:

1. **Discover** subsystem motifs (class sets) on N−1 animals (pool).
2. **Hold out** one animal; test whether the same motif appears with **blanket loss
   below that animal's own null band** and stable roles — i.e. the motif *generalizes*
   rather than being fit per-animal.
3. Report held-out reproducibility as the **"stronger success"** criterion (§9).

Caveats: partial class overlap between animals, per-animal sign/scale differences
(z-score per animal), and uneven behavioral-state coverage. Alignment is **by class,
never by ROI id**. This pooling is **unsupervised** — CePNEM/connectome remain eval-only.

---

## 7. Blockers & risks (honest)

Data availability is **not** a blocker. **Absence of ground truth is also not a blocker
here:** this is an *application*, not a correctness proof — we don't need labeled agents,
because (a) M0 synthetic benchmarks anchor that the estimators are sound, and (b) the
worm data itself carries enough structure to *interpret* roles (a neuron driven mainly by
the rest of the brain looks sensor-like; one mainly driving the rest looks action-like;
one retaining private state looks internal), with CePNEM/connectome as an independent
sanity check. Real-data claims stay operational ("candidate subsystem", "agent-like
signature"). The real risks are statistical/biological:

### 7.1 Short recordings (T≈1600)
CMI with multi-dimensional S/A conditioning is data-hungry. *Mitigation:* Gaussian/
partial-corr estimator as primary; keep role sets low-dimensional; lean on nulls for
significance; treat nonparametric kNN-CMI as confirmatory on the top-k only (likely
underpowered here). **This shapes the whole design.**

### 7.2 Slow, autocorrelated GCaMP (estimator-power risk, not a false-agent risk)
*What it means concretely.* GCaMP is a **calcium** indicator, and calcium is a slow,
low-pass, nonlinear proxy for spiking. Each spike/burst produces a fluorescence
transient with a fast rise and a **decay time constant of ~0.5–2 s** (these recordings
use NLS-GCaMP7f). At ~0.6 s/frame, a single underlying event therefore **smears across
several frames**, and the measured trace is roughly the true activity convolved with a
one-sided decay kernel plus a slow brain-state drift shared across many neurons.

*What it does NOT do.* It does **not** hand us false agents through the blanket check.
The blanket loss is a conditional-independence test: a non-agent with a shared slow drive
still injects a shared *innovation* into both `I_{t+1}` and `E_{t+1}` each step, so the
residual CMI stays **positive** and the random cut is correctly rejected — and more
autocorrelation does not drive that to zero. (Synthetic system 2 in M0 exists to confirm
this: strong correlation, no blanket ⇒ rejected.)

*Where it actually bites.*
- **Estimator power, not the criterion.** Slow signals collapse the effective sample
  size, `N_eff ≈ T·(1−ρ)/(1+ρ)`; at ρ→1, `T=1600` behaves like a few dozen independent
  samples, so the Gaussian partial-corr CMI is high-variance/biased and the accept
  boundary gets fuzzy. The criterion is sound; the *estimate at this T* is the weak link.
- **Auxiliary scores, not the blanket.** A neuron's own decaying transient makes its past
  "predict" its future (spurious `Δ_m`), and Granger/correlation **seeds** saturate
  (uninformative, push toward the giant cluster). These feed selection but are not the
  blanket test.

*Mitigations (mandatory, not optional).*
- **Whiten the signal** to raise `N_eff`: temporal derivative and/or a deconvolved /
  event-like representation (e.g. OASIS-style) alongside raw; report which was used.
- **Score relative to nulls, not an absolute ε:** the estimator's bias/variance depend on
  each candidate's autocorrelation and size, so a fixed ε is fragile. Accept the blanket
  when it beats **random matched partitions**; accept memory/Granger structure when it
  beats an **autocorrelation-preserving null** (circular/phase/block). Right null per
  statistic: random-partition for the blanket (lower=better), AR-matched for memory/seeds
  (higher=better).
- **Condition out self-history** in `Δ_m` (`I^C_t \ {m_t}` plus the neuron's own lagged
  terms) so "memory" isn't its own decay.

### 7.3 Irregular timestamps (mild — low risk)
Measured on `2023-01-23-01`: monotonic, `dt = 0.602 s ± 0.015` (**CV ≈ 2.5%**, range
0.53–0.67 s). That jitter is tiny relative to the ~1 s GCaMP timescale, so:
- **Linear interpolation onto a uniform median-`dt` grid is sufficient** — no GP/spline
  needed. (Cubic is optional; don't over-engineer.)
- **Resampling ≠ smoothing:** interpolate to a regular clock *first*, then apply any
  optional denoising as a separate, documented step, and compute derivatives on the
  uniform grid (derivatives amplify resampling artifacts, so order matters).
- Verify post-resample that the spectrum is essentially unchanged below Nyquist.

### 7.4 Partial connectome join
Only labeled (~60%) neurons map to `neuron_class`; the connectome dataset name for
`?datasets=` must be discovered from the connectome pages. Connectome is
**prior/eval/viz only**, never the answer.

### 7.5 Single-animal recordings
Cross-session reproducibility only works through shared `neuron_class` labels with
partial overlap (see §6).

---

## 8. Non-goals (v1)

- No CePNEM/connectome as **supervision** for discovery (eval/regularization only).
- No "the worm brain is an agent" / "we found the worm's agent" claims.
- No claim of measured dynamic synaptic edge activation (replay overlays wiring, not
  dynamic synapses).
- No full POMDP interpretation; no proto-goal inference until M0–M7 hold.

---

## 9. Minimum success criteria

On ≥1 NeuroPAL-labeled dataset: (1) a subsystem with blanket loss significantly below
null, (2) stable role assignments across bootstraps, (3) ≥1 neuron with significant
lagged memory, (4) behavior-predictive value beyond correlation clustering, (5) a
readable connectome visualization. **Stronger:** subsystem motifs recur across animals
via shared `neuron_class`.

---

## 10. Open decisions (would confirm before building)

- **Package vs scripts split** — keep heavy logic in `uad_worm/` and thin runners in
  `scripts/worm/` (matches repo), or a single CLI module? (Leaning: package + scripts.)
- **xarray dependency** — the brief wants it; repo currently doesn't use it. Add to
  `requirements.txt`, or stay with numpy + a dataclass schema to minimize deps?
  (Leaning: numpy/dataclass first, xarray only if it pays for itself.)
- **First target dataset** — propose `atanas_kim_2023-2023-01-23-01` (151 neurons, 91
  labeled, T=1600, already inspected) as the M1 smoke target.
- **Scope of v1** — recommend shipping **M0–M7** (discovery + memory + nulls + eval) and
  deferring proto-goals (§16) and inter-agent modeling (§17) to a follow-up E-line.

---

## 11. Immediate next actions

1. Log E20 stub in `docs/EXPERIMENTS.md` (title + this plan link).
2. Implement **M0 synthetic benchmarks + Gaussian CMI** first (`uad_worm/blanket.py`,
   `tests/test_worm_synth_*.py`) — provable, no network.
3. Implement **M1 ingestion** against `atanas_kim_2023-2023-01-23-01` (curl recipe §1).
4. Wire worm bundle → `SimulationResult` via `pack_trace` to reuse existing MI/spotlight.
5. Only then run M3–M7 on one labeled dataset and write the §9 report.
