# learn_agents / latent-UAD debug protocol — findings summary

**Date:** May 2026  
**Branch context:** `learn_agents` prototype + strict UAD downstream evaluation  
**Artifacts:** `results/debug_protocol_*_v3.json`, `results/candidate_uad_eval_*_v3_*.json`  
**Scripts:** `scripts/debug_learn_agents_protocol.py`, `scripts/evaluate_latent_candidates_with_uad.py`

---

## Goal

Determine whether failures in latent agent discovery come from (1) the simulated environment, (2) the learned representation, or (3) the candidate / validation pipeline — before investing in richer models.

The latent model is treated as a **candidate proposer**, not a final agent identifier. Success is measured by Recall@K (coverage of true agents in top candidates) and post-UAD precision/recall after strict Markov-blanket filtering.

---

## Protocol (six stages)

| Stage | What it tests |
|-------|----------------|
| 1. Oracle | Ground-truth clusters: Gaussian ε-blanket scores + strict discrete UAD |
| 2. Raw MI clustering | `agency_detect`-style agglomerative clustering without latent model |
| 3. Persistence | Slot assignment stability across time chunks |
| 4. Interface | Whether slots separate roles/agents (diagnostic uses simulator metadata only) |
| 5. Adaptation | ε-only local search vs MDL-regularized search on candidate variable sets |
| 6. Threshold sweep | Strict UAD pass rate vs blanket tolerance |

**Run settings (v3):** seed=1, T=4000, 50 training epochs, `copies_per_role=2`, 3 agents with 2 decoys / 8 agents with 0 decoys, slots 9 / 20.

---

## Methodological constraint (v3)

**Variable names are not used for role classification.** Names exist only in the simulator for debugging and ground-truth scoring (`metadata["var_role"]`, `metadata["var_agent"]`).

`classify_variables` in `agency_detect/markov_blanket.py` uses only:

- environment MI (high → sensor),
- lagged within-cluster MI (high → action),
- remainder → internal.

Using names to label S/A/I was removed as invalid for real data.

---

## Main findings

### 1. Environment structure is present

Gaussian oracle scores on ground-truth agent clusters show strong ε-blanket separability:

| Setting | Mean separation ratio (leakage / raw MI) |
|---------|------------------------------------------|
| 3 agents | ~0.04 |
| 8 agents | ~0.11 |

Lower is better. Conditional leakage is small relative to raw cross-boundary MI.

**Conclusion:** The simulator supports UAD-style structure; failures are not “no signal in the data.”

### 2. Statistical role classification is adequate on this simulator

After removing name hints, oracle clusters still pass strict discrete UAD at tolerance 1.0 (typical CMI ~0.25–0.29). Example 3-agent classification: S≈3, A≈1, I≈2 per agent.

**Conclusion:** On this benchmark, MI-based S/A/I assignment is workable. Real data may be harder; names must not be relied on.

### 3. Raw MI clustering beats the latent model at 8 agents

| Setting | Pre-validation MI recall (J≥0.3) | Post-validation pipeline recall | Latent Recall@30 |
|---------|----------------------------------|--------------------------------|------------------|
| 3 agents | 33% (one merged cluster) | 100% (1 cluster survives) | **100%** |
| 8 agents | **100%** (8/8, J=1.0) | **100%** | **25–38%** |

At 8 agents, agglomerative clustering on lagged MI recovers every agent perfectly **before** any neural model. The slot-attention model underperforms this baseline for candidate generation.

**Conclusion:** At scale, the bottleneck is **representation and slot→raw mapping**, not separability in raw variable space.

### 4. Latent slots are stable but do not align with agents

- Cross-chunk slot cosine similarity: **~0.99–1.0**
- Top-1 assignment persistence: **~0.82**
- Per-slot agent purity (metadata diagnostic): **~0.26–0.43** at 8 agents
- No slot achieves full sensor+internal+action interface in assignment mass

**Conclusion:** Training learns persistent latent factors that **mix agents and roles**, not unstable noise.

### 5. Strict UAD post-filtering can be misleading at 8 agents

With statistical classification only (v3), 8-agent candidate evaluation at K=30 showed:

- Pre-UAD Recall@30: **0.25**
- Post-UAD precision: **~0.16**
- Many strict survivors (`strict_count` high) that are **not** true agents

High `strict_count` with low precision means the validator accepts small/mixed partitions that look “clean” but do not match ground truth.

**Conclusion:** Treat strict UAD as **falsification**, not the sole success metric; pair with candidate-stage recall and mapping diagnostics (purity, `unique_agents_mean`).

### 6. ε-only adaptation shrinks; MDL mitigates but does not fix recall

Local search minimizing blanket violation alone collapses candidate size (mean |C| ~3–5 at 8 agents). Adding

`J = violation + λ · log((N+1)/|C|)` (default λ=0.15)

increases set size and can improve Recall@30 (e.g. 0.38 → 0.50 in one run) at the cost of lower post-UAD precision.

**Conclusion:** Anti-shrink regularization is necessary but **not sufficient** for 8-agent discovery.

### 7. `var_acc` alone is too harsh for the candidate framing

Low variable–agent assignment accuracy at 8 agents (~0.47 in sweeps) coexists with useful partial structure. Candidate Recall@K and mapping diagnostics are more informative for the proposer role.

---

## Failure-mode diagram

```
Environment OK (oracle separability)
        │
        ├─ 3 agents: latent candidates often OK (R@30 ≈ 1)
        │
        └─ 8 agents: raw MI clustering perfect
                    latent slots persistent but mixed
                    → REPRESENTATION / MAPPING bottleneck
                    strict UAD alone → many false positives
```

---

## MI-guided latent refinement (follow-up)

Implemented `refine_model_with_mi()` in `learn_agents.py`:

1. Lagged-MI agglomerative clustering → variable labels (`mi_cluster_variable_labels`)
2. Hungarian match MI clusters to slots using overlap with mean assignment
3. Fine-tune with KL alignment loss + recon/pred (`RefineConfig`, default 20–25 epochs)

**8 agents** (seed=1, T=4000, 50 pretrain + 25 refine epochs, no adaptation):

| Metric | Baseline slots | + MI refine |
|--------|----------------|-------------|
| Pre-UAD Recall@30 | 0.25 | **1.00** |
| Post-UAD recall@30 | 0.00 | **1.00** |
| Post-UAD precision@30 | 0.00 | **0.86** |
| Majority agent frac (top-20) | 0.59 | **0.77** |

**3 agents** (with decoys): MI refine did not help (Recall@30 stays ~0.33) because raw MI merges agents in this regime; refinement inherits that coarse partition.

```bash
.venv/bin/python scripts/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --mi-refine --refine-epochs 25 --no-adapt-blankets \
  --output-json results/candidate_uad_eval_8agents_mi_refine.json
```

---

## Agent-count sweep (1–12 agents, clean regime)

**Why 8 agents looked better than 3 in earlier runs:** those runs were not comparable. The strong 8-agent MI-refine result used `decoy_vars=0` (48 vars); many 3-agent runs used `decoy_vars=2` (20 vars) where MI merges agents. With a **fixed clean regime** (`decoy_vars=0`, `copies_per_role=2`, `T=4000`, 50 pretrain + 25 refine epochs, `num_slots = max(4, 3×agents)`), the picture is different.

Script: `scripts/learn_agents_agent_count_sweep.py` → `results/agent_count_sweep.json`

| Agents | Vars | MI recall | Baseline R@30 | Refine R@30 | Refine post prec |
|--------|------|-----------|---------------|-------------|------------------|
| 1–4 | 6–24 | 1.00 | **1.00** | 1.00 | high |
| 5 | 30 | 1.00 | 0.60 | **0.80** | 1.00 |
| 6 | 36 | 1.00 | 0.50 | **1.00** | 1.00 |
| 7 | 42 | 1.00 | 0.00 | **0.86** | 0.92 |
| 8 | 48 | 1.00 | 0.25 | **1.00** | 1.00 |
| 9 | 54 | 1.00 | 0.11 | **0.89** | 0.96 |
| 10 | 60 | 1.00 | 0.10 | **0.50** | 0.69 |
| 11 | 66 | 1.00 | 0.09 | **0.82** | 0.93 |
| 12 | 72 | 1.00 | 0.08 | **0.67** | 0.78 |

**Breaking points (seed=1):**

- **Raw MI partition:** perfect agent recall at all counts 1–12 (no MI breaking point in this simulator).
- **Latent baseline (no refine):** Recall@30 drops below 0.5 at **≥7 agents** (chance = 1/K).
- **MI refine helps meaningfully from ≥5 agents** (first count where refine beats baseline by >0.05).
- **MI refine stays ≥0.50 through 12 agents** but softens at **10 agents** (0.50) and **12 agents** (0.67) — likely slot/variable ratio pressure (`num_slots = 3×agents` may be tight at scale).

**Interpretation:** The bottleneck is **unsupervised slot discovery without MI init**, not MI separability. Refine transfers the perfect MI partition into slots until ~10–12 agents where alignment + candidate mapping degrade.

```bash
.venv/bin/python scripts/learn_agents_agent_count_sweep.py \
  --min-agents 1 --max-agents 12 --output-json results/agent_count_sweep.json
```

---

## Agent-count sweep with decoys (20% and 70% of variables)

Decoy count scales with agent variables: `decoy_vars = round(f × agent_vars / (1−f))` where `agent_vars = 3 × copies_per_role × num_agents`. Use `--decoy-fraction 0.20` or `0.70`.

Results: `results/agent_count_sweep_decoy20pct.json`, `results/agent_count_sweep_decoy70pct.json`

### 20% decoys

| Agents | Vars | MI recall | Baseline R@30 | Refine R@30 |
|--------|------|-----------|---------------|-------------|
| 1 | 8 | 1.00 | 1.00 | 1.00 |
| 2 | 15 | **0.50** | 1.00 | 1.00 |
| 3 | 22 | **0.33** | 1.00 | **0.33** |
| 4 | 30 | **0.00** | 0.75 | 0.50 |
| 5–7 | 38–52 | 0–0.43 | &lt;0.5 | ~0.29–0.33 |
| 8 | 60 | 0.38 | 0.12 | **0.38** |
| 10 | 75 | 0.10 | 0.00 | 0.00 |
| 12 | 90 | 0.33 | 0.00 | 0.25 |

**Breaking points:** MI recall &lt; 1.0 from **≥2 agents**; refine R@30 &lt; 0.5 from **≥3 agents**; baseline &lt; 0.5 from **≥5 agents**.

MI is the **first** failure mode: decoys pull agglomerative clusters off true agents. Refine cannot exceed MI quality (same ceiling at 8 agents: 0.38). Baseline slots can still look good briefly (3 agents: R@30=1.0) while MI is already broken — **misleading if you only watch latent metrics without checking MI init**.

### 70% decoys

| Agents | Vars | MI recall | Baseline R@30 | Refine R@30 |
|--------|------|-----------|---------------|-------------|
| 1 | 20 | 1.00 | 1.00 | 1.00 |
| 2+ | 40+ | **0.00** | ≈0 | **0.00** |

**Breaking points:** MI and refine both collapse from **≥2 agents**. The trace is mostly decoy/noise; lagged-MI clustering finds no agent structure to transfer.

### Comparison across decoy regimes

| Regime | MI breaks at | Refine useful? | Main limiter |
|--------|--------------|----------------|--------------|
| 0% decoys | never (1–12) | yes, through ~12 agents | slot learning without init |
| 20% decoys | ≥2 agents | only while MI recall &gt; 0 | **MI partition quality** |
| 70% decoys | ≥2 agents | no | **environment SNR / decoy load** |

This explains the earlier **8 &gt; 3** confusion: `decoy_vars=2` on 3 agents ≈ **25% decoys** (same order as the 20% sweep), where MI recall is already 0.33 and refine cannot recover.

```bash
.venv/bin/python scripts/learn_agents_agent_count_sweep.py \
  --decoy-fraction 0.20 --output-json results/agent_count_sweep_decoy20pct.json

.venv/bin/python scripts/learn_agents_agent_count_sweep.py \
  --decoy-fraction 0.70 --output-json results/agent_count_sweep_decoy70pct.json
```

---

## Recommendations

1. **Do not use variable names** in classification or validation; keep names/metadata for scoring only.
2. **Use MI clustering as a baseline and initializer** — at 8 agents it already solves partition discovery; latent training should refine or compress, not rediscover from scratch. **MI refine implements this.**
3. **Improve slot→agent alignment** (MI-guided init, cluster-aware losses, fewer slots, coarse-to-fine: MI partitions → latent refinement).
4. **Keep MDL (or similar) in candidate adaptation** — avoid ε-only shrink without a complexity term.
5. **Report candidate metrics** — Recall@K, mapping purity, post-UAD precision/recall, `strict_count` — not `var_acc` alone.
6. **Monitor MI partition quality before refine** — with ≥20% decoys, MI breaks before latent slots; refine inherits the coarse partition.
7. **Improve statistical S/A/I discovery** for real data (intervention signatures, persistence floors, null margins vs random partitions) without fixed agent-size priors.

---

## How to reproduce

```bash
# Six-stage debug protocol
.venv/bin/python scripts/debug_learn_agents_protocol.py \
  --num-agents 8 --copies-per-role 2 --decoy-vars 0 \
  --num-slots 20 --epochs 50 --T 4000 --seed 1 \
  --output-json results/debug_protocol_8agents_v3.json

# Candidate + strict UAD pipeline (no adaptation)
.venv/bin/python scripts/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --no-adapt-blankets \
  --output-json results/candidate_uad_eval_8agents_v3_no_adapt.json

# With MDL adaptation (default --adapt-objective mdl)
.venv/bin/python scripts/evaluate_latent_candidates_with_uad.py \
  --num-agents 8 --adapt-objective mdl \
  --output-json results/candidate_uad_eval_8agents_v3_mdl.json
```

---

## Related code changes

- `agency_detect/markov_blanket.py` — name-free MI role classification
- `scripts/debug_learn_agents_protocol.py` — six-stage ablation runner
- `scripts/evaluate_latent_candidates_with_uad.py` — four-step candidate pipeline + optional MDL adaptation
