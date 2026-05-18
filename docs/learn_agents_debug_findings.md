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

## Recommendations

1. **Do not use variable names** in classification or validation; keep names/metadata for scoring only.
2. **Use MI clustering as a baseline and initializer** — at 8 agents it already solves partition discovery; latent training should refine or compress, not rediscover from scratch. **MI refine implements this.**
3. **Improve slot→agent alignment** (MI-guided init, cluster-aware losses, fewer slots, coarse-to-fine: MI partitions → latent refinement).
4. **Keep MDL (or similar) in candidate adaptation** — avoid ε-only shrink without a complexity term.
5. **Report candidate metrics** — Recall@K, mapping purity, post-UAD precision/recall, `strict_count` — not `var_acc` alone.
6. **Improve statistical S/A/I discovery** for real data (intervention signatures, persistence floors, null margins vs random partitions) without fixed agent-size priors.

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
