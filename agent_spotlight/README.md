# agent-spotlight

Serial **spotlight** agent discovery: propose one MI cluster at a time, refine with small slot capacity, validate, peel, repeat.

Replaces global N-slot competition (see E8 in [`learn_agents/EXPERIMENTS.md`](../learn_agents/EXPERIMENTS.md)) with a human-inspired **one-agent-at-a-time** loop.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  agent_spotlight/                                       │
│  ├── config.py      SpotlightConfig (all ablation knobs)│
│  ├── proposal.py    MI partition → best single cluster  │
│  ├── refine.py      3-slot model, align to one cluster  │
│  ├── validation.py  candidate + strict UAD + agency gate │
│  ├── agency_gate.py cluster selection by gate mode      │
│  ├── diagnostics.py per-agent miss diagnosis            │
│  ├── metrics.py     per-pass measurements               │
│  └── peel.py        serial loop                         │
└─────────────────────────────────────────────────────────┘
         uses learn_agents simulator + slot model
         uses agency_detect MarkovBlanketValidator
```

## Phase E9a (implemented)

Per pass:

1. Background-factorize residual trace; MI at `proposal_mi_k` (default **16**)
2. Score each cluster by **continuous precursor + within-MI** (not binary pass/fail tie at 0.02)
3. Pretrain small model (`num_slots=3`); refine with high `lambda_align`
4. One candidate (`spotlight_slot` or `mi_cluster` mode)
5. UAD validate; record Jaccard vs ground truth
6. Peel cluster vars; repeat up to `max_passes`

**Default benchmark (E9e/E10):** 8 agents, 0 decoys, 12 exogenous world vars, `mi_cluster`, adapt1 strengths.

```bash
.venv/bin/python scripts/run_spotlight_e9a.py
.venv/bin/python scripts/run_spotlight_sweeps.py --fast
```

### Agency gate modes (`--agency-gate-mode`)

| Mode | Behavior |
|------|----------|
| `off` | No gate (default) |
| `score_penalty` | Subtract penalty for missing S/A/I roles in proposal ranking |
| `actions_only` | Skip clusters with no action vars (passive/world blobs) |
| `soft` | Prefer agency-passing clusters; fall back to top precursor |
| `strict` | Require S+A+I before train (~0.125 recall cost) |

Prefer `score_penalty` over strict — same recall as off on 8-agent benchmark. **Default: gate off** (no measured gain at current performance).

### Ablation examples

```bash
# MI cluster as candidate (skip slot mapping) — E9b preview
.venv/bin/python scripts/run_spotlight_e9a.py \
  --candidate-mode mi_cluster --output-json results/spotlight_mi_cluster.json

# Softer proposal partition
.venv/bin/python scripts/run_spotlight_e9a.py \
  --proposal-mi-k 12 --cluster-score precursor_x_size

# Stricter admission
.venv/bin/python scripts/run_spotlight_e9a.py \
  --require-uad-pass --require-jaccard-hit
```

## Key metrics (per pass)

| Field | Meaning |
|-------|---------|
| `cluster_score` | Precursor-based proposal ranking |
| `best_jaccard` | Overlap with nearest true agent |
| `cumulative_recall` | Fraction of true agents hit so far |
| `uad_valid` | Strict blanket pass |
| `refine_final_align` | Slot–cluster alignment loss |

## Planned phases

| Phase | Status |
|-------|--------|
| E9a | Cheap peel loop + measurements (`agent_spotlight/`) |
| E9b | MI-cluster candidates default |
| E9c | Un-peel / revision |
| E9d | Ablation sweep driver |

## Compare to E8

| | E8 global | E9a spotlight |
|--|-----------|---------------|
| MI K | 30 | 24 (pick **one** cluster) |
| Slots | 24 | **3** |
| Candidates/pass | 64 | **1** |
| Target metric | R@30=0.875 (exogenous) | cumulative recall **1.000** (8 agents, data-only peel, K=24) |

## Partial peel (7/8 agents)

MI proposes ~6-var clusters; admission at J≥0.3 matches the **right agent** with **2–4 vars**. Peel masks only those cluster vars, so overlapping serial passes orphan remaining vars. Data-only fix: grow peel set from refine alignment.

## 100% Recall Setup (E10)

The smallest data-only recovery change from `scripts/run_spotlight_recovery_sweep.py` is `proposal_mi_k=24` (now default). It keeps the exogenous world setting and cluster-only peel, recovering **8/8 agents** on seed 1 without ground-truth metadata.

```bash
.venv/bin/python scripts/run_spotlight_e9a.py \
  --proposal-mi-k 24 --output-json results/spotlight_exogenous_k24.json
```
