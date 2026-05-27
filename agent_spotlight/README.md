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
│  ├── validation.py  candidate + strict UAD            │
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

**Default setting = E8:** 8 agents, 12 decoys (20%), seed=1, T=4000.

```bash
.venv/bin/python scripts/run_spotlight_e9a.py \
  --output-json results/spotlight_peel_e8_decoy20.json
```

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
| MI K | 30 | 8 (pick **one** cluster) |
| Slots | 24 | **3** |
| Candidates/pass | 64 | **1** |
| Target metric | R@30=0.25 | cumulative recall @ 8 peels |
