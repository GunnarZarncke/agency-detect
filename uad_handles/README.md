# Handle-UAD toy benchmarks

Synthetic experiments for [*Handles Before Interventions: Access-Model UAD*](../docs/papers/access-uad/access-uad.tex) ([PDF](../docs/papers/access-uad/access-uad.pdf)).

The paper argues that real observers have **handles** (embedded observation and operation kernels), not ideal `do(X=x)` interventions. These scripts demonstrate that claim on a small binary synthetic world:

- A hidden agent loop: belief → policy → action → environment → sensor → belief, plus goal.
- **Passive alias handles** that mirror sensor/action readouts but are operationally inert.
- **Handle operations** (`sensor_flip`, `action_block`, `goal_flip`) that only affect the system when applied to true handles.
- **Plain UAD** scores loops from passive data only; **handle-UAD** adds interventional evidence and active test selection.

This is a proof-of-concept scaffold, not production UAD. It complements the main simulator line (`learn_agents/`, `agent_spotlight/`) and the worm application (`uad_worm/`).

## Layout

| Module | Role |
|--------|------|
| `minimal.py` | Single fixed world (1 alias pair); plain vs active handle-UAD |
| `scaling.py` | Sweep alias count, passive sample size, intervention rounds; **full rescore** after each test |
| `scaling_fast.py` | Same sweep; **targeted** score updates (faster first scaling pass) |

Runners live in `scripts/handles/`. Artifacts go to `results/handles/` (gitignored).

## Dependencies

From the repo root (needs `pandas`; optional `matplotlib` for plots):

```bash
.venv/bin/pip install pandas matplotlib
```

## Quick start

**Minimal demo** (one seed, default 800 passive steps, 3 active rounds):

```bash
PYTHONPATH=. .venv/bin/python scripts/handles/run_minimal.py
```

**Fast scaling sweep** (recommended first benchmark):

```bash
PYTHONPATH=. .venv/bin/python scripts/handles/run_scaling_fast.py \
  --alias-counts 0,1,2,4,8 \
  --passive-ns 600 \
  --seeds 1 \
  --max-rounds 10 \
  --intervention-batch 80 \
  --candidate-cap 800 \
  --verbose
```

**Full rescore scaling** (slower, more faithful to rescore-after-each-intervention):

```bash
PYTHONPATH=. .venv/bin/python scripts/handles/run_scaling.py \
  --alias-counts 0,1,2,4,8 \
  --passive-ns 300,600 \
  --seeds 4 \
  --max-rounds 8 \
  --verbose
```

Outputs include CSV summaries, example top-candidate tables, optional PNG plots, and a run-local `README.md`. Each scaling run also writes a `.zip` bundle next to the output directory.

## Interpreting results

- **Round 0** = passive UAD only (no handle tests).
- **Round > 0** = active handle tests on the current top candidate's claimed sensor/action/goal handles.
- **Success** = exact recovery of the true loop as rank-1 candidate.
- **Alias count** = number of passive alias pairs competing with real S/A handles; passive scores alone often prefer aliases; handle tests are meant to break that tie.

Early runs show recovery at zero aliases with enough intervention budget, and degraded recovery as alias decoys grow — consistent with the paper's access-model story. The fast rescoring variant is an approximation; compare against `scaling.py` when validating algorithm changes.

## Related docs

- Paper source: [`docs/papers/access-uad/`](../docs/papers/access-uad/)
- Experiment log entry: [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) (Handle-UAD toy benchmark section)

## Next steps (code)

1. **Negative interventional evidence** when blocking an alias handle has no effect.
2. **Smarter test scheduling** — probe real `{S,A,G}` and alias pools early, not only the top explanation's handles.
3. **Larger sweeps** — more seeds, higher `candidate-cap`, passive-N grid to separate sample count from handle budget.
4. Wire headline figures into the access-uad paper once the active policy stabilizes.
