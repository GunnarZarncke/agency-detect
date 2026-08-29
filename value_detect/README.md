# value_detect — Unsupervised value discovery (directional UAD)

**Status:** v1–v3 complete and wrapped up; the v3 entry is
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) §E21. This package is contributed from
its canonical home, https://github.com/SJ-Beard/value-detect, where the full record lives
(v1/v2 log entries in the same format, per-experiment write-ups, locked
pre-registrations, dated decisions log, archived result artifacts, paper draft). A live
v4 line applies the same instrument to the deployment-pipeline simulator:
https://github.com/SJ-Beard/deployment-pipeline-value-detect.

UAD locates agents' boundaries; this package asks what *drives* them — which component
plays the role of a discovered agent's **value structures**. Hypothesis (inverting the
classic AI-drives direction of explanation): whatever is maximally empowered and
minimally plastic thereby *serves as* the system's values, and that asymmetry is
passively detectable. The operationalisation is the **value signature** on
directed-information scores: system drive above a procedure-mirroring noise floor
(circular-shift nulls, p95 + z≥3), rankable, and net intake at floor or net drive ≥ 9×
net intake.

## Results summary

- **v1 (anchor):** on `uad_handles`' `SyntheticHandleWorld` (passive rollouts, world
  untouched), the signature picks out the planted goal, and only it, in every
  measurement convention able to see through the world's XOR composition (fused
  mega-state and fused+best-key: 100% of 20 seeds, unique in ≥95%); naive pairwise
  fails structurally (the XOR "cipher"). Controls: no-core, scrambles, decoys,
  `goal_progress` contrast.
- **v2 (which conventions survive more complex worlds):** across four worlds incl. an
  eight-agent 49-variable colony, only the block-level **fused-agents any-block**
  convention (fuse discovered agents; offer every block as a conditioning key) recovers
  every planted goal; all variable-level conventions hit a scale wall, caught by the
  no-core control.
- **v3 (E21 — how easily can the instrument be fooled):** fast captured goal refused
  20/20; a slow, history-triggered captured goal defeats every lag-1 passive test 20/20
  and is unmasked by the interventional yardstick 20/20; goal twins are passively
  indistinguishable below ~1–2% noise. The yardstick (a two-direction generalisation of
  the handle benchmark's `goal_flip`) resolves exactly those two cases — the access
  premium, quantified.

## Layout

| Module (`src/value_detect/`) | Role |
|------------------------------|------|
| `world.py`, `narrate.py`, `changes.py` | Passive-rollout wrapper for `uad_handles`, plain-English story printout, change-frequency audit |
| `directed.py` | Discrete plug-in directed-information estimators (α=0.1, nats; matches `agency_detect` conventions) |
| `scorer.py` | The measurement conventions (pairwise, fused mega-state, fused+best-key, fused-agents own-block/any-block, grown keys) |
| `floors.py`, `criteria.py` | Circular-shift noise floors (procedure-mirroring, p95, z-margin gates), signature evaluation |
| `variants.py`, `worlds_v2.py` | No-core/scramble/alias variants; the v2/v3 worlds (colony, deep-synergy, slow-meter, puppet- and alias-colonies) |
| `agentblocks.py` | Agent-block detection and fusion for the block-level conventions |
| `yardstick.py` | The interventional yardstick (goal-role + autonomy probes, do-style flips) |

Runners live in `value_detect/scripts/` (`chunk2–5*` = v1, `v2_*`, `v3_*`). Artifacts go
to the repo-root `results/` (gitignored here; archived copies with verdict tables and
figures in the canonical repo).

## Dependencies and tests

Python ≥3.9, `numpy`, `pandas`, `matplotlib` (see `pyproject.toml`); imports
`uad_handles` and `agency_detect` from this repository in place — nothing there is
modified. Set `PYTHONHASHSEED=0` for anything touching `agency_detect.agents` (its
per-agent parameters derive from `hash(name)`, which is per-process randomised
otherwise).

```bash
pip install -e value_detect
PYTHONHASHSEED=0 python -m pytest value_detect/tests -q
```

Reproduce v3 (writes `results/v3_0/`, `results/v3_5/`):

```bash
python value_detect/scripts/v3_build_audits.py
python value_detect/scripts/v3_sweep.py --seeds 20 --jobs 4   # ~37.5 h at 4 workers
python value_detect/scripts/v3_aggregate.py
```

## Provenance and licence

Written by SJ Beard's value-discovery programme (implementation and analysis by Claude
under SJ's direction); MIT-licensed (see `LICENSE` in this folder) and contributed to
this repository under its Apache-2.0 terms. Terminology follows the canonical repo's
`docs/DEFINITIONS.md`.
