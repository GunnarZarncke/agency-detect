# hierarchical_spotlight

E12 experiments for treating spotlight discoveries as **hierarchical chunks**.

The goal is deliberately narrower than `agent_spotlight/`: start from a spotlight
artifact, keep each discovered pass as a graph node, add data-only fusion edges,
and score whether each ground-truth agent is present anywhere in the resulting
graph or connected components.

## Current Experiment

```bash
.venv/bin/python hierarchical_spotlight/run_e12.py \
  --input-json results/spotlight/e11/spotlight_e11_rich_agents_cpr3_k24_p16.json \
  --output-json results/hierarchical/e12/hierarchical_e12_rich_agents.json
```

Each run writes a readable vertical component forest:

- JSON report: `results/hierarchical/e12/hierarchical_e12_rich_agents_<timestamp>.json`
- Graphviz DOT: `results/hierarchical/e12/hierarchical_e12_rich_agents_<timestamp>.dot`
- PNG rendering, when Graphviz `dot` is installed: `results/hierarchical/e12/hierarchical_e12_rich_agents_<timestamp>.png`

The JSON contains the full dense edge set. The DOT/PNG intentionally render a
summary forest: component nodes on the left, discovered chunks on the right.
Dashed fusion hints are hidden by default; enable them with
`--render-fusion-hints` for debugging.

Nodes are admitted spotlight chunks. Edges are data-only:

- mean lagged cross-MI between chunks
- optional UAD validation of the union
- optional precursor validation of the union

The default `min_cross_mi=0.70` intentionally keeps only strong chunk links.
Lower values are useful for graph-recall audits, but tend to bridge all agents
through weak cross-agent edges and produce one giant component.

Ground truth is used only for the final evaluation metric:

> Count detection as successful if a true agent appears in any discovered node or
> connected component at the configured Jaccard threshold.

E12 reports two families of metrics:

- `agent_graph_recall`: permissive coverage — an agent appears somewhere in the
  graph by Jaccard.
- `clean_graph_recall`: purity-aware coverage — the matching node/component must
  cover enough of the agent while staying below configured contamination limits
  for extra agent/world/decoy variables.

Current clean defaults are intentionally strict for non-agent contamination:
`clean_max_world_vars=0`, `clean_max_decoy_vars=0`, and
`clean_max_extra_fraction=0.5`.

## Why Separate Folder?

This is not another peel-loop variant. It is the start of a second-stage
hierarchical identity layer: local chunks first, fused agent hypotheses second.

