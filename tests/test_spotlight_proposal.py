"""Proposal ranking should prefer agent clusters over decoy blobs."""

from __future__ import annotations

import unittest

import numpy as np

from agent_spotlight.config import SpotlightConfig
from agent_spotlight.proposal import _score_single_cluster, apply_peel_mask, propose_best_cluster
from learn_agents.learn_agents import (
    TraceSimulationConfig,
    factorize_background,
    mi_partition_search,
    simulate_known_agent_trace,
)


def _sim_and_meta(cfg: SpotlightConfig):
    sim = simulate_known_agent_trace(
        TraceSimulationConfig(
            seed=cfg.seed,
            T=cfg.T,
            num_agents=cfg.num_agents,
            copies_per_role=cfg.copies_per_role,
            decoy_vars=cfg.decoy_vars,
            decoy_mode=cfg.decoy_mode,
        )
    )
    return sim.trace, sim.metadata


class SpotlightProposalTest(unittest.TestCase):
    def test_spotlight_proposal_prefers_agent_over_decoys(self):
        cfg = SpotlightConfig(proposal_mi_k=16)
        trace, meta = _sim_and_meta(cfg)
        decoys = {i for i, name in enumerate(meta["var_names"]) if "decoy" in name}
        agent_clusters = {int(k): list(v) for k, v in meta["agent_clusters"].items()}

        best, _ = propose_best_cluster(trace, cfg, peeled=set())
        self.assertIsNotNone(best)
        self.assertFalse(set(best.var_indices) & decoys)

        best_j = 0.0
        for vars_ in agent_clusters.values():
            inter = len(set(best.var_indices) & set(vars_))
            j = inter / len(set(best.var_indices) | set(vars_))
            best_j = max(best_j, j)
        self.assertGreaterEqual(best_j, 0.5)

        work = apply_peel_mask(trace, set(), cfg.peel_mode)
        mi_trace, _ = factorize_background(work, n_components=cfg.proposal_background_components)
        part = mi_partition_search(
            mi_trace,
            fixed_k=cfg.proposal_mi_k,
            bins=cfg.mi_bins,
            max_lag=cfg.mi_max_lag,
            k_selection="mdl",
        )
        decoy_pair_scores = []
        for cluster_id in np.unique(part.labels[part.labels >= 0]):
            idxs = np.where(part.labels == cluster_id)[0].tolist()
            if len(idxs) > 2 or not set(idxs) <= decoys:
                continue
            score, *_ = _score_single_cluster(work, idxs, cfg)
            decoy_pair_scores.append(score)

        if decoy_pair_scores:
            self.assertGreater(best.score, max(decoy_pair_scores))


if __name__ == "__main__":
    unittest.main()
