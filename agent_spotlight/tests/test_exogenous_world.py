"""Exogenous world must not be action-driven and should MI-split from agents."""

from __future__ import annotations

import unittest

from agent_spotlight.config import SpotlightConfig
from agent_spotlight.peel import _sim_config
from learn_agents.learn_agents import simulate_known_agent_trace


class ExogenousWorldTest(unittest.TestCase):
    def test_world_vars_not_in_agent_clusters(self):
        cfg = _sim_config(
            SpotlightConfig(
                env_vars_per_agent=2,
                env_action_coupling=0.0,
                world_vars=3,
                decoy_vars=0,
            )
        )
        sim = simulate_known_agent_trace(cfg)
        meta = sim.metadata
        agent_vars = {i for idxs in meta["agent_clusters"].values() for i in idxs}
        world_vars = set(meta["world_all_var_indices"])
        self.assertTrue(world_vars)
        self.assertFalse(agent_vars & world_vars)
        for name in meta["var_names"]:
            if name.startswith("world."):
                self.assertTrue(
                    name.startswith("world.local") or name.startswith("world.shared"),
                    msg=name,
                )


if __name__ == "__main__":
    unittest.main()
