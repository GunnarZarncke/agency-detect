"""Complex agent variant should not expose lockstep role copies."""

from __future__ import annotations

import unittest

import numpy as np

from learn_agents.learn_agents import TraceSimulationConfig, simulate_known_agent_trace


class ComplexAgentVariantTest(unittest.TestCase):
    def test_complex_variant_avoids_near_duplicate_role_channels(self):
        sim = simulate_known_agent_trace(
            TraceSimulationConfig(
                T=800,
                num_agents=2,
                copies_per_role=3,
                decoy_vars=0,
                world_vars=0,
                agent_variant_mode="complex",
                episodic=False,
                seed=3,
            )
        )
        role_indices = sim.metadata["role_indices"]

        for role in ("sensor", "internal", "action"):
            idxs = role_indices[(0, role)]
            corr = np.corrcoef(sim.trace[:, idxs], rowvar=False)
            offdiag = np.abs(corr[np.triu_indices_from(corr, k=1)])
            self.assertLess(float(offdiag.max()), 0.995, msg=role)


if __name__ == "__main__":
    unittest.main()
