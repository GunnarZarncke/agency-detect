#!/usr/bin/env python
"""
Basic Agency Detection Example

This example demonstrates the core functionality of the agency detection framework
with a simple two-agent system.
"""

import numpy as np
from agency_detect import (
    generate_decoupled_trace, 
    AgentDetector,
    SimulationConfig,
    DetectionConfig
)


def analyze_trace(trace):
    """Analyze and print basic trace statistics."""
    print("=== Trace Analysis ===")
    vars_ = list(trace[0].keys())
    data = np.array([[rec[v] for v in vars_] for rec in trace])
    
    print(f"Variables: {vars_}")
    print(f"Data shape: {data.shape}")
    
    # Check variance for each variable
    var_vals = data.var(axis=0)
    print("\nVariable variances:")
    for i, (var, variance) in enumerate(zip(vars_, var_vals)):
        print(f"  {var}: {variance:.6f}")
    
    active_vars = [vars_[i] for i in range(len(vars_)) if var_vals[i] > 0.0]
    print(f"\nActive variables (variance > 0): {len(active_vars)}")
    print(f"Active variables: {active_vars}")
    
    # Check some sample values
    print(f"\nFirst 10 timesteps of data:")
    for i in range(min(10, len(trace))):
        print(f"t={i}: {trace[i]}")
    
    return len(active_vars) >= 2


def main():
    """Main demonstration of agent detection."""
    # Set random seed for reproducible results
    np.random.seed(SimulationConfig.RANDOM_SEED)
    
    # Generate simulation data with specified configuration:
    # - 3 solar panels
    # - 1 wood factory
    # - 1 steel factory 
    # - 3 corn factories
    print("Generating simulation data...")
    
    factory_materials = ['wood', 'steel', 'corn', 'corn', 'corn']
    trace = generate_decoupled_trace(n_solar_panels=3, factory_materials=factory_materials)
    
    total_agents = 3 + len(factory_materials)  # 3 solar + 5 factories = 8 agents
    print(f"Created {total_agents} agents total")
    
    # Analyze trace
    if not analyze_trace(trace):
        print("ERROR: Insufficient active variables for clustering")
        return
    
    print(f"\nProceeding with clustering...")
    
    # Create detector and run detection
    detector = AgentDetector()
    clusters = detector.detect_agents(trace)
    
    # Print results
    detector.print_results(clusters)


if __name__ == '__main__':
    main()

