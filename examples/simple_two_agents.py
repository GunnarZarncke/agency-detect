#!/usr/bin/env python
"""
Simple Two Agent Example

Demonstrates the simplest possible case: one solar panel agent and one factory agent.
This is the minimal setup to see individual agent detection in action.
"""

from agency_detect import generate_decoupled_trace, AgentDetector


def main():
    """Run the simplest possible agent detection example."""
    print("=== Simple Two Agent Detection ===")
    print("Creating system with 1 solar panel + 1 steel factory...")
    
    # Generate minimal system: 1 solar panel + 1 steel factory
    trace = generate_decoupled_trace(n_solar_panels=1, factory_materials=['steel'])
    
    print(f"Generated {len(trace)} timesteps of data")
    print(f"Variables: {list(trace[0].keys())}")
    
    # Detect individual agents
    detector = AgentDetector()
    results = detector.detect_agents(trace)
    
    # Print results
    detector.print_results(results)
    
    # Summary
    agent_count = len([k for k in results.keys() if k != 'env'])
    print(f"\n=== Summary ===")
    print(f"Detected {agent_count} autonomous agents")
    
    for agent_id, agent_data in results.items():
        if agent_id != 'env':
            variables = agent_data['variables']
            classification = agent_data['classification']
            print(f"\nAgent {agent_id}:")
            print(f"  Variables: {variables}")
            print(f"  Sensors: {classification['S']}")
            print(f"  Actions: {classification['A']}")
            print(f"  Internal: {classification['I']}")


if __name__ == '__main__':
    main()

