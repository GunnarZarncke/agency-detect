#!/usr/bin/env python
"""
Test the Markov blanket validation fix.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agency_detect import DetectionConfig, generate_decoupled_trace, AgentDetector


def test_validation_fix():
    """Test that validation now properly catches invalid agents."""
    
    print("=== Testing Validation Fix ===")
    
    # Set up configuration with validation enabled
    config = DetectionConfig()
    config.N_AGENTS = 3  # Force over-clustering
    config.VALIDATE_BLANKETS = True  # Enable validation
    config.WEAK_THRESHOLD = 0.2
    
    print(f"N_AGENTS: {config.N_AGENTS}")
    print(f"VALIDATE_BLANKETS: {config.VALIDATE_BLANKETS}")
    
    # Generate data
    np.random.seed(42)
    trace = generate_decoupled_trace(steps=1000)  # Smaller for testing
    
    # Run detection with validation
    detector = AgentDetector(config)
    results = detector.detect_agents(trace)
    
    print("\n=== Results with Validation Enabled ===")
    
    for agent_id in sorted(k for k in results.keys() if k != 'env'):
        agent = results[agent_id]
        classification = agent['classification']
        validation = agent['blanket_validation']
        
        print(f"\nAgent {agent_id}:")
        print(f"  Variables: {agent['variables']}")
        print(f"  Sensors: {classification['S']}")
        print(f"  Actions: {classification['A']}")
        print(f"  Internal: {classification['I']}")
        print(f"  Valid: {validation['valid']}")
        print(f"  Violation: {validation['violation']:.4f}")
        print(f"  Details: {validation['details']}")
        
        # Check if this agent would be rejected
        if validation['valid'] == False:
            print(f"  🚫 REJECTED: {validation['details']}")
        elif validation['valid'] == True:
            print(f"  ✅ PASSED: Valid agent")
        else:
            print(f"  ⚠️  SKIPPED: Validation not performed")
    
    if 'env' in results:
        print(f"\nEnvironment: {results['env']['variables']}")
    
    print("\n=== Test Complete ===")


if __name__ == '__main__':
    test_validation_fix() 