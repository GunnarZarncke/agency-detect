#!/usr/bin/env python
"""
Test the adjusted validation threshold.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agency_detect import DetectionConfig, generate_decoupled_trace, AgentDetector


def test_validation_threshold():
    """Test validation with adjusted threshold."""
    
    print("=== Testing Adjusted Validation Threshold ===")
    
    # Generate data
    np.random.seed(42)
    trace = generate_decoupled_trace(steps=2000)
    
    config = DetectionConfig()
    config.VALIDATE_BLANKETS = True
    config.WEAK_THRESHOLD = 0.2
    
    detector = AgentDetector(config)
    
    print(f"Validation threshold: {config.BLANKET_TOLERANCE}")
    
    # Test N=2 (should work now)
    print(f"\n--- Testing N_AGENTS=2 ---")
    config.N_AGENTS = 2
    results_2 = detector.detect_agents(trace)
    
    agents_2 = [k for k in results_2.keys() if k != 'env']
    valid_2 = [k for k in agents_2 if results_2[k]['blanket_validation']['valid'] != False]
    
    print(f"Result: {len(valid_2)}/{len(agents_2)} agents passed validation")
    for agent_id in agents_2:
        validation = results_2[agent_id]['blanket_validation']
        status = "✅ PASSED" if validation['valid'] != False else "❌ FAILED"
        print(f"  Agent {agent_id}: {status} (CMI: {validation['violation']:.3f})")
    
    # Test N=3 (should still fail)
    print(f"\n--- Testing N_AGENTS=3 ---")
    config.N_AGENTS = 3
    results_3 = detector.detect_agents(trace)
    
    agents_3 = [k for k in results_3.keys() if k != 'env']
    valid_3 = [k for k in agents_3 if results_3[k]['blanket_validation']['valid'] != False]
    
    print(f"Result: {len(valid_3)}/{len(agents_3)} agents passed validation")
    for agent_id in agents_3:
        validation = results_3[agent_id]['blanket_validation']
        status = "✅ PASSED" if validation['valid'] != False else "❌ FAILED"
        print(f"  Agent {agent_id}: {status} (CMI: {validation['violation']:.3f})")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"N=2: {len(valid_2)}/{len(agents_2)} valid ({'✅ Good' if len(valid_2) >= 1 else '❌ Bad'})")
    print(f"N=3: {len(valid_3)}/{len(agents_3)} valid ({'✅ Good' if len(valid_3) == 0 else '❌ Bad'})")
    
    if len(valid_2) >= 1 and len(valid_3) == 0:
        print("🎯 Perfect! Threshold correctly distinguishes real agents from over-clustering")
    elif len(valid_2) >= 1:
        print("⚠️ Threshold may be too lenient (N=3 also passes)")
    else:
        print("⚠️ Threshold may still be too strict (N=2 fails)")


if __name__ == '__main__':
    test_validation_threshold() 