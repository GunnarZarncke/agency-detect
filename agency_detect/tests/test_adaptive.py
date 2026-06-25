#!/usr/bin/env python
"""
Test adaptive agent detection that automatically finds the right number of agents.
"""

import numpy as np

from agency_detect import DetectionConfig, generate_decoupled_trace, AgentDetector


def test_adaptive_detection():
    """Test adaptive detection starting with wrong N_AGENTS."""
    
    print("=== Testing Adaptive Agent Detection ===\n")
    
    # Generate data (we know this has 2 real agents)
    np.random.seed(42)
    trace = generate_decoupled_trace(steps=2000)
    
    # Set up detection with validation enabled and wrong N_AGENTS
    config = DetectionConfig()
    config.N_AGENTS = 4  # Intentionally too high
    config.VALIDATE_BLANKETS = True
    config.WEAK_THRESHOLD = 0.2
    
    detector = AgentDetector(config)
    
    print(f"🎯 True number of agents: 2")
    print(f"🔧 Starting with N_AGENTS: {config.N_AGENTS}")
    print(f"🧪 Validation enabled: {config.VALIDATE_BLANKETS}")
    
    # Test adaptive detection
    results = detector.adaptive_detect_agents(trace, min_success_rate=0.8)
    
    # Print final results
    print("\n" + "="*60)
    detector.print_results(results)


def test_normal_vs_adaptive():
    """Compare normal detection vs adaptive detection."""
    
    print("\n" + "="*60)
    print("=== Comparison: Normal vs Adaptive Detection ===")
    
    np.random.seed(42)
    trace = generate_decoupled_trace(steps=1000)
    
    config = DetectionConfig()
    config.N_AGENTS = 3  # Wrong number
    config.VALIDATE_BLANKETS = True
    config.WEAK_THRESHOLD = 0.2
    
    detector = AgentDetector(config)
    
    print(f"\n--- Normal Detection (N_AGENTS=3) ---")
    normal_results = detector.detect_agents(trace)
    normal_agents = len([k for k in normal_results.keys() if k != 'env'])
    normal_valid = len([k for k, v in normal_results.items() 
                       if k != 'env' and v['blanket_validation']['valid'] != False])
    print(f"Result: {normal_valid}/{normal_agents} valid agents")
    
    print(f"\n--- Adaptive Detection (starting with N_AGENTS=3) ---")
    adaptive_results = detector.adaptive_detect_agents(trace, min_success_rate=0.8)
    adaptive_agents = len([k for k in adaptive_results.keys() if k != 'env'])
    adaptive_valid = len([k for k, v in adaptive_results.items() 
                         if k != 'env' and v['blanket_validation']['valid'] != False])
    print(f"Result: {adaptive_valid}/{adaptive_agents} valid agents")
    
    print(f"\n🏆 Winner: {'Adaptive' if adaptive_valid > normal_valid else 'Normal'} detection!")


if __name__ == '__main__':
    test_adaptive_detection()
    test_normal_vs_adaptive() 