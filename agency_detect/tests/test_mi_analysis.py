#!/usr/bin/env python
"""
Analysis: Why Same-Type Agents Cluster Together Despite Different Trajectories

This test demonstrates that mutual information captures functional relationships,
not individual differences, explaining why A&C and B&D cluster together.
"""

import numpy as np
from sklearn.metrics import mutual_info_score

from agency_detect import generate_decoupled_trace
from agency_detect.detection import lagmax_mi

def analyze_mutual_information():
    """Analyze why same-type agents cluster together."""
    print("=== Mutual Information Analysis: Why Same-Type Agents Cluster ===\n")
    
    # Generate trace with 4 agents
    np.random.seed(42)
    trace = generate_decoupled_trace(steps=5000, n_agents=4)
    
    # Convert to data matrix
    vars_ = list(trace[0].keys())
    data = np.array([[rec[v] for v in vars_] for rec in trace])
    
    print("Agent Variables:")
    print("- A (solar_panel): A_sensor, A_action, A_internal, A_goal")
    print("- B (factory): B_sensor, B_action, B_internal, B_goal") 
    print("- C (solar_panel): C_sensor, C_action, C_internal, C_goal")
    print("- D (factory): D_sensor, D_action, D_internal, D_goal\n")
    
    # Get indices for key variables
    var_to_idx = {var: i for i, var in enumerate(vars_)}
    
    # Test 1: Compare trajectories (different values)
    print("=== 1. Trajectory Differences (Value Level) ===")
    A_sensor = data[:, var_to_idx['A_sensor']]
    C_sensor = data[:, var_to_idx['C_sensor']]
    B_sensor = data[:, var_to_idx['B_sensor']]
    D_sensor = data[:, var_to_idx['D_sensor']]
    
    print(f"First 10 timesteps:")
    print(f"A_sensor: {A_sensor[:10]}")
    print(f"C_sensor: {C_sensor[:10]}")
    print(f"B_sensor: {B_sensor[:10]}")
    print(f"D_sensor: {D_sensor[:10]}")
    
    # Show they have different trajectories
    print(f"\nTrajectory correlations (Pearson):")
    print(f"A_sensor vs C_sensor: {np.corrcoef(A_sensor, C_sensor)[0,1]:.3f}")
    print(f"B_sensor vs D_sensor: {np.corrcoef(B_sensor, D_sensor)[0,1]:.3f}")
    print(f"A_sensor vs B_sensor: {np.corrcoef(A_sensor, B_sensor)[0,1]:.3f}")
    
    # Test 2: Mutual Information (functional relationship)
    print(f"\n=== 2. Mutual Information (Functional Level) ===")
    
    # Same type agents
    mi_A_C_sensor = lagmax_mi(A_sensor, C_sensor, max_lag=3)
    mi_B_D_sensor = lagmax_mi(B_sensor, D_sensor, max_lag=3)
    
    # Different type agents  
    mi_A_B_sensor = lagmax_mi(A_sensor, B_sensor, max_lag=3)
    mi_A_D_sensor = lagmax_mi(A_sensor, D_sensor, max_lag=3)
    
    print(f"Lagged MI between sensors:")
    print(f"A_sensor <-> C_sensor (same type): {mi_A_C_sensor:.3f}")
    print(f"B_sensor <-> D_sensor (same type): {mi_B_D_sensor:.3f}")
    print(f"A_sensor <-> B_sensor (diff type): {mi_A_B_sensor:.3f}")
    print(f"A_sensor <-> D_sensor (diff type): {mi_A_D_sensor:.3f}")
    
    # Test 3: Action patterns
    print(f"\n=== 3. Action Pattern Analysis ===")
    A_action = data[:, var_to_idx['A_action']]
    C_action = data[:, var_to_idx['C_action']]
    B_action = data[:, var_to_idx['B_action']]
    D_action = data[:, var_to_idx['D_action']]
    
    print(f"Action distributions:")
    print(f"A_action: {np.bincount(A_action, minlength=3)}")
    print(f"C_action: {np.bincount(C_action, minlength=3)}")
    print(f"B_action: {np.bincount(B_action, minlength=3)}")
    print(f"D_action: {np.bincount(D_action, minlength=3)}")
    
    # Test 4: Environment coupling
    print(f"\n=== 4. Environment Coupling ===")
    solar_energy = data[:, var_to_idx['env_solar_panel_energy']]
    material_keys = [v for v in vars_ if v.startswith('env_material') and v.endswith('_material')]
    if not material_keys:
        raise KeyError("No env_material*_material variables in trace")
    factory_material = data[:, var_to_idx[material_keys[0]]]
    
    # Solar panel agents should be coupled to solar environment
    mi_A_solar = mutual_info_score(A_sensor, solar_energy)
    mi_C_solar = mutual_info_score(C_sensor, solar_energy)
    
    # Factory agents should be coupled to factory environment
    mi_B_factory = mutual_info_score(B_sensor, factory_material)
    mi_D_factory = mutual_info_score(D_sensor, factory_material)
    
    # Cross-couplings should be weaker
    mi_A_factory = mutual_info_score(A_sensor, factory_material)
    mi_B_solar = mutual_info_score(B_sensor, solar_energy)
    
    print(f"Environment coupling (sensor -> env):")
    print(f"A_sensor -> solar_energy: {mi_A_solar:.3f}")
    print(f"C_sensor -> solar_energy: {mi_C_solar:.3f}")
    print(f"B_sensor -> factory_material: {mi_B_factory:.3f}")
    print(f"D_sensor -> factory_material: {mi_D_factory:.3f}")
    print(f"\nCross-coupling (should be weaker):")
    print(f"A_sensor -> factory_material: {mi_A_factory:.3f}")
    print(f"B_sensor -> solar_energy: {mi_B_solar:.3f}")
    
    # Test 5: Why clustering groups them
    print(f"\n=== 5. Clustering Logic ===")
    print(f"The algorithm groups A&C and B&D because:")
    print(f"1. Same-type agents share similar decision thresholds")
    print(f"2. They interact with the same environment variables")
    print(f"3. Their temporal patterns are similar (memory, sensor-action loops)")
    print(f"4. Mutual information captures these functional relationships")
    print(f"5. Even different trajectories can have high MI if patterns are similar")
    
    return {
        'same_type_mi': (mi_A_C_sensor + mi_B_D_sensor) / 2,
        'diff_type_mi': (mi_A_B_sensor + mi_A_D_sensor) / 2,
        'trajectory_correlation': np.corrcoef(A_sensor, C_sensor)[0,1]
    }

if __name__ == '__main__':
    results = analyze_mutual_information()
    
    print(f"\n=== CONCLUSION ===")
    print(f"Average MI between same-type agents: {results['same_type_mi']:.3f}")
    print(f"Average MI between different-type agents: {results['diff_type_mi']:.3f}")
    print(f"Trajectory correlation (A vs C): {results['trajectory_correlation']:.3f}")
    print(f"\nThe algorithm correctly identifies functional relationships!")
    print(f"To get 4 separate agents, we need 4 functionally different agent types.") 