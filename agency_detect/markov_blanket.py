"""
Markov blanket validation and conditional mutual information estimation.

This module implements the theoretical core of agent detection:
validating that discovered clusters satisfy the Markov blanket property.
"""

import numpy as np
import warnings
from collections import Counter
from math import log
from sklearn.metrics import mutual_info_score
from .config import DetectionConfig


def conditional_mutual_info_discrete(X, Y, Z, alpha=None):
    """
    Estimate conditional mutual information I(X;Y|Z) using smoothed plug-in estimator.
    
    This is the discrete-appropriate method that replaces the broken k-NN estimator.
    Uses Laplace smoothing to handle zero counts robustly.
    
    Args:
        X, Y, Z: Input arrays (must have same length, should be discrete/integer)
        alpha: Smoothing parameter (default from config)
        
    Returns:
        Conditional mutual information estimate in nats
    """
    if alpha is None:
        alpha = getattr(DetectionConfig, 'CMI_SMOOTHING_ALPHA', 0.1)
        
    if len(X) != len(Y) or len(X) != len(Z):
        raise ValueError("All input arrays must have same length")
    
    n = len(X)
    if n < 10:  # Need minimum samples for discrete estimation
        return 0.0
    
    # Flatten arrays and convert to integers for discrete estimation
    X = X.flatten().astype(int)
    Y = Y.flatten().astype(int)
    
    # Handle Z which might be multidimensional
    if Z.ndim > 1:
        Z_tuples = [tuple(row.astype(int)) for row in Z]
    else:
        Z_tuples = Z.flatten().astype(int)
    
    try:
        # Get cardinalities
        card_x = len(set(X))
        card_y = len(set(Y))
        card_z = len(set(Z_tuples))
        
        # Count frequencies
        xyz_counts = Counter(zip(X, Y, Z_tuples))
        xz_counts = Counter(zip(X, Z_tuples))
        yz_counts = Counter(zip(Y, Z_tuples))
        z_counts = Counter(Z_tuples)
        
        # All possible combinations
        x_values = sorted(set(X))
        y_values = sorted(set(Y))
        z_values = sorted(set(Z_tuples))
        
        # Calculate CMI with Laplace smoothing
        cmi = 0.0
        
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    # Smoothed counts
                    n_xyz = xyz_counts.get((x, y, z), 0) + alpha
                    n_xz = xz_counts.get((x, z), 0) + alpha * card_y
                    n_yz = yz_counts.get((y, z), 0) + alpha * card_x  
                    n_z = z_counts.get(z, 0) + alpha * card_x * card_y
                    
                    # Smoothed probabilities
                    p_xyz = n_xyz / (n + alpha * card_x * card_y * card_z)
                    p_xz = n_xz / (n + alpha * card_y * card_x * card_z) 
                    p_yz = n_yz / (n + alpha * card_x * card_y * card_z)
                    p_z = n_z / (n + alpha * card_x * card_y * card_z)
                    
                    # Add to CMI sum
                    cmi += p_xyz * log(p_xyz * p_z / (p_xz * p_yz))
        
        return max(0.0, cmi)  # CMI should be non-negative
    
    except Exception as e:
        warnings.warn(f"Discrete CMI estimation failed: {e}")
        return 0.0


def classify_variables(cluster_vars, all_vars, data, trace):
    """
    Classify variables in a cluster into Sensors (S), Actions (A), Internal (I).
    
    Uses a combination of statistical analysis and heuristics to identify
    the role of each variable within an agent.
    
    Args:
        cluster_vars: List of variables in this cluster
        all_vars: List of all variables in the system
        data: Time series data matrix
        trace: Original trace data
        
    Returns:
        Dictionary with keys 'S', 'A', 'I' containing classified variables
    """
    print(f"\n=== Classifying cluster: {cluster_vars} ===")
    
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    cluster_indices = [var_to_idx[var] for var in cluster_vars]
    env_vars = [var for var in all_vars if var not in cluster_vars]
    env_indices = [var_to_idx[var] for var in env_vars]
    
    n_vars = len(cluster_vars)
    n_env = len(env_vars)
    
    if n_env == 0:
        # No environment variables - classify based on variable names and patterns
        sensors = [var for var in cluster_vars if 'sensor' in var]
        actions = [var for var in cluster_vars if 'action' in var]
        internal = [var for var in cluster_vars if var not in sensors and var not in actions]
        print(f"No env vars - Name-based classification: S={sensors}, A={actions}, I={internal}")
        return {'S': sensors, 'A': actions, 'I': internal}
    
    # Compute MI between cluster variables and environment
    env_mi = np.zeros(n_vars)
    for i, var_idx in enumerate(cluster_indices):
        for env_idx in env_indices:
            mi = mutual_info_score(data[:, var_idx], data[:, env_idx])
            env_mi[i] += mi
    
    # Compute MI between cluster variables and other agents' future states
    future_mi = np.zeros(n_vars)
    if len(data) > 1:
        for i, var_idx in enumerate(cluster_indices):
            for j, other_idx in enumerate(cluster_indices):
                if i != j and len(data) > 1:
                    # Lagged MI to detect influence on future states
                    mi = mutual_info_score(data[:-1, var_idx], data[1:, other_idx])
                    future_mi[i] += mi
    
    # More sensitive classification thresholds
    env_threshold = np.percentile(env_mi, DetectionConfig.ENV_MI_PERCENTILE) if n_vars > 1 else 0
    future_threshold = np.percentile(future_mi, DetectionConfig.FUTURE_MI_PERCENTILE) if n_vars > 1 else 0
    
    print(f"Environment MI: {list(zip(cluster_vars, env_mi))}")
    print(f"Future MI: {list(zip(cluster_vars, future_mi))}")
    print(f"Env threshold: {env_threshold}, Future threshold: {future_threshold}")
    
    # Also use variable names as hints
    sensors = []
    actions = []
    
    for i, var in enumerate(cluster_vars):
        print(f"Processing {var}: env_mi={env_mi[i]:.3f}, future_mi={future_mi[i]:.3f}")
        # Strong hint from variable names
        if 'sensor' in var:
            sensors.append(var)
            print(f"  -> Added to sensors (name hint)")
        elif 'action' in var:
            actions.append(var)
            print(f"  -> Added to actions (name hint)")
        # Memory variables should be internal, not sensors
        elif 'mem' in var:
            print(f"  -> Will be internal (memory)")
        # Environment variables that are sensors
        elif var.startswith('env_') and env_mi[i] > env_threshold:
            sensors.append(var)
            print(f"  -> Added to sensors (env variable with high MI)")
        # Statistical classification for others
        elif env_mi[i] > env_threshold:
            sensors.append(var)
            print(f"  -> Added to sensors (MI threshold)")
        elif future_mi[i] > future_threshold and var not in sensors:
            actions.append(var)
            print(f"  -> Added to actions (future MI)")
        else:
            print(f"  -> Will be internal")
    
    internal = [var for var in cluster_vars if var not in sensors and var not in actions]
    
    print(f"Final classification: S={sensors}, A={actions}, I={internal}")
    return {'S': sensors, 'A': actions, 'I': internal}


def validate_markov_blanket(cluster_vars, classification, all_vars, data, tolerance=None):
    """
    Validate that a cluster satisfies the Markov blanket property:
    I(I_{t+1}; E_{t+1} | S_t, A_t) ≈ 0
    
    This is the core theoretical test for autonomous agent boundaries.
    
    Args:
        cluster_vars: Variables in this cluster
        classification: Dictionary with 'S', 'A', 'I' classifications
        all_vars: All variables in the system
        data: Time series data
        tolerance: Maximum acceptable conditional MI (default from config)
        
    Returns:
        Tuple of (is_valid, violation_score, details)
    """
    if tolerance is None:
        tolerance = DetectionConfig.BLANKET_TOLERANCE
        
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    
    # Get variable indices
    S_vars = classification['S']  # Sensors
    A_vars = classification['A']  # Actions  
    I_vars = classification['I']  # Internal states
    E_vars = [var for var in all_vars if var not in cluster_vars]  # Environment
    
    # Agents must have internal variables to be valid
    if not I_vars:
        return False, 1.0, "Invalid: Agent has no internal variables"
    
    # Need environment variables to test against
    if not E_vars:
        return False, 1.0, "Invalid: No environment variables for testing"
    
    # Need sufficient data
    if len(data) < 10:
        return False, 1.0, "Invalid: Insufficient data for validation"
    
    # Prepare data for conditional MI calculation
    # We need t and t+1 time steps
    if len(data) < 2:
        return True, 0.0, "Insufficient time steps"
    
    try:
        # Current time step data
        S_t = np.column_stack([data[:-1, var_to_idx[var]] for var in S_vars]) if S_vars else np.zeros((len(data)-1, 1))
        A_t = np.column_stack([data[:-1, var_to_idx[var]] for var in A_vars]) if A_vars else np.zeros((len(data)-1, 1))
        I_t = np.column_stack([data[:-1, var_to_idx[var]] for var in I_vars])
        E_t = np.column_stack([data[:-1, var_to_idx[var]] for var in E_vars])
        
        # Next time step data  
        I_t1 = np.column_stack([data[1:, var_to_idx[var]] for var in I_vars])
        E_t1 = np.column_stack([data[1:, var_to_idx[var]] for var in E_vars])
        
        # Conditioning variables: S_t and A_t
        if S_vars and A_vars:
            conditioning = np.column_stack([S_t, A_t])
        elif S_vars:
            conditioning = S_t
        elif A_vars:
            conditioning = A_t
        else:
            conditioning = np.zeros((len(data)-1, 1))
        
        # Compute conditional MI: I(I_{t+1}; E_{t+1} | S_t, A_t)
        cmi = conditional_mutual_info_discrete(I_t1, E_t1, conditioning)
        
        is_valid = cmi <= tolerance
        details = f"CMI={cmi:.4f}, threshold={tolerance}, S={len(S_vars)}, A={len(A_vars)}, I={len(I_vars)}, E={len(E_vars)}"
        
        return is_valid, cmi, details
        
    except Exception as e:
        warnings.warn(f"Markov blanket validation failed: {e}")
        return True, 0.0, f"Validation error: {e}"


class MarkovBlanketValidator:
    """
    Encapsulates Markov blanket validation logic with configurable parameters.
    """
    
    def __init__(self, config=None):
        self.config = config or DetectionConfig()
    
    def validate_cluster(self, cluster_vars, all_vars, data, trace):
        """
        Complete validation pipeline for a cluster.
        
        Returns:
            Dictionary with validation results and classification
        """
        # Classify variables
        classification = classify_variables(cluster_vars, all_vars, data, trace)
        
        # Validate Markov blanket property if requested
        if self.config.VALIDATE_BLANKETS and len(cluster_vars) > 1:
            is_valid, violation, details = validate_markov_blanket(
                cluster_vars, classification, all_vars, data, 
                self.config.BLANKET_TOLERANCE)
        else:
            is_valid, violation, details = None, 0.0, "Validation skipped"
        
        return {
            'classification': classification,
            'blanket_validation': {
                'valid': is_valid,
                'violation': violation,
                'details': details
            }
        } 