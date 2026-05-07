"""
Core agent detection algorithm using mutual information clustering.

This module implements the main detection pipeline:
1. Mutual information clustering to find agent boundaries
2. Variable classification into sensors/actions/internal
3. Markov blanket validation (optional)
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering

from .config import DetectionConfig
from .markov_blanket import MarkovBlanketValidator


def lagmax_mi(x, y, max_lag=None):
    """
    Compute maximum mutual information over time lags.
    
    This captures temporal dependencies and memory effects between variables.
    Works with discrete integer values.
    
    Args:
        x, y: Time series arrays (discrete integer values)
        max_lag: Maximum lag to consider (default from config)
        
    Returns:
        Maximum mutual information across all tested lags
    """
    if max_lag is None:
        max_lag = DetectionConfig.MAX_LAG
        
    best = 0.0
    for τ in range(-max_lag, max_lag + 1):
        if τ == 0:
            w = mutual_info_score(x, y)  # synchronous influence
            best = max(best, w)
            continue
        if τ > 0: 
            xi, yi = x[:-τ], y[τ:]
        else:     
            xi, yi = x[-τ:], y[:τ]
        best = max(best, mutual_info_score(xi, yi))
    return best


def build_similarity_matrix(data, max_lag=None):
    """
    Build similarity matrix using lagged mutual information.
    
    Args:
        data: Time series data matrix (time x variables)
        max_lag: Maximum lag for temporal dependencies
        
    Returns:
        Similarity matrix and distance matrix for clustering
    """
    if max_lag is None:
        max_lag = DetectionConfig.MAX_LAG
        
    n_vars = data.shape[1]
    sim = np.zeros((n_vars, n_vars))
    
    # Compute pairwise lagged mutual information
    for i, j in combinations(range(n_vars), 2):
        w = lagmax_mi(data[:, i], data[:, j], max_lag=max_lag)
        sim[i, j] = sim[j, i] = w
    
    # Convert to distance matrix
    dist = 1.0 - sim / (sim.max() + 1e-12)  # normalize → distance
    
    return sim, dist


def filter_weak_connections(clusters, vars_active, sim, weak_thresh=None):
    """
    Remove weakly connected variables from clusters.
    
    Variables that don't have strong connections to their cluster
    are moved to the environment bucket.
    
    Args:
        clusters: Dictionary of cluster labels to variable lists
        vars_active: List of active variable names
        sim: Similarity matrix
        weak_thresh: Threshold for weak connections (default from config)
        
    Returns:
        Filtered clusters and list of environment variables
    """
    if weak_thresh is None:
        weak_thresh = DetectionConfig.WEAK_THRESHOLD
        
    env_bucket = []
    filtered_clusters = {}
    
    for lbl in list(clusters.keys()):
        mem = clusters[lbl]
        if len(mem) <= 1:  # Handle singletons later
            continue
            
        idx = [vars_active.index(v) for v in mem]
        sub_sim = sim[np.ix_(idx, idx)]
        
        # Average similarity between all pairs inside this cluster
        mean_intra = (sub_sim.sum() - np.trace(sub_sim)) / (len(idx) * (len(idx) - 1))
        
        # Test every variable against that average
        for v in mem[:]:  # iterate over a copy
            j = vars_active.index(v)
            sim_to_cluster = (sub_sim[idx.index(j)].sum() - 0) / (len(idx) - 1)
            if sim_to_cluster < weak_thresh * mean_intra:
                mem.remove(v)
                env_bucket.append(v)
        
        if mem:  # Cluster not empty after filtering
            filtered_clusters[lbl] = mem
    
    return filtered_clusters, env_bucket


class AgentDetector:
    """
    Main agent detection class that orchestrates the detection pipeline.
    """
    
    def __init__(self, config=None):
        self.config = config or DetectionConfig
        self.validator = MarkovBlanketValidator(config)
    
    def detect_agents(self, trace):
        """
        Main detection pipeline.
        
        Args:
            trace: List of dictionaries containing variable states over time
            
        Returns:
            Dictionary of detected agents with classifications and validation results
        """
        # Convert trace to data matrix
        vars_ = list(trace[0].keys())
        data = np.array([[rec[v] for v in vars_] for rec in trace])
        
        # Step 1: Remove variables that never change
        var_variance = data.var(axis=0)
        active_idx = np.where(var_variance > 0.0)[0]
        inactive_idx = np.where(var_variance == 0.0)[0]
        
        vars_active = [vars_[i] for i in active_idx]
        data_active = data[:, active_idx]
        
        print(f"Active variables: {len(vars_active)} out of {len(vars_)}")
        print(f"Variables with variance > 0: {vars_active}")
        
        if len(vars_active) < 2:
            print(f"ERROR: Only {len(vars_active)} active variables, need at least 2 for clustering")
            return {}
        
        # Step 2: Build similarity matrix using lagged mutual information
        sim, dist = build_similarity_matrix(data_active, self.config.MAX_LAG)
        
        # Step 3: Cluster variables
        clustering = AgglomerativeClustering(
            n_clusters=self.config.N_AGENTS, 
            metric='precomputed', 
            linkage='complete'
        )
        labels = clustering.fit_predict(dist)
        
        # Group variables by cluster
        clusters = defaultdict(list)
        for v, lbl in zip(vars_active, labels):
            clusters[lbl].append(v)
        
        print(f"Initial clusters: {dict(clusters)}")
        
        # Step 4: Filter weak connections
        filtered_clusters, env_bucket = filter_weak_connections(
            clusters, vars_active, sim, self.config.WEAK_THRESHOLD
        )
        
        print(f"Filtered clusters: {dict(filtered_clusters)}")
        print(f"Environment variables: {env_bucket}")
        
        # Step 5: Add inactive variables to environment
        for i in inactive_idx:
            env_bucket.append(vars_[i])
        
        # Step 6: Validate clusters and classify variables
        validated_clusters = {}
        failed_variables = []  # Track variables from failed agents
        
        for lbl, variables in filtered_clusters.items():
            if len(variables) > 0:
                result = self.validator.validate_cluster(variables, vars_, data, trace)
                
                # Check if validation failed - if so, track for reassignment
                if (result['blanket_validation']['valid'] == False and 
                    self.config.VALIDATE_BLANKETS):
                    print(f"Agent {lbl} failed Markov blanket validation: {result['blanket_validation']['details']}")
                    failed_variables.extend(variables)
                else:
                    validated_clusters[lbl] = {
                        'variables': variables,
                        **result
                    }
        
        # Failed variables go to environment (simpler approach)
        env_bucket.extend(failed_variables)
        
        # Add environment cluster
        if env_bucket:
            validated_clusters['env'] = {
                'variables': env_bucket,
                'classification': {'S': [], 'A': [], 'I': env_bucket},
                'blanket_validation': {
                    'valid': None,
                    'violation': 0.0,
                    'details': 'Environment variables'
                }
            }
        
        return validated_clusters
    
    def print_results(self, clusters):
        """
        Pretty print detection results.
        
        Args:
            clusters: Result from detect_agents()
        """
        print("=== Enhanced Agent Detection with Markov Blanket Validation ===\n")
        
        for lbl in sorted(k for k in clusters if k != 'env'):
            cluster_info = clusters[lbl]
            print(f"Agent {lbl}:")
            print(f"  Variables: {cluster_info['variables']}")
            
            classification = cluster_info['classification']
            print(f"  Sensors (S): {classification['S']}")
            print(f"  Actions (A): {classification['A']}")  
            print(f"  Internal (I): {classification['I']}")
            
            validation = cluster_info['blanket_validation']
            print(f"  Markov Blanket: Valid={validation['valid']}, Violation={validation['violation']:.4f}")
            print(f"  Details: {validation['details']}")
            print()
        
        if 'env' in clusters:
            print("Environment:")
            print(f"  Variables: {clusters['env']['variables']}")
            print(f"  Details: {clusters['env']['blanket_validation']['details']}")



    def adaptive_detect_agents(self, trace, min_success_rate=0.7, max_attempts=None):
        """
        Adaptive agent detection that tries different numbers of agents.
        
        Starts with the configured N_AGENTS and reduces until enough agents pass validation.
        
        Args:
            trace: Time series data
            min_success_rate: Minimum fraction of agents that must pass validation
            max_attempts: Maximum number of different N_AGENTS to try (default: original N_AGENTS)
            
        Returns:
            Detection results with optimal number of agents
        """
        if max_attempts is None:
            max_attempts = self.config.N_AGENTS
            
        original_n_agents = self.config.N_AGENTS
        
        print(f"=== Adaptive Agent Detection (starting with N_AGENTS={original_n_agents}) ===")
        
        best_results = None
        best_score = -1
        
        for attempt, n_agents in enumerate(range(original_n_agents, 0, -1)):
            if attempt >= max_attempts:
                break
                
            print(f"\n--- Attempt {attempt + 1}: Trying N_AGENTS={n_agents} ---")
            
            # Temporarily set N_AGENTS
            self.config.N_AGENTS = n_agents
            
            try:
                results = self.detect_agents(trace)
                
                # Count valid agents
                total_agents = len([k for k in results.keys() if k != 'env'])
                valid_agents = []
                failed_agents = []
                
                for agent_id, agent_data in results.items():
                    if agent_id != 'env':
                        is_valid = agent_data['blanket_validation']['valid']
                        if is_valid == True or is_valid is None:  # None means validation skipped
                            valid_agents.append(agent_id)
                        else:
                            failed_agents.append(agent_id)
                
                success_rate = len(valid_agents) / total_agents if total_agents > 0 else 0
                
                # Score based on number of valid agents and success rate
                score = len(valid_agents) + success_rate
                
                print(f"Results: {len(valid_agents)}/{total_agents} agents valid (success rate: {success_rate:.1%})")
                
                if len(valid_agents) > 0 and success_rate >= min_success_rate:
                    print(f"✅ Success! Found {len(valid_agents)} valid agents")
                    best_results = results
                    break
                elif score > best_score:
                    print(f"📈 Best so far (score: {score:.2f})")
                    best_results = results
                    best_score = score
                else:
                    print(f"📉 Worse than best (score: {score:.2f} vs {best_score:.2f})")
                    
            except Exception as e:
                print(f"❌ Error with N_AGENTS={n_agents}: {e}")
                continue
        
        # Restore original configuration
        self.config.N_AGENTS = original_n_agents
        
        if best_results is not None:
            valid_count = len([k for k in best_results.keys() if k != 'env' and 
                             best_results[k]['blanket_validation']['valid'] != False])
            print(f"\n🎯 Final result: {valid_count} valid agents detected")
            return best_results
        else:
            print(f"\n⚠️ No valid agents found at any clustering level")
            return {}


def detect_async_agents(trace, **kwargs):
    """
    Convenience function for backward compatibility.
    
    Args:
        trace: Time series data
        **kwargs: Configuration overrides
        
    Returns:
        Detection results
    """
    # Create config with overrides
    config = DetectionConfig()
    for key, value in kwargs.items():
        if hasattr(config, key.upper()):
            setattr(config, key.upper(), value)
    
    detector = AgentDetector(config)
    return detector.detect_agents(trace) 