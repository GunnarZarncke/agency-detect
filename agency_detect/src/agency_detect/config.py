"""
Configuration parameters for unsupervised agent discovery.

Split into separate classes for simulation and detection to allow
independent configuration of each component.
"""


class SimulationConfig:
    """Configuration for agent simulation parameters."""
    
    # Simulation parameters
    SIMULATION_STEPS = 50000  # Increased for better statistical reliability
    RANDOM_SEED = 42
    
    # Agent design parameters
    MEMORY_SIZE = 3
    N_AGENTS = 8  # Total agents - 3 solar + 5 factories (when using builder method)
    
    # Solar Panel agent parameters
    SOLAR_PANEL_ENERGY_RANGE = 10
    SOLAR_PANEL_FLOW_RANGE = 6
    SOLAR_PANEL_STATE_RANGE = 8
    SOLAR_PANEL_GOAL_RANGE = 10
    
    # Factory agent parameters  
    FACTORY_MATERIAL_RANGE = 8
    FACTORY_QUALITY_RANGE = 6
    FACTORY_STATE_RANGE = 6
    FACTORY_GOAL_RANGE = 8
    
    # Environment update probabilities
    SOLAR_PANEL_ENV_UPDATE_PROB = 0.4
    SOLAR_PANEL_FLOW_UPDATE_PROB = 0.3
    
    # Material-specific factory parameters for more diversity
    FACTORY_PARAMS = {
        'wood': {
            'env_update_prob': 0.4,        # Higher environmental change
            'quality_update_prob': 0.1,    # Lower natural restoration  
            'action_effect_prob': 0.9,     # Very high action effectiveness
            'production_efficiency': 1.5,   # Much more efficient
            'maintenance_cost_factor': 0.05 # Low maintenance cost
        },
        'steel': {
            'env_update_prob': 0.05,       # Very low environmental change (stable)
            'quality_update_prob': 0.5,    # Very high natural restoration
            'action_effect_prob': 0.4,     # Low action effectiveness (sluggish)
            'production_efficiency': 0.6,   # Much less efficient production
            'maintenance_cost_factor': 0.18 # Very high maintenance cost
        },
        'corn': {
            'env_update_prob': 0.35,       # High environmental change (volatile)
            'quality_update_prob': 0.05,   # Very low natural restoration  
            'action_effect_prob': 0.95,    # Very high action effectiveness (responsive)
            'production_efficiency': 1.8,   # Much higher efficiency
            'maintenance_cost_factor': 0.03 # Very low maintenance cost
        }
    }
    
    # Default factory parameters for unknown materials
    DEFAULT_FACTORY_PARAMS = {
        'env_update_prob': 0.2,
        'quality_update_prob': 0.25,
        'action_effect_prob': 0.7,
        'production_efficiency': 1.0,
        'maintenance_cost_factor': 0.08
    }


class DetectionConfig:
    """Configuration for agent detection algorithm."""
    
    # Clustering parameters
    N_AGENTS = 8  # Match number of simulated agents for individual clustering
    MAX_LAG = 3
    WEAK_THRESHOLD = 0.05  # Very low to keep individual agent actions
    
    # Classification parameters
    ENV_MI_PERCENTILE = 50  # Percentile for environment MI threshold
    FUTURE_MI_PERCENTILE = 50  # Percentile for future MI threshold
    
    # Markov blanket validation parameters
    VALIDATE_BLANKETS = True  # Can be enabled for strict validation
    BLANKET_TOLERANCE = 1.0  # Adjusted for discrete CMI estimation (was 5.0 for broken k-NN)
    CMI_SMOOTHING_ALPHA = 0.1  # Laplace smoothing parameter for discrete CMI


# Default configurations for backward compatibility
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_DETECTION_CONFIG = DetectionConfig() 