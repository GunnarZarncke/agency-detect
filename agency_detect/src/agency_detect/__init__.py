"""
Agency Detection Package

A principled framework for discovering autonomous agents within raw dynamical systems
without supervision, based on the theoretical foundations of Markov blankets and active inference.

Key modules:
- agents: Multi-agent simulation with material-specific behaviors
- detection: Core clustering algorithm with individual agent detection
- markov_blanket: Markov blanket validation using conditional mutual information
- config: Configuration parameters for simulation and detection
"""

from .agents import generate_decoupled_trace, IndependentAgent, DecoupledEnvironment
from .detection import AgentDetector, detect_async_agents
from .markov_blanket import MarkovBlanketValidator, conditional_mutual_info_discrete
from .config import SimulationConfig, DetectionConfig

__version__ = "0.1.0"
__author__ = "Agency Detection Research Team"

__all__ = [
    # Core classes
    "AgentDetector",
    "MarkovBlanketValidator", 
    "IndependentAgent",
    "DecoupledEnvironment",
    
    # Configuration
    "SimulationConfig",
    "DetectionConfig",
    
    # Main functions
    "generate_decoupled_trace",
    "detect_async_agents",
    "conditional_mutual_info_discrete",
]

