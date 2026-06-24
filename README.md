# Unsupervised Agent Discovery

This project implements a principled framework for discovering autonomous agents within raw dynamical systems without supervision, based on the theoretical foundations of **Markov blankets** and **active inference**.

To try it out, use [this 1 minute quickstart with screenshots](https://docs.google.com/document/d/1e4Xx-Ez0iuN5fhrYjcV9fjS26HiEx1e4RlEAPVBlCaA/edit?tab=t.0) (browser only required).


## Overview

Complex systems—from neural networks to multi-agent simulations—often contain hidden autonomous actors with their own sensors, actions, and internal states. Traditional approaches require manual labeling or domain knowledge. This framework automatically discovers these agents by detecting **Markov blanket structures** in time series data.

## Repository Layout

This repository uses top-level Python packages plus thin runner scripts:

| Path | Role |
|------|------|
| `agency_detect/` | Core UAD package: simulators, clustering, Markov-blanket validation, CLI |
| `learn_agents/` | Main telemetry simulator and latent/refinement experiment substrate |
| `agent_spotlight/`, `hierarchical_spotlight/` | Serial and hierarchical discovery experiments |
| `amortized_agency/` | Amortized same-agent affinity models and benchmarks |
| `intention_detect/`, `data_collect/` | Outcome-influence and real-machine intention probes |
| `uad_handles/` | Handle-aware UAD toy benchmarks for the access-model UAD paper |
| `uad_worm/` | WormWideWeb / C. elegans UAD application |
| `scripts/<line>/` | Command-line runners for each research line |
| `tests/<line>/` | Tests grouped by package or research line |
| `docs/` | Experiment log, findings, paper sources, and documentation index |
| `results/` | Gitignored experiment artifacts with an index README |

The repo intentionally keeps research packages at the top level for now. See
`docs/README.md`, `tests/README.md`, and `results/README.md` for the navigation
map.

## Key Concepts

**Markov Blankets**: A mathematical boundary that separates an agent's internal dynamics from the external world. For a true autonomous agent, the Markov blanket property ensures:

```
I(Internal_{t+1}; External_{t+1} | Sensors_t, Actions_t) ≈ 0
```

This means once you know the agent's sensors and actions, the internal and external dynamics become conditionally independent—the hallmark of true autonomy.

**Variable Classification**:
- **Sensors (S)**: Variables that bring information from the environment to the agent
- **Actions (A)**: Variables through which the agent influences the environment  
- **Internal (I)**: Variables that maintain the agent's private state and memory

## What This Implementation Does

1. **Discovers Agent Boundaries**: Uses mutual information clustering to find groups of variables that belong together
2. **Validates Autonomy**: Tests whether discovered clusters satisfy the Markov blanket property using conditional mutual information
3. **Classifies Variables**: Automatically identifies which variables serve as sensors, actions, or internal states
4. **Handles Temporal Dependencies**: Accounts for memory and lagged interactions between agents
5. **Individual Agent Detection**: Can discover individual agents where each agent's sensor, action, and memory variables cluster together
6. **Material-Specific Dynamics**: Supports different agent types with distinct behavioral patterns

## Simplest Example

### Quick Start - Two Individual Agents

```bash
python -c "
from agents import generate_decoupled_trace
from detection import AgentDetector

# Generate simple system: 1 solar panel + 1 steel factory
trace = generate_decoupled_trace(n_solar_panels=1, factory_materials=['steel'])

# Detect individual agents
detector = AgentDetector()
results = detector.detect_agents(trace)
detector.print_results(results)
"
```

This minimal example creates **two completely independent agents**:

**Solar Panel Agent**: Energy collection system
- **Sensors**: Reads solar energy levels and flow capacity
- **Actions**: 0=sleep, 1=charge, 2=discharge
- **Memory**: Recent sensor readings and internal energy state

**Steel Factory Agent**: Heavy industrial production system  
- **Sensors**: Reads steel material stock and equipment quality
- **Actions**: 0=cool, 1=smelt, 2=forge
- **Memory**: Production history and equipment condition

The framework automatically discovers that these are **separate autonomous agents** operating in **different domains** (energy vs. materials) with their own sensor-action-memory coupling.

## Advanced Usage

### Multiple Agents of Same Type

```python
from agents import generate_decoupled_trace
from detection import AgentDetector

# Create system with multiple similar agents
trace = generate_decoupled_trace(
    n_solar_panels=3,
    factory_materials=['wood', 'steel', 'corn', 'corn', 'corn']
)

detector = AgentDetector()
results = detector.detect_agents(trace)
detector.print_results(results)
```

This demonstrates **functional agent discovery**: 
- Multiple corn factories are detected as **one functional agent** (they work together)
- Wood and steel factories are detected as **separate individual agents** (different dynamics)
- Solar panels are detected as **one energy collective** (shared energy domain)

### Material-Specific Behaviors

Each factory type has distinct operational characteristics:

- **Wood**: Forestry with seasonal growth cycles and environmental sensitivity
- **Steel**: Heavy industry with temperature-based decisions and thermal cycling
- **Corn**: Agriculture with seasonal patterns and moisture-dependent operations

## Results Example

For the simplest two-agent system, the framework typically discovers:

**Agent 0: Solar Panel**
- **Variables**: `Solar1_sensor`, `Solar1_action`, `Solar1_mem0`, `Solar1_mem1`, `Solar1_mem2`
- **Structure**: Complete individual agent with tightly coupled sensor-action-memory

**Agent 1: Steel Factory** 
- **Variables**: `Steel1_sensor`, `Steel1_action`, `Steel1_mem0`, `Steel1_mem1`, `Steel1_mem2`
- **Structure**: Complete individual agent with material-specific dynamics

## Applications

Potential application include 

- **Neural Data Analysis**: Discover functional modules in brain recordings
- **Multi-Agent Systems**: Identify emergent agents in complex simulations
- **Biological Systems**: Find autonomous subsystems in cellular or ecological data
- **Robotics**: Detect modular controllers in complex robotic systems
- **Industrial Systems**: Identify independent production units in manufacturing

## Theoretical Foundation

Based on the paper "Foundations of Unsupervised Agent Discovery in Raw Dynamical Systems" which unifies:
- **Active Inference**: Agents as free-energy minimizers
- **Information Theory**: Mutual information and conditional independence
- **Markov Blanket Theory**: Formal boundaries of autonomous systems

## Getting Started

### Complete Demo

```bash
python detect.py
```

This runs the full demonstration with multiple agents and displays:
1. Simulation data generation with realistic agent dynamics
2. Variable variance analysis and clustering process
3. Individual agent detection with sensor-action-memory grouping
4. Classification results showing discovered agent boundaries

### Modular Architecture

The core package is organized into clean, modular components:

- **`agency_detect/config.py`**: Simulation and detection configuration parameters
- **`agency_detect/agents.py`**: Multi-agent simulation with material-specific behaviors
- **`agency_detect/markov_blanket.py`**: Markov blanket validation using conditional mutual information
- **`agency_detect/detection.py`**: Core clustering algorithm with individual agent detection
- **`detect.py`**: Backward-compatible demonstration entry point

### Custom Configurations

```python
from agents import generate_decoupled_trace
from detection import AgentDetector
from config import SimulationConfig, DetectionConfig

# Configure simulation
SimulationConfig.SIMULATION_STEPS = 20000  # More data for better statistics
SimulationConfig.RANDOM_SEED = 42

# Configure detection for individual agents
DetectionConfig.N_AGENTS = 8  # Match expected number of individual agents
DetectionConfig.WEAK_THRESHOLD = 0.05  # Very low to keep individual actions

# Generate data with specific composition
trace = generate_decoupled_trace(
    n_solar_panels=2, 
    factory_materials=['wood', 'steel']
)

# Run detection
detector = AgentDetector()
results = detector.detect_agents(trace)
detector.print_results(results)
```

### Backward Compatibility

The old API is still supported:

```python
from detect import detect_async_agents

results = detect_async_agents(trace, n_agents=2, weak_thresh=0.2)
```

See `dev.md` for detailed implementation notes and `docs/papers/README.md` for the paper collection.