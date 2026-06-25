"""
Agent simulation classes for generating test data.

This module contains independent Solar Panel and Factory agents operating
in separate domains to demonstrate clean agent boundary detection.

Solar Panel agents manage energy collection and discharge cycles.
Factory agents manage production and equipment maintenance.
"""

import numpy as np
from .config import SimulationConfig


class IndependentAgent:
    """
    An autonomous agent operating in a specific resource domain.
    
    Each agent has:
    - Private sensors that monitor its domain (energy levels or production capacity)
    - Actions that affect its domain (sleep/charge/discharge or idle/produce/maintain)
    - Internal state representing system condition
    - Goal-directed behavior with realistic dynamics
    """
    
    def __init__(self, name, agent_type, material_type=None, memory_size=None):
        if memory_size is None:
            memory_size = SimulationConfig.MEMORY_SIZE
            
        self.name = name
        self.agent_type = agent_type  # 'solar_panel' or 'factory'
        self.material_type = material_type  # For factories: 'wood', 'steel', 'corn', etc.
        self.memory = np.zeros(memory_size, dtype=int)
        self.internal_state = 0
        self.action = 0
        self.private_sensor = 0
        self.goal_progress = 0
        
        # CRITICAL: Add agent-specific ID for unique signatures
        # This creates distinct patterns for each individual agent
        self.agent_id = hash(name) % 1000  # Much larger range for more distinction
        self.agent_multiplier = (self.agent_id % 7) + 1  # Unique multiplier 1-7
        
        # Anti-stuck mechanism: Track recent actions to prevent getting trapped
        self.recent_actions = []
        self.steps_count = 0
        
        # Initialize with agent-specific starting conditions
        if agent_type == 'solar_panel':
            self.internal_state = 1 + (self.agent_id % 3)  # Vary by agent
            self.goal_progress = 0
        else:  # factory
            self.internal_state = 3 + (self.agent_id % 2)  # Vary by agent
            self.goal_progress = 2 + (self.agent_id % 3)  # Vary by agent
    
    def sense_private_domain(self, environment):
        """Each agent senses its domain with realistic sensor functions"""
        
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy level sensor with noise and saturation
            energy_level = environment.get('solar_panel_energy', 5)
            energy_flow = environment.get('solar_panel_flow', 1)
            
            # Realistic sensor combining energy state and flow rate
            base_reading = energy_level + energy_flow * 0.5
            sensor_noise = np.random.normal(0, 0.2)
            raw_sensor = base_reading + sensor_noise
            
            # Store as float but will be discretized in get_state_dict()
            self.private_sensor = max(0, min(8, raw_sensor))
            
        else:  # factory agent
            # Material-specific sensor functions for distinct behavioral signatures
            material_stock = environment.get(f'{self.material_type}_material', 3)
            material_quality = environment.get(f'{self.material_type}_quality', 2)
            
            # Add agent-specific sensor characteristics for individual signatures
            agent_noise_scale = 0.1 + (self.agent_multiplier * 0.02)  # Unique noise per agent
            agent_bias = (self.agent_multiplier - 4) * 0.03  # Unique bias per agent
            
            if self.material_type == 'steel':
                # Steel: Temperature/pressure sensor (emphasizes quality over quantity)
                temperature_reading = material_quality * 1.8 + material_stock * 0.2  # Quality-focused
                thermal_noise = np.random.normal(agent_bias, 0.25 + agent_noise_scale)  # Agent-specific noise
                raw_sensor = temperature_reading + thermal_noise
                
            elif self.material_type == 'corn':
                # Corn: Moisture/nutrients sensor (emphasizes stock over quality)  
                moisture_reading = material_stock * 1.5 + material_quality * 0.3  # Stock-focused
                seasonal_noise = np.random.normal(agent_bias, 0.1 + agent_noise_scale) * (1 + 0.5 * np.cos(len(self.memory) * 0.1))  # Agent-specific seasonal noise
                raw_sensor = moisture_reading + seasonal_noise
                
            elif self.material_type == 'wood':
                # Wood: Growth/biomass sensor (balanced but with growth dynamics)
                growth_reading = material_stock * 0.8 + material_quality * 1.0  # Balanced
                growth_noise = np.random.exponential(0.15 + agent_noise_scale) + agent_bias  # Agent-specific growth noise
                raw_sensor = growth_reading + growth_noise
                
            else:
                # Default sensor for unknown materials
                capacity_indicator = material_stock * 0.6 + material_quality * 1.2
                sensor_noise = np.random.normal(agent_bias, 0.15 + agent_noise_scale)
                raw_sensor = capacity_indicator + sensor_noise
            
            # Store as float but will be discretized in get_state_dict()
            self.private_sensor = max(0, min(8, raw_sensor))
    
    def decide_action(self):
        """Material-specific decision making with strong sensor coupling and anti-stuck mechanism"""
        self.steps_count += 1
        
        # Track recent actions to detect if we're stuck
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)  # Keep only last 10 actions
        
        # Normalize both influences to [0,1] scale
        memory_influence = np.mean(self.memory) / 8.0 if len(self.memory) > 0 else 0
        sensor_influence = self.private_sensor / 8.0  # Already normalized
        
        # CRITICAL: Make action VERY strongly dependent on THIS agent's own sensor
        # This creates tight coupling between sensor and action variables
        if self.agent_type == 'factory':
            # Factories need EVEN STRONGER sensor coupling to overcome low variance
            agent_specific_coupling = sensor_influence * 1.2  # Extremely strong for factories
        else:
            agent_specific_coupling = sensor_influence * 0.9  # Strong for solar panels
        
        # Add agent-specific decision characteristics for unique signatures
        agent_threshold_shift = (self.agent_multiplier - 4) * 0.02  # More pronounced unique shifts
        
        # Anti-stuck mechanism: If stuck in same action, inject variance
        stuck_variance = 0
        if len(self.recent_actions) >= 8:
            unique_actions = len(set(self.recent_actions[-8:]))  # Check last 8 actions
            if unique_actions == 1:  # Completely stuck
                stuck_variance = np.random.normal(0, 0.3)  # Large variance injection
            elif unique_actions == 2:  # Mostly stuck
                stuck_variance = np.random.normal(0, 0.15)  # Medium variance injection
        
        # Add realistic noise but keep it smaller to preserve coupling
        if self.agent_type == 'factory':
            # Factories need even less noise to strengthen sensor-action coupling
            noise = np.random.normal(0, 0.01)  # Very small noise for factories
            demand_signal = agent_specific_coupling + memory_influence * 0.05 + noise + stuck_variance
        else:
            noise = np.random.normal(0, 0.03)  # Slightly more noise for solar panels
            demand_signal = agent_specific_coupling + memory_influence * 0.1 + noise + stuck_variance
        
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy harvesting agent with agent-specific thresholds
            # Actions: 0=sleep, 1=charge, 2=discharge
            threshold1 = 0.35 + agent_threshold_shift
            threshold2 = 0.75 + agent_threshold_shift
            if demand_signal < threshold1:
                self.action = 0  # sleep
            elif demand_signal < threshold2:
                self.action = 1  # charge
            else:
                self.action = 2  # discharge
                
        elif self.material_type == 'steel':
            # Steel: Heavy industry with temperature-based decisions
            # Actions: 0=cool, 1=smelt, 2=forge (different action meanings)
            # Steel uses dynamic temperature-aware thresholds
            temperature_factor = np.sin(len(self.memory) * 0.3) * 0.2  # More thermal cycling
            steel_signal = demand_signal + temperature_factor
            
            # MUCH more dynamic thresholds to prevent getting stuck
            base_threshold1 = 0.3 + np.sin(len(self.memory) * 0.1) * 0.1  # Dynamic threshold
            base_threshold2 = 0.6 + np.cos(len(self.memory) * 0.15) * 0.1  # Dynamic threshold
            threshold1 = base_threshold1 + agent_threshold_shift
            threshold2 = base_threshold2 + agent_threshold_shift
            
            if steel_signal < threshold1:
                self.action = 0  # cool down
            elif steel_signal < threshold2:
                self.action = 1  # smelt/heat
            else:
                self.action = 2  # forge/shape
                
        elif self.material_type == 'corn':
            # Corn: Agriculture with seasonal/moisture-based decisions  
            # Actions: 0=plant, 1=harvest, 2=irrigate (different action meanings)
            # Corn uses highly dynamic seasonal patterns 
            seasonal_factor = np.cos(len(self.memory) * 0.08) * 0.25  # Stronger seasonal cycling
            weather_factor = np.sin(len(self.memory) * 0.12) * 0.15  # Weather variation
            moisture_sensitivity = memory_influence * 0.6  # More memory-dependent
            corn_signal = demand_signal + seasonal_factor + weather_factor + moisture_sensitivity
            
            # Dynamic agricultural thresholds based on season
            base_threshold1 = 0.25 + np.cos(len(self.memory) * 0.06) * 0.1  # Seasonal planting
            base_threshold2 = 0.55 + np.sin(len(self.memory) * 0.09) * 0.1  # Seasonal harvest
            threshold1 = base_threshold1 + agent_threshold_shift
            threshold2 = base_threshold2 + agent_threshold_shift
            
            if corn_signal < threshold1:
                self.action = 0  # plant
            elif corn_signal < threshold2:
                self.action = 1  # harvest
            else:
                self.action = 2  # irrigate
                
        elif self.material_type == 'wood':
            # Wood: Forestry with growth-based decisions
            # Actions: 0=idle, 1=produce, 2=maintain (original meanings)
            # Wood uses highly dynamic growth cycles
            growth_factor = memory_influence * 0.5  # More growth dependency
            seasonal_growth = np.sin(len(self.memory) * 0.04) * 0.2  # Seasonal growth
            weather_impact = np.cos(len(self.memory) * 0.07) * 0.15  # Weather effects
            wood_signal = demand_signal + growth_factor + seasonal_growth + weather_impact
            
            # Dynamic forestry thresholds based on growth cycles
            base_threshold1 = 0.25 + np.sin(len(self.memory) * 0.05) * 0.12  # Growth-based idle
            base_threshold2 = 0.65 + np.cos(len(self.memory) * 0.08) * 0.12  # Growth-based production
            threshold1 = base_threshold1 + agent_threshold_shift
            threshold2 = base_threshold2 + agent_threshold_shift
            
            if wood_signal < threshold1:
                self.action = 0  # idle/wait for growth
            elif wood_signal < threshold2:
                self.action = 1  # harvest/produce
            else:
                self.action = 2  # maintain/replant
        else:
            # Default factory behavior for unknown materials
            threshold1 = 0.35 + agent_threshold_shift
            threshold2 = 0.75 + agent_threshold_shift
            if demand_signal < threshold1:
                self.action = 0  # idle
            elif demand_signal < threshold2:
                self.action = 1  # produce
            else:
                self.action = 2  # maintain
        
        # Record this action in our recent actions list
        self.recent_actions.append(self.action)
        
        return self.action
    
    def update_state(self):
        """State updates with agent-specific memory coupling"""
        # CRITICAL: Make memory agent-specific by combining sensor + action + agent ID
        # This creates strong coupling between sensor, action, and memory variables
        base_signature = int(round(self.private_sensor)) + self.action * 2  # Sensor+action combo
        agent_signature = (base_signature + self.agent_multiplier) % 9  # Add unique agent pattern
        agent_signature = min(8, max(0, agent_signature))  # Keep in valid range
        
        # Update memory with agent-specific signature
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = agent_signature
        
        # More realistic internal state updates with momentum
        if self.agent_type == 'solar_panel':
            # Solar Panel: Energy system internal state with momentum
            state_change = (self.action - 1) * 0.3 + np.random.normal(0, 0.1)
            self.internal_state = max(0, min(SimulationConfig.SOLAR_PANEL_STATE_RANGE, 
                                           self.internal_state + state_change))
            
            # Goal progress with realistic advancement/decay
            if self.action == 1:  # charging advances goal
                progress_gain = min(0.5, self.private_sensor / 10.0)
                self.goal_progress = min(SimulationConfig.SOLAR_PANEL_GOAL_RANGE, 
                                       self.goal_progress + progress_gain)
            else:
                # Other actions cause slight goal decay
                self.goal_progress = max(0, self.goal_progress - 0.1)
                
        else:  # factory agent
            # Factory: Production system with wear-based state changes
            wear_factor = max(0.1, self.action / 3.0)  # more action = more wear
            state_change = self.action * 0.2 - wear_factor + np.random.normal(0, 0.05)
            self.internal_state = max(0, min(SimulationConfig.FACTORY_STATE_RANGE, 
                                           self.internal_state + state_change))
            
            # Goal progress based on production efficiency
            if self.action == 1:  # producing advances goal
                efficiency = max(0.1, self.internal_state / SimulationConfig.FACTORY_STATE_RANGE)
                progress_gain = efficiency * 0.4
                self.goal_progress = min(SimulationConfig.FACTORY_GOAL_RANGE, 
                                       self.goal_progress + progress_gain)
            elif self.action == 2:  # maintenance slightly advances goal
                self.goal_progress = min(SimulationConfig.FACTORY_GOAL_RANGE, 
                                       self.goal_progress + 0.1)
            # Idle (action=0) causes no goal change
    
    def get_state_dict(self):
        """Return agent state with all values discretized to integers"""
        return {
            f'{self.name}_sensor': int(round(self.private_sensor)),
            f'{self.name}_action': int(self.action),
            f'{self.name}_internal': int(round(self.internal_state)),
            f'{self.name}_goal': int(round(self.goal_progress)),
            **{f'{self.name}_mem{i}': int(self.memory[i]) for i in range(len(self.memory))}
        }


class DecoupledEnvironment:
    """
    Environment with separate energy and resource management domains.
    
    Solar Panel domain: Energy system with collection/discharge dynamics
    - solar_panel_energy: Available energy level (0 to SOLAR_PANEL_ENERGY_RANGE)
    - solar_panel_flow: Energy flow capacity (0 to SOLAR_PANEL_FLOW_RANGE)
    
    Factory domains: Resource production systems with maintenance (one per material type)
    - {material}_material: Raw material stock (0 to FACTORY_MATERIAL_RANGE)  
    - {material}_quality: Equipment condition (0 to FACTORY_QUALITY_RANGE)
    
    This design ensures clean agent separation with no shared variables.
    """
    
    def __init__(self, material_types=None):
        if material_types is None:
            material_types = ['wood', 'steel', 'corn']
            
        # Separate domains with no shared variables
        
        # Solar Panel domain: Energy collection system
        self.solar_panel_energy = 5    # Current stored energy
        self.solar_panel_flow = 1      # Energy flow/collection capacity
        
        # Factory domains: Resource production systems (one per material)
        self.material_types = material_types
        for material in material_types:
            setattr(self, f'{material}_material', 3)  # Raw material inventory
            setattr(self, f'{material}_quality', 2)   # Equipment condition/efficiency
        
        # Domains are completely independent!
        
    def update(self, agents):
        """Update with realistic resource dynamics using float computations"""
        solar_panel_agents = [a for a in agents if a.agent_type == 'solar_panel']
        factory_agents = [a for a in agents if a.agent_type == 'factory']
        
        # Solar Panel domain: Energy system with realistic dynamics
        # Aggregate actions from all solar panel agents
        if solar_panel_agents:
            # Count actions across all solar panel agents
            sleep_count = sum(1 for a in solar_panel_agents if a.action == 0)
            charge_count = sum(1 for a in solar_panel_agents if a.action == 1)
            discharge_count = sum(1 for a in solar_panel_agents if a.action == 2)
            
            # Apply dominant action with magnitude based on count
            total_agents = len(solar_panel_agents)
            if charge_count > discharge_count and charge_count > sleep_count:
                # Charge action dominates
                charge_strength = charge_count / total_agents
                energy_gain = min(2.0 * charge_strength, self.solar_panel_flow * 1.5)
                self.solar_panel_energy = min(SimulationConfig.SOLAR_PANEL_ENERGY_RANGE, 
                                      self.solar_panel_energy + energy_gain)
                self.solar_panel_flow = max(0, self.solar_panel_flow - 0.3 * charge_strength)
            elif discharge_count > sleep_count:
                # Discharge action dominates
                discharge_strength = discharge_count / total_agents
                self.solar_panel_energy = max(0, self.solar_panel_energy - 1.5 * discharge_strength)
                flow_restoration = min(0.5 * discharge_strength, 
                                     (SimulationConfig.SOLAR_PANEL_FLOW_RANGE - self.solar_panel_flow) * 0.4)
                self.solar_panel_flow = min(SimulationConfig.SOLAR_PANEL_FLOW_RANGE, 
                                    self.solar_panel_flow + flow_restoration)
            else:
                # Sleep action dominates (or tie)
                sleep_strength = max(sleep_count, 1) / total_agents
                self.solar_panel_energy = max(0, self.solar_panel_energy - 0.1 * sleep_strength)
                self.solar_panel_flow = self.solar_panel_flow * (0.95 ** sleep_strength)
        
        # Natural solar panel dynamics with exponential trends
        if np.random.random() < SimulationConfig.SOLAR_PANEL_ENV_UPDATE_PROB:
            # Environmental energy input with saturation
            energy_input = np.random.exponential(0.5)
            self.solar_panel_energy = min(SimulationConfig.SOLAR_PANEL_ENERGY_RANGE, 
                                   self.solar_panel_energy + energy_input)
        
        # Factory domains: Resource production with wear and maintenance (per material type)
        if factory_agents:
            # Group factory agents by material type
            agents_by_material = {}
            for agent in factory_agents:
                material = agent.material_type
                if material not in agents_by_material:
                    agents_by_material[material] = []
                agents_by_material[material].append(agent)
            
            # Update each material domain independently with material-specific parameters
            for material, material_agents in agents_by_material.items():
                # Get material-specific parameters
                params = SimulationConfig.FACTORY_PARAMS.get(material, SimulationConfig.DEFAULT_FACTORY_PARAMS)
                
                # Check if actions have effect for this material
                if np.random.random() >= params['action_effect_prob']:
                    continue  # Skip this material this step
                
                # Count actions for this material type
                idle_count = sum(1 for a in material_agents if a.action == 0)
                produce_count = sum(1 for a in material_agents if a.action == 1)
                maintain_count = sum(1 for a in material_agents if a.action == 2)
                
                total_agents = len(material_agents)
                material_stock = getattr(self, f'{material}_material')
                material_quality = getattr(self, f'{material}_quality')
                
                if produce_count > idle_count and produce_count > maintain_count:
                    # Production action dominates
                    produce_strength = produce_count / total_agents
                    production_rate = max(0.1, material_quality / SimulationConfig.FACTORY_QUALITY_RANGE)
                    production_rate *= params['production_efficiency']  # Material-specific efficiency
                    new_material = min(SimulationConfig.FACTORY_MATERIAL_RANGE, 
                                     material_stock + production_rate * produce_strength)
                    new_quality = max(0, material_quality - 0.2 * produce_strength)
                    setattr(self, f'{material}_material', new_material)
                    setattr(self, f'{material}_quality', new_quality)
                    
                elif maintain_count > idle_count:
                    # Maintenance action dominates
                    maintain_strength = maintain_count / total_agents
                    maintenance_cost = min(0.15, material_stock * params['maintenance_cost_factor']) * maintain_strength
                    new_material = max(0, material_stock - maintenance_cost)
                    
                    # Reduced maintenance effectiveness to prevent runaway quality gains
                    base_quality_gain = maintenance_cost * 5 + 0.2 * maintain_strength
                    # Add diminishing returns for high quality
                    diminishing_factor = max(0.1, 1.0 - (material_quality / SimulationConfig.FACTORY_QUALITY_RANGE) * 0.5)
                    quality_gain = min(0.8, base_quality_gain * diminishing_factor)
                    
                    new_quality = min(SimulationConfig.FACTORY_QUALITY_RANGE, 
                                    material_quality + quality_gain)
                    setattr(self, f'{material}_material', new_material)
                    setattr(self, f'{material}_quality', new_quality)
                else:
                    # Idle action dominates (or tie)
                    idle_strength = max(idle_count, 1) / total_agents
                    new_quality = max(0, material_quality - 0.05 * idle_strength)
                    setattr(self, f'{material}_quality', new_quality)
        
        # Natural factory dynamics with realistic decay (per material type)
        for material in self.material_types:
            # Get material-specific parameters
            params = SimulationConfig.FACTORY_PARAMS.get(material, SimulationConfig.DEFAULT_FACTORY_PARAMS)
            
            if np.random.random() < params['env_update_prob']:
                # Material naturally depletes (consumption/spoilage)
                depletion = np.random.exponential(0.3)
                current_material = getattr(self, f'{material}_material')
                setattr(self, f'{material}_material', max(0, current_material - depletion))
            
            if np.random.random() < params['quality_update_prob']:
                # Quality has some natural restoration
                restoration = np.random.gamma(2, 0.1)  # Small positive restoration
                current_quality = getattr(self, f'{material}_quality')
                new_quality = min(SimulationConfig.FACTORY_QUALITY_RANGE, 
                                current_quality + restoration)
                setattr(self, f'{material}_quality', new_quality)
        
    def get(self, key, default=0):
        """Get environment variable"""
        return getattr(self, key, default)
    
    def get_state_dict(self):
        """Return environment states with all values discretized to integers"""
        state_dict = {
            'env_solar_panel_energy': int(round(self.solar_panel_energy)),
            'env_solar_panel_flow': int(round(self.solar_panel_flow)),
        }
        
        # Add environment variables for each material type
        for material in self.material_types:
            state_dict[f'env_{material}_material'] = int(round(getattr(self, f'{material}_material')))
            state_dict[f'env_{material}_quality'] = int(round(getattr(self, f'{material}_quality')))
            
        return state_dict


def build_multi_agent_system(n_solar_panels=3, factory_materials=['wood', 'steel', 'corn', 'corn', 'corn']):
    """
    Builder method to create a multi-agent system with specified composition.
    
    Args:
        n_solar_panels: Number of solar panel agents
        factory_materials: List of materials for factory agents (one factory per material specified)
        
    Returns:
        Tuple of (agents_list, environment, material_types)
    """
    agents = []
    
    # Create solar panel agents with clear naming
    for i in range(n_solar_panels):
        agent_name = f'Solar{i+1}'
        agents.append(IndependentAgent(agent_name, 'solar_panel'))
    
    # Create factory agents for each material type with descriptive naming
    material_counters = {}
    for material in factory_materials:
        if material not in material_counters:
            material_counters[material] = 0
        material_counters[material] += 1
        
        agent_name = f'{material.title()}{material_counters[material]}'
        agents.append(IndependentAgent(agent_name, 'factory', material))
    
    # Get unique material types for environment
    unique_materials = list(set(factory_materials))
    environment = DecoupledEnvironment(unique_materials)
    
    return agents, environment, unique_materials


def generate_decoupled_trace(steps=None, n_agents=None, n_solar_panels=None, factory_materials=None):
    """
    Generate trace from independent Solar Panel and Factory agents.
    
    Creates realistic simulation of:
    - Solar Panel agent: Energy collection system with sleep/charge/discharge actions
      • sleep (0): Low power mode with slow energy decay
      • charge (1): Active energy collection with efficiency cost
      • discharge (2): Energy release with capacity restoration
      
    - Factory agent: Resource production system with idle/produce/maintain actions
      • idle (0): Equipment gradually degrades when unused
      • produce (1): Manufacture goods but wear equipment
      • maintain (2): Service equipment to improve quality
    
    Agents operate in separate domains with realistic dynamics including:
    - Sensor noise and saturation
    - Threshold-based decision making  
    - Exponential decay and momentum effects
    - Resource constraints and trade-offs
    
    Args:
        steps: Number of simulation steps (default from config)
        n_agents: Number of agents to simulate (default from config, ignored if using builder)
        n_solar_panels: Number of solar panel agents (uses builder method if specified)
        factory_materials: List of materials for factories (uses builder method if specified)
        
    Returns:
        List of dictionaries containing variable states at each timestep
    """
    if steps is None:
        steps = SimulationConfig.SIMULATION_STEPS
    
    # Use builder method if specific composition is requested
    if n_solar_panels is not None or factory_materials is not None:
        if n_solar_panels is None:
            n_solar_panels = 3
        if factory_materials is None:
            factory_materials = ['wood', 'steel', 'corn', 'corn', 'corn']
        agents, environment, _ = build_multi_agent_system(n_solar_panels, factory_materials)
    else:
        # Use simple alternating method for backward compatibility
        if n_agents is None:
            n_agents = SimulationConfig.N_AGENTS
            
        # Create agents alternating between solar_panel and factory types
        agents = []
        agent_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Support up to 10 agents
        
        for i in range(n_agents):
            agent_name = agent_names[i] if i < len(agent_names) else f'Agent{i}'
            agent_type = 'solar_panel' if i % 2 == 0 else 'factory'
            agent_material = None if agent_type == 'solar_panel' else f'material{i//2}'
            agents.append(IndependentAgent(agent_name, agent_type, agent_material))
        
        # Get unique materials for environment
        unique_materials = list(set(a.material_type for a in agents if a.material_type is not None))
        environment = DecoupledEnvironment(unique_materials if unique_materials else ['wood'])
    
    trace = []
    
    for _ in range(steps):
        # 1. Each agent senses only its own domain
        for agent in agents:
            agent.sense_private_domain(environment)
        
        # 2. Each agent decides completely independently
        for agent in agents:
            agent.decide_action()
        
        # 3. Environment updates independent domains
        environment.update(agents)
        
        # 4. Agents update states independently
        for agent in agents:
            agent.update_state()
        
        # 5. Record timestep
        record = {}
        for agent in agents:
            record.update(agent.get_state_dict())
        record.update(environment.get_state_dict())
        
        trace.append(record)
    
    return trace 