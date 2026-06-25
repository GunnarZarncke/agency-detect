#!/usr/bin/env python
"""
Command-line interface for agency detection framework.

This provides a convenient CLI for running agent detection on data files
or generating example traces for demonstration purposes.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import json

from .agents import generate_decoupled_trace
from .detection import AgentDetector
from .config import SimulationConfig, DetectionConfig


def load_trace_data(file_path):
    """Load trace data from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == '.npy':
        return np.load(file_path, allow_pickle=True).tolist()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_results(results, output_path):
    """Save detection results to file."""
    output_path = Path(output_path)
    
    # Convert numpy types to JSON-serializable types
    serializable_results = {}
    for agent_id, agent_data in results.items():
        serializable_results[agent_id] = {
            'variables': agent_data['variables'],
            'classification': {
                'S': agent_data['classification']['S'],
                'A': agent_data['classification']['A'], 
                'I': agent_data['classification']['I']
            },
            'blanket_validation': {
                'valid': agent_data['blanket_validation']['valid'],
                'violation': float(agent_data['blanket_validation']['violation']),
                'details': agent_data['blanket_validation']['details']
            }
        }
    
    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    else:
        # Default to JSON if no extension provided
        output_path = output_path.with_suffix('.json')
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def run_example(args):
    """Run the basic example demonstration."""
    print("=== Agency Detection Example ===")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed(SimulationConfig.RANDOM_SEED)
    
    # Generate trace data
    if args.simple:
        print("Generating simple 2-agent system (1 solar panel + 1 steel factory)...")
        trace = generate_decoupled_trace(n_solar_panels=1, factory_materials=['steel'])
    else:
        print("Generating complex multi-agent system...")
        factory_materials = ['wood', 'steel', 'corn', 'corn', 'corn']
        trace = generate_decoupled_trace(
            n_solar_panels=3, 
            factory_materials=factory_materials,
            steps=args.steps
        )
    
    print(f"Generated {len(trace)} timesteps of data")
    
    # Configure detection
    config = DetectionConfig()
    if args.n_agents:
        config.N_AGENTS = args.n_agents
    if args.threshold:
        config.WEAK_THRESHOLD = args.threshold
    if args.no_validation:
        config.VALIDATE_BLANKETS = False
    
    # Run detection
    detector = AgentDetector(config)
    
    if args.adaptive:
        print("Running adaptive agent detection...")
        results = detector.adaptive_detect_agents(trace)
    else:
        print(f"Running agent detection (N_AGENTS={config.N_AGENTS})...")
        results = detector.detect_agents(trace)
    
    # Print results
    detector.print_results(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    return results


def run_detection(args):
    """Run agent detection on provided data file."""
    print(f"Loading trace data from: {args.input}")
    
    try:
        trace = load_trace_data(args.input)
        print(f"Loaded {len(trace)} timesteps of data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Configure detection
    config = DetectionConfig()
    if args.n_agents:
        config.N_AGENTS = args.n_agents
    if args.threshold:
        config.WEAK_THRESHOLD = args.threshold
    if args.no_validation:
        config.VALIDATE_BLANKETS = False
    
    # Run detection
    detector = AgentDetector(config)
    
    if args.adaptive:
        print("Running adaptive agent detection...")
        results = detector.adaptive_detect_agents(trace)
    else:
        print(f"Running agent detection (N_AGENTS={config.N_AGENTS})...")
        results = detector.detect_agents(trace)
    
    # Print results
    detector.print_results(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unsupervised Agent Discovery Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simple 2-agent example
  agency-detect example --simple
  
  # Run complex multi-agent example
  agency-detect example --steps 10000 --adaptive
  
  # Detect agents in your own data
  agency-detect detect --input data.json --output results.json
  
  # Use custom detection parameters
  agency-detect example --n-agents 5 --threshold 0.1 --no-validation
        """
    )
    
    # Global arguments
    parser.add_argument('--version', action='version', version='agency-detect 0.1.0')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Example command
    example_parser = subparsers.add_parser('example', help='Run example demonstrations')
    example_parser.add_argument('--simple', action='store_true', 
                               help='Run simple 2-agent example instead of complex multi-agent')
    example_parser.add_argument('--steps', type=int, default=2000,
                               help='Number of simulation steps (default: 2000)')
    example_parser.add_argument('--adaptive', action='store_true',
                               help='Use adaptive detection to find optimal number of agents')
    example_parser.add_argument('--n-agents', type=int,
                               help='Number of agents to detect (overrides config)')
    example_parser.add_argument('--threshold', type=float,
                               help='Weak connection threshold (overrides config)')
    example_parser.add_argument('--no-validation', action='store_true',
                               help='Disable Markov blanket validation')
    example_parser.add_argument('--output', type=str,
                               help='Save results to file (JSON format)')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Run detection on data file')
    detect_parser.add_argument('--input', required=True, type=str,
                              help='Input data file (JSON or NPY format)')
    detect_parser.add_argument('--output', type=str,
                              help='Output file for results (JSON format)')
    detect_parser.add_argument('--adaptive', action='store_true',
                              help='Use adaptive detection to find optimal number of agents')
    detect_parser.add_argument('--n-agents', type=int,
                              help='Number of agents to detect (overrides config)')
    detect_parser.add_argument('--threshold', type=float,
                              help='Weak connection threshold (overrides config)')
    detect_parser.add_argument('--no-validation', action='store_true',
                              help='Disable Markov blanket validation')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'example':
        return run_example(args)
    elif args.command == 'detect':
        return run_detection(args)
    else:
        # Default to simple example if no command specified
        print("No command specified. Running simple example...")
        print("Use 'agency-detect --help' for more options.")
        
        # Create default args for simple example
        class DefaultArgs:
            simple = True
            steps = 2000
            adaptive = False
            n_agents = None
            threshold = None
            no_validation = False
            output = None
            seed = None
        
        return run_example(DefaultArgs())


if __name__ == '__main__':
    main()

