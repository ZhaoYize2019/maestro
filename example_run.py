"""
Example: Running Maestro Simulator
Demonstrates how to use the Maestro framework
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import simulator
from src.config import SimulationConfig
from src.simulator import MaestroSimulator


def main():
    """Main entry point"""
    
    # Create configuration
    config = SimulationConfig(
        NUM_NODES=25,
        NUM_PRIORITY_LEVELS=3,
        NUM_TASK_TYPES=3,
        SAMPLING_PERIOD=1.0,
        SIMULATION_DURATION=120.0,  # 2 minutes for testing
        LAMBDA_ARRIVAL_RATE=0.1,
    )
    
    # Validate configuration
    try:
        SimulationConfig.validate(config)
        print("✓ Configuration validated")
    except AssertionError as e:
        print(f"✗ Configuration error: {e}")
        return 1
    
    # Create simulator (RL disabled for now - no agent implementation)
    simulator = MaestroSimulator(config, rl_enabled=False)
    
    # Run simulation
    try:
        simulator.run()
        print("✓ Simulation completed successfully")
        return 0
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
