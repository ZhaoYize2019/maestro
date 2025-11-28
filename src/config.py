"""
Configuration module for Maestro Simulator
Centralized management of all simulation parameters and constants
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class SimulationConfig:
    """
    Configuration class for Maestro simulator parameters.
    
    All simulation parameters are centralized here to enable easy modification
    and maintain low coupling between modules.
    """
    
    # ==================== Network Configuration ====================
    NUM_NODES: int = 25
    """Total number of nodes in the network"""
    
    VOLTAGE_MAX: float = 4.3
    """Maximum voltage for node capacitor (Volts)"""
    
    # ==================== Task Priority Configuration ====================
    NUM_PRIORITY_LEVELS: int = 3
    """Number of priority levels (K in the paper)"""
    
    PRIORITY_LEVELS: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Priority levels from lowest to highest"""
    
    # ==================== Task Type Configuration ====================
    NUM_TASK_TYPES: int = 3
    """Number of different task types (M in the paper)"""
    
    TASK_TYPES: List[str] = field(
        default_factory=lambda: ["sampling", "computing", "communication"]
    )
    """Task type names"""
    
    # ==================== Simulation Time Configuration ====================
    SAMPLING_PERIOD: float = 1.0
    """Sampling period for Simulink control interface (seconds)"""
    
    SIMULATION_DURATION: float = 40
    """Total simulation duration (seconds)"""
    
    # ==================== Energy Configuration ====================
    ENERGY_THRESHOLD_MIN: float = 1.8
    """Minimum energy threshold for task execution (Volts)"""
    
    ENERGY_THRESHOLD_MAX: float = 3.8
    """Maximum energy threshold for task execution (Volts)"""
    
    # ==================== Poisson Process Configuration ====================
    LAMBDA_ARRIVAL_RATE: float = 0.1
    """Lambda parameter for Poisson process - task arrival rate (tasks per second)"""
    
    # ==================== State Vector Configuration ====================
    STATE_DIMENSION: int = 6
    """Dimension of state vector: [E_avg, σ_E², P_in, Q_len, Q_pri, T_wait]"""
    
    STATE_VARIABLE_NAMES: List[str] = field(
        default_factory=lambda: [
            "E_avg",      # Average network energy level
            "sigma_E2",   # Network energy variance
            "P_in",       # Energy collection rate
            "Q_len",      # Task queue length
            "Q_pri",      # Highest priority in queue
            "T_wait"      # Max waiting time
        ]
    )
    """Names of state variables for debugging and monitoring"""
    
    # ==================== RL Configuration ====================
    RL_ACTION_DIM: int = 3 + 3  # = 6
    """Dimension of action space (sampling period, priority, energy threshold, node count)"""


    # ==================== Data Logging Configuration ====================
    ENABLE_LOGGING: bool = True
    """Enable detailed logging"""
    
    LOG_LEVEL: str = "INFO"
    """Logging level"""
    
    SAVE_TRAJECTORY: bool = True
    """Save complete trajectory data for analysis"""
    
    TRAJECTORY_FILE: str = "./maestro_trajectory.pkl"
    """Path to save trajectory data"""
    
    @classmethod
    def validate(cls, config: 'SimulationConfig') -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        assert config.NUM_NODES > 0, "NUM_NODES must be positive"
        assert config.NUM_PRIORITY_LEVELS > 0, "NUM_PRIORITY_LEVELS must be positive"
        assert config.NUM_PRIORITY_LEVELS == len(config.PRIORITY_LEVELS), \
            "PRIORITY_LEVELS length must match NUM_PRIORITY_LEVELS"
        assert config.NUM_TASK_TYPES > 0, "NUM_TASK_TYPES must be positive"
        assert config.NUM_TASK_TYPES == len(config.TASK_TYPES), \
            "TASK_TYPES length must match NUM_TASK_TYPES"
        assert config.SAMPLING_PERIOD > 0, "SAMPLING_PERIOD must be positive"
        assert config.SIMULATION_DURATION > 0, "SIMULATION_DURATION must be positive"
        assert config.ENERGY_THRESHOLD_MIN < config.ENERGY_THRESHOLD_MAX, \
            "ENERGY_THRESHOLD_MIN must be less than ENERGY_THRESHOLD_MAX"
        assert config.VOLTAGE_MAX > 0, "VOLTAGE_MAX must be positive"
        assert config.LAMBDA_ARRIVAL_RATE > 0, "LAMBDA_ARRIVAL_RATE must be positive"
        assert config.STATE_DIMENSION == len(config.STATE_VARIABLE_NAMES), \
            "STATE_DIMENSION must match length of STATE_VARIABLE_NAMES"
        
        return True
