"""
Configuration module for Maestro Simulator
Centralized management of all simulation parameters and constants
"""

from dataclasses import dataclass, field
from typing import List, Tuple
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

    # ==================== Task Stream Configuration (Heterogeneous) ====================
    NUM_TASK_STREAMS: int = 7
    """Number of independent heterogeneous task streams (devices)"""

    # Random generation ranges for task streams (Min, Max)
    STREAM_LAMBDA_RANGE: Tuple[float, float] = (0.05, 0.5)
    """Range for Poisson arrival rate lambda (tasks/sec)"""

    STREAM_ENERGY_RANGE: Tuple[float, float] = (1.8, 3.8)
    """Range for energy threshold (Volts)"""

    STREAM_NODES_RANGE: Tuple[int, int] = (1, 6)
    """Range for required number of nodes"""

    STREAM_PRIORITY_RANGE: Tuple[int, int] = (1, 3)
    """Range for task priority"""

    STREAM_DELAY_FACTOR_RANGE: Tuple[float, float] = (0.5, 3.0)
    """Range for max delay factor (MaxDelay = MeanInterval * Factor)"""

    RL_ACTION_DIM: int = 1 + 15 + 15  # = 31

    # ==================== Simulation Time Configuration ====================
    SAMPLING_PERIOD: float = 1.0
    """Sampling period for Simulink control interface (seconds)"""

    SIMULATION_DURATION: float = 40.0
    """Total simulation duration (seconds)"""

    # ==================== Energy Configuration (Global Limits) ====================
    ENERGY_THRESHOLD_MIN: float = 1.8
    """Absolute minimum energy threshold possible"""

    ENERGY_THRESHOLD_MAX: float = 3.8
    """Absolute maximum energy threshold possible"""

    # ==================== Poisson Process Configuration (Legacy/Fallback) ====================
    LAMBDA_ARRIVAL_RATE: float = 0.1
    """Default lambda parameter (used only if streams not configured correctly)"""

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
    RL_ACTION_DIM: int = 4
    """Dimension of action space"""

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
            bool: True if valid, raises ValueError or AssertionError otherwise
        """
        assert config.NUM_NODES > 0, "NUM_NODES must be positive"
        assert config.NUM_PRIORITY_LEVELS > 0, "NUM_PRIORITY_LEVELS must be positive"
        assert config.NUM_PRIORITY_LEVELS == len(config.PRIORITY_LEVELS), \
            "PRIORITY_LEVELS length must match NUM_PRIORITY_LEVELS"

        # [Updated Validation for Streams]
        assert config.NUM_TASK_STREAMS > 0, "NUM_TASK_STREAMS must be positive"
        assert config.STREAM_LAMBDA_RANGE[0] < config.STREAM_LAMBDA_RANGE[1], "Invalid Lambda Range"
        assert config.STREAM_ENERGY_RANGE[0] < config.STREAM_ENERGY_RANGE[1], "Invalid Energy Range"

        assert config.SAMPLING_PERIOD > 0, "SAMPLING_PERIOD must be positive"
        assert config.SIMULATION_DURATION > 0, "SIMULATION_DURATION must be positive"
        assert config.ENERGY_THRESHOLD_MIN < config.ENERGY_THRESHOLD_MAX, \
            "ENERGY_THRESHOLD_MIN must be less than ENERGY_THRESHOLD_MAX"
        assert config.VOLTAGE_MAX > 0, "VOLTAGE_MAX must be positive"

        assert config.STATE_DIMENSION == len(config.STATE_VARIABLE_NAMES), \
            "STATE_DIMENSION must match length of STATE_VARIABLE_NAMES"

        return True