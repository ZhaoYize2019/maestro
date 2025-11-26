"""
State Calculator Module - State Vector Computation
Converts raw sensor data and queue information into normalized MDP state vectors.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class EnergySnapshot:
    """
    Energy snapshot at a specific timestamp.
    
    Attributes:
        timestamp: Time when snapshot was taken
        node_voltages: List of voltage readings from each node (Volts)
        normalized_voltages: Normalized voltage values [0, 1]
    """
    timestamp: float
    node_voltages: np.ndarray
    normalized_voltages: np.ndarray = field(default=None, init=False)
    
    def __post_init__(self):
        """Normalize voltages after initialization"""
        if self.normalized_voltages is None:
            self.normalized_voltages = self.normalize_voltages(self.node_voltages, 4.3)
    
    @staticmethod
    def normalize_voltages(voltages: np.ndarray, v_max: float = 4.3) -> np.ndarray:
        """Normalize voltages to [0, 1] range"""
        return np.clip(voltages / v_max, 0.0, 1.0)


class IStateCalculator(ABC):
    """Abstract interface for state calculator implementations"""
    
    @abstractmethod
    def update_energy_data(self, voltages: np.ndarray, timestamp: float) -> None:
        """Update with latest energy data"""
        pass
    
    @abstractmethod
    def compute_state_vector(self,
                            queue_length: int,
                            queue_priority: Optional[int],
                            max_wait_time: float) -> np.ndarray:
        """Compute 6-dimensional state vector"""
        pass


class StateCalculator(IStateCalculator):
    """
    State Vector Computation Engine
    
    Converts raw energy measurements and queue statistics into normalized
    6-dimensional state vectors for RL agent consumption.
    
    State vector: s_t = [E_avg, σ_E², P_in, Q_len, Q_pri, T_wait]
    
    High cohesion: All state computation logic is self-contained
    Low coupling: Depends only on numpy and data classes
    """
    
    # State vector indices
    STATE_E_AVG = 0      # Average network energy
    STATE_SIGMA_E2 = 1   # Energy variance
    STATE_P_IN = 2       # Energy collection rate
    STATE_Q_LEN = 3      # Queue length
    STATE_Q_PRI = 4      # Queue priority
    STATE_T_WAIT = 5     # Max waiting time
    
    def __init__(self, 
                 num_nodes: int,
                 state_dimension: int = 6,
                 history_size: int = 100,
                 voltage_max: float = 4.3):
        """
        Initialize state calculator.
        
        Args:
            num_nodes: Number of nodes in network
            state_dimension: Dimension of state vector (should be 6)
            history_size: Maximum number of energy snapshots to keep
            voltage_max: Maximum voltage for normalization
        """
        self.num_nodes = num_nodes
        self.state_dimension = state_dimension
        self.history_size = history_size
        self.voltage_max = voltage_max
        
        # Energy history buffer
        self._energy_history: List[EnergySnapshot] = []
        
        # Previous values for rate calculation
        self._previous_e_avg: float = 0.0
        self._previous_timestamp: float = 0.0
        
        # Configuration normalization ranges
        self._max_queue_length = 1000  # For normalizing queue length
        self._max_wait_time = 3600.0   # For normalizing wait time (1 hour)
        
        logger.debug(f"StateCalculator initialized: nodes={num_nodes}, "
                    f"state_dim={state_dimension}, history_size={history_size}")
    
    def update_energy_data(self, voltages: np.ndarray, timestamp: float) -> None:
        """
        Update energy history with new measurements.
        
        Args:
            voltages: Array of voltage readings from each node
            timestamp: Timestamp of measurement
            
        Raises:
            ValueError: If voltage array has wrong dimension
        """
        if len(voltages) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} voltage readings, "
                           f"got {len(voltages)}")
        
        # Create snapshot
        snapshot = EnergySnapshot(
            timestamp=timestamp,
            node_voltages=np.array(voltages, dtype=np.float32)
        )
        
        # Maintain history buffer
        self._energy_history.append(snapshot)
        if len(self._energy_history) > self.history_size:
            self._energy_history.pop(0)
        
        logger.debug(f"Energy snapshot recorded: timestamp={timestamp}, "
                    f"mean_voltage={snapshot.node_voltages.mean():.2f}V")
    
    def _compute_average_energy(self) -> float:
        """
        Compute average network energy level (normalized).
        
        E_avg = mean(normalized voltages of all nodes) ∈ [0, 1]
        
        Returns:
            Average normalized energy level
            
        Raises:
            RuntimeError: If no energy data available
        """
        if not self._energy_history:
            raise RuntimeError("No energy data available. Call update_energy_data first.")
        
        latest = self._energy_history[-1]
        e_avg = np.mean(latest.normalized_voltages)
        
        return np.clip(e_avg, 0.0, 1.0)
    
    def _compute_energy_variance(self) -> float:
        """
        Compute network energy variance.
        
        σ_E² = var(normalized voltages of all nodes)
        
        Returns:
            Energy variance
        """
        if not self._energy_history:
            return 0.0
        
        latest = self._energy_history[-1]
        variance = np.var(latest.normalized_voltages)
        
        return float(variance)
    
    def _compute_energy_collection_rate(self) -> float:
        """
        Compute energy collection/consumption rate.
        
        P_in = ΔE / Δt where:
            ΔE = E_avg_current - E_avg_previous
            Δt = timestamp_current - timestamp_previous
        
        Returns:
            Energy rate (Volts per second) normalized to [-1, 1]
        """
        if len(self._energy_history) < 2:
            return 0.0
        
        current_e_avg = self._compute_average_energy()
        current_timestamp = self._energy_history[-1].timestamp
        
        delta_e = current_e_avg - self._previous_e_avg
        delta_t = current_timestamp - self._previous_timestamp
        
        if delta_t <= 0:
            return 0.0
        
        # Energy rate: ΔE / Δt, clipped to [-1, 1]
        rate = delta_e / delta_t
        rate = np.clip(rate, -1.0, 1.0)
        
        return float(rate)
    
    def compute_state_vector(self,
                            queue_length: int,
                            queue_priority: Optional[int],
                            max_wait_time: float) -> np.ndarray:
        """
        Compute complete 6-dimensional state vector.
        
        State vector: s_t = [E_avg, σ_E², P_in, Q_len, Q_pri, T_wait]
        
        Args:
            queue_length: Current total tasks in queue
            queue_priority: Highest priority in queue (or None if empty)
            max_wait_time: Maximum waiting time of head task (in seconds)
            
        Returns:
            Normalized state vector of shape (6,)
            
        Raises:
            RuntimeError: If state cannot be computed due to missing data
        """
        # Compute energy-related states
        e_avg = self._compute_average_energy()
        sigma_e2 = self._compute_energy_variance()
        p_in = self._compute_energy_collection_rate()
        
        # Normalize queue states
        q_len = np.clip(queue_length / self._max_queue_length, 0.0, 1.0)
        q_pri = (queue_priority if queue_priority is not None else 0) / 10.0  # Normalize priority
        t_wait = np.clip(max_wait_time / self._max_wait_time, 0.0, 1.0)
        
        # Construct state vector
        state = np.array([
            e_avg,
            sigma_e2,
            p_in,
            q_len,
            q_pri,
            t_wait
        ], dtype=np.float32)
        
        # Update previous values for next rate calculation
        self._previous_e_avg = e_avg
        self._previous_timestamp = self._energy_history[-1].timestamp
        
        logger.debug(f"State vector computed: {state}")
        
        return state
    
    def get_energy_history_stats(self) -> Dict:
        """
        Get statistics about energy history.
        
        Returns:
            Dictionary with energy statistics
        """
        if not self._energy_history:
            return {
                'num_snapshots': 0,
                'time_span': 0.0,
                'mean_voltage': 0.0,
                'std_voltage': 0.0,
                'min_voltage': 0.0,
                'max_voltage': 0.0,
            }
        
        timestamps = [s.timestamp for s in self._energy_history]
        all_voltages = np.concatenate([s.node_voltages for s in self._energy_history])
        
        return {
            'num_snapshots': len(self._energy_history),
            'time_span': timestamps[-1] - timestamps[0],
            'mean_voltage': float(np.mean(all_voltages)),
            'std_voltage': float(np.std(all_voltages)),
            'min_voltage': float(np.min(all_voltages)),
            'max_voltage': float(np.max(all_voltages)),
        }
    
    def reset(self) -> None:
        """Reset state calculator to initial state"""
        self._energy_history.clear()
        self._previous_e_avg = 0.0
        self._previous_timestamp = 0.0
        
        logger.info("StateCalculator reset")
