"""
Reinforcement Learning Interface Module
Provides abstract interface for RL integration while keeping the core framework
independent of specific RL algorithms.

This module defines the contract for RL agents without implementing any
specific algorithm. This design allows easy swapping of different RL implementations.
"""

import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import Enum

if TYPE_CHECKING:
    from .task_manager import TaskManager
    from .simulink_interface import SimulinkInterface

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Enumeration of possible RL action types"""
    SAMPLING_PERIOD = "sampling_period"      # Adjust Simulink sampling period
    TASK_PRIORITY = "task_priority"          # Adjust task priority
    ENERGY_THRESHOLD = "energy_threshold"    # Adjust energy threshold
    REQUIRED_NODES = "required_nodes"        # Adjust nodes required for task


class RLAction:
    """
    Represents an action taken by RL agent.
    
    The action space is 4-dimensional:
    - sampling_period: float - new sampling period for data collection
    - task_priority: int - priority level to adjust
    - energy_threshold: float - new energy threshold
    - required_nodes: int - number of nodes required
    """
    
    def __init__(self,
                 sampling_period: float = None,
                 task_priority: int = None,
                 energy_threshold: float = None,
                 required_nodes: int = None):
        """
        Initialize RL action.
        
        Args:
            sampling_period: New sampling period (seconds)
            task_priority: Task priority level to modify
            energy_threshold: New energy threshold (Volts)
            required_nodes: New required node count
        """
        self.sampling_period = sampling_period
        self.task_priority = task_priority
        self.energy_threshold = energy_threshold
        self.required_nodes = required_nodes
    
    def to_vector(self) -> np.ndarray:
        """Convert action to vector form for RL agent"""
        return np.array([
            self.sampling_period or 0.0,
            self.task_priority or 0,
            self.energy_threshold or 0.0,
            self.required_nodes or 0
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, action_vector: np.ndarray) -> 'RLAction':
        """Construct action from vector output by RL agent"""
        if len(action_vector) != 4:
            raise ValueError(f"Expected 4-dim action, got {len(action_vector)}")
        
        return cls(
            sampling_period=float(action_vector[0]),
            task_priority=int(action_vector[1]),
            energy_threshold=float(action_vector[2]),
            required_nodes=int(action_vector[3])
        )


class RLReward:
    """
    Reward signal for RL agent.
    
    The reward is designed to optimize:
    - Task completion rate (maximize)
    - Energy efficiency (minimize waste)
    - Response time (minimize)
    """
    
    def __init__(self,
                 base_reward: float = 0.0,
                 task_completion_bonus: float = 0.0,
                 energy_efficiency_bonus: float = 0.0,
                 response_time_penalty: float = 0.0,
                 deadline_miss_penalty: float = 0.0):
        """
        Initialize reward object.
        
        Args:
            base_reward: Base reward value
            task_completion_bonus: Bonus for completed tasks
            energy_efficiency_bonus: Bonus for efficient energy use
            response_time_penalty: Penalty for slow response
            deadline_miss_penalty: Penalty for missed deadlines
        """
        self.base_reward = base_reward
        self.task_completion_bonus = task_completion_bonus
        self.energy_efficiency_bonus = energy_efficiency_bonus
        self.response_time_penalty = response_time_penalty
        self.deadline_miss_penalty = deadline_miss_penalty
    
    @property
    def total(self) -> float:
        """Calculate total reward"""
        return (self.base_reward +
                self.task_completion_bonus +
                self.energy_efficiency_bonus -
                self.response_time_penalty -
                self.deadline_miss_penalty)
    
    def __repr__(self) -> str:
        return f"Reward(total={self.total:.3f}, completion={self.task_completion_bonus:.3f}, " \
               f"efficiency={self.energy_efficiency_bonus:.3f}, " \
               f"response_penalty={self.response_time_penalty:.3f})"


class IRLAgent(ABC):
    """Abstract interface for RL agent implementations"""
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> RLAction:
        """Select action based on current state"""
        pass
    
    @abstractmethod
    def update(self, 
               state: np.ndarray,
               action: RLAction,
               reward: RLReward,
               next_state: np.ndarray,
               done: bool) -> None:
        """Update agent based on experience (S-A-R-S')"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent model/policy"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent model/policy"""
        pass


class RLInterface:
    """
    Reinforcement Learning Interface Module
    
    Provides abstraction layer between simulation framework and RL agent.
    Translates simulation states and actions, manages RL training loop.
    
    This interface is algorithm-agnostic - specific RL implementations
    (CQL, DQN, A3C, etc.) are plugged in as IRLAgent implementations.
    
    High cohesion: All RL interface logic is self-contained
    Low coupling: Depends only on abstract IRLAgent interface
    """
    
    def __init__(self,
                 state_dimension: int = 6,
                 action_dimension: int = 4,
                 rl_agent: Optional[IRLAgent] = None,
                 enabled: bool = False):
        """
        Initialize RL interface.
        
        Args:
            state_dimension: Dimension of state vector (typically 6)
            action_dimension: Dimension of action space (typically 4)
            rl_agent: RL agent implementation (None for placeholder)
            enabled: Whether RL is enabled for this simulation
        """
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.rl_agent = rl_agent
        self.enabled = enabled and (rl_agent is not None)
        
        # Training statistics
        self._episode_count = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._rewards_history: List[float] = []
        
        logger.info(f"RL Interface initialized: enabled={self.enabled}, "
                   f"state_dim={state_dimension}, action_dim={action_dimension}")
    
    def select_action(self, state: np.ndarray) -> RLAction:
        """
        Select action using RL agent or return default action.
        
        Args:
            state: Current state vector (6-dimensional)
            
        Returns:
            RLAction object
        """
        if not self.enabled or self.rl_agent is None:
            # Return default action (no-op)
            logger.debug("RL disabled, returning default action")
            return RLAction()
        
        if len(state) != self.state_dimension:
            raise ValueError(f"Expected state dim {self.state_dimension}, got {len(state)}")
        
        try:
            action = self.rl_agent.get_action(state)
            logger.debug(f"RL action selected: {action}")
            return action
        except Exception as e:
            logger.error(f"Error selecting RL action: {e}")
            return RLAction()  # Fallback to default
    
    def compute_reward(self,
                       tasks_completed: int,
                       energy_used: float,
                       avg_response_time: float,
                       deadlines_missed: int,
                       time_step: float) -> RLReward:
        """
        Compute reward signal based on simulation metrics.
        
        Reward design:
        - Task completion: +1.0 per completed task
        - Energy efficiency: based on energy consumption rate
        - Response time: penalty proportional to average response time
        - Deadline miss: -10.0 per missed deadline
        
        Args:
            tasks_completed: Number of tasks completed in this step
            energy_used: Energy consumed (Joules or normalized)
            avg_response_time: Average response time (seconds)
            deadlines_missed: Number of missed deadlines
            time_step: Time elapsed in this step (seconds)
            
        Returns:
            RLReward object
        """
        base = 0.0
        completion_bonus = tasks_completed * 1.0
        efficiency_bonus = max(0.0, 1.0 - (energy_used / 100.0))  # Normalize energy
        response_penalty = min(5.0, avg_response_time * 0.1)
        deadline_penalty = deadlines_missed * 10.0
        
        reward = RLReward(
            base_reward=base,
            task_completion_bonus=completion_bonus,
            energy_efficiency_bonus=efficiency_bonus,
            response_time_penalty=response_penalty,
            deadline_miss_penalty=deadline_penalty
        )
        
        logger.debug(f"Reward computed: {reward}")
        return reward
    
    def update_agent(self,
                     state: np.ndarray,
                     action: RLAction,
                     reward: RLReward,
                     next_state: np.ndarray,
                     done: bool = False) -> None:
        """
        Update RL agent with experience tuple (S, A, R, S').
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        if not self.enabled or self.rl_agent is None:
            logger.debug("RL disabled, skipping agent update")
            return
        
        try:
            self.rl_agent.update(state, action, reward, next_state, done)
            self._step_count += 1
            self._total_reward += reward.total
            
            if done:
                self._episode_count += 1
                self._rewards_history.append(self._total_reward)
                logger.info(f"Episode {self._episode_count} completed. "
                           f"Total reward: {self._total_reward:.3f}")
                self._total_reward = 0.0
        
        except Exception as e:
            logger.error(f"Error updating RL agent: {e}")
    
    def apply_action(self,
                     action: RLAction,
                     task_manager: 'TaskManager',
                     simulink_interface: 'SimulinkInterface') -> None:
        """
        Apply RL action to system components.
        
        Args:
            action: RLAction to apply
            task_manager: TaskManager instance to update
            simulink_interface: SimulinkInterface to update
        """
        if action.sampling_period is not None and action.sampling_period > 0:
            try:
                simulink_interface.set_sampling_period(action.sampling_period)
            except Exception as e:
                logger.error(f"Error applying sampling period action: {e}")
        
        if action.task_priority is not None and action.task_priority > 0:
            # TODO: Implement task priority adjustment
            logger.debug(f"Task priority adjustment not yet implemented")
        
        if action.energy_threshold is not None:
            # TODO: Implement energy threshold adjustment
            logger.debug(f"Energy threshold adjustment not yet implemented")
        
        if action.required_nodes is not None and action.required_nodes > 0:
            # TODO: Implement required nodes adjustment
            logger.debug(f"Required nodes adjustment not yet implemented")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get RL training statistics.
        
        Returns:
            Dictionary containing training metrics
        """
        avg_reward = np.mean(self._rewards_history) if self._rewards_history else 0.0
        max_reward = np.max(self._rewards_history) if self._rewards_history else 0.0
        
        return {
            'episodes': self._episode_count,
            'steps': self._step_count,
            'current_episode_reward': self._total_reward,
            'average_reward': avg_reward,
            'max_reward': max_reward,
            'enabled': self.enabled,
        }
    
    def save_model(self, path: str) -> None:
        """Save RL model"""
        if self.rl_agent is not None:
            self.rl_agent.save(path)
            logger.info(f"RL model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load RL model"""
        if self.rl_agent is not None:
            self.rl_agent.load(path)
            logger.info(f"RL model loaded from {path}")
