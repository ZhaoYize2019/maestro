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
                 energy_threshold: List[float] = None,  # 修改为列表
                 required_nodes: int = None):
        self.sampling_period = sampling_period
        self.task_priority = task_priority
        self.energy_threshold = energy_threshold
        self.required_nodes = required_nodes
        self.required_nodes = required_nodes

    def to_vector(self) -> np.ndarray:
        """Convert action to vector form for RL agent"""
        # [修改] 1. 处理 energy_threshold 是列表的情况
        if isinstance(self.energy_threshold, list):
            eth_values = self.energy_threshold
        elif isinstance(self.energy_threshold, (float, int)):
            eth_values = [float(self.energy_threshold)]
        else:
            eth_values = [0.0]

        # [修改] 2. 构建扁平化的列表 (Flat List)
        # 顺序: [sampling_period, task_priority, eth_1, eth_2, ..., eth_N, required_nodes]
        # 注意：这里我们把阈值列表展平塞进去
        components = [
            self.sampling_period or 0.0,
            float(self.task_priority or 0)
        ]
        components.extend(eth_values)  # 关键：使用 extend 而不是 append
        components.append(self.required_nodes or 0.0)

        return np.array(components, dtype=np.float32)

    @classmethod
    def from_vector(cls, action_vector: np.ndarray) -> 'RLAction':
        """Construct action from vector output by RL agent"""
        # 注意：如果向量长度变了，这里解析逻辑也需要变。
        # 在多阈值情况下，解析会比较麻烦，需要知道阈值的数量。
        # 简单起见，这里假设阈值只有一个或者是定长的，或者直接取剩余部分。
        # 既然是离线数据收集，如果不回放数据进行训练，可以暂时简化处理或略过此方法。

        # 假设 action_vector = [period, prio, eth_1...eth_N, nodes]
        sampling_period = float(action_vector[0])
        task_priority = int(action_vector[1])
        required_nodes = int(action_vector[-1])  # 最后一个是 nodes

        # 中间部分全是阈值
        energy_thresholds = action_vector[2:-1].tolist()

        # 如果只有一个阈值，转回 float (为了兼容旧代码)
        if len(energy_thresholds) == 1:
            energy_thresholds = energy_thresholds[0]

        return cls(
            sampling_period=sampling_period,
            task_priority=task_priority,
            energy_threshold=energy_thresholds,
            required_nodes=required_nodes
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
                     simulink_interface: 'SimulinkInterface',
                     task_queue: 'TaskQueue') -> None:
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
            current_task_types = task_manager.get_all_task_types()

            # 校验维度是否匹配
            if len(action.energy_threshold) == len(current_task_types):
                for i, task_type in enumerate(current_task_types):
                    new_threshold = action.energy_threshold[i]
                    try:
                        # 核心修改：针对特定任务名更新属性
                        task_manager.update_task_type_attribute(
                            task_type.name,
                            "energy_threshold",
                            new_threshold
                        )
                        logger.debug(f"Updated {task_type.name} threshold -> {new_threshold:.2f}V")
                    except ValueError as e:
                        logger.warning(f"Failed to update {task_type.name}: {e}")
            else:
                logger.error(f"Action threshold dim ({len(action.energy_threshold)}) "
                             f"mismatch with task types ({len(current_task_types)})")
        
        if action.required_nodes is not None and action.required_nodes > 0:
            # TODO: Implement required nodes adjustment
            logger.debug(f"Required nodes adjustment not yet implemented")

        scheduling_mode = action.task_priority

        if scheduling_mode == 1:
            self._apply_priority_inversion(task_queue)
        elif scheduling_mode == 2:
            self._apply_queue_shuffle(task_queue)

    def _apply_priority_inversion(self, task_queue):
        """
        策略示例：将低优先级队列的任务临时提升，模拟 '饿死' 高优先级任务的场景
        这有助于收集极端情况下的数据 (Corner Cases)
        """
        # 注意：这需要访问 TaskQueue 的私有属性 _queues，或者在 TaskQueue 中增加相应公共方法
        # 这里演示直接操作逻辑（Python 允许访问 _queues，虽然不推荐）
        if hasattr(task_queue, '_queues'):
            # 简单粗暴：交换 高(3) 和 低(1) 优先级的整个队列
            q1 = task_queue._queues.get(1)
            q3 = task_queue._queues.get(3)
            if q1 and q3:
                # 交换内容
                task_queue._queues[1], task_queue._queues[3] = q3, q1
                logger.warning("EXPLORATION: Swapped Priority 1 and 3 Queues!")

    def _apply_queue_shuffle(self, task_queue):
        """随机打乱所有队列的顺序"""
        import random
        for p_level, deque_obj in task_queue._queues.items():
            if len(deque_obj) > 1:
                temp_list = list(deque_obj)
                random.shuffle(temp_list)
                task_queue._queues[p_level] = type(deque_obj)(temp_list)

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
