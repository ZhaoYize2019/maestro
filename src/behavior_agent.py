import numpy as np
import random
from .rl_interface import IRLAgent, RLAction, RLReward
import logging
logger = logging.getLogger(__name__)


class BehaviorAgent(IRLAgent):
    """
    Data Collection Agent (Behavior Policy).
    Implements the strategy described in the Maestro paper:
    Mix of Frugal Strategy (80%) and Random Exploration (20%).
    """

    def __init__(self, action_space_limits: dict, task_types: list):
        self.limits = action_space_limits
        self.task_types = task_types  # 记录任务类型，例如 ['sampling', 'computing', 'communication']
        self.exploration_rate = 0.2
        self.task_queue = None  # 新增：用于持有队列引用

    def bind_task_queue(self, task_queue):
        """注入 TaskQueue 实例"""
        self.task_queue = task_queue

    def get_action(self, state: np.ndarray) -> RLAction:
        # 1. 感知：检查队列状态
        queue_pressure = 0.0
        if self.task_queue:
            # 例如：计算排队任务的总等待时间作为“压力值”
            # 注意：这里需要访问 TaskQueue 的内部或提供新接口
            # 假设我们只看长度：
            queue_pressure = self.task_queue.get_total_length()

        # 2. 决策：基于压力值调整策略
        # 如果队列压力大，以一定概率尝试“激进调度”（例如反转优先级）来探索极限
        force_exploration = False
        if queue_pressure > 10 and random.random() < 0.3:
            logger.info(f"High Queue Pressure ({queue_pressure})! Triggering Aggressive Scheduling.")
            force_exploration = True

        if force_exploration or random.random() < self.exploration_rate:
            return self._get_random_action(force_reorder=force_exploration)
        else:
            return self._get_frugal_action()

    def _get_random_action(self, force_reorder=False) -> RLAction:
        random_thresholds = [
            random.uniform(*self.limits['energy_threshold'])
            for _ in self.task_types
        ]
        # 新增：调度模式 (0: 默认FIFO, 1: 优先级反转/随机洗牌)
        scheduling_mode = 1 if force_reorder else 0

        return RLAction(
            sampling_period=random.uniform(*self.limits['sampling_period']),
            energy_threshold=random_thresholds,  # 传入列表
            required_nodes=random.randint(*self.limits['required_nodes']),
            task_priority=scheduling_mode  # 复用 task_priority 字段，或新增字段
        )

    def _get_frugal_action(self) -> RLAction:
        # [修改] 节俭模式下，所有任务都使用最大阈值（或者您可以定义某类任务更保守）
        max_energy = self.limits['energy_threshold'][1]
        frugal_thresholds = [max_energy for _ in self.task_types]

        return RLAction(
            sampling_period=self.limits['sampling_period'][1],
            energy_threshold=frugal_thresholds, # 传入列表
            required_nodes=self.limits['required_nodes'][0],
            task_priority=max(self.limits['priority_levels'])
        )

    def update(self, state, action, reward, next_state, done):
        # Offline data collection agent does not learn online.
        # It just acts. Data is saved by the Simulator class.
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass