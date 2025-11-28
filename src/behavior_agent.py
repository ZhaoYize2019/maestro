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

    def __init__(self, action_space_limits: dict, task_streams: list):
        self.limits = action_space_limits
        self.task_streams = task_streams  # 这是一个 PoissonTaskStream 对象列表
        self.exploration_rate = 0.2
        self.task_queue = None

        # [新增] 用于感知紧急程度
        self.current_time = 0.0

    def bind_task_queue(self, task_queue):
        self.task_queue = task_queue

    def set_current_time(self, time_val: float):
        """由 Simulator 每步调用，更新 Agent 的时间感知"""
        self.current_time = time_val

    def get_action(self, state: np.ndarray) -> RLAction:
        # 1. 决定模式
        # 如果队列压力大，或者随机骰子，进入探索模式
        # 这里为了演示用户需求：通常模式 vs 随机模式
        if random.random() < self.exploration_rate:
            return self._get_random_exploration_action()
        else:
            return self._get_smart_priority_action()

    def _get_random_exploration_action(self) -> RLAction:
        """随机探索模式：所有参数随机"""
        num_streams = len(self.task_streams)

        # 随机生成 N 个阈值
        rand_energy = [random.uniform(*self.limits['energy_threshold']) for _ in range(num_streams)]
        # 随机生成 N 个优先级
        rand_prio = [random.choice(self.limits['priority_levels']) for _ in range(num_streams)]

        return RLAction(
            sampling_period=random.uniform(*self.limits['sampling_period']),
            energy_threshold=rand_energy,
            task_priority=rand_prio
        )

    def _get_smart_priority_action(self) -> RLAction:
        num_streams = len(self.task_streams)

        # [调整 1] 默认能量阈值不要设为最大 (3.8)，设为中位数或稍高 (例如 3.0)
        # 这样非紧急任务也有机会执行，而不是必须等到满电
        default_energy = 3.0
        default_prio = 1

        priorities = [default_prio] * num_streams
        energies = [default_energy] * num_streams

        if self.task_queue:
            id_to_idx = {stream.stream_id: i for i, stream in enumerate(self.task_streams)}

            for q in self.task_queue._queues.values():
                for task in q:
                    wait_time = self.current_time - task.arrival_time
                    wait_ratio = wait_time / task.max_delay

                    # 如果等待时间超过 70%
                    if wait_ratio > 0.7:
                        stream_id = task.task_type
                        if stream_id in id_to_idx:
                            idx = id_to_idx[stream_id]

                            # [调整 2] 紧急状态下：提权 + 降阈值
                            priorities[idx] = 3  # 提高优先级
                            energies[idx] = 1.8  # 降低门槛到最小值，确保能立刻执行！

        return RLAction(
            sampling_period=self.limits['sampling_period'][1],
            energy_threshold=energies,
            task_priority=priorities
        )

    def update(self, state, action, reward, next_state, done):
        # Offline data collection agent does not learn online.
        # It just acts. Data is saved by the Simulator class.
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass