import numpy as np
import random
from .rl_interface import IRLAgent, RLAction, RLReward


class BehaviorAgent(IRLAgent):
    """
    Data Collection Agent (Behavior Policy).
    Implements the strategy described in the Maestro paper:
    Mix of Frugal Strategy (80%) and Random Exploration (20%).
    """

    def __init__(self, action_space_limits: dict):
        """
        Args:
            action_space_limits: Dictionary defining min/max for actions.
            Example:
            {
                'sampling_period': (1.0, 10.0),
                'energy_threshold': (1.8, 3.8),
                'required_nodes': (1, 10),
                'priority_levels': [1, 2, 3]
            }
        """
        self.limits = action_space_limits
        self.exploration_rate = 0.2  # 20% Random actions

    def get_action(self, state: np.ndarray) -> RLAction:
        """
        Select action based on behavior policy.
        """
        # Determine strategy: Random vs Frugal
        if random.random() < self.exploration_rate:
            return self._get_random_action()
        else:
            return self._get_frugal_action()

    def _get_random_action(self) -> RLAction:
        """Generate a completely random action for exploration."""
        return RLAction(
            sampling_period=random.uniform(*self.limits['sampling_period']),
            energy_threshold=random.uniform(*self.limits['energy_threshold']),
            required_nodes=random.randint(*self.limits['required_nodes']),
            task_priority=random.choice(self.limits['priority_levels'])
        )

    def _get_frugal_action(self) -> RLAction:
        """
        Generate 'Frugal' strategy action.
        Characteristics: Low frequency (High period), High energy threshold, Few nodes.

        """
        # Frugal: Maximize sampling period (Low frequency)
        max_period = self.limits['sampling_period'][1]

        # Frugal: High energy threshold (wait until battery is full)
        max_energy = self.limits['energy_threshold'][1]

        # Frugal: Minimum required nodes
        min_nodes = self.limits['required_nodes'][0]

        # Frugal: Lowest priority (defer tasks)
        # Assuming larger number = lower priority (or keep default 3)
        low_priority = max(self.limits['priority_levels'])

        return RLAction(
            sampling_period=max_period,
            energy_threshold=max_energy,
            required_nodes=min_nodes,
            task_priority=low_priority
        )

    def update(self, state, action, reward, next_state, done):
        # Offline data collection agent does not learn online.
        # It just acts. Data is saved by the Simulator class.
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass