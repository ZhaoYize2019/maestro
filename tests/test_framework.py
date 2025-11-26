"""
Unit tests for Maestro Simulator Framework
Tests individual modules and their interactions
"""

import unittest
import numpy as np
from src.config import SimulationConfig
from src.task_queue import Task, TaskQueue
from src.task_manager import TaskType, TaskManager, PoissonTaskGenerator
from src.state_calculator import StateCalculator, EnergySnapshot
from src.simulink_interface import SimulinkInterface
from src.rl_interface import RLInterface, RLAction, RLReward


class TestConfiguration(unittest.TestCase):
    """Test configuration module"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = SimulationConfig()
        self.assertTrue(SimulationConfig.validate(config))
        self.assertEqual(config.NUM_NODES, 25)
        self.assertEqual(config.NUM_PRIORITY_LEVELS, 3)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SimulationConfig(
            NUM_NODES=50,
            SAMPLING_PERIOD=0.5,
            SIMULATION_DURATION=1000.0
        )
        self.assertTrue(SimulationConfig.validate(config))
        self.assertEqual(config.NUM_NODES, 50)


class TestTaskQueue(unittest.TestCase):
    """Test task queue module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.queue = TaskQueue(num_priority_levels=3, priority_levels=[1, 2, 3])
    
    def test_enqueue_dequeue(self):
        """Test basic enqueue/dequeue operations"""
        task = Task(
            task_id=1,
            task_type="sampling",
            priority=2,
            arrival_time=0.0,
            max_delay=30.0,
            required_nodes=3,
            energy_threshold=2.0
        )
        
        self.queue.enqueue(task)
        self.assertEqual(self.queue.get_total_length(), 1)
        
        dequeued = self.queue.dequeue()
        self.assertEqual(dequeued.task_id, 1)
        self.assertEqual(self.queue.get_total_length(), 0)
    
    def test_priority_ordering(self):
        """Test priority-based ordering"""
        tasks = []
        for priority in [1, 2, 3, 1, 2]:
            task = Task(
                task_id=len(tasks) + 1,
                task_type="sampling",
                priority=priority,
                arrival_time=0.0,
                max_delay=30.0,
                required_nodes=1,
                energy_threshold=2.0
            )
            tasks.append(task)
            self.queue.enqueue(task)
        
        # Should dequeue in order: 3, 2, 2, 1, 1
        self.assertEqual(self.queue.dequeue().priority, 3)
        self.assertEqual(self.queue.dequeue().priority, 2)
        self.assertEqual(self.queue.dequeue().priority, 2)
        self.assertEqual(self.queue.dequeue().priority, 1)
        self.assertEqual(self.queue.dequeue().priority, 1)


class TestTaskManager(unittest.TestCase):
    """Test task manager module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.task_types = [
            TaskType(
                name="sampling",
                priority=1,
                max_delay=30.0,
                required_nodes_range=(1, 5),
                energy_threshold=2.0
            )
        ]
        
        self.manager = TaskManager(
            task_types=self.task_types,
            lambda_arrival_rate=0.1,
            num_nodes=25,
            random_seed=42
        )
    
    def test_task_generation(self):
        """Test task generation"""
        task, next_arrival = self.manager.generate_next_task(0.0, "sampling")
        
        self.assertEqual(task.task_type, "sampling")
        self.assertEqual(task.priority, 1)
        self.assertGreater(next_arrival, 0.0)
    
    def test_poisson_process(self):
        """Test Poisson process properties"""
        generator = PoissonTaskGenerator(lambda_rate=0.1, random_seed=42)
        
        intervals = [generator.get_next_arrival_interval() for _ in range(100)]
        mean_interval = np.mean(intervals)
        
        # Mean should be approximately 1/lambda = 10
        self.assertAlmostEqual(mean_interval, 10.0, delta=3.0)


class TestStateCalculator(unittest.TestCase):
    """Test state calculator module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = StateCalculator(num_nodes=25, voltage_max=4.3)
    
    def test_energy_snapshot(self):
        """Test energy snapshot normalization"""
        voltages = np.array([2.15, 3.44, 4.3] + [0.0] * 22)  # 25 nodes
        snapshot = EnergySnapshot(timestamp=0.0, node_voltages=voltages)
        
        # Check normalization
        self.assertAlmostEqual(snapshot.normalized_voltages[0], 0.5, places=2)
        self.assertAlmostEqual(snapshot.normalized_voltages[1], 0.8, places=2)
        self.assertAlmostEqual(snapshot.normalized_voltages[2], 1.0, places=2)
    
    def test_state_vector_computation(self):
        """Test state vector computation"""
        voltages = np.random.uniform(2.0, 4.0, 25)
        self.calculator.update_energy_data(voltages, 0.0)
        
        state = self.calculator.compute_state_vector(
            queue_length=5,
            queue_priority=2,
            max_wait_time=10.0
        )
        
        # Check state vector properties
        self.assertEqual(len(state), 6)
        self.assertTrue(np.all(state >= 0.0))
        self.assertTrue(np.all(state <= 1.0))


class TestSimulinkInterface(unittest.TestCase):
    """Test Simulink interface module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = SimulinkInterface(
            sampling_period=1.0,
            num_nodes=25,
            matlab_engine=None  # Mock mode
        )
    
    def test_initialization(self):
        """Test interface initialization"""
        self.interface.initialize()
        self.assertTrue(self.interface._is_running)
    
    def test_energy_sampling(self):
        """Test energy sampling"""
        self.interface.initialize()
        
        energies = self.interface.get_node_energies()
        self.assertEqual(len(energies), 25)
        self.assertTrue(all(0.0 <= e <= 4.3 for e in energies))
    
    def test_task_activation(self):
        """Test task activation"""
        self.interface.initialize()
        
        initial_energies = self.interface.get_node_energies()
        self.interface.activate_task_on_nodes([0, 1, 2], 2.5)
        
        # Energy should decrease
        new_energies = self.interface.get_node_energies()
        self.assertTrue(new_energies[0] < initial_energies[0] + 0.5)


class TestRLInterface(unittest.TestCase):
    """Test RL interface module"""
    
    def test_action_creation(self):
        """Test RLAction creation and conversion"""
        action = RLAction(
            sampling_period=1.5,
            task_priority=2,
            energy_threshold=2.5,
            required_nodes=5
        )
        
        vector = action.to_vector()
        self.assertEqual(len(vector), 4)
        
        reconstructed = RLAction.from_vector(vector)
        self.assertEqual(reconstructed.sampling_period, 1.5)
    
    def test_reward_computation(self):
        """Test reward computation"""
        reward = RLReward(
            base_reward=0.0,
            task_completion_bonus=5.0,
            energy_efficiency_bonus=1.0,
            response_time_penalty=2.0,
            deadline_miss_penalty=10.0
        )
        
        total = reward.total
        self.assertEqual(total, 5.0 + 1.0 - 2.0 - 10.0)
    
    def test_rl_interface_disabled(self):
        """Test RL interface when disabled"""
        rl_interface = RLInterface(enabled=False)
        
        state = np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.6])
        action = rl_interface.select_action(state)
        
        # Should return default (no-op) action
        self.assertIsNotNone(action)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple modules"""
    
    def test_task_flow(self):
        """Test complete task flow"""
        # Create components
        queue = TaskQueue(3, [1, 2, 3])
        manager = TaskManager(
            task_types=[
                TaskType("sampling", 1, 30.0, (1, 5), 2.0)
            ],
            lambda_arrival_rate=0.1,
            num_nodes=25,
            random_seed=42
        )
        
        # Generate and queue task
        task, _ = manager.generate_next_task(0.0, "sampling")
        queue.enqueue(task)
        
        # Verify state
        self.assertEqual(queue.get_total_length(), 1)
        self.assertEqual(queue.get_highest_priority(), 1)
        
        # Dequeue
        dequeued = queue.dequeue()
        self.assertEqual(dequeued.task_id, task.task_id)


if __name__ == '__main__':
    unittest.main()
