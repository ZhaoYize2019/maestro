"""
Task Manager Module - Task Generation and Management
Handles task type definitions, Poisson process task generation,
and task attribute management
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .task_queue import Task

logger = logging.getLogger(__name__)


@dataclass
class TaskType:
    """
    Task type definition containing attributes for a specific task category.
    
    Attributes:
        name: Task type identifier (e.g., "sampling", "computing")
        priority: Priority level [1, K]
        max_delay: Maximum delay tolerance in seconds
        required_nodes_range: Tuple (min, max) nodes needed
        energy_threshold: Energy threshold in Volts
    """
    name: str
    priority: int
    max_delay: float
    required_nodes_range: Tuple[int, int]
    energy_threshold: float
    
    def validate(self, num_nodes: int) -> bool:
        """Validate task type configuration"""
        assert self.priority > 0, "Priority must be positive"
        assert self.max_delay > 0, "Max delay must be positive"
        assert 0 < self.required_nodes_range[0] <= self.required_nodes_range[1] <= num_nodes, \
            f"Invalid node range for {self.name}"
        assert 1.8 <= self.energy_threshold <= 3.8, \
            f"Energy threshold must be in [1.8, 3.8], got {self.energy_threshold}"
        return True


class PoissonTaskGenerator:
    """
    Generates tasks according to a Poisson process.
    
    Inter-arrival times follow exponential distribution with parameter lambda.
    This generator maintains state for next arrival time calculation.
    
    High cohesion: All Poisson generation logic is encapsulated
    Low coupling: Depends only on numpy and configuration
    """
    
    def __init__(self, lambda_rate: float, random_seed: Optional[int] = None):
        """
        Initialize Poisson task generator.
        
        Args:
            lambda_rate: Lambda parameter (tasks per second)
            random_seed: Random seed for reproducibility
        """
        self.lambda_rate = lambda_rate
        self.mean_inter_arrival = 1.0 / lambda_rate
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.debug(f"Poisson generator initialized: lambda={lambda_rate}, "
                    f"mean_inter_arrival={self.mean_inter_arrival}s")
    
    def get_next_arrival_interval(self) -> float:
        """
        Generate next inter-arrival time from exponential distribution.
        
        Returns:
            Inter-arrival time in seconds
        """
        interval = np.random.exponential(self.mean_inter_arrival)
        return interval
    
    def get_next_arrival_time(self, current_time: float) -> float:
        """
        Calculate the next task arrival time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Next arrival time
        """
        interval = self.get_next_arrival_interval()
        return current_time + interval


class ITaskManager(ABC):
    """Abstract interface for task manager implementations"""
    
    @abstractmethod
    def get_task_type(self, task_type_name: str) -> Optional[TaskType]:
        """Retrieve task type definition"""
        pass
    
    @abstractmethod
    def generate_next_task(self, current_time: float) -> Tuple['Task', float]:
        """Generate next task and its arrival time"""
        pass
    
    @abstractmethod
    def get_all_task_types(self) -> List[TaskType]:
        """Get all registered task types"""
        pass


class TaskManager(ITaskManager):
    """
    Task Manager Implementation
    
    Manages task type definitions, generates tasks according to Poisson process,
    and tracks task scheduling statistics.
    
    High cohesion: Encapsulates all task generation and management logic
    Low coupling: Depends on TaskType and PoissonTaskGenerator abstractions
    """
    
    def __init__(self, 
                 task_types: List[TaskType],
                 lambda_arrival_rate: float,
                 num_nodes: int,
                 random_seed: Optional[int] = None):
        """
        Initialize task manager.
        
        Args:
            task_types: List of TaskType definitions
            lambda_arrival_rate: Poisson process lambda parameter
            num_nodes: Total number of network nodes
            random_seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self._task_types: Dict[str, TaskType] = {}
        self._task_id_counter = 0
        
        # Validate and store task types
        for task_type in task_types:
            task_type.validate(num_nodes)
            self._task_types[task_type.name] = task_type
        
        # Initialize task generator
        self._generator = PoissonTaskGenerator(lambda_arrival_rate, random_seed)
        
        # Statistics
        self._tasks_generated = 0
        self._task_type_counts: Dict[str, int] = {name: 0 for name in self._task_types}
        self._next_arrival_times: Dict[str, float] = {}
        
        logger.info(f"TaskManager initialized with {len(task_types)} task types: "
                   f"{list(self._task_types.keys())}")
    
    def get_task_type(self, task_type_name: str) -> Optional[TaskType]:
        """
        Retrieve task type definition by name.
        
        Args:
            task_type_name: Name of task type
            
        Returns:
            TaskType object or None if not found
        """
        return self._task_types.get(task_type_name)
    
    def get_all_task_types(self) -> List[TaskType]:
        """
        Get all registered task types.
        
        Returns:
            List of TaskType objects
        """
        return list(self._task_types.values())
    
    def generate_next_task(self, 
                          current_time: float,
                          task_type_name: Optional[str] = None) -> Tuple['Task', float]:
        """
        Generate the next task.
        
        Args:
            current_time: Current simulation time
            task_type_name: Specific task type to generate (random if None)
            
        Returns:
            Tuple of (Task object, next arrival time)
            
        Raises:
            ValueError: If task type not found
        """
        from .task_queue import Task  # Avoid circular import
        
        # Select task type
        if task_type_name is None:
            # Random selection from available types
            task_type_name = np.random.choice(list(self._task_types.keys()))
        
        task_type = self.get_task_type(task_type_name)
        if task_type is None:
            raise ValueError(f"Unknown task type: {task_type_name}")
        
        # Generate task instance
        self._task_id_counter += 1
        
        # Randomly select required nodes within range
        required_nodes = np.random.randint(
            task_type.required_nodes_range[0],
            task_type.required_nodes_range[1] + 1
        )
        
        task = Task(
            task_id=self._task_id_counter,
            task_type=task_type_name,
            priority=task_type.priority,
            arrival_time=current_time,
            max_delay=task_type.max_delay,
            required_nodes=required_nodes,
            energy_threshold=task_type.energy_threshold,
            enqueue_timestamp=current_time
        )
        
        # Calculate next arrival time for this task type
        next_arrival_time = self._generator.get_next_arrival_time(current_time)
        
        self._tasks_generated += 1
        self._task_type_counts[task_type_name] += 1
        self._next_arrival_times[task_type_name] = next_arrival_time

        logger.info(f"[TASK GEN] ID:{task.task_id:03d} | Type:{task_type_name[:4].upper()} | "
                    f"Prio:{task.priority} | Nodes:{required_nodes} | "
                    f"Deadline:{current_time + task.max_delay:.2f}s")
        
        return task, next_arrival_time
    
    def generate_initial_tasks(self, current_time: float) -> Dict[str, float]:
        """
        Generate initial tasks for all task types and schedule their first arrivals.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping task type to its next arrival time
        """
        next_arrivals = {}
        
        for task_type_name in self._task_types.keys():
            _, next_time = self.generate_next_task(current_time, task_type_name)
            next_arrivals[task_type_name] = next_time
        
        logger.info(f"Initial tasks generated. Next arrivals: {next_arrivals}")
        
        return next_arrivals
    
    def get_next_task_arrival_time(self) -> Tuple[str, float]:
        """
        Get the task type and time of the next scheduled arrival.
        
        Returns:
            Tuple of (task_type_name, arrival_time)
        """
        if not self._next_arrival_times:
            return None, float('inf')
        
        task_type = min(self._next_arrival_times, 
                       key=self._next_arrival_times.get)
        arrival_time = self._next_arrival_times[task_type]
        
        return task_type, arrival_time
    
    def update_task_arrival(self, task_type_name: str, current_time: float) -> None:
        """
        Update the next arrival time for a task type after generation.
        
        Args:
            task_type_name: Task type that was just generated
            current_time: Current simulation time
        """
        next_time = self._generator.get_next_arrival_time(current_time)
        self._next_arrival_times[task_type_name] = next_time
    
    def get_statistics(self) -> Dict:
        """
        Get task generation statistics.
        
        Returns:
            Dictionary containing task statistics
        """
        return {
            'total_generated': self._tasks_generated,
            'type_counts': self._task_type_counts,
            'next_arrivals': self._next_arrival_times,
            'num_task_types': len(self._task_types),
        }
    
    def update_task_type_attribute(self,
                                   task_type_name: str,
                                   attribute: str,
                                   value: float) -> None:
        """
        Update a task type attribute (called by RL interface).
        
        This method allows RL agent to dynamically modify task attributes
        such as priority, energy threshold, and required node count.
        
        Args:
            task_type_name: Task type to update
            attribute: Attribute name ('priority', 'energy_threshold', etc.)
            value: New value
            
        Raises:
            ValueError: If task type or attribute not found
        """
        task_type = self.get_task_type(task_type_name)
        if task_type is None:
            raise ValueError(f"Unknown task type: {task_type_name}")
        
        if not hasattr(task_type, attribute):
            raise ValueError(f"Task type {task_type_name} has no attribute {attribute}")
        
        old_value = getattr(task_type, attribute)
        setattr(task_type, attribute, value)
        
        logger.info(f"Updated {task_type_name}.{attribute}: {old_value} -> {value}")
