"""
Task Queue Module - Multi-level Priority Queue Management
Implements a K-level priority queue system for task scheduling
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    Task data structure with metadata.
    
    Attributes:
        task_id: Unique task identifier
        task_type: Type of task (e.g., "sampling", "computing", "communication")
        priority: Priority level [1, K]
        arrival_time: When the task arrived (seconds)
        max_delay: Maximum delay tolerance (deadline) (seconds)
        required_nodes: Number of nodes needed for execution
        energy_threshold: Minimum energy threshold required (Volts)
        enqueue_timestamp: Timestamp when enqueued
    """
    task_id: int
    task_type: str
    priority: int
    arrival_time: float
    max_delay: float
    required_nodes: int
    energy_threshold: float
    enqueue_timestamp: float = field(default_factory=time.time)

    def current_wait_time(self, current_time: float = None) -> float:
        """Calculate current waiting time since enqueuing.

        If current_time is provided, it is treated as the simulation time
        (preferred). If omitted, wall-clock time is used as a fallback.
        """
        if current_time is None:
            return time.time() - self.enqueue_timestamp
        return float(current_time - self.enqueue_timestamp)
    
    @property
    def deadline(self) -> float:
        """Calculate absolute deadline"""
        return self.arrival_time + self.max_delay


class ITaskQueue(ABC):
    """Abstract interface for task queue implementations"""
    
    @abstractmethod
    def enqueue(self, task: Task) -> None:
        """Enqueue a task"""
        pass
    
    @abstractmethod
    def dequeue(self) -> Optional[Task]:
        """Dequeue a task from highest priority"""
        pass
    
    @abstractmethod
    def peek(self) -> Optional[Task]:
        """Peek at the highest priority task without removing"""
        pass
    
    @abstractmethod
    def get_total_length(self) -> int:
        """Get total number of tasks across all queues"""
        pass
    
    @abstractmethod
    def get_queue_state(self) -> Dict[int, int]:
        """Get length of each priority queue"""
        pass


class TaskQueue(ITaskQueue):
    """
    Multi-level Priority Queue Implementation
    
    Maintains K independent FIFO queues, one for each priority level.
    Tasks are enqueued to their respective priority queues and dequeued
    in order of priority (highest priority first), with FIFO ordering
    within each priority level.
    
    High cohesion: All queue operations are self-contained
    Low coupling: Depends only on Task dataclass and configuration
    """
    
    def __init__(self, num_priority_levels: int, priority_levels: List[int]):
        """
        Initialize the task queue system.
        
        Args:
            num_priority_levels: Number of priority levels K
            priority_levels: List of priority level values (should be sorted ascending)
        """
        self.num_priority_levels = num_priority_levels
        self.priority_levels = sorted(priority_levels, reverse=True)  # Descending order for easy iteration
        
        # Create a FIFO queue for each priority level
        # Key: priority level, Value: deque of tasks
        self._queues: Dict[int, deque] = {
            priority: deque() for priority in priority_levels
        }
        
        # Metadata for statistics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._enqueue_times: Dict[int, List[float]] = {
            priority: [] for priority in priority_levels
        }
        self._dequeue_times: Dict[int, List[float]] = {
            priority: [] for priority in priority_levels
        }
        
        logger.debug(f"TaskQueue initialized with {num_priority_levels} priority levels: {priority_levels}")
    
    def enqueue(self, task: Task) -> None:
        """
        Enqueue a task to its respective priority queue.
        
        Args:
            task: Task to enqueue
            
        Raises:
            ValueError: If task priority is not valid
        """
        if task.priority not in self._queues:
            raise ValueError(f"Invalid task priority {task.priority}. "
                           f"Valid priorities: {list(self._queues.keys())}")
        
        self._queues[task.priority].append(task)
        self._total_enqueued += 1
        self._enqueue_times[task.priority].append(time.time())
        
        logger.debug(f"Task {task.task_id} (priority {task.priority}) enqueued. "
                    f"Queue length: {self.get_total_length()}")
    
    def dequeue(self) -> Optional[Task]:
        """
        Dequeue the highest priority task.
        
        Returns:
            Task with highest priority, or None if all queues are empty
        """
        for priority in self.priority_levels:
            if self._queues[priority]:
                task = self._queues[priority].popleft()
                self._total_dequeued += 1
                self._dequeue_times[priority].append(time.time())
                
                logger.debug(f"Task {task.task_id} (priority {priority}) dequeued. "
                            f"Remaining: {self.get_total_length()}")
                return task
        
        return None
    
    def peek(self) -> Optional[Task]:
        """
        Peek at the highest priority task without removing it.
        
        Returns:
            Highest priority task, or None if all queues are empty
        """
        for priority in self.priority_levels:
            if self._queues[priority]:
                return self._queues[priority][0]
        
        return None
    
    def get_total_length(self) -> int:
        """
        Get total number of tasks across all priority queues.
        
        Returns:
            Total task count
        """
        return sum(len(queue) for queue in self._queues.values())
    
    def get_queue_state(self) -> Dict[int, int]:
        """
        Get the length of each priority queue.
        
        Returns:
            Dictionary mapping priority level to queue length
        """
        return {priority: len(self._queues[priority]) for priority in self.priority_levels}
    
    def get_highest_priority(self) -> Optional[int]:
        """
        Get the highest priority value of all tasks in queue.
        
        Returns:
            Highest priority level, or None if queue is empty
        """
        for priority in self.priority_levels:
            if self._queues[priority]:
                return priority
        
        return None
    
    def get_max_waiting_time(self, current_time: float = None) -> float:
        """
        Get the waiting time of the task at the head of highest priority queue.
        
        Returns:
            Waiting time in seconds, or 0.0 if queue is empty
        """
        task = self.peek()
        if task is None:
            return 0.0
        # Use simulation time when available to compute wait times
        return task.current_wait_time(current_time=current_time)
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """
        Find a task by its ID (useful for debugging).
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task object if found, None otherwise
        """
        for queue in self._queues.values():
            for task in queue:
                if task.task_id == task_id:
                    return task
        
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get queue statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing queue statistics
        """
        stats = {
            'total_enqueued': self._total_enqueued,
            'total_dequeued': self._total_dequeued,
            'current_length': self.get_total_length(),
            'queue_state': self.get_queue_state(),
            'highest_priority': self.get_highest_priority(),
            # Default to wall-clock when no simulation time is provided
            'max_waiting_time': self.get_max_waiting_time(current_time=None),
        }
        
        return stats
    
    def clear(self) -> None:
        """Clear all tasks from the queue (useful for resetting simulation)"""
        for queue in self._queues.values():
            queue.clear()
        
        logger.info("All queues cleared")
