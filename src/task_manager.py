"""
Task Manager Module - Task Generation and Management
Handles heterogeneous Poisson task streams generation and management.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_queue import Task
    from .config import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class PoissonTaskStream:
    """
    Represents an independent task stream (e.g., a specific IoT device).
    Attributes are fixed at initialization to simulate device heterogeneity.
    """
    stream_id: str          # Unique identifier, e.g., "stream_001"

    # === Independent Attributes ===
    lambda_rate: float      # Arrival rate (tasks per second)
    mean_interval: float    # 1 / lambda_rate
    priority: int           # Task priority level
    required_nodes: int     # Number of nodes required
    energy_threshold: float # Minimum energy threshold (Volts)
    max_delay: float        # Maximum allowed delay (Deadline - Arrival Time)

    # === Dynamic State ===
    next_arrival_time: float = 0.0

    @property
    def name(self) -> str:
        """Alias for stream_id to maintain compatibility with interfaces expecting 'name'."""
        return self.stream_id

    def schedule_next_arrival(self, current_time: float) -> None:
        """
        Calculate the next arrival time based on exponential distribution (Poisson Process).
        Updates self.next_arrival_time.
        """
        # Inter-arrival time follows exponential distribution
        interval = np.random.exponential(self.mean_interval)
        # The next arrival is relative to the previous scheduled arrival time to maintain average rate,
        # or relative to current_time if it's the first scheduling or after a reset.
        # Using max(current_time, ...) ensures we don't schedule in the past.
        self.next_arrival_time = max(current_time, self.next_arrival_time) + interval


class TaskManager:
    """
    Task Manager Implementation

    Manages N independent, heterogeneous Poisson task streams.
    Instead of fixed task types (sampling/computing), it generates diverse streams
    based on configuration ranges.
    """

    def __init__(self, config: 'SimulationConfig', random_seed: Optional[int] = None):
        """
        Initialize task manager with random heterogeneous streams.

        Args:
            config: SimulationConfig object containing STREAM_*_RANGE parameters.
            random_seed: Seed for reproducibility of stream generation and task arrivals.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.config = config
        self._streams: Dict[str, PoissonTaskStream] = {}
        self._task_id_counter = 0

        # Initialize the random heterogeneous streams
        self._initialize_random_streams()

    def _initialize_random_streams(self) -> None:
        """
        Generates fully heterogeneous task streams based on configuration ranges.
        Each stream simulates a unique device with specific characteristics.
        """
        num_streams = getattr(self.config, 'NUM_TASK_STREAMS', 15)
        logger.info(f"Initializing {num_streams} fully heterogeneous task streams...")

        for i in range(num_streams):
            stream_id = f"stream_{i:03d}"

            # 1. Randomly generate attributes based on Config ranges

            # Lambda & Interval
            l_min, l_max = self.config.STREAM_LAMBDA_RANGE
            lambda_val = np.random.uniform(l_min, l_max)
            mean_interval = 1.0 / lambda_val

            # Energy Threshold
            e_min, e_max = self.config.STREAM_ENERGY_RANGE
            # Round to 2 decimals for realistic voltage values
            energy_val = round(np.random.uniform(e_min, e_max), 2)

            # Required Nodes
            n_min, n_max = self.config.STREAM_NODES_RANGE
            nodes_val = np.random.randint(n_min, n_max + 1)

            # Priority
            p_min, p_max = self.config.STREAM_PRIORITY_RANGE
            prio_val = np.random.randint(p_min, p_max + 1)

            # Max Delay (Deadline)
            # Calculated as a factor of the mean interval (e.g., 1.5x interval)
            d_min, d_max = self.config.STREAM_DELAY_FACTOR_RANGE
            delay_factor = np.random.uniform(d_min, d_max)
            max_delay_val = mean_interval * delay_factor

            # 2. Create the Stream object
            stream = PoissonTaskStream(
                stream_id=stream_id,
                lambda_rate=lambda_val,
                mean_interval=mean_interval,
                priority=prio_val,
                required_nodes=nodes_val,
                energy_threshold=energy_val,
                max_delay=max_delay_val
            )

            # 3. Schedule the first arrival (Random phase start)
            # Use exponential distribution for the first arrival to stagger start times
            first_interval = np.random.exponential(mean_interval)
            stream.next_arrival_time = first_interval

            self._streams[stream_id] = stream

            logger.debug(f"Created {stream_id}: Prio={prio_val}, Nodes={nodes_val}, "
                         f"Energy={energy_val}V, Lambda={lambda_val:.3f}, Interval={mean_interval:.1f}s")

        logger.info(f"Task Manager ready with {len(self._streams)} streams.")

    def get_next_task_arrival_time(self) -> Tuple[Optional[str], float]:
        """
        Find the earliest scheduled arrival time among all streams.

        Returns:
            Tuple (stream_id, arrival_time).
            Returns (None, inf) if no streams exist.
        """
        if not self._streams:
            return None, float('inf')

        # Find the stream with the minimum next_arrival_time
        best_stream = min(self._streams.values(), key=lambda s: s.next_arrival_time)
        return best_stream.stream_id, best_stream.next_arrival_time

    def generate_next_task(self, current_time: float, stream_id: str) -> Tuple['Task', float]:
        """
        Generate a Task instance for the specified stream and schedule its next arrival.

        Args:
            current_time: Current simulation time.
            stream_id: The ID of the stream that is ready to generate a task.

        Returns:
            Tuple (Task object, next_arrival_time for this stream).
        """
        # Local import to avoid circular dependency issues at module level
        from .task_queue import Task

        stream = self._streams.get(stream_id)
        if not stream:
            raise ValueError(f"Unknown stream_id: {stream_id}")

        self._task_id_counter += 1

        # 1. Instantiate the Task
        # Note: We use stream.next_arrival_time as the task's official arrival time
        # to ensure mathematical precision of the Poisson process.
        task = Task(
            task_id=self._task_id_counter,
            task_type=stream.stream_id,  # Use stream ID as the type identifier
            priority=stream.priority,
            arrival_time=stream.next_arrival_time,
            max_delay=stream.max_delay,
            required_nodes=stream.required_nodes,
            energy_threshold=stream.energy_threshold,
            enqueue_timestamp=current_time  # Record when it actually entered the system
        )

        # 2. Schedule the next arrival for this specific stream
        stream.schedule_next_arrival(stream.next_arrival_time)

        logger.debug(f"[GEN] {stream_id} (#{task.task_id}) generated. "
                     f"Arr: {task.arrival_time:.2f}s, Next: {stream.next_arrival_time:.2f}s")

        return task, stream.next_arrival_time

    def get_all_task_types(self) -> List[PoissonTaskStream]:
        """
        Returns a list of all stream objects.
        Used by RL Interface to perceive available 'task types' (streams) and modify them.
        """
        return list(self._streams.values())

    def get_task_type(self, stream_id: str) -> Optional[PoissonTaskStream]:
        """
        Retrieve a specific stream by ID.
        """
        return self._streams.get(stream_id)

    def update_task_type_attribute(self, stream_id: str, attribute: str, value: float) -> None:
        """
        Allows RL agent (or other controllers) to dynamically modify stream attributes.
        LOGS ONLY IF VALUE CHANGES.

        Args:
            stream_id: The ID of the stream to modify.
            attribute: The attribute name (e.g., 'priority', 'energy_threshold').
            value: The new value.
        """
        stream = self._streams.get(stream_id)
        if stream and hasattr(stream, attribute):
            old_val = getattr(stream, attribute)

            # [修改] 仅当值真正发生变化时才更新和打印日志
            # 对于浮点数，!= 判断通常足够，因为 RL Agent 即使输出 3.800000001 也是一种变化
            # 对于整数优先级，这能完美过滤 1 -> 1 的情况
            if old_val != value:
                setattr(stream, attribute, value)
                logger.info(f"Updated {stream_id}.{attribute}: {old_val} -> {value}")
        else:
            logger.warning(f"Failed to update {stream_id}: attribute '{attribute}' not found or stream unknown.")

    def generate_initial_tasks(self, current_time: float) -> None:
        """Legacy method for compatibility."""
        pass

    def get_statistics(self) -> Dict:
        """Get task generation statistics."""
        return {
            'total_generated': self._task_id_counter,
            'num_streams': len(self._streams),
            'next_arrival': self.get_next_task_arrival_time()[1]
        }