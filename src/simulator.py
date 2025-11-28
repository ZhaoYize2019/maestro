"""
Main Simulator Module
Orchestrates all components to execute complete S-A-R-S' simulation loops.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 关键行：必须放在 pyplot 之前！
import matplotlib.pyplot as plt

from .config import SimulationConfig
from .task_queue import TaskQueue, Task
# [修改] 移除了 TaskType 的导入，因为该类已在 TaskManager 重构中废弃
from .task_manager import TaskManager
from .state_calculator import StateCalculator
from .simulink_interface import SimulinkInterface
from .rl_interface import RLInterface, RLAction, RLReward
from .rl_interface import IRLAgent


logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation"""
    current_time: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    energy_consumed: float = 0.0
    deadlines_missed: int = 0
    episode_reward: float = 0.0


class MaestroSimulator:
    """
    Main Maestro Simulator Orchestrator

    Coordinates simulation of all modules and executes the complete
    S-A-R-S' (State-Action-Reward-State) reinforcement learning loop.
    """

    def __init__(self, config: SimulationConfig, rl_enabled: bool = False, agent: Optional[IRLAgent] = None):
        """
        Initialize Maestro simulator.

        Args:
            config: SimulationConfig object with all parameters
            rl_enabled: Enable RL agent integration
        """
        SimulationConfig.validate(config)
        self.config = config

        num_streams = config.NUM_TASK_STREAMS
        config.RL_ACTION_DIM = 1 + num_streams + num_streams

        self.rl_interface = RLInterface(
            state_dimension=config.STATE_DIMENSION,
            action_dimension=config.RL_ACTION_DIM,
            rl_agent=agent,
            enabled=rl_enabled
        )

        # Initialize core components
        self.task_queue = TaskQueue(
            config.NUM_PRIORITY_LEVELS,
            config.PRIORITY_LEVELS
        )

        # [修改] 不再手动定义 task_types 列表
        # 现在的 TaskManager 会根据 config 中的范围参数自动生成随机异构任务流
        self.task_manager = TaskManager(
            config=config,
            random_seed=42
        )

        self.state_calculator = StateCalculator(
            num_nodes=config.NUM_NODES,
            state_dimension=config.STATE_DIMENSION,
            voltage_max=config.VOLTAGE_MAX
        )

        self.simulink_interface = SimulinkInterface(
            sampling_period=config.SAMPLING_PERIOD,
            num_nodes=config.NUM_NODES,
            simulation_duration=config.SIMULATION_DURATION,
            energy_threshold_min=config.ENERGY_THRESHOLD_MIN,
            energy_threshold_max=config.ENERGY_THRESHOLD_MAX,
            matlab_engine=None,
            data_file = "ENERGY_SOURCE_50s.csv",
            scale_factor = 0.01
        )

        self.voltage_history: List[List[float]] = []
        self.time_history: List[float] = []

        # Simulation state
        self.current_time = 0.0
        self.metrics = SimulationMetrics()
        self._trajectory_data: List[Dict] = []

        self.node_release_times = [0.0] * config.NUM_NODES
        logger.info("Node locking mechanism initialized.")

        logger.info(f"MaestroSimulator initialized with config: {config}")

    def initialize(self) -> None:
        """Initialize simulation and start Simulink interface"""
        logger.info("=" * 60)
        logger.info("Initializing Maestro Simulator")
        logger.info("=" * 60)

        # Start Simulink
        self.simulink_interface.initialize()

        # Generate first batch of tasks
        # TaskManager 已经在 init 中预排了时间，这里调用一下是安全的（虽然在泊松模式下可能不做实际操作）
        self.task_manager.generate_initial_tasks(self.current_time)

        # Initialize energy data
        initial_voltages, _ = self.simulink_interface.get_node_energies()
        self.state_calculator.update_energy_data(initial_voltages, self.current_time)

        logger.info("Simulator initialization complete")

    def _get_next_event_time(self) -> float:
        """Determine the next event time (task arrival or sample time)"""
        # Next scheduled sampling time
        current_period = self.simulink_interface.sampling_period
        next_sample_time = self.current_time + current_period

        # Next task arrival time
        # [修改] 变量名适配: stream_id
        stream_id, next_task_time = self.task_manager.get_next_task_arrival_time()

        # Return minimum of both
        return min(next_sample_time, next_task_time)

    def _check_and_create_tasks(self) -> int:
        """Check for scheduled task arrivals and create them"""
        tasks_created = 0

        while True:
            stream_id, next_task_time = self.task_manager.get_next_task_arrival_time()

            if next_task_time > self.current_time:
                break

            # Generate the task
            # [修改] 现在的 TaskManager 在生成任务时会自动调度下一次到达
            # 所以不需要手动调用 update_task_arrival
            task, arrival_time = self.task_manager.generate_next_task(
                self.current_time,
                stream_id
            )

            # Enqueue task
            self.task_queue.enqueue(task)

            tasks_created += 1
            logger.debug(f"Task {task.task_id} arrived at {self.current_time}s")

        return tasks_created

    def _process_tasks(self) -> Tuple[int, int]:
        """
        Process queued tasks with Energy AND Status checks.
        Prevents assigning multiple tasks to the same node simultaneously.
        """
        tasks_executed = 0
        tasks_failed = 0

        # 1. 获取最新能量数据
        current_energies, _ = self.simulink_interface.get_node_energies()

        # 2. 打印排队概况
        q_stats = self.task_queue.get_queue_state()
        logger.info(
            f"[QUEUE] High:{q_stats.get(3, 0)} Mid:{q_stats.get(2, 0)} Low:{q_stats.get(1, 0)} | Total: {self.task_queue.get_total_length()}")

        # --- 任务调度循环 ---
        while True:
            task = self.task_queue.peek()
            if task is None:
                break

            # 找出所有“可用”节点
            candidate_nodes = []
            busy_nodes_count = 0

            for i in range(self.config.NUM_NODES):
                is_energized = current_energies[i] >= task.energy_threshold
                is_idle = self.current_time >= self.node_release_times[i]

                if is_energized and is_idle:
                    candidate_nodes.append(i)
                elif not is_idle:
                    busy_nodes_count += 1

            available_count = len(candidate_nodes)

            decision_msg = (f"[CHECK] Task {task.task_id:03d} (Prio {task.priority}) | "
                            f"Need {task.required_nodes} nodes | "
                            f"Candidates: {available_count} (Busy: {busy_nodes_count})")

            if available_count >= task.required_nodes:
                # --- 资源充足，执行任务 ---
                task = self.task_queue.dequeue()

                selected_nodes = candidate_nodes[:task.required_nodes]

                # 锁定节点状态
                task_duration = self.config.SAMPLING_PERIOD
                release_time = self.current_time + task_duration

                for node_id in selected_nodes:
                    self.node_release_times[node_id] = release_time

                # 激活 Simulink
                self.simulink_interface.activate_task_on_nodes(selected_nodes, task.energy_threshold)

                tasks_executed += 1
                self.metrics.tasks_completed += 1

                logger.info(f"{decision_msg} -> EXECUTE on {selected_nodes} (Locked until {release_time:.1f}s)")
            else:
                # --- 资源不足 ---
                if busy_nodes_count > 0 and (available_count + busy_nodes_count >= task.required_nodes):
                    logger.info(f"{decision_msg} -> BLOCKED (Nodes Busy)")
                else:
                    logger.info(f"{decision_msg} -> BLOCKED (Insufficient Energy)")

                # 发生阻塞，跳出循环
                break

        # --- 超时检查循环 ---
        while True:
            peeked_task = self.task_queue.peek()
            if peeked_task is None:
                break

            # 使用仿真时间计算等待时长
            wait_time = peeked_task.current_wait_time(current_time=self.current_time)

            if wait_time > peeked_task.max_delay:
                dead_task = self.task_queue.dequeue()
                tasks_failed += 1
                self.metrics.deadlines_missed += 1
                self.metrics.tasks_failed += 1
                logger.error(f"[TIMEOUT] Task {dead_task.task_id} dropped! (Waited {wait_time:.1f}s)")
            else:
                break

        return tasks_executed, tasks_failed

    def _simulation_step(self) -> None:
        """Execute one simulation step"""
        # Update simulation time
        self.simulink_interface.update_env_data(self.current_time)

        next_sample_time = self.current_time + self.config.SAMPLING_PERIOD

        current_energies, _ = self.simulink_interface.get_node_energies()

        if current_energies:
            self.time_history.append(self.current_time)
            self.voltage_history.append(current_energies)

            # 统计电压分布
            low = sum(1 for v in current_energies if v < 1.8)
            mid = sum(1 for v in current_energies if 1.8 <= v < 2.5)
            high = sum(1 for v in current_energies if v >= 2.5)
            avg_v = sum(current_energies) / len(current_energies)

            logger.info(f"[ENERGY] Time:{self.current_time:.1f}s | Avg:{avg_v:.2f}V | "
                        f"Dist: Low[{low}] Mid[{mid}] High[{high}]")

        self.simulink_interface.pause_at_time(next_sample_time)

        # Wait for Simulink pause
        self.simulink_interface.wait_for_next_pause()

        # Update current time
        self.current_time = self.simulink_interface.get_current_time()

        if self.rl_interface.enabled and hasattr(self.rl_interface.rl_agent, "set_current_time"):
            self.rl_interface.rl_agent.set_current_time(self.current_time)

        if current_energies:
            volts_str = ", ".join([f"{v:.2f}" for v in current_energies])
            logger.info(f"Step [{self.current_time:.1f}s] Energy States:")
            logger.info(f"  -> Voltages: [{volts_str}]")
        else:
            logger.warning("Step data is empty!")

        self.state_calculator.update_energy_data(current_energies, self.current_time)

        # Check for new task arrivals
        self._check_and_create_tasks()

        # Compute current state
        queue_length = self.task_queue.get_total_length()
        queue_priority = self.task_queue.get_highest_priority()
        max_wait_time = self.task_queue.get_max_waiting_time(current_time=self.current_time)

        state = self.state_calculator.compute_state_vector(
            queue_length,
            queue_priority,
            max_wait_time
        )

        # Get RL action
        action = self.rl_interface.select_action(state)

        # Apply RL action
        self.rl_interface.apply_action(
            action,
            self.task_manager,
            self.simulink_interface,
            self.task_queue
        )

        # Process tasks
        tasks_executed, tasks_failed = self._process_tasks()

        # Sample next state
        next_energies, _ = self.simulink_interface.get_node_energies()
        self.state_calculator.update_energy_data(next_energies, self.current_time)

        next_queue_length = self.task_queue.get_total_length()
        next_queue_priority = self.task_queue.get_highest_priority()
        next_max_wait_time = self.task_queue.get_max_waiting_time(current_time=self.current_time)

        next_state = self.state_calculator.compute_state_vector(
            next_queue_length,
            next_queue_priority,
            next_max_wait_time
        )

        # Compute reward
        reward = self.rl_interface.compute_reward(
            tasks_completed=tasks_executed,
            energy_used=0.0,
            avg_response_time=0.0,
            deadlines_missed=self.metrics.deadlines_missed,
            time_step=self.config.SAMPLING_PERIOD
        )

        # Update RL agent
        done = self.current_time >= self.config.SIMULATION_DURATION
        self.rl_interface.update_agent(state, action, reward, next_state, done)

        # Log step data
        trajectory_entry = {
            'time': self.current_time,
            'state': state,
            'action': action.to_vector(),
            'reward': reward.total,
            'next_state': next_state,
            'tasks_executed': tasks_executed,
            'queue_length': queue_length,
        }
        self._trajectory_data.append(trajectory_entry)

        logger.debug(
            f"Step {len(self._trajectory_data)}: time={self.current_time:.1f}s, "
            f"reward={reward.total:.3f}, queue_len={queue_length}"
        )

    def run(self) -> None:
        """Run complete simulation"""
        logger.info("=" * 60)
        logger.info("Starting Maestro Simulation")
        logger.info(f"Duration: {self.config.SIMULATION_DURATION}s")
        logger.info("=" * 60)

        self.initialize()
        start_wall_time = time.time()

        try:
            step_count = 0
            while self.current_time < self.config.SIMULATION_DURATION:
                self._simulation_step()
                step_count += 1

                if step_count % 100 == 0:
                    elapsed = time.time() - start_wall_time
                    progress = self.current_time / self.config.SIMULATION_DURATION * 100
                    logger.info(f"Progress: {progress:.1f}% (sim_time={self.current_time:.1f}s, "
                               f"wall_time={elapsed:.1f}s, tasks={self.metrics.tasks_completed})")

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup and save results"""
        logger.info("=" * 60)
        logger.info("Finalizing Simulation")
        logger.info("=" * 60)

        self.simulink_interface.stop()

        full_times, full_voltages = self.simulink_interface.get_full_history()

        if len(full_times) > 0:
            self._plot_voltage_curves(full_times, full_voltages)
        else:
            logger.warning("No full history data found. Using coarse Python logs.")
            self._plot_voltage_curves(np.array(self.time_history), np.array(self.voltage_history))

        # Log final metrics
        self.metrics.current_time = self.current_time
        logger.info(f"Simulation complete")
        logger.info(f"  Tasks completed: {self.metrics.tasks_completed}")
        logger.info(f"  Tasks failed: {self.metrics.tasks_failed}")
        logger.info(f"  Deadlines missed: {self.metrics.deadlines_missed}")
        logger.info(f"  Total steps: {len(self._trajectory_data)}")

        # Save trajectory if enabled
        if self.config.SAVE_TRAJECTORY:
            self._save_trajectory()

    def _plot_voltage_curves(self, times: np.ndarray, data: np.ndarray) -> None:
        """Visualizes the voltage history using provided data arrays."""
        if len(times) == 0 or len(data) == 0:
            logger.warning("Data empty, skipping plot.")
            return

        try:
            plt.figure(figsize=(12, 6))

            nodes_count = data.shape[1] if len(data.shape) > 1 else 1

            for node_id in range(nodes_count):
                plt.plot(times, data[:, node_id], label=f'Node {node_id}', linewidth=1, alpha=0.7)

            plt.title('High-Resolution Node Voltage Trajectories (Simulink Log)')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Voltage (V)')

            plt.axhline(y=self.config.ENERGY_THRESHOLD_MIN, color='r', linestyle='--', label='Min Threshold')
            plt.axhline(y=self.config.ENERGY_THRESHOLD_MAX, color='g', linestyle='--', label='Max Threshold')

            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # Legend might be too big if many nodes, can adjust
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1, fontsize='small')
            plt.tight_layout()

            output_file = 'voltage_result_full.png'
            plt.savefig(output_file, dpi=300)
            plt.close()

            logger.info(f"High-res plot saved to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to plot results: {e}")

    def _save_trajectory(self) -> None:
        """Save trajectory data for analysis"""
        import pickle

        try:
            with open(self.config.TRAJECTORY_FILE, 'wb') as f:
                pickle.dump({
                    'config': self.config,
                    'metrics': self.metrics,
                    'trajectory': self._trajectory_data,
                }, f)
            logger.info(f"Trajectory saved to {self.config.TRAJECTORY_FILE}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")