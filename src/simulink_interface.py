"""
Simulink Control Interface Module
Handles communication with MATLAB/Simulink simulation environment.
"""

import logging
import time
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import matlab.engine
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class NodeEnergyState:
    """Energy state of a network node."""
    node_id: int
    voltage: float
    is_active: bool = True
    last_update: float = 0.0


class ISimulinkInterface(ABC):
    """Abstract interface for Simulink integration"""

    @abstractmethod
    def _generate_next_step_env(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize Simulink connection"""
        pass
    
    @abstractmethod
    def pause_at_time(self, pause_time: float) -> None:
        """Set pause time"""
        pass
    
    @abstractmethod
    def get_current_time(self) -> float:
        """Get current simulation time"""
        pass

    @abstractmethod
    def get_full_history(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def update_env_data(self, current_sim_time: float) -> None:
        pass
    
    @abstractmethod
    def get_node_energies(self) -> List[float]:
        """Read energy levels"""
        pass
    
    @abstractmethod
    def activate_task_on_nodes(self, node_ids: List[int], task_threshold: float) -> None:
        """Activate task on nodes"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop simulation"""
        pass


class SimulinkInterface(ISimulinkInterface):
    """
    Simulink Control Interface Implementation (Custom Block Control Version)
    """

    def __init__(self,
                 sampling_period: float = 0.2,
                 num_nodes: int = 25,
                 simulation_duration: float = 100.0,
                 energy_threshold_min: float = 1.8,
                 energy_threshold_max: float = 3.8,
                 matlab_engine: Optional[object] = None,
                 data_file: str = "ENERGY_SOURCE_50s.csv",
                 scale_factor: float = 0.01):

        self.sampling_period = sampling_period
        self.num_nodes = num_nodes
        self.simulation_duration = simulation_duration
        self.matlab_engine = matlab_engine
        self._is_running = False
        self._last_sample_time = 0.0
        self.scale_factor = scale_factor

        self.energy_threshold_min = energy_threshold_min
        self.energy_threshold_max = energy_threshold_max

        # --- 配置路径 ---
        self.model_dir = r"D:\Document\Prj_file\MATLAB\Maestro"
        self.model_name = "node_model"
        self.pause_block_name = "pause_time"

        self._node_energies: Dict[int, float] = {i: 0.0 for i in range(num_nodes)}

        self.last_env_values: np.ndarray = np.zeros((1, self.num_nodes))

        # --- 加载真实数据 ---
        try:
            # 读取扩展后的 CSV
            data = pd.read_csv(data_file)
            self.trace_time = data['time'].values
            self.trace_value = data['voltage'].values

            # 创建插值函数 (线性插值)
            # fill_value="extrapolate" 防止浮点误差导致超出范围报错
            self.interp_func = interp1d(self.trace_time, self.trace_value,
                                        kind='linear', fill_value="extrapolate")

            logger.info(f"Loaded real energy trace from {data_file}. Scale factor: {scale_factor}")
        except Exception as e:
            logger.error(f"Failed to load energy trace: {e}")
            raise e

        logger.debug(f"SimulinkInterface initialized. Target Model: {self.model_name}")

    def _generate_next_step_env(self) -> np.ndarray:
        """
        [修改版] 电压源充能模式
        利用 Ohm's Law (I = (V_source - V_node) / R) 计算输入电流。
        """
        # 1. 获取时间点
        t_start = self.get_current_time()
        t_end = t_start + self.sampling_period

        # 2. 从 CSV 获取 "源电压" (Source Voltage)
        v_source_start = float(self.interp_func(t_start))
        v_source_end = float(self.interp_func(t_end))

        # 3. 获取 "节点电压" (Node Voltage)
        # 注意：我们需要的是当前的节点电压。
        # self._node_energies 是一个 (N, 1) 的 numpy 数组
        current_node_volts = self._node_energies.flatten()  # 转换为 (N,)

        # 4. 定义充电参数
        # 充电电导 G = 1/R。值越大，充电越快；值越小，充电越慢。
        # 建议根据 CSV 电压量级调整。假设 R=100欧姆 -> G=0.01
        G_conductance = 0.01

        # 5. 计算输入电流 (核心修改)
        # 逻辑：I = (V_source - V_node) * G
        # 如果 V_source < V_node，电流为负(放电)；如果加了二极管保护，需 clip 到 0

        # 计算 Start 时刻的电流 (对每个节点)
        # 假设所有节点面对的环境源电压相同 v_source_start
        delta_v = v_source_start - current_node_volts
        i_start = delta_v * G_conductance

        # 物理约束：假设带有二极管，不允许倒灌 (即不允许源电压低于电池时放电)
        i_start = np.maximum(i_start, 0.0)

        # 物理约束：最大充电电流限制 (例如 50mA)
        i_start = np.clip(i_start, 0.0, 0.05)

        # 6. 预测 End 时刻的电流
        # 由于不知道 End 时刻的节点电压(那是Simulink算出来的)，
        # 我们简单假设它与 Start 时刻近似，或者略微上升。
        # 这里为了稳定，直接使用 Start 时刻的计算值作为基准，
        # 或者使用 v_source_end 计算，但忽略节点电压在这一步内的微小变化。
        delta_v_end = v_source_end - current_node_volts
        i_end = delta_v_end * G_conductance
        i_end = np.clip(np.maximum(i_end, 0.0), 0.0, 0.05)

        # 7. 格式化数据 (Reshape to 1xN)
        row_start = i_start.reshape(1, self.num_nodes)
        row_end = i_end.reshape(1, self.num_nodes)

        # 添加一点微小的随机扰动，避免所有节点完全同步
        noise = np.random.normal(0, 0.0005, (1, self.num_nodes))
        row_start += noise
        row_end += noise

        # 再次 Clip 确保安全
        row_start = np.clip(row_start, 0.0, 0.05)
        row_end = np.clip(row_end, 0.0, 0.05)

        # 8. 更新 & 返回
        self.last_env_values = row_end
        return np.vstack((row_start, row_end))

    def initialize(self) -> None:
        """
        Initialize Simulink environment.
        Adapted for 'From Workspace' blocks for ALL inputs (Energy, Task, Initial_Voltages).
        """
        # 1. 启动或连接 MATLAB 引擎
        if self.matlab_engine is None:
            logger.info("Starting MATLAB Engine...")
            self.matlab_engine = matlab.engine.start_matlab()

        # 2. 切换工作目录
        if os.path.exists(self.model_dir):
            self.matlab_engine.cd(self.model_dir, nargout=0)
        else:
            logger.warning(f"Model directory not found: {self.model_dir}")


        # 3. 初始化关键变量 (这也是防止 Sim 报错的关键)
        logger.info("Initializing Workspace variables...")

        self.matlab_engine.workspace['NUM_NODES'] = float(self.num_nodes)
        logger.debug(f"Pushed NUM_NODES={self.num_nodes} to MATLAB workspace.")

        self.matlab_engine.workspace['Min_energy_threshold'] = float(self.energy_threshold_min)
        self.matlab_engine.workspace['Max_energy_threshold'] = float(self.energy_threshold_max)

        # --- A. 初始化电压向量 (State) ---
        # 对应模型中的 'Initial_Voltages'，被 Constant 模块读取
        # 必须是 <num_nodes x 1> 的二维 double 数组 (列向量)
        # 初始默认设为 0V 或者满电 (e.g., 2.5V)
        self._node_energies = np.random.uniform(0.0, 4.2, (self.num_nodes, 1))
        self.matlab_engine.workspace['Initial_Voltages'] = matlab.double(self._node_energies.tolist())

        # --- B. 自动生成全周期环境获能数据 ---
        # 使用 config 中的总时长
        base_current = 0.02
        self.last_env_values = np.full((1, self.num_nodes), base_current)
        # 添加一点初始随机性
        self.last_env_values += np.random.normal(0, 0.002, (1, self.num_nodes))
        self.last_env_values = np.clip(self.last_env_values, 0.0, 0.05)

        # --- C. 生成并推送 Step 0 的输入数据 ---
        # 调用刚刚写的即时生成函数
        env_slice_step0 = self._generate_next_step_env()

        # 构建时间向量 [0; dt] (Simulink 内部时间每次都重置)
        t_vector = matlab.double([[0.0], [float(self.sampling_period)]])
        v_vector = matlab.double(env_slice_step0.tolist())

        # 1. 设置截断时间
        self.matlab_engine.workspace['pause_time'] = float(self.sampling_period)

        # 2. 推送 Energy_Harvesting
        self.matlab_engine.workspace['temp_t'] = t_vector
        self.matlab_engine.workspace['temp_v'] = v_vector
        self.matlab_engine.eval("Energy_Harvesting.time = temp_t;", nargout=0)
        self.matlab_engine.eval("Energy_Harvesting.signals.values = temp_v;", nargout=0)
        self.matlab_engine.eval(f"Energy_Harvesting.signals.dimensions = {self.num_nodes};", nargout=0)

        # 3. 推送 Task_Activation_Signal (Step 0 默认全 0)
        self.matlab_engine.eval("Task_Activation_Signal.time = temp_t;", nargout=0)
        self.matlab_engine.eval(f"Task_Activation_Signal.signals.values = zeros(2, {self.num_nodes});", nargout=0)
        self.matlab_engine.eval(f"Task_Activation_Signal.signals.dimensions = {self.num_nodes};", nargout=0)

        # --- D. 加载模型 ---
        self.matlab_engine.load_system(self.model_name, nargout=0)

        safe_stop_time = self.simulation_duration + 5.0
        logger.info(f"Setting Simulink StopTime to {safe_stop_time}s (Duration + Buffer)")
        self.matlab_engine.set_param(self.model_name, 'StopTime', str(safe_stop_time), nargout=0)

        self._is_running = True
        logger.info("Initialization complete. Ready for Step 0.")

    def _mock_initialize(self, init_voltages: list) -> None:
        """Helper for mock mode initialization"""
        for i, voltage in enumerate(init_voltages):
            self._node_energies[i] = float(voltage)
        logger.info("Mock mode: initialized")
        self._is_running = True
        self._last_sample_time = 0.0

    def update_env_data(self, current_sim_time: float) -> None:
        """
        生成下一个时间步的环境获能数据，并推送到 MATLAB Workspace。

        Args:
            current_sim_time: 当前仿真的截止时间 (即下一帧的起始时间)
        """
        if self.matlab_engine is None:
            # Mock模式下的简单处理
            self._generate_next_step_env()  # 仅为了更新内部 last_env_values
            return

        try:
            # 1. 生成数据 (返回 2xN 矩阵: [Start_Values; End_Values])
            # 这一步利用了您已经写好的随机游走逻辑
            env_data = self._generate_next_step_env()

            # 2. 构建时间向量 [t_current; t_next]
            # 必须使用 float 强转，防止 numpy 类型导致 MATLAB 引擎报错
            next_sim_time = current_sim_time + self.sampling_period
            t_vector = matlab.double([[float(current_sim_time)], [float(next_sim_time)]])

            # 3. 构建数值矩阵
            v_vector = matlab.double(env_data.tolist())

            # 4. 推送到 MATLAB 工作区
            # 使用临时变量避免直接操作结构体出错
            self.matlab_engine.workspace['temp_t'] = t_vector
            self.matlab_engine.workspace['temp_v'] = v_vector

            # 5. 更新结构体 (关键步骤)
            # 这会覆盖 Energy_Harvesting 变量。
            # Simulink 的 'From Workspace' 模块若设置了 "Interpolate data"，
            # 它会在 t_current 到 t_next 之间对这两个点进行线性插值。
            self.matlab_engine.eval("Energy_Harvesting.time = temp_t;", nargout=0)
            self.matlab_engine.eval("Energy_Harvesting.signals.values = temp_v;", nargout=0)

            logger.debug(f"Environment data updated for window [{current_sim_time:.2f}, {next_sim_time:.2f}]")

        except Exception as e:
            logger.error(f"Failed to update environment data: {e}")
            raise e

    def pause_at_time(self, pause_time: float) -> None:
        """
        更新模型中的 Constant 模块数值，并让仿真继续运行。
        """
        if not self._is_running:
            raise RuntimeError("Interface not running")

        if self.matlab_engine is None:
            self._last_sample_time = pause_time
            return

        try:
            # 1. 构建模块的完整路径
            block_path = f"{self.model_name}/{self.pause_block_name}"

            # [优化] 使用格式化字符串防止浮点数精度导致的字符串转换问题
            # 例如防止 20.0000000001 导致 Simulink 无法精确匹配
            self.matlab_engine.set_param(block_path, 'Value', f"{pause_time:.6f}", nargout=0)

            status = self.matlab_engine.get_param(self.model_name, 'SimulationStatus')

            if status == 'stopped':
                # 检查当前时间。如果当前时间已经是目标时间或更晚，说明已经跑完了，不需要 continue
                sim_time = float(self.matlab_engine.get_param(self.model_name, 'SimulationTime'))

                if sim_time >= pause_time:
                    logger.warning(f"Simulink already reached target time {pause_time} (Current: {sim_time}).")
                    return

                logger.info(f"Starting/Continuing simulation to t={pause_time}...")

                # 只有在 t=0 时才使用 start，中间意外停止应该报错或检查配置
                if self._last_sample_time == 0.0:
                    self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'start', nargout=0)
                else:
                    # 如果在这里是 stopped 状态且不是 t=0，说明 Simulink 因为 StopTime 或 错误而自行停止了
                    # 尝试 continue 可能会失败，取决于 MATLAB 版本
                    try:
                        self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'continue', nargout=0)
                    except Exception as e:
                        logger.error("Simulation was stopped and cannot be continued. Check 'StopTime' in .slx file.")
                        raise e

            elif status == 'paused':
                # 如果是 Assertion 模块导致的暂停
                logger.info(f"Resuming simulation to t={pause_time}...")
                self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'continue', nargout=0)

            elif status == 'running':
                logger.warning("Simulink is already running.")

        except Exception as e:
            logger.error(f"Failed to control Simulink block/time: {e}")
            raise

    def wait_for_next_pause(self, timeout: float = 60.0) -> bool:
        if self.matlab_engine is None:
            return True

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                status = self.matlab_engine.get_param(self.model_name, 'SimulationStatus')
                sim_time = float(self.matlab_engine.get_param(self.model_name, 'SimulationTime'))

                # 更新当前时间记录
                self._last_sample_time = sim_time

                if status in ['paused', 'stopped'] and sim_time > 0:
                    # [可选] 可以在这里再次检查 sim_time 是否接近我们设定的目标时间
                    # if abs(sim_time - target_time) < 1e-3: ...
                    return True

            except Exception as e:
                logger.error(f"Error checking status: {e}")
                return False

            time.sleep(0.1)

        logger.error(f"Timeout waiting for Simulink to reach next pause. Status: {status}")
        return False

    def get_current_time(self) -> float:
        if self.matlab_engine is None:
            return self._last_sample_time

        try:
            # 1. (可选) 先检查仿真状态，防止在已停止的模型上查询导致异常
            # status = self.matlab_engine.get_param(self.model_name, 'SimulationStatus')
            # if status == 'stopped':
            #     return self._last_sample_time

            # 2. 获取时间字符串
            time_str = self.matlab_engine.get_param(self.model_name, 'SimulationTime')

            # 3. 转换为浮点数
            return float(time_str)

        except Exception as e:
            # 打印详细错误帮助调试
            logger.warning(f"Failed to get SimulationTime: {e}")
            # 如果读取失败（例如模型刚关闭），返回上一次记录的时间，避免程序崩溃
            return self._last_sample_time

    def get_node_energies(self) -> tuple[List[float], List[float]]:
        """
        获取当前节点能量 (State) 和 当前环境输入 (Debug Input).

        Returns:
            Tuple[List[float], List[float]]: (节点电压列表, 环境输入电流列表)
        """
        node_volts = []
        input_debug = []

        # 1. 获取节点电压 (原有逻辑)
        try:
            raw_data = self.matlab_engine.workspace['Node_Voltages']
            new_energies = np.array(raw_data)
            if new_energies.size == self.num_nodes:
                self._node_energies = new_energies.reshape((self.num_nodes, 1))
            node_volts = self._node_energies.flatten().tolist()
        except Exception:
            # Fallback for t=0 or error
            if isinstance(self._node_energies, np.ndarray):
                node_volts = self._node_energies.flatten().tolist()
            else:
                node_volts = list(np.array(self._node_energies).flatten())

        # 2. [新增] 获取环境输入探针 (用于验证数据是否更新)
        try:
            # 读取 To Workspace 模块输出的变量 'Debug_Input_Monitor'
            # 只有当仿真运行过至少一步，这个变量才会存在
            if self.matlab_engine.exist('Debug_Input_Monitor'):
                # 获取最后一行数据 (当前时刻的值)
                # MATLAB 语法: Debug_Input_Monitor(end, :)
                raw_input = self.matlab_engine.eval("Debug_Input_Monitor(end, :)")
                input_debug = np.array(raw_input).flatten().tolist()
            else:
                # 仿真刚开始或变量不存在
                input_debug = [0.0] * self.num_nodes

        except Exception as e:
            logger.warning(f"Failed to read Debug_Input_Monitor: {e}")
            input_debug = [-1.0] * self.num_nodes  # 使用 -1 标记读取失败

        return node_volts, input_debug

    def get_full_history(self) -> tuple[np.ndarray, np.ndarray]:
        """
        在仿真结束后，从 MATLAB 工作区提取高分辨率的全过程日志。

        Returns:
            times: (T,) 时间向量
            voltages: (T, N) 电压矩阵
        """
        if self.matlab_engine is None:
            return np.array([]), np.array([])

        try:
            logger.info("Fetching full high-resolution history from MATLAB...")

            # 1. 检查变量是否存在
            if not self.matlab_engine.exist('Voltage_Log'):
                logger.warning("Variable 'Voltage_Log' not found in MATLAB workspace.")
                return np.array([]), np.array([])

            # 2. 提取时间轴 (Time)
            # 假设 Save format 是 Structure with Time
            time_data = self.matlab_engine.eval("Voltage_Log.time")
            times = np.array(time_data).flatten()

            # 3. 提取数据值 (Values)
            # 注意：Simulink Structure 的 values 维度通常是 (N, 1, T) 或 (T, N)，需要根据实际情况调整
            val_data = self.matlab_engine.eval("Voltage_Log.signals.values")
            voltages = np.array(val_data)

            # 维度清洗：确保它是 (Time, Nodes) 的形状
            # 如果 Simulink 输出是 (Nodes, Time)，则转置
            if voltages.shape[0] == self.num_nodes and voltages.shape[1] != self.num_nodes:
                voltages = voltages.T

            # 如果有多余的维度 (例如 (T, 1, N)), 压缩它
            voltages = np.squeeze(voltages)

            logger.info(f"Retrieved history: {len(times)} time steps.")
            return times, voltages

        except Exception as e:
            logger.error(f"Failed to fetch full history: {e}")
            return np.array([]), np.array([])

    def activate_task_on_nodes(self,
                               node_ids: List[int],
                               task_threshold: float) -> None:
        if not self._is_running:
            raise RuntimeError("Simulink interface not running")

        # 1. 构建控制向量 (1xN)
        # Python list: [0.0, 1.0, 0.0, ...]
        control_vector = [0.0] * self.num_nodes
        for node_id in node_ids:
            if 0 <= node_id < self.num_nodes:
                control_vector[node_id] = 1.0
            else:
                logger.warning(f"Ignored invalid node_id: {node_id}")

        if self.matlab_engine is not None:
            try:
                # [关键修改] 使用 set_param 直接修改 Constant 模块的值

                # 1. 将列表转换为 MATLAB 格式的字符串向量
                # 例如: "[0 1 0 0 ...]"
                # 注意：使用空格分隔比逗号更符合 MATLAB 原生习惯，尽管两者通常都行
                vec_str = "[" + " ".join([str(int(x)) for x in control_vector]) + "]"

                # 2. 构造模块路径
                # 请确保这里的 'Task_Input' 与您Simulink里新加的Constant模块名字一致
                block_path = f"{self.model_name}/Task_Input"

                # 3. 直接注入参数
                self.matlab_engine.set_param(block_path, 'Value', vec_str, nargout=0)

                logger.debug(f"Directly updated {block_path} with active nodes: {node_ids}")

            except Exception as e:
                logger.error(f"Failed to update task control signal via set_param: {e}")
                # 发生错误时尝试打印更多信息帮助调试
                logger.error(f"Target Block Path: {block_path}")
        else:
            # Mock Mode (模拟模式) - 这里的代码保持不变
            for node_id in node_ids:
                if 0 <= node_id < self.num_nodes:
                    discharge = 0.1
                    self._node_energies[node_id] = max(0.0, float(self._node_energies[node_id]) - discharge)
            logger.info(f"Mock task activated on nodes {node_ids}")

    def stop(self) -> None:
        if self.matlab_engine is not None:
            self.matlab_engine.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self._is_running = False
    
    def get_statistics(self) -> Dict:
        """Get interface statistics."""
        return {
            'is_running': self._is_running,
            'sampling_period': self.sampling_period,
            'last_sample_time': self._last_sample_time,
            'next_pause_time': self._next_pause_time,
            'num_nodes': self.num_nodes,
        }
    
    def set_sampling_period(self, new_period: float) -> None:
        """Update sampling period (called by RL interface)."""
        if new_period <= 0:
            raise ValueError("Sampling period must be positive")
        
        old_period = self.sampling_period
        self.sampling_period = new_period
        logger.info(f"Sampling period updated: {old_period}s -> {new_period}s")
