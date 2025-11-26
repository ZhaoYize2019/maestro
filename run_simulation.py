import sys
import os
import logging
import matlab.engine

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 关键步骤：确保 Python 能找到 src 包 ---
# 获取当前文件所在目录，并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 现在可以安全地从 src 包导入模块了
from src.config import SimulationConfig
from src.simulator import MaestroSimulator

# [新增 1] 导入行为代理 (假设文件名为 behavior_agent.py 且位于 src 目录下)
# 如果文件在根目录，请使用: from behavior_agent import BehaviorAgent
from src.behavior_agent import BehaviorAgent


def main():
    logger.info("Starting Maestro Simulation System (Data Collection Mode)...")

    # 1. 启动 MATLAB 引擎
    logger.info("Launching MATLAB Engine (this may take a while)...")
    eng = matlab.engine.start_matlab()
    # eng.desktop(nargout=0) # 取消注释以查看 MATLAB 界面

    try:
        # 2. 加载配置
        config = SimulationConfig()

        # [新增 2] 开启轨迹保存功能 (离线 RL 必需)
        # 这样 simulator.py 的 cleanup() 才会把数据写入硬盘
        config.SAVE_TRAJECTORY = True
        config.TRAJECTORY_FILE = "offline_rl_dataset_v1.pkl"

        # 可选：调整仿真时长
        # config.SIMULATION_DURATION = 1000.0

        # [新增 3] 实例化行为代理 (80% Frugal + 20% Random)
        # 定义动作空间的物理边界 (需根据 Simulink 参数范围调整)
        action_limits = {
            'sampling_period': (0.5, 5.0),  # 采样周期范围 (秒)
            'energy_threshold': (1.8, 3.8),  # 能量阈值范围 (V)
            'required_nodes': (1, 5),  # 所需节点数范围
            'priority_levels': [1, 2, 3]  # 优先级列表
        }
        collector_agent = BehaviorAgent(action_space_limits=action_limits)

        # 3. 初始化仿真器并注入代理
        # 注意：这里假设你已经修改了 simulator.py 的 __init__ 以接收 agent 参数
        # sim = MaestroSimulator(config, rl_enabled=True, agent=collector_agent)

        # 【备选方案】如果你还没有修改 simulator.py 的 __init__，请使用下面这种“手动注入”方式：
        sim = MaestroSimulator(config, rl_enabled=True)
        sim.rl_interface.rl_agent = collector_agent
        sim.rl_interface.enabled = True
        logger.info(f"Agent injected: {type(collector_agent).__name__}")

        # --- 注入 MATLAB Engine ---
        # 保持你原有的逻辑，将启动好的 engine 塞进去
        if hasattr(sim, 'simulink_interface'):
            sim.simulink_interface.matlab_engine = eng
            logger.info("MATLAB Engine injected into SimulinkInterface.")
        else:
            logger.warning("Could not find simulink_interface attribute in simulator.")

        # 4. 运行仿真
        logger.info(f"Running simulation. Data will be saved to: {config.TRAJECTORY_FILE}")
        sim.run()

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
    finally:
        # 5. 清理
        logger.info("Shutting down MATLAB Engine...")
        eng.quit()
        logger.info("Done.")


if __name__ == "__main__":
    main()