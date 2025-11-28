import sys
import os
import logging
import matlab.engine

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 关键步骤：确保 Python 能找到 src 包 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.config import SimulationConfig
from src.simulator import MaestroSimulator
from src.behavior_agent import BehaviorAgent


def main():
    logger.info("Starting Maestro Simulation System (Heterogeneous Stream Mode)...")

    # 1. 启动 MATLAB 引擎
    logger.info("Launching MATLAB Engine (this may take a while)...")
    eng = matlab.engine.start_matlab()
    # eng.desktop(nargout=0) # 如需查看 MATLAB 界面可取消注释

    try:
        # 2. 加载配置
        config = SimulationConfig()
        config.SAVE_TRAJECTORY = True

        # [可选] 在这里覆盖 config 的默认值，例如：
        # config.NUM_TASK_STREAMS = 20
        # config.SIMULATION_DURATION = 100.0

        # 3. 初始化仿真器 (Simulator)
        # 注意：这里我们先不传入 agent，因为 agent 需要依赖 simulator 生成的任务流信息
        sim = MaestroSimulator(config, rl_enabled=True, agent=None)
        logger.info("Simulator initialized. Generating random task streams...")

        # 4. 获取生成的任务流列表
        # 因为任务现在是随机生成的，我们需要问 TaskManager：“你到底生成了哪些流？”
        # 这将返回一个 PoissonTaskStream 对象的列表
        task_streams_list = sim.task_manager.get_all_task_types()
        logger.info(f"Retrieved {len(task_streams_list)} generated task streams.")

        # 5. 定义动作空间的物理边界
        action_limits = {
            'sampling_period': (0.1, 3.0),  # 采样周期范围 (秒)
            'energy_threshold': (1.8, 3.8),  # 能量阈值范围 (V)
            'required_nodes': (1, 6),  # 所需节点数范围 (需覆盖 config 中的最大值)
            'priority_levels': [1, 2, 3]  # 优先级列表
        }

        # 6. 初始化行为代理 (BehaviorAgent)
        # 注意：这里传入的是 task_streams 而不是旧的 task_types
        collector_agent = BehaviorAgent(
            action_space_limits=action_limits,
            task_streams=task_streams_list
        )

        # 7. 将 Agent 注入到仿真器中
        # 步骤 A: 注入到 RL 接口
        sim.rl_interface.rl_agent = collector_agent
        sim.rl_interface.enabled = True

        # 步骤 B: 让 Agent 绑定任务队列 (用于感知队列压力)
        collector_agent.bind_task_queue(sim.task_queue)

        logger.info(f"Agent injected: {type(collector_agent).__name__}")

        # 8. 注入 MATLAB Engine 到 Simulink 接口
        if hasattr(sim, 'simulink_interface'):
            sim.simulink_interface.matlab_engine = eng
            logger.info("MATLAB Engine injected into SimulinkInterface.")
        else:
            logger.warning("Could not find simulink_interface attribute in simulator.")

        # 9. 运行仿真
        logger.info(f"Running simulation. Data will be saved to: {config.TRAJECTORY_FILE}")
        sim.run()

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
    finally:
        # 10. 清理
        logger.info("Shutting down MATLAB Engine...")
        eng.quit()
        logger.info("Done.")


if __name__ == "__main__":
    main()