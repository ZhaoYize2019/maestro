import pickle
import sys
import os


def inspect_pickle(file_path):
    # 确保当前目录在 python path 中，以便加载 src 模块中的类
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    print(f"正在尝试加载: {file_path} ...")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("\n✅ 加载成功! 数据概览:")
        print("=" * 40)
        print(f"顶层 Keys: {list(data.keys())}")

        # 1. 检查配置
        if 'config' in data:
            print("\n[1] 仿真配置 (Config):")
            print(data['config'])

        # 2. 检查统计指标
        if 'metrics' in data:
            print("\n[2] 统计指标 (Metrics):")
            print(data['metrics'])

        # 3. 检查 S-A-R 轨迹
        if 'trajectory' in data:
            traj = data['trajectory']
            print(f"\n[3] 轨迹数据 (Trajectory):")
            print(f"    总步数 (Steps): {len(traj)}")

            if len(traj) > 0:
                print("\n    --- 第 1 步数据样本 ---")
                sample = traj[0]
                for key, value in sample.items():
                    # 针对 numpy 数组打印形状，避免刷屏
                    if hasattr(value, 'shape'):
                        print(f"    {key}: shape={value.shape}, value={value}")
                    else:
                        print(f"    {key}: {value}")

                print("\n    --- S-A-R 确认 ---")
                print(f"    State (S): {sample.get('state')}")
                print(f"    Action (A): {sample.get('action')}")
                print(f"    Reward (R): {sample.get('reward')}")
                print(f"    Next State (S'): {sample.get('next_state')}")

    except ModuleNotFoundError as e:
        print(f"\n❌ 错误: 找不到模块定义。请确保此脚本在项目根目录运行，且 'src' 文件夹存在。\n详细信息: {e}")
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")


if __name__ == "__main__":
    # 默认文件名，如果您的文件名不同请修改
    target_file = 'maestro_trajectory.pkl'
    inspect_pickle(target_file)