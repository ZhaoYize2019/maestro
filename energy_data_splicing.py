import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_csv('ENERGY_SOURCE.csv')
time_orig = df['time'].values
volt_orig = df['voltage'].values

# 2. 计算参数
dt = 0.025  # 原始采样间隔
max_t_orig = time_orig[-1]
target_duration = 50.0

# 3. 循环拼接数据
# 计算需要拼接的次数
repeats = int(np.ceil(target_duration / (max_t_orig + dt))) + 1

new_times = []
new_volts = []

for i in range(repeats):
    # 每一轮的时间都要加上偏移量
    offset = i * (max_t_orig + dt)
    t_chunk = time_orig + offset

    new_times.append(t_chunk)
    new_volts.append(volt_orig)

# 4. 合并并裁剪
full_time = np.concatenate(new_times)
full_volt = np.concatenate(new_volts)

# 仅保留 0 - 50s 的数据
mask = full_time <= target_duration
final_time = full_time[mask]
final_volt = full_volt[mask]

# 5. 保存结果
df_extended = pd.DataFrame({'time': final_time, 'voltage': final_volt})
df_extended.to_csv('ENERGY_SOURCE_50s.csv', index=False)
print("已生成扩展数据文件: ENERGY_SOURCE_50s.csv")