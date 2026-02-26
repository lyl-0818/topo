"""
generate_dataset.py
功能：生成具身智能训练数据 (进阶版)
特点：
1. 引入频域分析 (FFT) 识别高频抖动。
2. 引入轨迹平滑度识别异常失控。
3. 模拟真实物理环境，制造数据重叠，倒逼 AI 学习深层规律。
"""
import numpy as np
import pandas as pd
import config as cfg
from node import Node

# 定义要生成多少个样本
SAMPLE_COUNT = 10000
data_rows = []

print("正在生成高维特征训练数据...")
print("特征包含: [平均速度, 速度标准差, 航向标准差, 高频能量(FFT), 轨迹平滑度]")

for i in range(SAMPLE_COUNT):
    # 1. 创建节点
    node = Node(i)
    
    # 2. 强制均衡分配标签 (确保数据 1:1:1，防止 AI 偷懒猜概率)
    label = i % 3
    if label == 0: 
        node.status = cfg.STATUS_NORMAL
    elif label == 1: 
        node.status = cfg.STATUS_FAULTY
    elif label == 2: 
        node.status = cfg.STATUS_OVERLOAD
    
    # 3. 模拟运动 (跑 15 步，取最后 10 步，消除启动时的不稳定)
    # 注意：这里依赖 node.py 中的 move 逻辑 (需确保 node.py 已更新为"真实/模糊"版本)
    for _ in range(15):
        node.move()
        
    # --- 4. 特征提取 (Feature Engineering) ---
    # 获取最近 10 帧的历史速度向量
    # history_vel 是一个 list of numpy arrays
    if len(node.history_vel) < 10:
        continue # 数据不足跳过
        
    hist_vel = np.array(node.history_vel) # Shape: (10, 2)
    
    # --- 特征 1: 平均速度 (Speed Mean) ---
    # 区分静止/慢速和高速，但有重叠区
    speeds = np.linalg.norm(hist_vel, axis=1)
    feat_speed_mean = np.mean(speeds)
    
    # --- 特征 2: 速度稳定性 (Speed Std) ---
    # 故障节点往往忽快忽慢
    feat_speed_std = np.std(speeds)
    
    # --- 特征 3: 航向稳定性 (Heading Std) ---
    # 区分"锁定方向"(过载)和"随意转弯"(正常/故障)
    # 使用 arctan2 计算角度，并处理 -pi 到 pi 的跳变
    angles = np.arctan2(hist_vel[:, 1], hist_vel[:, 0])
    # 使用复数平均法计算循环数据的标准差 (更准确)
    complex_angles = np.exp(1j * angles)
    if np.abs(np.mean(complex_angles)) < 1e-9:
        feat_angle_std = 1.0 # 极度不稳定
    else:
        mean_angle_complex = np.mean(complex_angles)
        mean_angle = np.angle(mean_angle_complex)
        # 计算角度差
        angle_diffs = np.angle(complex_angles / np.exp(1j * mean_angle))
        feat_angle_std = np.std(angle_diffs)
    
    # --- 特征 4: 微多普勒特征 / 高频能量 (FFT High Freq Energy) ---
    # 核心创新点：正常转弯是低频变化，故障抖动是高频变化
    # 对速度模值做快速傅里叶变换
    speed_fluctuation = speeds - np.mean(speeds)
    fft_coeffs = np.fft.fft(speed_fluctuation)
    fft_power = np.abs(fft_coeffs) ** 2
    # 取后半部分(高频)的能量和
    feat_high_freq_energy = np.sum(fft_power[len(fft_power)//2:])
    
    # --- 特征 5: 轨迹平滑度 (Trajectory Smoothness) ---
    # 计算相邻速度向量的余弦相似度均值
    # 正常节点轨迹圆滑(接近1)，故障节点轨迹杂乱(接近0或负数)
    cos_sims = []
    for k in range(len(hist_vel)-1):
        v1 = hist_vel[k]
        v2 = hist_vel[k+1]
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod > 1e-6:
            cos_sim = np.dot(v1, v2) / norm_prod
            cos_sims.append(cos_sim)
        else:
            cos_sims.append(0)
    feat_smoothness = np.mean(cos_sims) if cos_sims else 0.0

    # 5. 保存数据
    data_rows.append([feat_speed_mean, feat_speed_std, feat_angle_std, 
                      feat_high_freq_energy, feat_smoothness, label])

# 写入文件
cols = ['speed_mean', 'speed_std', 'angle_std', 'fft_energy', 'smoothness', 'label']
df = pd.DataFrame(data_rows, columns=cols)
df.to_csv('embodied_ai_data.csv', index=False)

print(f"成功生成 {len(df)} 条数据，已保存为 'embodied_ai_data.csv'")
print("数据预览:")
print(df.head())