"""
radar_asic.py
ASIC辅助通感一体化拓扑仿真平台 - 核心算法层 (修复警告版)
---------------------------------------------------------
修复内容：
在 NPU 推理阶段，将输入数据转换为带有列名的 DataFrame，
以匹配训练时的格式，消除 sklearn 的 UserWarning。
"""

import numpy as np
import pandas as pd  # 【新增】需要导入 pandas
import joblib
import config as cfg

class RadarASIC:
    def __init__(self, mode='SMART'):
        self.mode = mode
        self.ai_model = None
        
        # --- 加载训练好的 AI 模型 ---
        if self.mode == 'SMART':
            try:
                self.ai_model = joblib.load('asic_model.pkl')
            except FileNotFoundError:
                print("[Error] 未找到 asic_model.pkl！")

    def run_topology_control(self, nodes):
        """
        执行拓扑控制循环 (保持不变)
        """
        predicted_status = None

        if self.mode == 'SMART':
            predicted_status = []
            for node in nodes:
                predicted_status.append(
                self._npu_inference(node.history_vel)
                )
        N = len(nodes)
        adjacency_matrix = np.zeros((N, N))
        positions = np.array([n.pos for n in nodes])
        
        if self.mode == 'SMART':
            R_max = cfg.R_DIR_NORMAL
            handshake_prob = cfg.EFFICIENCY_SMART
            FoV = cfg.FOV_DIR
        elif self.mode == 'RANDOM_DIR':
            R_max = cfg.R_DIR_NORMAL
            handshake_prob = cfg.EFFICIENCY_RANDOM
            FoV = cfg.FOV_DIR
        else: 
            R_max = cfg.R_OMNI_JAMMED
            handshake_prob = 1.0
            FoV = cfg.FOV_OMNI

        if FoV < 360:
            self._update_headings(nodes, positions, R_max, FoV)

        for i in range(N):
            node_i = nodes[i]
            for j in range(i + 1, N):
                node_j = nodes[j]
                
                # A. 物理判定
                vec_ij = positions[j] - positions[i]
                dist = np.linalg.norm(vec_ij)
                if dist > R_max: continue
                
                cos_theta_i = np.dot(node_i.heading, vec_ij) / dist
                cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
                angle_i = np.degrees(np.arccos(cos_theta_i))
                
                vec_ji = -vec_ij
                cos_theta_j = np.dot(node_j.heading, vec_ji) / dist
                cos_theta_j = np.clip(cos_theta_j, -1.0, 1.0)
                angle_j = np.degrees(np.arccos(cos_theta_j))
                
                seen_by_i = angle_i <= (FoV / 2)
                seen_by_j = angle_j <= (FoV / 2)
                
                if not (seen_by_i or seen_by_j): continue

                # B. 具身智能判定
                if self.mode == 'SMART':
                   if predicted_status[i] != cfg.STATUS_NORMAL or predicted_status[j] != cfg.STATUS_NORMAL:
                        continue 

                # C. 握手概率
                if np.random.rand() <= handshake_prob:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                    
        return adjacency_matrix

    def _npu_inference(self, velocity_history):
        """
        【修复重点】ASIC NPU 推理逻辑
        将输入转换为 DataFrame，包含特征名称
        """
        if self.ai_model is None or len(velocity_history) < 5:
            return cfg.STATUS_NORMAL
            
        hist_vel = np.array(velocity_history)
        
        # 1. 特征提取 (与 generate_dataset.py 保持一致)
        speeds = np.linalg.norm(hist_vel, axis=1)
        feat_speed_mean = np.mean(speeds)
        feat_speed_std = np.std(speeds)
        
        angles = np.arctan2(hist_vel[:, 1], hist_vel[:, 0])
        complex_angles = np.exp(1j * angles)
        if np.abs(np.mean(complex_angles)) < 1e-9:
            feat_angle_std = 1.0
        else:
            mean_angle_complex = np.mean(complex_angles)
            mean_angle = np.angle(mean_angle_complex)
            angle_diffs = np.angle(complex_angles / np.exp(1j * mean_angle))
            feat_angle_std = np.std(angle_diffs)
            
        speed_fluctuation = speeds - np.mean(speeds)
        fft_coeffs = np.fft.fft(speed_fluctuation)
        fft_power = np.abs(fft_coeffs) ** 2
        feat_high_freq_energy = np.sum(fft_power[len(fft_power)//2:])
        
        if len(hist_vel) > 1:
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
        else:
            feat_smoothness = 1.0
            
        # --- 【关键修改】构造 DataFrame ---
        # 这里的列名必须和 generate_dataset.py 里保存 CSV 时的列名一模一样！
        feature_names = ['speed_mean', 'speed_std', 'angle_std', 'fft_energy', 'smoothness']
        
        input_df = pd.DataFrame([[
            feat_speed_mean, 
            feat_speed_std, 
            feat_angle_std, 
            feat_high_freq_energy, 
            feat_smoothness
        ]], columns=feature_names)
        
        # 使用 DataFrame 进行预测
        prediction = self.ai_model.predict(input_df)
        return prediction[0]

    def _update_headings(self, nodes, positions, R_max, FoV):
        """
        更新波束朝向 (保持不变)
        """
        N = len(nodes)
        for i in range(N):
            node = nodes[i]
            if self.mode == 'RANDOM_DIR':
                random_angle = np.random.uniform(-np.pi, np.pi)
                node.heading = np.array([np.cos(random_angle), np.sin(random_angle)])
            elif self.mode == 'SMART':
                diffs = positions - positions[i]
                dists = np.linalg.norm(diffs, axis=1)
                neighbor_idx = np.where((dists <= R_max) & (dists > 0))[0]
                num_neighbors = len(neighbor_idx)
                
                if num_neighbors > 15:
                    alpha = 0.5 
                    rand_angle = np.random.uniform(-np.pi, np.pi)
                    vec_target = np.array([np.cos(rand_angle), np.sin(rand_angle)])
                elif num_neighbors > 0:
                    alpha = 0.9 
                    neighbor_pos = positions[neighbor_idx]
                    centroid = np.mean(neighbor_pos, axis=0)
                    vec_target = centroid - positions[i]
                else:
                    alpha = 0.0 
                    vec_target = cfg.MAP_CENTER - positions[i]
                
                vec_center = cfg.MAP_CENTER - positions[i]
                if np.linalg.norm(vec_center) > 0: vec_center /= np.linalg.norm(vec_center)
                if np.linalg.norm(vec_target) > 0: vec_target /= np.linalg.norm(vec_target)
                final_vec = alpha * vec_target + (1 - alpha) * vec_center
                if np.linalg.norm(final_vec) > 0:
                    node.heading = final_vec / np.linalg.norm(final_vec)