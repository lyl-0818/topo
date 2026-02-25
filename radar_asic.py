"""
radar_asic.py
ASIC辅助通感一体化拓扑仿真平台 - 核心算法层
---------------------------------------------------------
模拟节点搭载的 "毫米波雷达 + ASIC芯片"。
核心功能：
1. 具身智能推理 (NPU)：根据雷达观测到的速度，推断目标好坏。
2. 智能波束转向 (Smart Steering)：计算邻居密度和向心引力，调整朝向。
3. 连接判定 (Link Setup)：结合物理层(距离/角度)、AI层(状态过滤)和协议层(握手)进行最终判定。
"""

import numpy as np
import config as cfg

class RadarASIC:
    def __init__(self, mode='SMART'):
        """
        初始化 ASIC
        :param mode: 'RANDOM' (随机/全向策略) 或 'SMART' (智能ASIC策略)
        """
        self.mode = mode

    def run_topology_control(self, nodes):
        """
        执行一轮完整的拓扑控制循环
        :param nodes: 所有节点的列表
        :return: 邻接矩阵 (NxN numpy array), 1表示连接, 0表示断开
        """
        N = len(nodes)
        adjacency_matrix = np.zeros((N, N))
        
        # 1. 提取所有节点的位置和速度 (模拟雷达全局扫描)
        # 在实际硬件中，这是通过雷达回波并行获取的
        positions = np.array([n.pos for n in nodes])
        velocities = np.array([n.vel for n in nodes])
        
        # 2. 确定物理参数 (根据模式选择 R 和 握手效率)
        if self.mode == 'SMART':
            R_max = cfg.R_DIR_NORMAL
            handshake_prob = cfg.EFFICIENCY_SMART
            FoV = cfg.FOV_DIR
        elif self.mode == 'RANDOM_DIR':
            R_max = cfg.R_DIR_NORMAL
            handshake_prob = cfg.EFFICIENCY_RANDOM
            FoV = cfg.FOV_DIR
        else: # OMNI (Baseline)
            R_max = cfg.R_OMNI_JAMMED
            handshake_prob = 1.0 # 全向默认不需要对准握手
            FoV = cfg.FOV_OMNI

        # 3. 调整波束朝向 (仅针对定向模式)
        if FoV < 360:
            self._update_headings(nodes, positions, R_max, FoV)

        # 4. 计算连接矩阵 (N^2 复杂度，模拟硬件流水线)
        for i in range(N):
            node_i = nodes[i]
            
            for j in range(i + 1, N):
                node_j = nodes[j]
                
                # --- A. 物理层判定 (距离 & 角度) ---
                vec_ij = positions[j] - positions[i]
                dist = np.linalg.norm(vec_ij)
                
                if dist > R_max:
                    continue # 距离太远，看不见
                
                # 计算 i 看 j 是否在视场内
                # 逻辑：cos(theta) = (heading · vec_ij) / dist
                # 注意：heading已经是归一化向量
                cos_theta_i = np.dot(node_i.heading, vec_ij) / dist
                # 限制数值范围防止 arccos 报错
                cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0) 
                angle_i = np.degrees(np.arccos(cos_theta_i))
                
                # 计算 j 看 i 是否在视场内 (双向握手的前提)
                vec_ji = -vec_ij
                cos_theta_j = np.dot(node_j.heading, vec_ji) / dist
                cos_theta_j = np.clip(cos_theta_j, -1.0, 1.0)
                angle_j = np.degrees(np.arccos(cos_theta_j))
                
                seen_by_i = angle_i <= (FoV / 2)
                seen_by_j = angle_j <= (FoV / 2)
                
                # --- B. 协议层判定 (ASIC 握手逻辑) ---
                # 逻辑：只要有一方看见，且握手成功(概率)，就建立连接
                # 这里的 OR 逻辑 (|) 对应我们之前讨论的 "雷达唤醒"
                physically_connected = (seen_by_i or seen_by_j)
                
                if not physically_connected:
                    continue

                # --- C. 具身智能判定 (Embodied AI Inference) ---
                # 只有 SMART 模式开启此功能
                if self.mode == 'SMART':
                    # i 推断 j 的状态
                    status_j_pred = self._npu_inference(node_j.vel)
                    # j 推断 i 的状态
                    status_i_pred = self._npu_inference(node_i.vel)
                    
                    # 决策：如果推断对方是坏的(故障/过载)，则拒绝连接
                    if status_j_pred != cfg.STATUS_NORMAL or status_i_pred != cfg.STATUS_NORMAL:
                        continue # 具身智能发挥作用：切断无效连接！

                # --- D. 最终握手概率 (模拟扫描时延) ---
                if np.random.rand() <= handshake_prob:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                    
        return adjacency_matrix

    def _npu_inference(self, target_velocity):
        """
        【创新点】模拟 ASIC NPU 的推理功能
        根据观测到的物理特征(速度)，推断内在状态
        """
        speed = np.linalg.norm(target_velocity)
        
        # 1. 规则推理 (模拟训练好的神经网络)
        # 正常节点通常全速飞行 (18~60)
        # 故障/过载节点通常速度较慢 (<18)
        if speed < 18: 
            predicted_label = cfg.STATUS_FAULTY # 泛指异常
        else:
            predicted_label = cfg.STATUS_NORMAL
            
        # 2. 模拟推理误差
        if np.random.rand() > cfg.NPU_ACCURACY:
            # 以 1-Accuracy 的概率翻转预测结果
            predicted_label = cfg.STATUS_NORMAL if predicted_label == cfg.STATUS_FAULTY else cfg.STATUS_FAULTY
            
        return predicted_label

    def _update_headings(self, nodes, positions, R_max, FoV):
        """
        更新所有节点的朝向
        """
        N = len(nodes)
        FoV_rad = np.radians(FoV)
        
        for i in range(N):
            node = nodes[i]
            
            if self.mode == 'RANDOM_DIR':
                # 随机模式：随机选一个方向，或者保持速度方向
                # 为了模拟"乱看"，我们给速度方向加一个巨大的随机扰动
                random_angle = np.random.uniform(-np.pi, np.pi)
                node.heading = np.array([np.cos(random_angle), np.sin(random_angle)])
                
            elif self.mode == 'SMART':
                # 【创新点】智能模式：密度感知 + 向心引力
                
                # 1. 找出所有潜在邻居
                diffs = positions - positions[i]
                dists = np.linalg.norm(diffs, axis=1)
                # 排除自己，排除太远的
                neighbor_idx = np.where((dists <= R_max) & (dists > 0))[0]
                
                # 2. 策略融合
                # A. 密度向量
                vec_density = np.array([0.0, 0.0])
                has_neighbors = False
                
                if len(neighbor_idx) > 0:
                    has_neighbors = True
                    # 简化版密度计算：直接指向邻居的几何中心 (质心)
                    # (比滑动窗口快，且效果近似)
                    neighbor_pos = positions[neighbor_idx]
                    centroid = np.mean(neighbor_pos, axis=0)
                    vec_density = centroid - positions[i]
                    if np.linalg.norm(vec_density) > 0:
                        vec_density /= np.linalg.norm(vec_density)
                
                # B. 向心向量
                vec_center = cfg.MAP_CENTER - positions[i]
                if np.linalg.norm(vec_center) > 0:
                    vec_center /= np.linalg.norm(vec_center)
                    
                # C. 融合
                alpha = cfg.WEIGHT_DENSITY if has_neighbors else 0.0
                final_vec = alpha * vec_density + (1 - alpha) * vec_center
                
                # 更新节点朝向
                if np.linalg.norm(final_vec) > 0:
                    node.heading = final_vec / np.linalg.norm(final_vec)