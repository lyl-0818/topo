"""
node.py
ASIC辅助通感一体化拓扑仿真平台 - 节点实体类 (最终版)
---------------------------------------------------------
核心职责：
1. 模拟无人机节点的物理运动 (RWP模型 + 异常状态模拟)。
2. 生成供 ASIC NPU 学习的物理特征 (速度、航向抖动、微动)。
3. 维护历史轨迹数据。
"""

import numpy as np
import config as cfg

class Node:
    def __init__(self, node_id):
        """
        初始化节点
        """
        self.id = node_id
        
        # 1. 位置初始化
        self.pos = np.random.rand(2) * np.array([cfg.AREA_WIDTH, cfg.AREA_HEIGHT])
        
        # 2. 状态初始化 (按照 config 概率分配)
        rand_val = np.random.rand()
        if rand_val < cfg.RATIO_FAULTY:
            self.status = cfg.STATUS_FAULTY
        elif rand_val < (cfg.RATIO_FAULTY + cfg.RATIO_OVERLOAD):
            self.status = cfg.STATUS_OVERLOAD
        else:
            self.status = cfg.STATUS_NORMAL
            
        # 3. 初始朝向 (随机单位向量)
        angle = np.random.uniform(0, 2 * np.pi)
        self.heading = np.array([np.cos(angle), np.sin(angle)])
        
        # 4. 初始速度向量
        self.vel = np.array([0.0, 0.0])
        
        # 5. 【关键】历史数据缓存 (用于 AI 时序分析)
        # 存储过去 10 帧的速度向量
        self.history_vel = []

    def move(self, dt=1.0):
        """
        物理运动引擎：根据不同状态表现出不同的运动特征
        这是具身智能推理的物理基础。
        """
        
        # ==========================================
        # 场景 A: 故障节点 (Faulty)
        # 特征：失去动力(慢)，姿态失控(乱转)，引擎震动(位移抖动)
        # ==========================================
        if self.status == cfg.STATUS_FAULTY:
            # 1. 速度极慢 (0 ~ 5 m/s)
            target_speed = np.random.uniform(0, 5)
            
            # 2. 航向完全随机 (模拟失控旋转)
            # 这一帧和下一帧的朝向毫无关系 -> 方差极大
            random_angle = np.random.uniform(-np.pi, np.pi)
            self.heading = np.array([np.cos(random_angle), np.sin(random_angle)])
            
            # 3. 合成速度向量
            self.vel = self.heading * target_speed
            
            # 4. 额外的物理位置抖动 (Jitter)
            jitter = np.random.normal(0, 5.0, 2) # 剧烈震动
            self.pos += jitter

        # ==========================================
        # 场景 B: 过载/占用节点 (Overloaded)
        # 特征：为了维持链路，航向死死锁定(方差极小)，速度较快且稳定
        # ==========================================
        elif self.status == cfg.STATUS_OVERLOAD:
            # 1. 速度较快且稳定 (15 ~ 25 m/s)
            target_speed = np.random.uniform(15, 25)
            
            # 2. 航向高度锁定 (模拟波束对准)
            # 几乎不改变方向，只有极其微小的扰动
            noise_angle = np.random.normal(0, 0.02) # 极小方差
            # 旋转一个小角度
            c, s = np.cos(noise_angle), np.sin(noise_angle)
            rotation_matrix = np.array(((c, -s), (s, c)))
            self.heading = rotation_matrix.dot(self.heading)
            
            # 3. 合成速度向量
            self.vel = self.heading * target_speed

        # ==========================================
        # 场景 C: 正常节点 (Normal)
        # 特征：平滑飞行，有规律的转向 (RWP)
        # ==========================================
        else:
            # 1. 全速机动 (20 ~ 60 m/s)
            target_speed = np.random.uniform(cfg.V_MIN, cfg.V_MAX)
            
            # 2. 平滑转向 (Smooth Turning)
            # 相比过载节点，正常节点会巡逻，所以会有一定的转向幅度
            turn_angle = np.random.normal(0, 0.3) # 中等方差
            c, s = np.cos(turn_angle), np.sin(turn_angle)
            rotation_matrix = np.array(((c, -s), (s, c)))
            self.heading = rotation_matrix.dot(self.heading)
            
            # 3. 合成速度向量
            self.vel = self.heading * target_speed

        # --- 更新位置 ---
        self.pos += self.vel * dt

        # --- 边界反弹处理 (防止飞出地图) ---
        if self.pos[0] < 0 or self.pos[0] > cfg.AREA_WIDTH:
            self.vel[0] *= -1
            self.heading[0] *= -1 # 物理反弹
            self.pos[0] = np.clip(self.pos[0], 0, cfg.AREA_WIDTH)
            
        if self.pos[1] < 0 or self.pos[1] > cfg.AREA_HEIGHT:
            self.vel[1] *= -1
            self.heading[1] *= -1
            self.pos[1] = np.clip(self.pos[1], 0, cfg.AREA_HEIGHT)

        # --- 维护历史数据 (供 AI 训练和推理) ---
        # 记录当前速度向量的副本
        self.history_vel.append(self.vel.copy())
        # 保持窗口长度为 10
        if len(self.history_vel) > 10:
            self.history_vel.pop(0)