"""
node.py
ASIC辅助通感一体化拓扑仿真平台 - 节点实体类
---------------------------------------------------------
定义网络中的单个节点（无人机/车辆）。
核心功能：
1. 初始化：生成位置、分配状态（正常/故障）。
2. 运动：根据状态生成不同的速度矢量 (这是具身智能推断的物理基础)。
3. 更新：在仿真地图中移动，处理边界碰撞。
"""

import numpy as np
import config as cfg  

class Node:
    def __init__(self, node_id):
        """
        初始化一个节点
        :param node_id: 节点的唯一ID
        """
        self.id = node_id
        
        # 1. 初始化位置 (在地图范围内随机分布)
        self.pos = np.random.rand(2) * np.array([cfg.AREA_WIDTH, cfg.AREA_HEIGHT])
        
        # 2. 初始化状态 (Ground Truth)
        # 根据配置文件的比例，随机分配身份
        rand_val = np.random.rand()
        if rand_val < cfg.RATIO_FAULTY:
            self.status = cfg.STATUS_FAULTY
        elif rand_val < (cfg.RATIO_FAULTY + cfg.RATIO_OVERLOAD):
            self.status = cfg.STATUS_OVERLOAD
        else:
            self.status = cfg.STATUS_NORMAL
            
        # 3. 初始化速度 (体现具身特征)
        self.vel = self._generate_velocity_by_status()
        
        # 4. 初始化雷达朝向 (默认朝向速度方向，或者随机)
        # 归一化速度向量作为朝向
        speed = np.linalg.norm(self.vel)
        if speed > 0:
            self.heading = self.vel / speed
        else:
            self.heading = np.array([1.0, 0.0]) # 静止时默认朝东

    def _generate_velocity_by_status(self):
        """
        【关键逻辑】根据内在状态生成物理运动特征
        这是后续 ASIC NPU 进行推理的物理依据。
        """
        # 生成一个随机方向
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # 根据状态决定速度大小 (Magnitude)
        if self.status == cfg.STATUS_NORMAL:
            # 正常节点：高速机动 (V_MIN ~ V_MAX)
            speed = np.random.uniform(cfg.V_MIN, cfg.V_MAX)
            
        elif self.status == cfg.STATUS_FAULTY:
            # 故障节点：速度极慢或停滞 (0 ~ V_MIN/2)
            # 模拟动力系统受损
            speed = np.random.uniform(0, cfg.V_MIN / 2)
            
        elif self.status == cfg.STATUS_OVERLOAD:
            # 过载节点：速度受限 (V_MIN)
            # 模拟因为计算任务重，不敢飞太快
            speed = cfg.V_MIN
            
        return direction * speed

    def move(self, dt=1.0):
        """
        更新节点位置 (模拟物理移动)
        :param dt: 时间步长
        """
        # 简单位移公式: s = v * t
        self.pos += self.vel * dt
        
        # --- 边界处理 (反弹/Bouncing) ---
        # 如果撞墙了，就把速度反向，模拟在区域内巡逻
        if self.pos[0] < 0 or self.pos[0] > cfg.AREA_WIDTH:
            self.vel[0] *= -1
            self.heading[0] *= -1 # 掉头
            # 修正位置防止出界
            self.pos[0] = np.clip(self.pos[0], 0, cfg.AREA_WIDTH)
            
        if self.pos[1] < 0 or self.pos[1] > cfg.AREA_HEIGHT:
            self.vel[1] *= -1
            self.heading[1] *= -1 # 掉头
            self.pos[1] = np.clip(self.pos[1], 0, cfg.AREA_HEIGHT)