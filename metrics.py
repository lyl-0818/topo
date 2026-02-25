"""
metrics.py
功能：计算除连通度之外的高级指标。
1. 平均建链时延 (Average Link Latency)
2. 有效连接比例 (Quality of Connectivity)
"""
import numpy as np
import networkx as nx
import config as cfg

def calculate_latency(nodes, adjacency_matrix, mode):
    """
    计算网络的平均建链时延
    """
    G = nx.from_numpy_array(adjacency_matrix)
    edges = G.edges()
    if len(edges) == 0:
        return 0.0
    
    total_latency = 0
    
    # 基础握手时间 (ms)
    T_handshake = 2.0 
    
    for u, v in edges:
        # 1. 物理传输时延 (距离/光速) - 很小，可忽略或设为固定
        dist = np.linalg.norm(nodes[u].pos - nodes[v].pos)
        t_prop = dist / 3e8 * 1000 # ms
        
        # 2. 协议/扫描时延 (核心差异)
        if mode == 'RANDOM_DIR':
            # 随机组：没有ASIC，需要扫描，且握手成功率低(重传)
            # 模拟：成功率0.5意味着平均发2次才能成功，或者需要等待扫描周期
            t_protocol = 50.0 # 假设平均扫描等待50ms
        elif mode == 'SMART':
            # 智能组：ASIC秒级锁定 + 具身智能免握手
            t_protocol = 0.1  # ASIC处理时间，极快
        else: # OMNI
            t_protocol = 5.0  # 传统全向握手
            
        total_latency += (t_prop + t_protocol)
        
    return total_latency / len(edges)

def calculate_valid_link_ratio(nodes, adjacency_matrix):
    """
    计算有效连接比例：(连接到好节点的边) / (总边数)
    用于证明具身智能过滤了坏节点。
    """
    rows, cols = np.where(adjacency_matrix == 1)
    total_links = len(rows)
    
    if total_links == 0:
        return 0.0
    
    valid_links = 0
    for i, j in zip(rows, cols):
        # 只有当两端都是正常节点时，才算“高质量有效连接”
        if nodes[i].status == cfg.STATUS_NORMAL and nodes[j].status == cfg.STATUS_NORMAL:
            valid_links += 1
            
    return valid_links / total_links