"""
simulation.py
ASIC辅助通感一体化拓扑仿真平台 - 主程序 (完整版)
---------------------------------------------------------
更新内容：
1. 集成了 visualization.py，生成拓扑快照图。
2. 集成了 metrics.py，计算时延和连接质量。
3. 输出了三张核心对比图：连通度、时延、有效连接比。
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 导入所有模块
import config as cfg
from node import Node
from radar_asic import RadarASIC
import visualization as viz  
import metrics as met        

def get_effective_connectivity(adjacency_matrix, nodes, survivor_indices):
    """
    计算有效连通度 (只统计正常存活节点)
    """
    G = nx.from_numpy_array(adjacency_matrix)
    # 筛选出存活且状态正常的节点索引
    good_survivors = [i for i in survivor_indices if nodes[i].status == cfg.STATUS_NORMAL]
    
    if len(good_survivors) == 0: return 0.0
    
    sub_graph = G.subgraph(good_survivors)
    if sub_graph.number_of_nodes() == 0: return 0.0
    
    largest_cc = max(nx.connected_components(sub_graph), key=len)
    return len(largest_cc) / len(good_survivors)

def run_simulation_full():
    """
    运行全量仿真：包含蒙特卡洛循环、可视化快照、多维指标统计
    """
    # 准备存储结果 [算法数, 损毁比例数]
    # 0: OMNI, 1: RANDOM, 2: SMART
    res_conn = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_latency = np.zeros((3, len(cfg.BREAK_RATIOS))) # 时延
    res_quality = np.zeros((3, len(cfg.BREAK_RATIOS))) # 质量
    
    print(f"开始全量仿真... (N={cfg.NODE_COUNT}, Rounds={cfg.MONTE_CARLO_ROUNDS})")
    
    # --- 循环 1: 遍历损毁比例 ---
    for i_ratio, break_ratio in enumerate(cfg.BREAK_RATIOS):
        
        # 临时存储每次蒙特卡洛的结果
        temp_conn = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_lat = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_qual = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        
        # --- 循环 2: 蒙特卡洛多次平均 ---
        for r in range(cfg.MONTE_CARLO_ROUNDS):
            # 1. 初始化节点 (重置状态)
            nodes = [Node(i) for i in range(cfg.NODE_COUNT)]
            
            # 模拟对抗
            num_destroy = int(cfg.NODE_COUNT * break_ratio)
            survivor_indices = np.random.choice(cfg.NODE_COUNT, 
                                              cfg.NODE_COUNT - num_destroy, 
                                              replace=False)
            
            # ==========================================
            # 1. 运行全向 (Baseline 1)
            # ==========================================
            asic_omni = RadarASIC(mode='OMNI')
            adj_omni = asic_omni.run_topology_control(nodes)
            
            # ==========================================
            # 2. 运行随机定向 (Baseline 2)
            # ==========================================
            # 为了防止朝向被后续步骤覆盖，我们需要在这里【先保存状态】或者【立刻画图】
            
            # 重置节点朝向为随机 (因为node初始化是速度方向，这里确保它是乱的)
            # 注意：这一步其实在 run_topology_control 内部会做，但为了保险起见，
            # 我们依靠 RadarASIC 内部的 _update_headings 来随机化
            
            asic_rand = RadarASIC(mode='RANDOM_DIR')
            adj_rand = asic_rand.run_topology_control(nodes)
            
            # 【关键修改】趁热画图！此时 nodes 的 heading 是随机的
            if r == 0 and i_ratio == 0:
                print("  正在生成随机拓扑快照...")
                viz.draw_topology_snapshot(nodes, adj_rand, 
                                         "Baseline: Random Directional (No ASIC)", 
                                         "topo_random.png")

            # 计算随机组指标
            temp_conn[1, r] = get_effective_connectivity(adj_rand, nodes, survivor_indices)
            temp_lat[1, r] = met.calculate_latency(nodes, adj_rand, 'RANDOM_DIR')
            temp_qual[1, r] = met.calculate_valid_link_ratio(nodes, adj_rand)

            # ==========================================
            # 3. 运行智能定向 (Ours)
            # ==========================================
            # 此时 nodes 会被传入智能算法，heading 会被修改为“指向邻居/中心”
            
            asic_smart = RadarASIC(mode='SMART')
            adj_smart = asic_smart.run_topology_control(nodes)
            
            # 【关键修改】趁热画图！此时 nodes 的 heading 是智能的
            if r == 0 and i_ratio == 0:
                print("  正在生成智能拓扑快照...")
                viz.draw_topology_snapshot(nodes, adj_smart, 
                                         "Ours: ASIC-Smart Topology (Density+Center)", 
                                         "topo_smart.png")
            
            # 计算智能组指标
            temp_conn[2, r] = get_effective_connectivity(adj_smart, nodes, survivor_indices)
            temp_lat[2, r] = met.calculate_latency(nodes, adj_smart, 'SMART')
            temp_qual[2, r] = met.calculate_valid_link_ratio(nodes, adj_smart)

            # 全向的指标最后算也没关系，因为它不依赖 heading
            temp_conn[0, r] = get_effective_connectivity(adj_omni, nodes, survivor_indices)
            temp_lat[0, r] = met.calculate_latency(nodes, adj_omni, 'OMNI')
            temp_qual[0, r] = met.calculate_valid_link_ratio(nodes, adj_omni)
            
        # 取平均并存储
        res_conn[:, i_ratio] = np.mean(temp_conn, axis=1)
        res_latency[:, i_ratio] = np.mean(temp_lat, axis=1)
        res_quality[:, i_ratio] = np.mean(temp_qual, axis=1)
        
        print(f"进度: 损毁 {break_ratio*100:.0f}% 完成。")

    return res_conn, res_latency, res_quality

def plot_all_metrics(res_conn, res_lat, res_qual):
    """
    绘制三张核心图表
    """
    x = cfg.BREAK_RATIOS
    models = ['Omni (Jammed)', 'Random Dir', 'ASIC-Smart (Ours)']
    colors = ['g', 'r', 'b']
    markers = ['s', 'o', '^']
    
    # 1. 连通度图
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_conn[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 1. Effective Network Connectivity")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Connectivity")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("result_connectivity.png")
    
    # 2. 时延图 (你的ASIC优势)
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_lat[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 2. Average Link Setup Latency")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("result_latency.png")
    
    # 3. 连接质量图 (你的具身智能优势)
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_qual[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 3. Valid Link Ratio (Embodied AI Filtering)")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Ratio of Healthy Links")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("result_quality.png")
    
    print("\n所有图表已生成：result_connectivity.png, result_latency.png, result_quality.png")
    print("拓扑快照已生成：topo_random.png, topo_smart.png")

if __name__ == "__main__":
    r_conn, r_lat, r_qual = run_simulation_full()
    plot_all_metrics(r_conn, r_lat, r_qual)