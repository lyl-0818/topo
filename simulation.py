"""
simulation.py
ASIC辅助通感一体化拓扑仿真平台 - 主程序 (最终修复版)
---------------------------------------------------------
更新日志：
1. [Fix] 添加预热阶段 (Warm-up Phase)，解决 AI 冷启动导致的误判问题。
2. [Fix] 修正快照生成顺序，确保随机组和智能组的箭头朝向展示正确。
3. [Feature] 采用'有效连通度'指标，只统计正常节点，体现具身智能的过滤价值。
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
    计算有效连通度 (Effective Connectivity)
    逻辑：只统计【存活】且【状态正常】的节点之间的连通性。
    这样可以剔除故障/过载节点带来的虚假连通度。
    """
    G = nx.from_numpy_array(adjacency_matrix)
    
    # 筛选出存活且状态正常的节点索引
    good_survivors = [i for i in survivor_indices if nodes[i].status == cfg.STATUS_NORMAL]
    
    # 如果没有好节点存活，连通度为0
    if len(good_survivors) == 0: return 0.0
    
    # 提取子图
    sub_graph = G.subgraph(good_survivors)
    if sub_graph.number_of_nodes() == 0: return 0.0
    
    # 计算最大连通分量
    largest_cc = max(nx.connected_components(sub_graph), key=len)
    
    # 连通度 = 最大连通子图节点数 / 有效存活节点总数
    return len(largest_cc) / len(good_survivors)

def run_simulation_full():
    """
    运行全量仿真：包含预热、蒙特卡洛循环、可视化快照、多维指标统计
    """
    # 准备存储结果 [算法数, 损毁比例数]
    # 0: OMNI, 1: RANDOM, 2: SMART
    res_conn = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_latency = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_quality = np.zeros((3, len(cfg.BREAK_RATIOS)))
    
    print(f"开始全量仿真... (N={cfg.NODE_COUNT}, Rounds={cfg.MONTE_CARLO_ROUNDS})")
    print("---------------------------------------------------------------")
    
    # --- 循环 1: 遍历损毁比例 ---
    for i_ratio, break_ratio in enumerate(cfg.BREAK_RATIOS):
        
        # 临时存储每次蒙特卡洛的结果
        temp_conn = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_lat = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_qual = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        
        # --- 循环 2: 蒙特卡洛多次平均 ---
        for r in range(cfg.MONTE_CARLO_ROUNDS):
            # 1. 初始化节点
            nodes = [Node(i) for i in range(cfg.NODE_COUNT)]
            
            # --- 【关键修复】预热阶段 (Warm-up) ---
            # 让节点先跑 15 步，积累历史数据，让 AI 能够识别出"乱动"的故障节点
            # 如果不加这一步，AI 会因为数据不足默认所有节点都是正常的
            for _ in range(15):
                for node in nodes:
                    node.move()
            
            # 2. 模拟对抗：随机摧毁
            num_destroy = int(cfg.NODE_COUNT * break_ratio)
            survivor_indices = np.random.choice(cfg.NODE_COUNT, 
                                              cfg.NODE_COUNT - num_destroy, 
                                              replace=False)
            
            # ------------------------------------------
            # 方案 A: 全向 (Baseline 1)
            # ------------------------------------------
            asic_omni = RadarASIC(mode='OMNI')
            adj_omni = asic_omni.run_topology_control(nodes)
            
            # ------------------------------------------
            # 方案 B: 随机定向 (Baseline 2)
            # ------------------------------------------
            # 注意：我们需要先跑随机，趁热画图，然后再跑智能
            # 否则节点的 heading 属性会被智能算法覆盖
            asic_rand = RadarASIC(mode='RANDOM_DIR')
            adj_rand = asic_rand.run_topology_control(nodes)
            
            # 【可视化】仅在第一轮且无损毁时，保存随机组的快照
            if r == 0 and i_ratio == 0:
                print("  [Snapshot] 正在生成随机拓扑快照 (topo_random.png)...")
                viz.draw_topology_snapshot(nodes, adj_rand, 
                                         "Baseline: Random Directional (No ASIC)", 
                                         "topo_random.png")
            
            # ------------------------------------------
            # 方案 C: 智能定向 (Ours)
            # ------------------------------------------
            asic_smart = RadarASIC(mode='SMART')
            # 这一步会修改 nodes 里的 heading (指向邻居/中心)
            adj_smart = asic_smart.run_topology_control(nodes)
            
            # 【可视化】保存智能组的快照 (此时箭头是有序的)
            if r == 0 and i_ratio == 0:
                print("  [Snapshot] 正在生成智能拓扑快照 (topo_smart.png)...")
                viz.draw_topology_snapshot(nodes, adj_smart, 
                                         "Ours: ASIC-Smart Topology (AI Filtered)", 
                                         "topo_smart.png")
            
            # ------------------------------------------
            # 指标计算
            # ------------------------------------------
            
            # 1. 有效连通度 (Effective Connectivity)
            temp_conn[0, r] = get_effective_connectivity(adj_omni, nodes, survivor_indices)
            temp_conn[1, r] = get_effective_connectivity(adj_rand, nodes, survivor_indices)
            temp_conn[2, r] = get_effective_connectivity(adj_smart, nodes, survivor_indices)
            
            # 2. 平均时延 (Latency)
            temp_lat[0, r] = met.calculate_latency(nodes, adj_omni, 'OMNI')
            temp_lat[1, r] = met.calculate_latency(nodes, adj_rand, 'RANDOM_DIR')
            temp_lat[2, r] = met.calculate_latency(nodes, adj_smart, 'SMART')
            
            # 3. 连接质量 (Valid Link Ratio)
            temp_qual[0, r] = met.calculate_valid_link_ratio(nodes, adj_omni)
            temp_qual[1, r] = met.calculate_valid_link_ratio(nodes, adj_rand)
            temp_qual[2, r] = met.calculate_valid_link_ratio(nodes, adj_smart)
            
        # 取平均并存储
        res_conn[:, i_ratio] = np.mean(temp_conn, axis=1)
        res_latency[:, i_ratio] = np.mean(temp_lat, axis=1)
        res_quality[:, i_ratio] = np.mean(temp_qual, axis=1)
        
        print(f"进度: 损毁比例 {break_ratio*100:.0f}% 计算完成。")

    return res_conn, res_latency, res_quality

def plot_all_metrics(res_conn, res_lat, res_qual):
    """
    绘制三张核心论文图表
    """
    x = cfg.BREAK_RATIOS
    models = ['Omni (Jammed)', 'Random Dir', 'ASIC-Smart (Ours)']
    colors = ['g', 'r', 'b']
    markers = ['s', 'o', '^']
    
    # --- 图 1: 有效连通度 ---
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_conn[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 1. Effective Network Connectivity (Robustness)")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Effective Connectivity")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.savefig("result_connectivity.png")
    
    # --- 图 2: 建链时延 ---
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_lat[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 2. Average Link Setup Latency (Real-time)")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("result_latency.png")
    
    # --- 图 3: 连接质量 (AI 过滤效果) ---
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_qual[i], color=colors[i], marker=markers[i], label=models[i], lw=2)
    plt.title("Fig 3. Valid Link Ratio (Embodied AI Filtering)")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Ratio of Healthy Links")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.savefig("result_quality.png")
    
    print("\n✅ 所有图表生成完毕！")
    print("1. result_connectivity.png (看看蓝线是不是反超了红线)")
    print("2. result_latency.png (看看时延是不是极低)")
    print("3. result_quality.png (看看是否过滤了坏节点)")
    print("4. topo_smart.png (看看红叉是不是孤立了)")

if __name__ == "__main__":
    r_conn, r_lat, r_qual = run_simulation_full()
    plot_all_metrics(r_conn, r_lat, r_qual)