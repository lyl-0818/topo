"""
simulation.py
ASIC辅助通感一体化拓扑仿真平台 - 时间拓扑版本（可视化增强版）
---------------------------------------------------------
新增：
1. 时间拓扑稳定度（Topology Stability）
2. 最后时间步拓扑快照可视化
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import config as cfg
from node import Node
from radar_asic import RadarASIC
import visualization as viz
import metrics as met


def get_effective_connectivity(adjacency_matrix, nodes, survivor_indices):

    G = nx.from_numpy_array(adjacency_matrix)

    good_survivors = [
        i for i in survivor_indices
        if nodes[i].status == cfg.STATUS_NORMAL
    ]

    if len(good_survivors) == 0:
        return 0.0

    sub_graph = G.subgraph(good_survivors)

    if sub_graph.number_of_nodes() == 0:
        return 0.0

    largest_cc = max(nx.connected_components(sub_graph), key=len)

    return len(largest_cc) / len(good_survivors)


def run_simulation_full():

    res_conn = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_latency = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_quality = np.zeros((3, len(cfg.BREAK_RATIOS)))
    res_stability = np.zeros((3, len(cfg.BREAK_RATIOS)))

    print(f"开始全量仿真... (N={cfg.NODE_COUNT}, Rounds={cfg.MONTE_CARLO_ROUNDS})")
    print("---------------------------------------------------------------")

    for i_ratio, break_ratio in enumerate(cfg.BREAK_RATIOS):

        temp_conn = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_lat = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_qual = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))
        temp_stab = np.zeros((3, cfg.MONTE_CARLO_ROUNDS))

        for r in range(cfg.MONTE_CARLO_ROUNDS):

            # -------------------------
            # 初始化节点
            # -------------------------
            nodes = [Node(i) for i in range(cfg.NODE_COUNT)]

            # -------------------------
            # 预热
            # -------------------------
            for _ in range(15):
                for node in nodes:
                    node.move()

            # -------------------------
            # 对抗损毁
            # -------------------------
            num_destroy = int(cfg.NODE_COUNT * break_ratio)
            survivor_indices = np.random.choice(
                cfg.NODE_COUNT,
                cfg.NODE_COUNT - num_destroy,
                replace=False
            )

            # -------------------------
            # 三种算法
            # -------------------------
            asic_omni = RadarASIC(mode='OMNI')
            asic_rand = RadarASIC(mode='RANDOM_DIR')
            asic_smart = RadarASIC(mode='SMART')

            prev_adj_omni = None
            prev_adj_rand = None
            prev_adj_smart = None

            stab_omni = []
            stab_rand = []
            stab_smart = []

            snapshot_adj_omni = None
            snapshot_adj_rand = None
            snapshot_adj_smart = None
            snapshot_nodes = None

            adj_omni = None
            adj_rand = None
            adj_smart = None

            # ==================================================
            # 时间拓扑演化
            # ==================================================
            for t in range(cfg.TEMPORAL_STEPS):

                # 1. 所有节点同步运动
                for node in nodes:
                    node.move()

                # 2. 本时间步构建一次拓扑（非常重要：只做一次）
                adj_omni = asic_omni.run_topology_control(nodes)
                adj_rand = asic_rand.run_topology_control(nodes)
                adj_smart = asic_smart.run_topology_control(nodes)

                # 3. 稳定度
                stab_omni.append(
                    met.calculate_topology_stability(prev_adj_omni, adj_omni)
                )
                stab_rand.append(
                    met.calculate_topology_stability(prev_adj_rand, adj_rand)
                )
                stab_smart.append(
                    met.calculate_topology_stability(prev_adj_smart, adj_smart)
                )

                prev_adj_omni = adj_omni.copy()
                prev_adj_rand = adj_rand.copy()
                prev_adj_smart = adj_smart.copy()

                # 4. 保存最后一帧用于可视化
                if t == cfg.TEMPORAL_STEPS - 1:
                    snapshot_adj_omni = adj_omni.copy()
                    snapshot_adj_rand = adj_rand.copy()
                    snapshot_adj_smart = adj_smart.copy()
                    snapshot_nodes = nodes

            # ==================================================
            # 指标
            # ==================================================
            temp_conn[0, r] = get_effective_connectivity(
                adj_omni, nodes, survivor_indices
            )
            temp_conn[1, r] = get_effective_connectivity(
                adj_rand, nodes, survivor_indices
            )
            temp_conn[2, r] = get_effective_connectivity(
                adj_smart, nodes, survivor_indices
            )

            temp_lat[0, r] = met.calculate_latency(nodes, adj_omni, 'OMNI')
            temp_lat[1, r] = met.calculate_latency(nodes, adj_rand, 'RANDOM_DIR')
            temp_lat[2, r] = met.calculate_latency(nodes, adj_smart, 'SMART')

            temp_qual[0, r] = met.calculate_valid_link_ratio(nodes, adj_omni)
            temp_qual[1, r] = met.calculate_valid_link_ratio(nodes, adj_rand)
            temp_qual[2, r] = met.calculate_valid_link_ratio(nodes, adj_smart)

            temp_stab[0, r] = np.mean(stab_omni)
            temp_stab[1, r] = np.mean(stab_rand)
            temp_stab[2, r] = np.mean(stab_smart)

            # ==================================================
            # 只画一次最终时间步拓扑
            # ==================================================
            if r == 0 and i_ratio == 0:

                print("  [Snapshot] 生成最终时间步拓扑快照...")

                viz.draw_topology_snapshot(
                    snapshot_nodes,
                    snapshot_adj_rand,
                    "Baseline: Random Directional (Final t)",
                    "topo_random_final.png"
                )

                viz.draw_topology_snapshot(
                    snapshot_nodes,
                    snapshot_adj_smart,
                    "Ours: ASIC-SMART (Final t)",
                    "topo_smart_final.png"
                )

        res_conn[:, i_ratio] = np.mean(temp_conn, axis=1)
        res_latency[:, i_ratio] = np.mean(temp_lat, axis=1)
        res_quality[:, i_ratio] = np.mean(temp_qual, axis=1)
        res_stability[:, i_ratio] = np.mean(temp_stab, axis=1)

        print(f"进度: 损毁比例 {break_ratio*100:.0f}% 计算完成。")

    return res_conn, res_latency, res_quality, res_stability


def plot_all_metrics(res_conn, res_lat, res_qual, res_stab):

    x = cfg.BREAK_RATIOS
    models = ['Omni (Jammed)', 'Random Dir', 'ASIC-Smart (Ours)']
    colors = ['g', 'r', 'b']
    markers = ['s', 'o', '^']

    # Fig 1
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_conn[i], color=colors[i], marker=markers[i],
                 label=models[i], lw=2)
    plt.title("Fig 1. Effective Network Connectivity")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Effective Connectivity")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.savefig("result_connectivity.png")

    # Fig 2
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_lat[i], color=colors[i], marker=markers[i],
                 label=models[i], lw=2)
    plt.title("Fig 2. Average Link Setup Latency")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("result_latency.png")

    # Fig 3
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_qual[i], color=colors[i], marker=markers[i],
                 label=models[i], lw=2)
    plt.title("Fig 3. Valid Link Ratio")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Ratio of Healthy Links")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.savefig("result_quality.png")

    # Fig 4
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(x, res_stab[i], color=colors[i], marker=markers[i],
                 label=models[i], lw=2)
    plt.title("Fig 4. Topology Stability over Time")
    plt.xlabel("Node Destruction Ratio")
    plt.ylabel("Link Survival Ratio")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 1.1)
    plt.savefig("result_stability.png")

    print("\n✅ 所有图表生成完毕！")
    print("  - topo_random_final.png")
    print("  - topo_smart_final.png")


if __name__ == "__main__":
    r_conn, r_lat, r_qual, r_stab = run_simulation_full()
    plot_all_metrics(r_conn, r_lat, r_qual, r_stab)