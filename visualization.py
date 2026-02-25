"""
visualization.py
功能：绘制网络拓扑的瞬时快照，用于论文插图。
展示：节点位置(颜色区分状态)、雷达朝向(箭头)、通信链路(连线)。
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # 确保导入 Line2D
import numpy as np
import config as cfg

def draw_topology_snapshot(nodes, adjacency_matrix, title, filename=None):
    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.gca()
    
    # 1. 绘制链路 (Edges)
    rows, cols = np.where(adjacency_matrix == 1)
    for i, j in zip(rows, cols):
        if i < j: # 避免重复画
            p1 = nodes[i].pos
            p2 = nodes[j].pos
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', alpha=0.3, linewidth=0.5)

    # 2. 绘制节点 (Nodes)
    for n in nodes:
        # 根据状态选颜色
        if n.status == cfg.STATUS_NORMAL:
            color = 'blue'
            marker = 'o'
        else:
            color = 'red' # 故障/过载节点
            marker = 'x'
            
        plt.scatter(n.pos[0], n.pos[1], c=color, marker=marker, s=30, zorder=5)
        
        # 3. 绘制雷达朝向 (Heading Arrows)
        # 箭头长度
        arrow_len = 2000 
        plt.arrow(n.pos[0], n.pos[1], 
                  n.heading[0] * arrow_len, n.heading[1] * arrow_len,
                  head_width=500, head_length=800, fc='gray', ec='gray', alpha=0.5)

    plt.title(title)
    plt.xlim(0, cfg.AREA_WIDTH)
    plt.ylim(0, cfg.AREA_HEIGHT)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    
    # --- 【修复部分】创建图例 ---
    # 使用 marker='>' 来代替 arrowprops，避免报错
    custom_lines = [
        Line2D([0], [0], color='green', lw=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='x', color='red', markersize=8),
        Line2D([0], [0], color='gray', lw=1, marker='>') # 修复：用简单的三角箭头符号
    ]
    ax.legend(custom_lines, ['Link', 'Normal Node', 'Faulty/Overload', 'Radar Beam'])

    if filename:
        plt.savefig(filename)
    # plt.show() # 如果你想在后台跑，可以注释掉这行；如果想看图，保留它