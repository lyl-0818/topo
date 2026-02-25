My_Topology_Sim/
│
├── config.py           # 【配置文件】存放所有参数 (N, R, FoV, 区域大小等)
├── node.py             # 【节点类】定义节点的物理属性和运动逻辑
├── radar_asic.py       # 【核心算法】智能朝向、状态推理、连接判定
├── metrics.py          # 【工具库】计算连通度、时延、图论指标
├── simulation.py       # 【主程序】运行单次或蒙特卡洛仿真
└── visualization.py    # 【画图】实时显示拓扑动画、绘制结果曲线