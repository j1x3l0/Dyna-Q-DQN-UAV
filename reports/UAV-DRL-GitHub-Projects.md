# UAV辅助无线传感器网络中深度强化学习相关GitHub项目整理

> 主题：Deep Reinforcement Learning for Joint Trajectory Planning, Transmission Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks
>
> 整理日期：2026-07-10

---

## 说明

本文档整理了与 **"无人机辅助无线传感器网络中联合轨迹规划、传输调度和访问控制的深度强化学习"** 主题相关的GitHub开源项目。这些项目涵盖了UAV轨迹规划、深度强化学习算法实现、传输调度优化、无线传感器网络仿真等核心技术方向，可作为研究参考或代码复现的基础。

---

## 高相关度项目

### 1. UAV-DDPG

| 属性 | 内容 |
|------|------|
| **项目链接** | https://github.com/fangvv/UAV-DDPG |
| **核心内容** | 无人机辅助移动边缘计算（MEC）中的计算卸载优化 |
| **主要算法** | DDPG、DQN、Actor-Critic |
| **编程语言** | Python（TensorFlow 1.X） |
| **关联论文** | Computation Offloading Optimization for UAV-assisted Mobile Edge Computing: A Deep Deterministic Policy Gradient Approach (Wireless Networks, 2021) |

**技术特点**：
- 联合优化用户调度、任务卸载比例、无人机飞行角度与飞行速度
- 目标：最小化最大处理延迟
- 针对非凸问题、高维状态空间和连续动作空间的优化
- 实验表明DDPG算法相比DQN等基线算法在处理延迟方面有显著提升

**与目标论文的关联性**：该项目与目标论文的技术框架高度接近，均涉及UAV辅助网络中的联合优化问题，使用DRL解决连续动作空间中的决策问题，是最具参考价值的项目之一。

**项目结构**：
```
UAV-DDPG/
├── Actor-Critic/
├── DDPG/
├── DQN/
├── Edge_only/
├── Local_only/
└── README.md
```

---

### 2. UAV-RIS-EnergyHarvesting

| 属性 | 内容 |
|------|------|
| **项目链接** | https://github.com/Haoran-Peng/UAV-RIS_EnergyHarvesting |
| **核心内容** | 基于鲁棒深度强化学习的无人机能量收集可重构智能表面 |
| **主要算法** | DDPG、TD3 |
| **编程语言** | Python |
| **关联论文** | Energy Harvesting Reconfigurable Intelligent Surface for UAV Based on Robust Deep Reinforcement Learning (IEEE Trans. Wireless Commun., vol.22, no.10) |

**技术特点**：
- UAV轨迹设计（density-aware和Fermat point-based算法）
- 优化RIS调度矩阵、能量收集时长和发射功率
- 使用鲁棒深度强化学习方法应对环境不确定性
- 实现了TD3-SingleUT等多种变体

**与目标论文的关联性**：该项目包含UAV轨迹设计和深度强化学习的结合实现，与目标论文的轨迹规划部分直接相关，可作为轨迹优化模块的参考。

---

### 3. VN-MADDPG

| 属性 | 内容 |
|------|------|
| **项目链接** | https://github.com/fangvv/VN-MADDPG |
| **核心内容** | 基于多智能体深度强化学习的车联网通信资源分配优化 |
| **主要算法** | MADDPG |
| **编程语言** | Python |
| **关联论文** | 基于多智能体深度强化学习的车联网通信资源分配优化 |

**技术特点**：
- 多智能体深度确定性策略梯度算法实现
- 适用于多UAV/多车辆协作场景
- 解决通信资源分配优化问题
- 与OpenAI Multi-Agent Particle Environment兼容

**与目标论文的关联性**：如果研究涉及多UAV协作场景下的联合决策，该项目的MADDPG实现具有重要参考价值。

---

## 路径规划基础项目

### 4. PathPlanning

| 属性 | 内容 |
|------|------|
| **项目链接** | https://github.com/zhm-real/PathPlanning |
| **核心内容** | 机器人技术中常见的路径规划算法集合 |
| **主要算法** | A*、Dijkstra、RRT、RRT*、APF、PSO等 |
| **编程语言** | Python |
| **Stars** | 4900+ |

**技术特点**：
- 包含基于搜索的算法和基于采样的算法
- 提供Python实现代码、运行过程动画
- 附带相关论文链接
- 可作为传统路径规划算法的基准对比

**与目标论文的关联性**：该项目提供了传统路径规划算法的完整实现，可作为DRL-based轨迹规划方法的对比基准，也可用于构建仿真环境中的基础路径模块。

---

## 其他相关项目

### 5. intelligent-uavpath-planning

| 属性 | 内容 |
|------|------|
| **项目描述** | 智能无人机路径规划仿真系统 |
| **核心功能** | UAV路径规划仿真与可视化 |
| **技术特点** | 集成多种路径规划算法的仿真平台 |

---

### 6. UAV Obstacle Avoidance Controller

| 属性 | 内容 |
|------|------|
| **项目链接** | https://github.com/abhiksingla/uav_obstacle_avoidance_controller |
| **核心内容** | 使用深度循环强化学习的无人机避障 |
| **主要算法** | Deep Recurrent Reinforcement Learning with Temporal Attention |
| **编程语言** | Python |

**技术特点**：
- 结合时序注意力的深度循环强化学习
- 适用于动态障碍物环境下的UAV控制
- 可作为UAV动态环境决策的参考

---

## 推荐学习路径

### 如果复现目标论文

1. **基础框架**：以 [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) 为代码基础
   - 其联合优化框架与目标论文最接近
   - 已实现DDPG/DQN等核心算法
   - 代码结构清晰，易于扩展

2. **轨迹设计参考**：参考 [UAV-RIS-EnergyHarvesting](https://github.com/Haoran-Peng/UAV-RIS_EnergyHarvesting)
   - 了解DDPG/TD3在UAV轨迹优化中的具体实现
   - 学习轨迹设计的奖励函数构造方法

3. **多UAV扩展**：如需多无人机场景，参考 [VN-MADDPG](https://github.com/fangvv/VN-MADDPG)
   - MADDPG算法适用于多智能体协作
   - 通信资源分配优化具有借鉴意义

4. **算法对比**：使用 [PathPlanning](https://github.com/zhm-real/PathPlanning) 中的传统算法作为对比基准

---

## 关键技术要点总结

| 技术方向 | 相关项目 | 核心算法 |
|---------|---------|---------|
| UAV轨迹规划 | UAV-DDPG, UAV-RIS-EnergyHarvesting | DDPG, TD3 |
| 传输调度/资源分配 | UAV-DDPG, VN-MADDPG | DDPG, MADDPG |
| 访问控制/用户调度 | UAV-DDPG | DDPG, DQN |
| 多智能体协作 | VN-MADDPG | MADDPG |
| 传统路径规划 | PathPlanning | A*, RRT, etc. |
| 避障/动态环境 | uav_obstacle_avoidance_controller | DRRL |

---

## 注意事项

1. **环境依赖**：部分项目使用TensorFlow 1.X，在新环境中可能需要调整或降级TensorFlow版本
2. **代码维护**：部分开源代码由研究生作者在学业期间完成，毕业后可能不再维护，仅供参考使用
3. **论文对应**：当前未发现与目标论文完全匹配的开源代码实现，建议以相关项目为基础进行二次开发
4. **仿真环境**：大多数项目需要自行构建或适配UAV-WSN仿真环境

---

*本文档基于公开网络资源整理，项目链接和信息可能随时间变化，请以GitHub页面实际内容为准。*
