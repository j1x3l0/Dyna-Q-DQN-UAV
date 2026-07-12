# UAV辅助无线传感器网络深度强化学习项目

## 项目概述

本项目基于论文 **"Deep Reinforcement Learning for Joint Trajectory Planning, Transmission Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks"**（Sensors 2023, 23, 4691）实现，旨在通过多智能体深度强化学习方法，联合优化UAV轨迹规划、传输调度和接入控制策略，以最大化系统能效。

### 核心问题

在UAV辅助的无线功率传感器网络中，联合优化以下四个控制变量：
- **$z$**：地面用户（GU）传输模式选择（反向散射 vs. RF主动通信）
- **$X$**：GU接入控制策略
- **$L$**：UAV轨迹规划策略
- **$y$**：UAV传输调度策略

## 目录结构

```
Deep Reinforcement Learning for Joint Trajectory Planning/
├── src/                    # 核心源代码
│   ├── system_model.py     # 系统模型（UAV、GU、信道模型）
│   ├── maddpg_agent.py     # MADDPG算法实现
│   ├── hierarchical_agent.py # 分层学习框架（含Dyna-Q）
│   └── compare_results.py  # 结果对比工具
├── scripts/                # 运行脚本
│   ├── train_maddpg.py     # MADDPG训练脚本
│   ├── train_hierarchical.py # 分层学习训练脚本
│   ├── run_test_training.py # 测试训练与报告生成
│   ├── run_hierarchical_no_dyna.py # 不含Dyna-Q的分层训练
│   ├── run_dyna_validation.py # Dyna-Q验证实验
│   ├── analyze_convergence.py # 收敛分析脚本
│   ├── run_maddpg_small.py # 小规模测试
│   └── test_dyna_q.py      # Dyna-Q单元测试
├── results/                # 训练结果
│   ├── *.npy               # 奖励数据
│   ├── *.png               # 奖励曲线图表
│   └── *.txt               # 实验报告
├── logs/                   # 日志文件
│   ├── system_model.log    # 系统模型日志
│   ├── maddpg_agent.log    # MADDPG日志
│   ├── hierarchical_agent.log # 分层代理日志
│   └── training_*.log      # 训练脚本日志
└── reports/                # 文档报告
    ├── 项目报告.md         # 论文翻译与整理
    ├── sensors-23-04691.pdf # 原始论文
    └── dyna-q-implementation-plan/ # Dyna-Q实现方案
```

## 核心模块

### 1. 系统模型 [system_model.py](file:///d:/task/xchallenge/Deep%20Reinforcement%20Learning%20for%20Joint%20Trajectory%20Planning,%20Transmission%20Scheduling,%20and%20Access%20Control%20in%20UAV-Assisted%20Wireless%20Sensor%20Networks/src/system_model.py)

实现UAV辅助无线传感器网络的完整仿真环境：

- **Config类**：系统参数配置（UAV数量、GU数量、资源块数量等）
- **UAV类**：无人机状态管理、位置更新、碰撞检测
- **GU类**：地面用户状态（能量、缓冲区、信道增益）
- **Channel类**：Rician衰落信道模型
- **Environment类**：环境仿真核心，包含状态重置和步进函数

### 2. MADDPG算法 [maddpg_agent.py](file:///d:/task/xchallenge/Deep%20Reinforcement%20Learning%20for%20Joint%20Trajectory%20Planning,%20Transmission%20Scheduling,%20and%20Access%20Control%20in%20UAV-Assisted%20Wireless%20Sensor%20Networks/src/maddpg_agent.py)

多智能体深度确定性策略梯度算法实现：

- **Actor网络**：输出连续动作（轨迹方向、速度、调度决策）
- **Critic网络**：评估动作价值，考虑所有智能体的联合状态
- **经验回放**：存储和采样(s, a, r, s')元组
- **软更新**：目标网络平滑更新

### 3. 分层学习框架 [hierarchical_agent.py](file:///d:/task/xchallenge/Deep%20Reinforcement%20Learning%20for%20Joint%20Trajectory%20Planning,%20Transmission%20Scheduling,%20and%20Access%20Control%20in%20UAV-Assisted%20Wireless%20Sensor%20Networks/src/hierarchical_agent.py)

实现论文提出的双层强化学习框架：

#### 3.1 HierarchicalAgent（含Dyna-Q）
- **上层**：MADDPG（`UpperActor` + `UpperCritic`），负责UAV轨迹规划和资源调度
- **下层**：DQN（`LowerDQN`），负责GU模式选择和接入控制
- **Dyna-Q模块**：
  - `Model`网络：预测奖励和下一状态
  - `model_predict`：基于(s, a)预测(r_pred, s_pred)
  - `update_model`：更新模型网络
  - `dyna_plan`：从经验回放采样，用模型预测结果做DQN更新（梯度累积优化）

#### 3.2 HierarchicalNoDynaAgent（不含Dyna-Q）
- 结构与HierarchicalAgent相同
- 移除Model网络和Dyna-Q规划模块
- 用于验证Dyna-Q框架的加速作用

## 算法对比

| 算法 | 上层决策 | 下层决策 | 模型规划 | 核心特点 |
|------|---------|---------|---------|---------|
| MADDPG | 无（单层） | 无（单层） | 无 | 纯多智能体DDPG，集中式训练分布式执行 |
| Hierarchical (DQN) | MADDPG | DQN | 无 | 分层架构，下层纯经验回放 |
| Hierarchical (Dyna-Q) | MADDPG | DQN | Dyna-K=5 | 分层架构+模型预测，从虚拟经验学习 |

## 实验结果

### 500轮训练对比

| 指标 | MADDPG | Hierarchical (DQN) | Hierarchical (Dyna-Q) |
|------|--------|-------------------|----------------------|
| Final Avg Reward | -4238.19 | -1022.69 | -1022.69 |
| Best Reward | -0.18 | -0.67 | -0.67 |
| Improvement | - | 75.87% | 75.87% |

### 5000轮训练对比

| 指标 | MADDPG | Hierarchical (Dyna-Q) |
|------|--------|----------------------|
| Final Avg Reward | -2996.17 | -1107.98 |
| Best Reward | -1.20 | -0.28 |
| Improvement | - | 63.02% |

### 收敛分析

- **MADDPG**：训练过程中出现性能波动，标准差约2800，训练不稳定
- **Hierarchical (Dyna-Q)**：稳定收敛在-1000~-1200区间，标准差约1500，比MADDPG稳定1.87倍

## 运行方式

### 环境配置

```powershell
# 创建conda环境
conda create -n uav-drl python=3.9
conda activate uav-drl

# 安装依赖
pip install torch numpy matplotlib scipy
```

### 训练脚本

```powershell
# 运行MADDPG训练
python scripts/train_maddpg.py

# 运行分层学习训练（含Dyna-Q）
python scripts/train_hierarchical.py

# 运行分层学习训练（不含Dyna-Q）
python scripts/run_hierarchical_no_dyna.py

# 运行测试训练并生成报告
python scripts/run_test_training.py

# 运行Dyna-Q验证实验（三种算法对比）
python scripts/run_dyna_validation.py
```

### 结果分析

```powershell
# 收敛分析
python scripts/analyze_convergence.py
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| N（UAV数量） | 3 | 无人机数量 |
| M（GU数量） | 6 | 地面用户数量 |
| F（资源块数量） | 4 | 可用资源块数 |
| Learning Rate | 1e-4 | 学习率 |
| Gamma | 0.95 | 折扣因子 |
| Tau | 0.01 | 软更新系数 |
| Batch Size | 32 | 批处理大小 |
| Memory Size | 2000 | 经验回放容量 |
| Dyna-K | 5 | Dyna-Q规划步数 |

## 日志系统

- **日志级别**：DEBUG（文件）、WARNING（控制台）
- **日志格式**：`时间戳 - 模块名 - 级别 - 消息`
- **文件大小**：10MB，保留5个备份
- **核心日志文件**：
  - `logs/system_model.log`：系统模型日志
  - `logs/maddpg_agent.log`：MADDPG日志
  - `logs/hierarchical_agent.log`：分层代理日志

## 引用论文

```
罗晓玲, 陈澈, 曾春年, 等. Deep Reinforcement Learning for Joint Trajectory Planning, Transmission Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks[J]. Sensors, 2023, 23(10): 4691.
```

DOI: https://doi.org/10.3390/s23104691
