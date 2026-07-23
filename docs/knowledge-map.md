# 系统认知地图 — UAV-DRL 项目

> 维护者：@knowledge-map-maintainer | 更新频率：每轮开发后
> 本文档回答"系统长什么样"——模块、依赖、数据流、理解度。
> 每次开发前 Agent 应读取本文件建立上下文。

---

## 1. 模块职责矩阵

| 模块 | 文件 | 职责 | 输入 | 输出 | 理解度 | 最后理解 |
|------|------|------|------|------|--------|---------|
| 系统模型 | `src/system_model.py` | 环境仿真：UAV/GU/信道/能量/Reward | Config 参数 | (state, reward, done) | 80% | 2026-07 |
| MADDPG | `src/maddpg_agent.py` | 多智能体 DDPG (Actor+Critic+Replay) | 联合状态 | 联合动作 | 85% | 2026-07 |
| 分层-DynaQ | `src/hierarchical_agent.py` | 双层 RL + Dyna-Q 规划 | 全局状态 | (UAV动作, GU动作) | 75% | 2026-07 |
| 分层-NoDyna | `src/hierarchical_agent.py` | 双层 RL 无模型规划 | 全局状态 | (UAV动作, GU动作) | 75% | 2026-07 |
| MATD3 | `src/matd3_agent.py` | Twin Critic + 延迟更新 | 联合状态 | 联合动作 | 60% | 2026-07 |
| IDDPG | `src/iddpg_agent.py` | 独立 DDPG (无 CTDE) | 局部状态 | 局部动作 | 50% | 2026-07 |
| CoP-MADDPG | `src/cop_maddpg_agent.py` | 通信预测 + MADDPG | 局部状态+消息 | 联合动作 | 50% | 2026-07 |
| 训练工具 | `scripts/training_utils.py` | 训练循环/日志/checkpoint | 算法实例 | 训练结果 | 70% | 2026-07 |
| 基准测试 | `scripts/run_full_benchmark.py` | 统一入口运行所有算法 | 参数配置 | .npy + 报告 | 65% | 2026-07 |
| 结果分析 | `scripts/analyze_*.py` | 收敛/奖励/可视化分析 | .npy 文件 | 图表 + 报告 | 60% | 2026-07 |
| 奖励分解 | `scripts/reward_decomposition.py` | 按 reward 成分分析 | .npy 文件 | 分解报告 | 40% | 2026-07 |

## 2. 依赖图

```
system_model.py ◄─────────────────────────────┐
       │                                        │
       ├── maddpg_agent.py ─────────────────────┤
       │       │                                 │
       ├── hierarchical_agent.py ───────────────┤
       │       │ (含 Dyna-Q / NoDyna 两分支)     │
       ├── matd3_agent.py ──────────────────────┤
       ├── iddpg_agent.py ──────────────────────┤
       └── cop_maddpg_agent.py ────────────────┘
               │
               ▼
       scripts/training_utils.py
               │
               ▼
       scripts/run_full_benchmark.py
               │
               ▼
       results/*.npy → scripts/analyze_*.py → results/*.png + reports/*.md
```

**关键依赖关系**：
- 所有 Agent 依赖 `system_model.Environment` 作为训练环境
- `training_utils.py` 被所有训练脚本调用（训练循环抽象）
- 结果分析脚本之间独立，可并行运行
- 不存在循环依赖 ✅

## 3. 核心数据流

```
Episode 开始
  │
  ├─► Environment.reset()
  │     └─► 初始化 UAV 位置、GU 能量/缓冲区、信道
  │
  ├─► for t in 0..T:
  │     │
  │     ├─► 上层 Agent (MADDPG).act(state)
  │     │     └─► UAV 轨迹动作: (Δx, Δy, speed, scheduling_decision)
  │     │     └─► Environment.step(): UAV 移动到新位置
  │     │
  │     ├─► 下层 Agent (DQN).act(gu_states)
  │     │     └─► GU 模式动作: (mode_backscatter_or_RF, access_decision)
  │     │
  │     ├─► Environment.step(): 
  │     │     └─► 计算信道增益 (Rician)
  │     │     └─► 计算吞吐量 (Shannon)
  │     │     └─► 更新 GU 能量/缓冲区
  │     │     └─► 计算 Reward (能效 Ξ)
  │     │
  │     ├─► [Dyna-Q 分支] Model.predict(s, a) → (r_pred, s_pred)
  │     │     └─► 对虚拟经验做 K 步 DQN 更新
  │     │
  │     └─► Replay Buffer 存储 (s, a, r, s')
  │
  └─► Episode 结束 → 训练更新 (replay sample → actor/critic loss → backprop)
```

## 4. 雾图 (Fog Map)

标记当前理解最薄弱的模块（"雾区"）：

| 模块 | 雾度 | 不清晰的部分 |
|------|------|-------------|
| Reward 分解 | 🔴 60% 未知 | 各 reward 成分对策略的实际贡献权重未知 |
| CoP-MADDPG 通信 | 🔴 50% 未知 | 通信消息对 critic 的影响路径未分析 |
| MATD3 双 Critic | 🟡 40% 未知 | Twin Q 对本项目的实际改善幅度待验证 |
| Dyna-Q 模型误差 | 🟡 35% 未知 | 长期训练中模型预测误差是否累积 |
| 能量采集边界 | 🟢 20% 未知 | 线性模型假设的合理性已验证 |

## 5. 认知快照表

| 角色 | 最熟悉模块 | 理解度 | 上次深入 |
|------|-----------|--------|---------|
| 开发者 | system_model.py | 高 | 2026-07 |
| 开发者 | hierarchical_agent.py | 中高 | 2026-07 |
| 开发者 | maddpg_agent.py | 中 | 2026-07 |
| AI Agent 集群 | 全部 | 基于代码分析 | 2026-07-23 |

## 6. 关键概念速查表

| 概念 | 定义 | 首次出现 |
|------|------|---------|
| **能效 Ξ** | RBS 总吞吐量 / UAV 总能耗的时间平均 | 论文 Eq.9 |
| **CTDE** | Centralized Training Decentralized Execution | MADDPG 论文 |
| **Dyna-Q** | 环境模型 + 虚拟经验 → 加速 Q-learning | Sutton & Barto |
| **Backscatter** | GU 反射 UAV 信号通信，不消耗自身能量 | 论文 §2.2 |
| **RF Active** | GU 使用自身能量主动发送信号 | 论文 §2.2 |
| **Rician K-factor** | LOS 功率 / 散射功率比 — UAV 信道特征 | 论文 Eq.1 |
| **Resource Block** | 频域资源分配的基本单位 | 论文 §2.1 |
| **Soft Update** | θ_target = τ·θ + (1-τ)·θ_target | DDPG 论文 |

---

*最后更新: 2026-07-23 | 由 @knowledge-map-maintainer 维护*
