# 系统认知地图 — UAV-DRL 项目

> 维护者：@knowledge-map-maintainer | 更新频率：每轮开发后
> 本文档回答"系统长什么样"——模块、依赖、数据流、理解度。
> 每次开发前 Agent 应读取本文件建立上下文。

---

## 1. 模块职责矩阵

| 模块 | 文件 | 职责 | 输入 | 输出 | 理解度 | 最后理解 |
|------|------|------|------|------|--------|---------|
| 系统模型 | `src/system_model.py` | 环境仿真：UAV/GU/信道/能量/Reward；53 维局部状态 | Config 参数 | (state, reward, done) | 85% | 2026-08 |
| MADDPG | `src/maddpg_agent.py` | 多智能体 DDPG (Actor+Critic+Replay) | 联合状态 | 联合动作 | 85% | 2026-07 |
| 分层-DynaQ | `src/hierarchical_agent.py` | 双层 RL + Dyna-Q 规划；归一化网络；执行动作 replay | 全局状态 | (UAV动作, GU动作) | 80% | 2026-08 |
| 分层-NoDyna | `src/hierarchical_agent.py` | 双层 RL 无模型规划；归一化网络；执行动作 replay | 全局状态 | (UAV动作, GU动作) | 80% | 2026-08 |
| MATD3 | `src/matd3_agent.py` | Twin Critic + 延迟更新 | 联合状态 | 联合动作 | 60% | 2026-07 |
| IDDPG | `src/iddpg_agent.py` | 独立 DDPG (无 CTDE) | 局部状态 | 局部动作 | 50% | 2026-07 |
| CoP-MADDPG | `src/cop_maddpg_agent.py` | 通信预测 + MADDPG | 局部状态+消息 | 联合动作 | 50% | 2026-07 |
| 训练工具 | `scripts/training_utils.py` | 训练循环/日志/checkpoint | 算法实例 | 训练结果 | 70% | 2026-07 |
| 基准测试 | `scripts/run_full_benchmark.py` | 统一入口；独立无噪声评估与收敛统计 | 参数配置 | .npy + 报告 | 75% | 2026-08 |
| 结果分析 | `scripts/analyze_*.py` | 收敛/奖励/可视化分析 | .npy 文件 | 图表 + 报告 | 60% | 2026-07 |
| 奖励分解 | `scripts/reward_decomposition.py` | 按 reward 成分分析 | .npy 文件 | 分解报告 | 40% | 2026-07 |
| Dyna-Q 消融 | `scripts/run_dyna_ablation.py` | K / 模型 warm-up 参数化对照 | seeds + sweep 参数 | 逐轮指标 + JSON | 70% | 2026-07 |

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
  │     │     └─► 计算 Reward（由 reward_mode 选择）
  │     │
  │     ├─► [Dyna-Q 分支] Model.predict(s, a) → (r_pred, s_pred)
  │     │     └─► 对虚拟经验做 K 步 DQN 更新
  │     │
  │     └─► Replay Buffer 存储 (s, a_commanded, a_executed, r, s')
  │
  ├─► 每 10 个训练 Episode
  │     └─► 固定测试 seed 的独立 Environment + noise=False 策略评估
  │           └─► eval_rewards → 无噪声收敛回合统计
  │
  └─► Episode 结束 → 训练更新 (replay sample → actor/critic loss → backprop)
```

## 4. 雾图 (Fog Map)

标记当前理解最薄弱的模块（"雾区"）：

| 模块 | 雾度 | 不清晰的部分 |
|------|------|-------------|
| Reward 分解 | 🟡 35% 未知 | 已定位历史代理目标与论文 Ξ 的偏差；长期训练收益待验证 |
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
| **ee_ratio 奖励** | `(GU→UAV 接收量 + γ_forward × UAV→RBS 转发量) / UAV 能耗`；历史默认，保留旧实验兼容性 | `src/system_model.py` |
| **paper_xi 奖励** | `UAV→RBS 转发量 / UAV 能耗`；与论文 Eq.9 的系统目标直接对齐 | `src/system_model.py` |
| **additive 奖励** | 数据收益减能耗惩罚的加性代理目标，用于奖励分解消融 | `src/system_model.py` |
| **CTDE** | Centralized Training Decentralized Execution | MADDPG 论文 |
| **Dyna-Q** | 环境模型 + 虚拟经验 → 加速 Q-learning | Sutton & Barto |
| **Backscatter** | GU 反射 UAV 信号通信，不消耗自身能量 | 论文 §2.2 |
| **RF Active** | GU 使用自身能量主动发送信号 | 论文 §2.2 |
| **Rician K-factor** | LOS 功率 / 散射功率比 — UAV 信道特征 | 论文 Eq.1 |
| **Resource Block** | 频域资源分配的基本单位 | 论文 §2.1 |
| **Soft Update** | θ_target = τ·θ + (1-τ)·θ_target | DDPG 论文 |
| **硬安全投影** | Actor 动作执行前联合投影到地理围栏、速度和 `d_min` 约束；安全干预与通信奖励分开统计 | `Environment.prepare_step` |
| **执行动作 replay** | 环境保留命令动作与投影后的等价执行动作；Critic 学习执行动作，Actor 用二者残差学习减少安全干预 | `Environment.get_last_upper_actions` / `HierarchicalAgent.update_upper` |
| **状态归一化** | Upper Actor/Critic、Lower DQN、target 网络与世界模型共享物理量尺度；缩放 buffer 不进入 checkpoint | `src/hierarchical_agent.py` |
| **分动作噪声衰减** | 航向、速度、调度分数使用不同高斯噪声标准差，并在 150 回合内线性衰减 | `HierarchicalAgent.upper_act` |
| **无噪声评估** | 固定测试 seed 的独立环境每 10 个训练回合执行一次 `noise=False` 策略；报告收敛仅依据评估曲线 | `scripts/run_full_benchmark.py` |

### 2026-07 阶段 C 诊断

50 轮短测显示，Hier-DynaQ 在历史 `ee_ratio` 奖励下能够收集 GU 数据，但很少向 RBS
转发：平均转发/接收比约 1.5%，46/50 个 episode 为零转发，终局 UAV 缓冲区持续累积。
动作索引检查未发现错误。当前假设是历史奖励中的“接收量”项允许上层策略在不完成 RBS
交付时仍获得较高回报，因此新增 `paper_xi` 模式用于阶段 C 对照；在完成长期验证前，
`ee_ratio` 仍是默认值。

---

*最后更新: 2026-08-09 | 由 @knowledge-map-maintainer 维护*
