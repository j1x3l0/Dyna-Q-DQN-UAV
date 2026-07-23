# CLAUDE.md — UAV-DRL 项目心智模型

> AI 协作编程入口文件。所有 Agent 工作前必读本文档与 `docs/knowledge-map.md`。
> 基于 jxl_better_vibe_coding 认知债务防御体系 (V2)。

---

## 1. 项目一句话定位

复现并改进论文 **"Deep Reinforcement Learning for Joint Trajectory Planning, Transmission Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks"** (Sensors 2023)，实现 MADDPG + 分层 DRL (Dyna-Q) 算法，联合优化 UAV 轨迹规划、传输调度和接入控制。

## 2. 核心心智模型

```
┌──────────────────────────────────────────────────────────────────┐
│                        UAV-Assisted WSN                          │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                      │
│  │  UAV 1  │    │  UAV 2  │    │  UAV 3  │   ← 轨迹+调度决策    │
│  └────┬────┘    └────┬────┘    └────┬────┘                      │
│       │              │              │                            │
│       ▼              ▼              ▼                            │
│  ┌─────────────────────────────────────────┐                     │
│  │         GU 1 .. GU 6 (地面用户)          │  ← 模式+接入决策   │
│  └─────────────────────────────────────────┘                     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐                                                     │
│  │   RBS   │  ← 总吞吐量 / UAV 能耗 = 系统能效 Ξ                 │
│  └─────────┘                                                     │
│                                                                  │
│  双层决策架构：                                                   │
│  上层 (MADDPG): UAV 轨迹 (连续) + 调度 y_i(t)                     │
│  下层 (DQN):   GU 模式 z_m(t) + 接入 x_{m,i}(t)                  │
│  Dyna-Q:       环境模型 → 虚拟经验 → 加速下层 DQN 收敛            │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计理念**：
- 分层不是架构偏好，而是**解耦两个不同时间尺度的决策**：UAV 轨迹（慢）vs GU 接入（快）
- Dyna-Q 不是"锦上添花"——在 GU 状态转移有明确物理模型的前提下，模型预测可以显著减少真实交互次数
- 三个算法（MADDPG / Hier-DQN / Hier-DynaQ）是**递进关系**，不是并列关系——每个后续算法解决前一个的特定缺陷

## 3. 目录结构

```
project/
├── src/                          # 核心源代码
│   ├── system_model.py           # 环境仿真（UAV, GU, Channel, Environment）
│   ├── maddpg_agent.py           # MADDPG 算法
│   ├── hierarchical_agent.py     # 分层学习 + Dyna-Q
│   ├── matd3_agent.py            # MATD3 算法
│   ├── iddpg_agent.py            # IDDPG 算法
│   └── cop_maddpg_agent.py       # CoP-MADDPG 算法
├── scripts/                      # 运行与分析脚本
│   ├── train_maddpg.py           # MADDPG 训练
│   ├── train_hierarchical.py     # 分层训练
│   ├── run_full_benchmark.py     # 全量基准测试
│   ├── analyze_convergence.py    # 收敛分析
│   ├── plot_*.py                 # 可视化脚本
│   └── reward_decomposition.py   # 奖励分解分析
├── results/                      # 实验输出（.npy, .png, .txt）
├── logs/                         # 训练日志
├── reports/                      # 文档报告
├── docs/                         # 项目文档
│   ├── knowledge-map.md          # 系统认知地图
│   └── lessons-learned.md        # 教训知识库
├── .claude/agents/               # 智能体定义（12个）
├── CLAUDE.md                     # 本文件
└── memory/                       # 持久化记忆（GPU连接等）
```

## 4. 关键设计决策及其 WHY

| 决策 | 选择 | 为什么不选其他 |
|------|------|----------------|
| 上层算法 | MADDPG (CTDE) | 单智能体 DDPG 无法处理 UAV 间协调；完全集中式 Critic 入维度过高 |
| 下层算法 | DQN (离散动作) | GU 模式选择是天然离散的（Backscatter / RF）；DDPG 的连续输出不适合 |
| Dyna-Q 集成方式 | 只加速下层 DQN | 上层 MADDPG 的状态转移没有明确物理模型；下层 GU 能量/缓冲区有闭式方程 |
| 能量采集模型 | 线性 (论文公式7) | 论文基准；非线性模型是未来工作方向 |
| 信道模型 | Rician 衰落 | UAV-地面信道有强 LOS 分量，Rayleigh 不适合 |
| 动作空间 | UAV: (Δx, Δy, 速度, 调度) | 论文原设；简化于端到端视觉导航方案（本项目的对照方向） |

## 5. 算法演进路径

```
MADDPG (基线)
  │  问题：单层架构，状态-动作空间爆炸
  ▼
Hierarchical-DQN (分层)
  │  问题：下层 DQN 纯经验回放，样本效率低
  ▼
Hierarchical-DynaQ (分层+模型)
  │  已验证：比 MADDPG 提升 63-76%
  │  未验证：Dyna-K 的最优取值、模型预测误差对长期训练的影响
  ▼
未来方向：非线性 EH + 改进探索 + UAV 直连通信
```

## 6. AI 协作约定

每次生成本项目代码时：
1. 先读 `CLAUDE.md`(本文件) + `docs/knowledge-map.md`
2. 一个 Prompt = 一个模块（≤200 行），不要一次生成整个实验脚本
3. 交付时附带知识包：位置 + WHY + 数据流 + 边界 + 风险点
4. 实验结果分析 → 先用 `@experiment-analyzer`，不要手动读 npy 文件
5. 训练监控 → 用 `@training-monitor`，检查 GPU 服务器状态
6. 重要输出（分析报告/论文段落/代码改动）→ `@output-reviewer` 终审
7. 开发会话结束后 → `@lesson-capturer` 记录教训

## 7. 实验结果速查

### 500 轮训练
| 指标 | MADDPG | Hier-DQN | Hier-DynaQ |
|------|--------|----------|------------|
| Final Avg Reward | -4238.19 | -1022.69 | -1022.69 |
| Improvement vs MADDPG | — | 75.87% | 75.87% |

### 5000 轮训练
| 指标 | MADDPG | Hier-DynaQ |
|------|--------|------------|
| Final Avg Reward | -2996.17 | -1107.98 |
| Improvement | — | 63.02% |

### 关键发现
- MADDPG 训练不稳定（std ≈ 2800），Hier-DynaQ 稳定 1.87×
- Dyna-Q 在 500 轮时加速效果不明显（K=5）；需验证更大 K 值
- 分层架构是主要增益来源，Dyna-Q 是增量改进

## 8. 红线

- 禁止修改 `src/system_model.py` 的物理模型公式而不更新对应文档
- 禁止在未确认 GPU 空闲时启动全量训练
- 禁止删除 `results/` 下的原始 .npy 文件（只增不删原则）
- 实验参数变更必须同步更新 readme.md 的关键参数表

## 9. 相关文档索引

- 系统认知地图 → `docs/knowledge-map.md`
- 教训知识库 → `docs/lessons-learned.md`
- 智能体协作协议 → `docs/agent-collaboration-protocol.md`
- 项目报告 → `reports/项目报告.md`
- 未来工作 → `reports/未来工作子课题清单.md`
- 论文原文 → `reports/sensors-23-04691.pdf`
- GPU 服务器 → memory `server-connection.md`
