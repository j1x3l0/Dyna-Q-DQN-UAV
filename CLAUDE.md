# CLAUDE.md — UAV-DRL 项目心智模型

> AI 协作编程入口文件。所有工作前必读本文档与 `docs/knowledge-map.md`。
> 基于 jxl_better_vibe_coding 认知债务防御体系 (V3 — 顶级模型优化版)。

---

## 0. 顶级模型使用范式

> **当前主力模型：Codex + GPT-5.6-SOL。本节覆盖所有历史「过程性提示词」约定。**

### 核心原则

| 原则 | 含义 | 反模式（不要） |
|------|------|--------------|
| **目标 > 过程** | 定义目的地（目标 + 约束 + 输出格式），不规定路线 | 不要写"先做A→再做B→再检查C" |
| **约束 > 步骤** | 明确硬边界（禁止什么、必须包含什么），放开方法 | 不要用"think step by step" |
| **示例 > 描述** | 一个具体的输入/输出对胜过一段文字说明 | 不要用大段文字描述期望格式 |
| **单次完成 > 多轮把关** | 顶级模型一次性做到质量层 3 个 Agent 的事 | 不要跑完 @understanding-reviewer 再跑 @understanding-gate |
| **知识为人 > 知识为模型** | 记忆/认知地图/教训库是给你看的，不是给模型看的 | 不要为了"模型理解"写冗长的上下文 |

### 对顶级模型有害的提示词模式

- ~~"请仔细思考每一步"~~ → 冗余，模型内部推理已超过人类预设
- ~~"先分析 X，再设计 Y，最后实现 Z"~~ → 压制模型自主规划能力
- ~~"如果你是 XXX 专家，你应该…"~~ → 角色扮演分散注意力
- ~~"请确认你理解了，再继续"~~ → 浪费 token，不提升质量
- ~~多轮 Agent 串联（spec → architect → implement → review → gate）~~ → 顶级模型一次完成等价质量

### 对顶级模型有效的提示词模式

- **一句话目标** + **硬约束** + **输出格式示例**：最简形式，最高质量
- **"这是输入：[X]，我需要输出：[Y]，不能：[Z]"**：比 500 字描述更有效
- **直接给代码示例**：模型从示例中学到的比从指令中多
- **标注不确定性**："这部分我不确定…"比假装知道更好——模型会帮你验证

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

## 6. Agent 体系

### 6.1 当前模式：顶级模型 + 精简 Agent

> Codex + GPT-5.6-SOL 的推理能力已超过大多数 Agent 的"判断/审查"职责。Agent 体系精简为：**模型直接完成 + 关键 Agent 兜底**。

**模型直接完成（不需要 Agent）**：
- ~~需求分析~~ → 直接告诉模型目标 + 约束，它自己完成 spec 级别的结构设计
- ~~模块拆分~~ → 顶级模型能自己判断"这个任务应该拆成几步"
- ~~可读性审查~~ → 代码生成质量已足够高，独立审查收益递减
- ~~知识包生成~~ → 改动后让模型在回复中简要说明 WHY，不需要单独 Agent

**保留的 Agent（有不可替代的专用能力）**：

| Agent | 什么时候用 | 为什么不能省 |
|-------|-----------|-------------|
| `@experiment-analyzer` | 分析 .npy 结果、生成论文图表 | 专用 DRL 分析逻辑，需要读大量数据文件 |
| `@training-monitor` | 检查 GPU 状态、排查训练异常 | 需要 SSH 到服务器，检查外部状态 |
| `@fast-debugger` | 贴错误日志 → 最小修复 | 错误定位需要 grep 代码库，Agent 有搜索工具 |
| `@knowledge-map-maintainer` | 一轮开发结束后 | 认知地图是给你看的，更新它不需要模型审查 |

**按需使用的 Agent**：

| Agent | 什么时候用 |
|-------|-----------|
| `@output-reviewer` | 论文段落、投稿材料——你无法承受幻觉，需要第二意见 |
| `@lesson-capturer` | 踩坑后——但更推荐你直接告诉模型"记录这条教训"，不用独立 Agent |
| `@microworld-builder` | 新成员需要上手复杂模块时 |

### 6.2 能力边界升级路径（已废弃）

旧版模型级联路径（Haiku→Sonnet→Opus→人类）不适用于顶级模型。当前替代：
```
直接问 Codex/GPT-5.6-SOL → 不确定时手动验证 → 涉及安全/合规/不可逆操作时人工确认
```

### 6.3 Ponytail — 代码精简（保留）

已安装 Ponytail skills（`npx skills add DietrichGebert/ponytail`）：
- `ponytail` — 核心精简规则：少写代码，用内置方案代替过度工程
- `ponytail-debt` — 技术债务检查
- `ponytail-gain` — 收益评估
- `ponytail-review` — 代码审查时精简

### 6.4 Matt Pocock 工程 Skills（降级为按需）

以下 Skills 在顶级模型范式下多数已冗余，保留 `/research` 和 `diagnosing-bugs` 作为后备：

| Skill | 状态 | 说明 |
|-------|------|------|
| `/research` | 保留 | 后台调研高信任源，模型不能替代实时搜索 |
| `diagnosing-bugs` | 保留 | 系统化诊断循环，复杂 bug 可能需要 |
| `/code-review` | 降级 | 顶级模型一次生成质量已够，双轴审查收益递减 |
| `/grill-with-docs` | 降级 | 可以直接问模型"质询我这个方案" |
| 其他 | 降级 | 过程性脚手架，顶级模型不需要 |

## 7. AI 协作约定

> 以下约定为**人类认知辅助**，不为约束模型行为。顶级模型不需要过程性指令。

### 每次开发时
1. 确保模型已读取 `CLAUDE.md` + `docs/knowledge-map.md`（获取项目上下文）
2. 一个 Prompt = 一个明确目标（可以 >200 行——顶级模型不会因为文件长而出错）
3. 要求模型在改动后**简要说明** WHAT + WHY（1-3 句话，不要展开成知识包）

### 实验与分析时
4. 分析 .npy 结果 → `@experiment-analyzer`（不要手动读）
5. 检查 GPU 训练状态 → `@training-monitor`
6. 重要报告/论文段落 → `@output-reviewer` 终审（你无法承受 DRL 数据的幻觉）

### 开发结束后
7. 一句话确认认知地图是否需要更新 → `@knowledge-map-maintainer`
8. 踩坑后直接告诉模型"记录到 lessons-learned.md"

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
