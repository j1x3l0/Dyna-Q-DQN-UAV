# 智能体协作协议 — UAV-DRL 项目

> 定义本项目 12 个智能体之间的协作规范、工作流、上下文传递协议。
> 基于 jxl_better_vibe_coding 体系优化，适配 DRL 研究项目场景。

---

## 一、智能体全景图（12 个）

### 规划层（写代码之前）
| Agent | 模型 | 工具 | 触发时机 |
|-------|------|------|---------|
| `@spec-writer` | sonnet | Read, Grep, Glob | 新功能需求模糊时 |
| `@modular-architect` | sonnet | Read, Grep, Glob | 预计 >200 行代码 |
| `@knowledge-map-maintainer` | haiku | Read, Grep, Glob, Edit | 一轮开发结束后 |

### 执行层（写代码 + 调试）
| Agent | 模型 | 工具 | 触发时机 |
|-------|------|------|---------|
| `主对话` | 继承 | Read, Write, Edit, Bash | 逐模块实现（≤200行/模块） |
| `@fast-debugger` | sonnet | Read, Grep, Glob, Edit, Bash | 任何报错 |
| `@microworld-builder` | sonnet | Read, Grep, Glob, Write | 复杂模块需深度理解 |

### 研究专用层（本项目特有）
| Agent | 模型 | 工具 | 触发时机 |
|-------|------|------|---------|
| `@training-monitor` | sonnet | Read, Grep, Glob, Bash | 训练状态检查 / 异常检测 |
| `@experiment-analyzer` | sonnet | Read, Grep, Glob, Bash | 实验结果对比 / 分析报告 |

### 质量层（交付前）
| Agent | 模型 | 工具 | 触发时机 |
|-------|------|------|---------|
| `@knowledge-pack-generator` | sonnet | Read, Grep, Glob | 每次代码改动后 |
| `@understanding-reviewer` | opus | Read, Grep, Glob | PR Review |
| `@understanding-gate` | opus | Read, Grep, Glob | 合并前最后一步 |
| `@output-reviewer` | opus | Read, Grep, Glob | 交付重要输出前 |

### 反思层（开发后）
| Agent | 模型 | 工具 | 触发时机 |
|-------|------|------|---------|
| `@lesson-capturer` | haiku | Read, Grep, Glob, Edit | 开发会话结束后 |

---

## 二、上下文传递协议

### 协议 1：项目基线上下文（所有 Agent 共享）
每个 Agent 启动时自动读取（按优先级）：
1. `CLAUDE.md` — 项目心智模型
2. `docs/knowledge-map.md` — 系统认知地图

### 协议 2：Agent 间硬传递（通过文件）
上游 Agent 输出 → 写入指定文件 → 下游 Agent 读取：

```
@spec-writer          → docs/specs/[feature].md
@modular-architect    → docs/plans/[feature]-modules.md
@experiment-analyzer  → results/analysis/[exp_name]_report.md
@knowledge-pack-gen   → 输出到对话（Markdown 知识包）
@understanding-reviewer → 输出到对话（审查报告）
@output-reviewer      → 输出到对话（审查报告）
@lesson-capturer      → docs/lessons-learned.md
```

### 协议 3：Agent 间软传递（通过对话）
短生命周期 Agent（fast-debugger, training-monitor）的输出直接在对话中消费，不需文件持久化。

### 协议 4：升级传递（模型层级）
```
haiku 输出不确定 → 升级到 sonnet 重做
sonnet 输出需要审查 → 升级到 opus 审查
opus 输出涉及决策 → 标记 ⚠️ 等待人类确认
```

---

## 三、标准工作流

### 工作流 1：新实验开发

```
1. 需求分析
   ├─ 模糊 → @spec-writer 生成实验规格
   └─ >200行 → @modular-architect 拆分模块

2. 逐模块实现（主对话，≤200行/模块）
   ├─ 每模块 → @knowledge-pack-generator
   └─ 报错 → @fast-debugger

3. 实验验证
   ├─ GPU 训练监控 → @training-monitor
   └─ 结果分析 → @experiment-analyzer

4. 交付前
   ├─ @understanding-reviewer → 可读性
   ├─ @output-reviewer → 终审
   └─ @understanding-gate → 合并门槛

5. 收尾
   ├─ @knowledge-map-maintainer
   └─ @lesson-capturer
```

### 工作流 2：训练问题诊断

```
发现问题（NaN / 发散 / 停滞）
  │
  ├─ @training-monitor
  │     └─ 日志解析 + 健康指标 + 异常分类
  │
  ├─ 如果是代码 bug
  │     └─ @fast-debugger（定位 + 最小修复）
  │
  ├─ 如果是超参数问题
  │     └─ 主对话调整参数
  │
  └─ 修复后验证
        └─ @training-monitor（确认恢复）
```

### 工作流 3：论文对比分析

```
新论文
  │
  ├─ 主对话：pdftotext 提取全文
  │
  ├─ 主对话：提炼摘要/方法/实验/局限
  │
  ├─ @experiment-analyzer：与本项目结果逐项对比
  │     └─ 输出：差异表 + 可复用组件评估 + 切入点识别
  │
  ├─ @output-reviewer：终审对比分析的准确性
  │
  └─ 更新 `reports/` 下的相关文档
```

### 工作流 4：结果分析 → 论文图表

```
实验完成（.npy 文件就绪）
  │
  ├─ @experiment-analyzer
  │     └─ 指标计算 + 对比分析 + 洞察提炼
  │
  ├─ @dataviz skill
  │     └─ 论文级图表（reward 曲线 / 收敛对比 / 消融实验）
  │
  ├─ @output-reviewer
  │     └─ 图表 + 分析报告终审
  │
  └─ 交付（可写入论文 / 报告）
```

---

## 四、协作规则

### 串行 vs 并行

| 可并行 | 必须串行 |
|--------|---------|
| spec-writer + modular-architect（同 phase） | Spec 确认 → 模块拆分 |
| knowledge-pack-gen × N 个模块 | 模块拆分 → 逐模块实现 |
| training-monitor + experiment-analyzer | 实现 → knowledge-pack |
| understanding-reviewer（不同维度） | knowledge-pack → 审查 |
| knowledge-map-maintainer + lesson-capturer | 审查 → 合并门槛 |

### 模型选择决策树

```
任务需要"判断"还是"执行"？
  ├─ 纯执行（格式检查/查找替换）→ haiku
  ├─ 需要推理（生成/设计/调试）→ sonnet
  └─ 需要审查（找茬/验证/质量把关）→ opus

任务涉及"决策"吗？
  ├─ 技术选型/不可逆操作/安全敏感 → 暂停，等待人类
  └─ 实现细节/参数选择 → AI 可决定（标注 ⚠️ 不确定项）
```

### 手动接管信号
以下信号出现时停止 AI 自动流转，等待人类介入：
- 任何 Agent 输出含 `⚠️ 不确定` 且涉及核心决策
- 同一建议反复出现但无进展（死循环）
- 涉及删除 `results/` 原始数据、修改 `system_model.py` 物理公式
- 涉及 GPU 服务器上的破坏性操作（kill 训练进程、清理 screen 会话）
- 涉及安全/合规/论文署名/数据造假风险

---

## 五、Agent 能力边界速查

| Agent | 擅长 | 不擅长 | 升级到 |
|-------|------|--------|--------|
| haiku agents | 格式检查、机械维护、简单编辑 | 深度推理、模糊需求、架构判断 | sonnet |
| sonnet agents | 代码生成、调试、分析、设计 | 对抗性验证、极端边界、安全审计 | opus |
| opus agents | 深度审查、交叉验证、质量把关 | 替代领域专家、100% 事实保证 | 人类 |
| training-monitor | 日志解析、指标监控、异常检测 | 架构级性能优化、新算法设计 | opus |
| experiment-analyzer | 指标计算、对比分析、洞察提炼 | 统计显著性检验（需 scipy）、因果推断 | opus |

---

## 六、成本优化策略

| 策略 | 效果 |
|------|------|
| haiku 做机械活（knowledge-map, lesson-capture） | 单次 ~$0.03 |
| sonnet 做生成/分析活（主力） | 单次 ~$0.10-0.30 |
| opus 只做审查（不超过总调用 20%） | 单次 ~$0.50 |
| 训练监控用 sonnet 而非 opus | 避免 expensive 的日常检查 |
| 多个 sonnet agent 并行（独立任务） | 减少总等待时间 |

---

*最后更新: 2026-07-23 | 基于 jxl_better_vibe_coding V2 体系*
