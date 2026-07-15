# Dyna-Q-DQN-UAV

面向 UAV 辅助无线传感器网络的多智能体深度强化学习实验代码。项目实现联合轨迹规划、传输调度、接入控制和传输模式选择，并提供可复现的 Dyna-Q 规划步数 `k` 对比流程。

本项目参考论文：*Deep Reinforcement Learning for Joint Trajectory Planning, Transmission Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks*（Sensors 2023, 23, 4691）。

## 当前实现

- 上层：MADDPG，负责 UAV 轨迹与调度动作。
- 下层：DQN，负责地面用户接入和 RF/反向散射模式选择。
- Dyna-Q：学习环境模型，并使用模型生成的虚拟经验更新下层 DQN。
- 可复现实验：独立的环境、动作、经验回放、模型和 Dyna 随机数流。
- 训练诊断：奖励、碰撞、数据采集、基站送达、能耗、采能和模型误差。

### 奖励与通信修复

当前奖励公式保持为：

```text
数据送达收益（或未调度时的 0.5 × 数据采集收益）
- eta × 碰撞事件数
- eta1 × 消耗能量
```

关键参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `eta` | 10.0 | 单次碰撞惩罚 |
| `eta1` | 0.1 | 能耗惩罚系数 |
| `d_min` | 5.0 | UAV 最小安全距离 |
| `Dyna-k` | 默认 1 | 每次真实更新后的模型规划次数 |

通信速率按照当前 UAV–地面用户链路逐步计算，避免多个 UAV 覆盖同一用户时相互覆盖速率。`k=0` 表示不执行 Dyna 规划，可作为无模型规划基线。

## 目录结构

```text
.
├── src/
│   ├── system_model.py          # 环境、信道、UAV、地面用户和奖励
│   ├── hierarchical_agent.py    # 分层代理与 Dyna-Q
│   ├── maddpg_agent.py          # MADDPG 基线
│   └── compare_results.py
├── scripts/
│   ├── run_k_experiment.py      # 推荐：多 k、多随机种子实验
│   ├── train_all.py             # 兼容入口，转发到 run_k_experiment.py
│   ├── train_hierarchical.py
│   ├── run_hierarchical_no_dyna.py
│   └── train_maddpg.py
├── tests/
│   └── test_k_experiment_support.py
├── reports/                     # Markdown 研究文档
├── results/                     # 精选历史报告；生成的训练产物默认忽略
├── requirements.txt
└── run_on_server.sh
```

## 环境安装

建议使用 Python 3.8–3.11。服务器验证环境为 Python 3.8、PyTorch 2.4.1 CPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU 环境建议先安装 PyTorch CPU wheel：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU 环境请按实际驱动和 CUDA 版本从 PyTorch 官方安装页面选择 wheel，然后执行：

```bash
python -m pip install -r requirements.txt
```

## 运行实验

推荐的 9 组实验：

```bash
python scripts/run_k_experiment.py \
  --k-values 0 1 3 \
  --seeds 42 123 2026 \
  --episodes 1000 \
  --case 1
```

CPU 服务器后台运行：

```bash
EPISODES=1000 ./run_on_server.sh
```

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--k-values` | `0 1 3` | 要比较的 Dyna 规划步数 |
| `--seeds` | `42 123 2026` | 独立随机种子 |
| `--episodes` | `500` | 每组训练回合数 |
| `--case` | `1` | 环境初始化场景 |
| `--warmup-steps` | `1000` | Dyna 规划开始前的真实环境步数 |

每个实验组会生成：

- `config.json`：运行环境和奖励参数；
- `metrics.csv`：逐回合指标；
- `summary.json`：单次运行汇总；
- `checkpoint_latest.pt`：最新检查点；
- `training_diagnostics.png`：单次运行诊断图；
- `aggregate_by_k.csv` 和 `comparison_diagnostics.png`：跨种子汇总。

这些生成物默认不提交 GitHub，避免仓库被模型和重复实验数据撑大。需要公开结果时，建议只提交经过筛选的汇总表/报告，或使用 GitHub Release、Git LFS 和外部数据仓库。

## 测试

```bash
python -m compileall -q src scripts tests
python -m unittest tests/test_k_experiment_support.py -v
```

测试覆盖随机种子复现、随机数流隔离、`k=0`、学习率调度、环境指标、通信数据采集/送达和 `eta=10` 碰撞惩罚。

## 结果解释注意事项

`results/` 中部分历史报告生成于通信速率修复和奖励参数统一之前，不能与当前实验直接横向比较。新的 k 值结论应以相同代码、相同奖励参数、相同回合数和多随机种子结果为准。

## 贡献流程

1. 从最新 `main` 创建功能分支。
2. 不提交密码、SSH 密钥、日志、模型检查点或原始批量训练结果。
3. 提交前运行全部测试。
4. 使用清晰的提交信息，并通过 Pull Request 合并。

## 引用

```text
Luo, X.; Chen, C.; Zeng, C.; et al.
Deep Reinforcement Learning for Joint Trajectory Planning, Transmission
Scheduling, and Access Control in UAV-Assisted Wireless Sensor Networks.
Sensors 2023, 23, 4691.
```

DOI: https://doi.org/10.3390/s23104691

## 许可证

当前仓库尚未声明开源许可证。若计划允许他人复制、修改或分发代码，仓库所有者应根据研究和发布要求选择并添加许可证。
