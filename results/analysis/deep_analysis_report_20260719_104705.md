# UAV-DRL 深度分析报告

> 生成时间: 2026-07-19 10:47:10
> 数据来源: benchmark_report_20260718_112755.json + checkpoints/

## 1. 固定轮数性能对比

| Episode | Dyna-Q | NoDyna | MADDPG | Dyna-Q vs NoDyna |
|---------|--------|--------|--------|-----------------|
| 500 | -1.63 | -2.73 | -1.83 | +40.3% |
| 1000 | -5.61 | -4.04 | -1.86 | -39.0% |
| 1500 | -4.67 | -1.32 | -3.77 | -253.7% |
| 2000 | -5.93 | -5.56 | N/A | -6.6% |
| 2500 | -2.68 | -2.41 | N/A | -11.3% |

**关键发现**: Dyna-Q 在 2000 轮后持续改善，MADDPG 和 NoDyna 在 ~2000-3000 轮收敛。

## 2. 局部最优分析

Dyna-Q 和 NoDyna 在相同初始状态下运行策略对比：

| 指标 | Dyna-Q | NoDyna |
|------|--------|--------|
| Dyna-Q 总奖励 | 0.0000 | — |
| Dyna-Q 碰撞次数 | 0 | — |
| Dyna-Q 平均接入率 | 0.277 | — |
| Dyna-Q 最终 GU Buffer | 10.86 | — |
| NoDyna 总奖励 | 0.0000 | — |
| NoDyna 碰撞次数 | 0 | — |
| NoDyna 平均接入率 | 0.244 | — |
| NoDyna 最终 GU Buffer | 34.38 | — |

## 3. 缓冲区与能耗分析

| 算法 | 平均 GU Buffer | 最大 GU Buffer | 最终 GU Buffer | 平均 UAV Buffer |
|------|---------------|---------------|---------------|----------------|
| Dyna-Q s=42 | 60.01 | 609.93 | 172.40 | 0.00 |
| Dyna-Q s=123 | 10.86 | 14.68 | 10.86 | 0.00 |
| Dyna-Q s=2026 | 96.99 | 723.35 | 162.03 | 0.00 |
| NoDyna s=123 | 19.06 | 151.39 | 34.38 | 0.00 |
| NoDyna s=2026 | 127.15 | 547.05 | 311.03 | 0.00 |

## 4. 生成图表

- [fixed_episode_comparison_20260719_104705.png](fixed_episode_comparison_20260719_104705.png)
- [local_optimum_analysis_20260719_104705.png](local_optimum_analysis_20260719_104705.png)
- [buffer_dynamics_20260719_104705.png](buffer_dynamics_20260719_104705.png)
- [performance_gradient_20260719_104705.png](performance_gradient_20260719_104705.png)