"""
Comprehensive offline analysis of benchmark results.
1. Fixed-episode performance comparison
2. Local optimum / strategy divergence analysis
3. Buffer dynamics analysis
4. Visualization charts

Runs entirely offline from downloaded checkpoints + JSON data.
"""
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from system_model import Config, Environment
from maddpg_agent import MADDPGAgent
from hierarchical_agent import HierarchicalAgent, HierarchicalNoDynaAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_PATH = os.path.join(RESULTS_DIR, 'benchmark_report_20260718_112755.json')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    bench_data = json.load(f)

runs_by_algo = defaultdict(list)
for run in bench_data['runs']:
    runs_by_algo[run['algo']].append(run)


def load_agent(algo, ckpt_path, config):
    """Load an agent from checkpoint."""
    state_dim, action_dim = 30, 4 + 2 * config.M + 1
    if algo == 'maddpg':
        agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'dyna':
        agent = HierarchicalAgent(state_dim, action_dim, config.N, config, dyna_k=config.dyna_k)
    elif algo == 'nodyna':
        agent = HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config)
    else:
        raise ValueError(algo)
    agent.load_checkpoint(ckpt_path)
    return agent, state_dim, action_dim


# ===================================================================
# 1. FIXED-EPISODE COMPARISON
# ===================================================================
print("=== 1. Fixed-Episode Performance Comparison ===")

def compute_rolling(rewards, window=200):
    if len(rewards) < window:
        return np.array([np.mean(rewards)] * len(rewards))
    rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return np.concatenate([np.full(window-1, rolling[0]), rolling])

checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: raw reward curves with markers at checkpoints
ax = axes[0]
colors = {'dyna': '#1f77b4', 'nodyna': '#2ca02c', 'maddpg': '#ff7f0e'}
labels = {'dyna': 'Dyna-Q', 'nodyna': 'NoDyna', 'maddpg': 'MADDPG'}
markers = {'dyna': 'o', 'nodyna': 's', 'maddpg': '^'}

cp_data = {algo: {} for algo in ['dyna', 'nodyna', 'maddpg']}

for algo in ['dyna', 'nodyna', 'maddpg']:
    runs = runs_by_algo[algo]
    all_rewards = [np.array(r['rewards']) for r in runs]
    min_len = min(len(r) for r in all_rewards)
    aligned = np.array([r[:min_len] for r in all_rewards])
    mean_curve = aligned.mean(axis=0)
    std_curve = aligned.std(axis=0, ddof=1) if aligned.shape[0] > 1 else np.zeros(min_len)

    x = np.arange(len(mean_curve))
    ax.plot(x, mean_curve, color=colors[algo], label=labels[algo], linewidth=1.5, alpha=0.9)
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                    color=colors[algo], alpha=0.1)

    # Rolling average at checkpoints
    for cp in checkpoints:
        if cp <= min_len:
            rolling = compute_rolling(mean_curve[:cp], window=min(200, cp))
            cp_data[algo][cp] = float(rolling[-1])
            if cp % 1000 == 0:
                ax.axvline(x=cp, color=colors[algo], linestyle=':', alpha=0.3)

ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward')
ax.set_title('Reward Curves with Convergence Checkpoints')
ax.legend()
ax.grid(True, alpha=0.25)

# Right: bar chart at fixed episodes
ax = axes[1]
algos_plot = ['dyna', 'nodyna', 'maddpg']
x_pos = np.arange(len(checkpoints))
width = 0.25

for i, algo in enumerate(algos_plot):
    values = [cp_data[algo].get(cp, np.nan) for cp in checkpoints]
    bars = ax.bar(x_pos + i * width, values, width, label=labels[algo],
                  color=colors[algo], alpha=0.85)

ax.set_xlabel('Episode')
ax.set_ylabel('Rolling Average Reward')
ax.set_title('Performance at Fixed Training Intervals')
ax.set_xticks(x_pos + width)
ax.set_xticklabels([str(cp) for cp in checkpoints], rotation=45)
ax.legend()
ax.grid(True, alpha=0.25, axis='y')

fig.suptitle('Fixed-Episode Performance Comparison', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
path1 = os.path.join(OUTPUT_DIR, f'fixed_episode_comparison_{TIMESTAMP}.png')
fig.savefig(path1, dpi=150)
plt.close(fig)
print(f"  Saved: {path1}")

# Print convergence table
print("\n  Rolling Avg at Fixed Episodes:")
header = f"{'Episode':>8s}"
for algo in algos_plot:
    header += f"  {labels[algo]:>10s}"
print(f"  {header}")
for cp in checkpoints:
    row = f"  {cp:>8d}"
    for algo in algos_plot:
        v = cp_data[algo].get(cp)
        row += f"  {v:>10.2f}" if v is not None else f"  {'N/A':>10s}"
    print(row)


# ===================================================================
# 2. LOCAL OPTIMUM / STRATEGY DIVERGENCE ANALYSIS
# ===================================================================
print("\n=== 2. Local Optimum Analysis ===")

# Compare NoDyna vs Dyna-Q at 3000 episodes (near NoDyna convergence)
# Use best checkpoints for each algorithm, seed 123 (best performing for both)

def collect_episode_trace(algo, ckpt_path, seed, case=1):
    """Run one episode with loaded agent and collect per-step data."""
    config = Config(seed=seed)
    env = Environment(config)
    agent, state_dim, action_dim = load_agent(algo, ckpt_path, config)

    is_hier = algo in ('dyna', 'nodyna')

    trace = {
        'gu_buffers': [],      # (M,) per step
        'gu_energy': [],       # (M,) per step
        'uav_positions': [],   # (N, 3) per step
        'uav_buffers': [],     # (N,) per step
        'access_decisions': [],# (N, M) per step
        'collisions': [],      # int per step
        'rewards': [],         # float per step
        'total_reward': 0.0,
    }

    states = env.reset(case)
    total_reward = 0.0

    while True:
        # Collect state
        gu_buffers = np.array([gu.buffer for gu in env.gus])
        gu_energy = np.array([gu.energy for gu in env.gus])
        uav_pos = np.array([uav.pos.copy() for uav in env.uavs])
        uav_buf = np.array([uav.buffer for uav in env.uavs])

        trace['gu_buffers'].append(gu_buffers)
        trace['gu_energy'].append(gu_energy)
        trace['uav_positions'].append(uav_pos)
        trace['uav_buffers'].append(uav_buf)

        if is_hier:
            upper_actions = agent.upper_act(states, noise=False)
            lower_actions = agent.lower_act(states)
            full_actions = np.array([
                np.concatenate([upper_actions[i], lower_actions[i]])
                for i in range(config.N)
            ])
            next_states, rewards, done = env.step(full_actions)
            access = np.array([la[:config.M] for la in lower_actions])
        else:
            actions = agent.act(states, noise=False)
            next_states, rewards, done = env.step(actions)
            access = actions[:, :config.M]

        trace['access_decisions'].append(access)
        trace['collisions'].append(env.last_step_info.get('totals', {}).get('collision_events', 0))
        trace['rewards'].append(float(np.sum(rewards)))
        total_reward += float(np.sum(rewards))
        states = next_states
        if done:
            break

    trace['total_reward'] = total_reward
    # Convert lists to arrays where possible
    trace['gu_buffers'] = np.array(trace['gu_buffers'])
    trace['gu_energy'] = np.array(trace['gu_energy'])
    trace['uav_positions'] = np.array(trace['uav_positions'])
    trace['uav_buffers'] = np.array(trace['uav_buffers'])
    trace['access_decisions'] = np.array(trace['access_decisions'])
    trace['rewards'] = np.array(trace['rewards'])
    return trace


# Find best checkpoints
best_ckpts = {}
for f in sorted(os.listdir(CKPT_DIR)):
    if 'best' in f:
        for algo in ['dyna', 'nodyna', 'maddpg']:
            if f.startswith(algo):
                seed = int(f.split('seed')[1].split('_')[0])
                best_ckpts[(algo, seed)] = os.path.join(CKPT_DIR, f)

print(f"  Available best checkpoints: {list(best_ckpts.keys())}")

# Run traces for Dyna-Q and NoDyna with same seed, same initial state
comparison_pairs = []
for (algo, seed), ckpt_path in best_ckpts.items():
    if seed in (123, 2026):
        trace = collect_episode_trace(algo, ckpt_path, seed)
        comparison_pairs.append((algo, seed, trace))
        print(f"  {algo} seed={seed}: total_reward={trace['total_reward']:.4f}, "
              f"collisions={sum(trace['collisions']):.0f}, "
              f"avg_access_rate={np.mean(trace['access_decisions']):.3f}")

# Strategy comparison charts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_pairs = [(algo, seed, trace) for algo, seed, trace in comparison_pairs
              if seed == 123]  # Focus on seed 123 (best performer)

for idx, (algo, seed, trace) in enumerate(comparison_pairs[:2]):
    color = colors.get(algo, '#333')
    label = f'{labels.get(algo, algo)} (seed={seed})'

    # GU buffer dynamics
    ax = axes[0, 0]
    mean_buf = trace['gu_buffers'].mean(axis=1)
    ax.plot(mean_buf, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean GU Buffer')
    ax.set_title('GU Buffer Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # UAV buffer dynamics
    ax = axes[0, 1]
    mean_uav_buf = trace['uav_buffers'].mean(axis=1)
    ax.plot(mean_uav_buf, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean UAV Buffer')
    ax.set_title('UAV Buffer Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Access rate
    ax = axes[0, 2]
    access_rate = trace['access_decisions'].mean(axis=(1, 2))
    ax.plot(access_rate, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Access Rate')
    ax.set_title('GU Access Rate Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # UAV trajectory (top-down, 2D)
    ax = axes[1, 0]
    for uav_idx in range(3):
        x = trace['uav_positions'][:, uav_idx, 0]
        y = trace['uav_positions'][:, uav_idx, 1]
        ax.plot(x, y, color=color, alpha=0.3 + 0.2 * uav_idx, linewidth=1)
        ax.scatter(x[0], y[0], color=color, marker='o', s=30)
        ax.scatter(x[-1], y[-1], color=color, marker='x', s=40)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'UAV Trajectories ({label})')
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')

    # GU energy
    ax = axes[1, 1]
    mean_e = trace['gu_energy'].mean(axis=1)
    ax.plot(mean_e, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean GU Energy')
    ax.set_title('GU Energy Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Cumulative reward
    ax = axes[1, 2]
    cum_reward = np.cumsum(trace['rewards'])
    ax.plot(cum_reward, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

fig.suptitle('Strategy & Dynamics Comparison: Dyna-Q vs NoDyna (seed=123)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
path2 = os.path.join(OUTPUT_DIR, f'local_optimum_analysis_{TIMESTAMP}.png')
fig.savefig(path2, dpi=150)
plt.close(fig)
print(f"  Saved: {path2}")


# ===================================================================
# 3. BUFFER DYNAMICS ANALYSIS (DETAILED)
# ===================================================================
print("\n=== 3. Buffer Dynamics Analysis ===")

# Run longer comparison: all seeds, both algorithms
all_traces = {}
for (algo, seed), ckpt_path in best_ckpts.items():
    trace = collect_episode_trace(algo, ckpt_path, seed)
    all_traces[(algo, seed)] = trace
    avg_buf = trace['gu_buffers'].mean()
    max_buf = trace['gu_buffers'].max()
    final_buf = trace['gu_buffers'][-1].mean()
    print(f"  {algo} s={seed}: avg_gu_buf={avg_buf:.2f}, max_gu_buf={max_buf:.2f}, "
        f"final_gu_buf={final_buf:.2f}, total_r={trace['total_reward']:.4f}")

# Aggregate by algorithm
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# GU Buffer comparison (average over seeds)
for algo in ['dyna', 'nodyna']:
    algo_traces = [t for (a, s), t in all_traces.items() if a == algo]
    if not algo_traces:
        continue
    # Align to min length
    min_len = min(len(t['gu_buffers']) for t in algo_traces)
    buf_curves = np.array([t['gu_buffers'][:min_len].mean(axis=1) for t in algo_traces])
    mean_buf = buf_curves.mean(axis=0)
    std_buf = buf_curves.std(axis=0, ddof=1) if buf_curves.shape[0] > 1 else np.zeros(min_len)

    ax = axes[0, 0]
    x = np.arange(len(mean_buf))
    ax.plot(x, mean_buf, color=colors[algo], label=labels[algo], linewidth=2)
    ax.fill_between(x, mean_buf - std_buf, mean_buf + std_buf, color=colors[algo], alpha=0.15)

axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Mean GU Buffer Size')
axes[0, 0].set_title('GU Buffer Dynamics (avg over seeds)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.25)

# UAV Buffer comparison
for algo in ['dyna', 'nodyna']:
    algo_traces = [t for (a, s), t in all_traces.items() if a == algo]
    if not algo_traces:
        continue
    min_len = min(len(t['uav_buffers']) for t in algo_traces)
    buf_curves = np.array([t['uav_buffers'][:min_len].mean(axis=1) for t in algo_traces])
    mean_buf = buf_curves.mean(axis=0)
    std_buf = buf_curves.std(axis=0, ddof=1) if buf_curves.shape[0] > 1 else np.zeros(min_len)

    ax = axes[0, 1]
    x = np.arange(len(mean_buf))
    ax.plot(x, mean_buf, color=colors[algo], label=labels[algo], linewidth=2)
    ax.fill_between(x, mean_buf - std_buf, mean_buf + std_buf, color=colors[algo], alpha=0.15)

axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Mean UAV Buffer Size')
axes[0, 1].set_title('UAV Buffer Dynamics (avg over seeds)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.25)

# GU Energy comparison
for algo in ['dyna', 'nodyna']:
    algo_traces = [t for (a, s), t in all_traces.items() if a == algo]
    if not algo_traces:
        continue
    min_len = min(len(t['gu_energy']) for t in algo_traces)
    energy_curves = np.array([t['gu_energy'][:min_len].mean(axis=1) for t in algo_traces])
    mean_e = energy_curves.mean(axis=0)
    std_e = energy_curves.std(axis=0, ddof=1) if energy_curves.shape[0] > 1 else np.zeros(min_len)

    ax = axes[1, 0]
    x = np.arange(len(mean_e))
    ax.plot(x, mean_e, color=colors[algo], label=labels[algo], linewidth=2)
    ax.fill_between(x, mean_e - std_e, mean_e + std_e, color=colors[algo], alpha=0.15)

axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Mean GU Energy')
axes[1, 0].set_title('GU Energy Dynamics (avg over seeds)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.25)

# Access rate comparison
for algo in ['dyna', 'nodyna']:
    algo_traces = [t for (a, s), t in all_traces.items() if a == algo]
    if not algo_traces:
        continue
    min_len = min(len(t['access_decisions']) for t in algo_traces)
    access_curves = np.array([t['access_decisions'][:min_len].mean(axis=(1, 2)) for t in algo_traces])
    mean_acc = access_curves.mean(axis=0)
    std_acc = access_curves.std(axis=0, ddof=1) if access_curves.shape[0] > 1 else np.zeros(min_len)

    ax = axes[1, 1]
    x = np.arange(len(mean_acc))
    ax.plot(x, mean_acc, color=colors[algo], label=labels[algo], linewidth=2)
    ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color=colors[algo], alpha=0.15)

axes[1, 1].set_xlabel('Time Step')
axes[1, 1].set_ylabel('Mean Access Rate')
axes[1, 1].set_title('GU Access Rate (avg over seeds)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.25)

fig.suptitle('Buffer & Energy Dynamics: Dyna-Q vs NoDyna', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
path3 = os.path.join(OUTPUT_DIR, f'buffer_dynamics_{TIMESTAMP}.png')
fig.savefig(path3, dpi=150)
plt.close(fig)
print(f"  Saved: {path3}")


# ===================================================================
# 4. PERFORMANCE GRADIENT & FINAL SUMMARY CHARTS
# ===================================================================
print("\n=== 4. Performance Gradient Visualization ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4a. Final performance bar chart
ax = axes[0]
algo_names = []
algo_means = []
algo_stds = []
algo_colors_bar = []

for algo in ['dyna', 'nodyna', 'maddpg']:
    s = next((s for s in bench_data['summary'] if s['algo'] == algo), None)
    if s:
        algo_names.append(s['algo_name'])
        algo_means.append(s['final_reward'][0])
        algo_stds.append(s['final_reward'][1])
        algo_colors_bar.append(colors[algo])

bars = ax.bar(algo_names, algo_means, color=algo_colors_bar, alpha=0.85)
ax.errorbar(algo_names, algo_means, yerr=algo_stds, fmt='none', ecolor='black', capsize=5)
for bar, mean in zip(bars, algo_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{mean:.1f}', ha='center', fontweight='bold')
ax.set_ylabel('Final Reward (last 50)')
ax.set_title('Algorithm Performance Comparison')
ax.grid(True, alpha=0.25, axis='y')

# 4b. Convergence duration comparison
ax = axes[1]
algo_names_short = ['dyna', 'nodyna', 'maddpg']
eps_by_algo = {algo: [r['episodes_completed'] for r in runs_by_algo[algo]] for algo in algo_names_short}

positions = []
data = []
box_colors = []
for i, algo in enumerate(algo_names_short):
    eps_list = [r['episodes_completed'] for r in runs_by_algo[algo]]
    positions.append(i + 1)
    data.append(eps_list)
    box_colors.append(colors[algo])

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels([labels[a] for a in algo_names_short])
ax.set_ylabel('Episodes to Convergence')
ax.set_title('Convergence Duration')
ax.grid(True, alpha=0.25, axis='y')

# 4c. Improvement breakdown (stacked bar)
ax = axes[2]
# dyna beats maddpg by 55.9%: 16.0% from hierarchical + 47.4% from dyna (approx)
improvements = {
    'iDDPG → MADDPG': ('Centralized\nTraining', 0),   # will fill after iDDPG done
    'MADDPG → NoDyna': ('Hierarchical\nDecomposition', 16.0),
    'NoDyna → Dyna-Q': ('Model-Based\nPlanning (Dyna-Q)', 47.4),
}

# Show incremental contributions
contrib_names = ['Hierarchical\n(+16.0%)', 'Dyna-Q\n(+47.4%)']
contrib_values = [16.0, 47.4]
contrib_colors = ['#2ca02c', '#1f77b4']

bottom = 0
bars_list = []
for name, val, color in zip(contrib_names, contrib_values, contrib_colors):
    bar = ax.bar(['MADDPG → Dyna-Q'], [val], bottom=bottom, color=color, alpha=0.85)
    bars_list.append(bar)
    bottom += val

ax.set_ylabel('Cumulative Improvement (%)')
ax.set_title('Incremental Contribution Breakdown')
ax.legend([b[0] for b in bars_list], contrib_names, fontsize=9)
ax.grid(True, alpha=0.25, axis='y')

fig.suptitle('Performance Gradient Analysis', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
path4 = os.path.join(OUTPUT_DIR, f'performance_gradient_{TIMESTAMP}.png')
fig.savefig(path4, dpi=150)
plt.close(fig)
print(f"  Saved: {path4}")


# ===================================================================
# 5. SUMMARY REPORT
# ===================================================================
print("\n=== 5. Writing Summary Report ===")

report_path = os.path.join(OUTPUT_DIR, f'deep_analysis_report_{TIMESTAMP}.md')
lines = []
lines.append('# UAV-DRL 深度分析报告')
lines.append(f'\n> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
lines.append(f'> 数据来源: benchmark_report_20260718_112755.json + checkpoints/')

lines.append('\n## 1. 固定轮数性能对比\n')
lines.append('| Episode | Dyna-Q | NoDyna | MADDPG | Dyna-Q vs NoDyna |')
lines.append('|---------|--------|--------|--------|-----------------|')
for cp in checkpoints:
    dv = cp_data['dyna'].get(cp)
    nv = cp_data['nodyna'].get(cp)
    mv = cp_data['maddpg'].get(cp)
    if dv is not None and nv is not None:
        diff = ((dv - nv) / abs(nv) * 100) if nv != 0 else 0
        mv_str = f'{mv:.2f}' if mv is not None else 'N/A'
        lines.append(f'| {cp} | {dv:.2f} | {nv:.2f} | {mv_str} | {diff:+.1f}% |')

lines.append('\n**关键发现**: Dyna-Q 在 2000 轮后持续改善，MADDPG 和 NoDyna 在 ~2000-3000 轮收敛。')

lines.append('\n## 2. 局部最优分析\n')
lines.append('Dyna-Q 和 NoDyna 在相同初始状态下运行策略对比：\n')
lines.append('| 指标 | Dyna-Q | NoDyna |')
lines.append('|------|--------|--------|')
for (algo, seed), trace in all_traces.items():
    if seed == 123:
        name = labels[algo]
        lines.append(f'| {name} 总奖励 | {trace["total_reward"]:.4f} | — |')
        lines.append(f'| {name} 碰撞次数 | {sum(trace["collisions"]):.0f} | — |')
        lines.append(f'| {name} 平均接入率 | {np.mean(trace["access_decisions"]):.3f} | — |')
        lines.append(f'| {name} 最终 GU Buffer | {trace["gu_buffers"][-1].mean():.2f} | — |')

lines.append('\n## 3. 缓冲区与能耗分析\n')
lines.append('| 算法 | 平均 GU Buffer | 最大 GU Buffer | 最终 GU Buffer | 平均 UAV Buffer |')
lines.append('|------|---------------|---------------|---------------|----------------|')
for (algo, seed), trace in sorted(all_traces.items()):
    lines.append(f'| {labels[algo]} s={seed} | {trace["gu_buffers"].mean():.2f} | '
                 f'{trace["gu_buffers"].max():.2f} | {trace["gu_buffers"][-1].mean():.2f} | '
                 f'{trace["uav_buffers"].mean():.2f} |')

lines.append('\n## 4. 生成图表\n')
for p in [path1, path2, path3, path4]:
    lines.append(f'- [{os.path.basename(p)}]({os.path.basename(p)})')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Report saved: {report_path}")

print(f"\n{'='*60}")
print(f"Analysis complete! All outputs in: {OUTPUT_DIR}")
print(f"{'='*60}")
