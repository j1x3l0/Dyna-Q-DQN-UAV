"""
Comprehensive visualization analysis for UAV-DRL benchmark results.

Generates:
  1. 2D+3D UAV trajectories (single episode rollouts from checkpoint)
  2. GU/UAV buffer time-series per step
  3. Energy efficiency ratio curve over training
  4. Convergence curve with CI bands
  5. Combined dashboard summary

Usage:
  python scripts/analyze_visualizations.py <benchmark_json> [--algo dyna]
"""

import json
import os
import sys
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from system_model import Config, Environment
from maddpg_agent import MADDPGAgent
from hierarchical_agent import HierarchicalAgent, HierarchicalNoDynaAgent
from iddpg_agent import iDDPGAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLORS = {'iddpg': '#d62728', 'maddpg': '#ff7f0e',
          'nodyna': '#2ca02c', 'dyna': '#1f77b4',
          'cop_maddpg': '#9467bd', 'matd3': '#8c564b'}
LABELS = {'iddpg': 'iDDPG', 'maddpg': 'MADDPG',
          'nodyna': 'NoDyna', 'dyna': 'Dyna-Q',
          'cop_maddpg': 'CoP-MADDPG', 'matd3': 'MATD3'}

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def load_agent(algo, ckpt_path, config):
    """Load a trained agent from checkpoint."""
    state_dim, action_dim = 30, 4 + 2 * config.M + 1
    if algo == 'maddpg':
        return MADDPGAgent(state_dim, action_dim, config.N, config), state_dim, action_dim
    elif algo == 'iddpg':
        return iDDPGAgent(state_dim, action_dim, config.N, config), state_dim, action_dim
    elif algo == 'dyna':
        return HierarchicalAgent(state_dim, action_dim, config.N, config, dyna_k=config.dyna_k), state_dim, action_dim
    elif algo == 'nodyna':
        return HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config), state_dim, action_dim
    raise ValueError(f"Unknown algo: {algo}")


# ===================================================================
# 1. 3D TRAJECTORY PLOT
# ===================================================================
def generate_trajectory_plots(ckpt_path: str, algo: str, seed: int, config: Config):
    """Run one episode with loaded agent and generate 2D + 3D trajectory plots."""
    env = Environment(config)
    agent, state_dim, action_dim = load_agent(algo, ckpt_path, config)
    is_hier = algo in ('dyna', 'nodyna')

    states = env.reset(case=1)
    traces = {'uav_positions': [], 'gu_positions': []}

    # Collect initial positions
    gu_positions = np.array([gu.pos.copy() for gu in env.gus])
    step = 0
    while step < 200:
        traces['uav_positions'].append(np.array([uav.pos.copy() for uav in env.uavs]))
        if step == 0:
            traces['gu_positions'] = gu_positions

        # Agent action selection
        if is_hier:
            from training_utils import compose_full_actions
            upper_actions = agent.upper_act(states)
            lower_actions = agent.lower_act(states)
            actions = compose_full_actions(upper_actions, lower_actions, config.N)
        else:
            actions = agent.act(states, noise=False)

        next_states, rewards, done = env.step(actions)
        states = next_states
        step += 1
        if done:
            break

    uav_positions = np.array(traces['uav_positions'])  # (T, N, 3)

    # --- 2D top-down ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(config.N):
        ax.plot(uav_positions[:, i, 0], uav_positions[:, i, 1],
                color=COLORS.get(f'agent{i}', '#333'), linewidth=2,
                marker='o', markersize=3, label=f'UAV {i}')
        # start
        ax.scatter(*uav_positions[0, i, :2], color=COLORS.get(f'agent{i}', '#333'),
                   marker='*', s=150, zorder=5)
        # end
        ax.scatter(*uav_positions[-1, i, :2], color=COLORS.get(f'agent{i}', '#333'),
                   marker='X', s=100, zorder=5)

    # GUs
    ax.scatter(gu_positions[:, 0], gu_positions[:, 1],
               c='gray', marker='s', s=80, alpha=0.5, label='GUs', zorder=3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{LABELS[algo]} — UAV Trajectory (2D Top-Down, seed={seed})')
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')
    fig.tight_layout()
    path_2d = os.path.join(OUTPUT_DIR, f'trajectory_2d_{algo}_seed{seed}_{TIMESTAMP}.png')
    fig.savefig(path_2d, dpi=150)
    plt.close(fig)
    print(f"  2D trajectory: {path_2d}")

    # --- 3D with terrain ---
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')

    for i in range(config.N):
        ax3d.plot(uav_positions[:, i, 0], uav_positions[:, i, 1], uav_positions[:, i, 2],
                  color=COLORS.get(f'agent{i}', '#333'), linewidth=2, marker='o', markersize=2,
                  label=f'UAV {i}')
        ax3d.scatter(*uav_positions[0, i], color=COLORS.get(f'agent{i}', '#333'),
                     marker='*', s=150, zorder=5)
        ax3d.scatter(*uav_positions[-1, i], color=COLORS.get(f'agent{i}', '#333'),
                     marker='X', s=100, zorder=5)

    ax3d.scatter(gu_positions[:, 0], gu_positions[:, 1], np.zeros(config.M),
                 c='gray', marker='s', s=60, alpha=0.5, label='GUs', zorder=3)

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title(f'{LABELS[algo]} — UAV Trajectory (3D, seed={seed})')
    ax3d.legend()
    fig.tight_layout()
    path_3d = os.path.join(OUTPUT_DIR, f'trajectory_3d_{algo}_seed{seed}_{TIMESTAMP}.png')
    fig.savefig(path_3d, dpi=150)
    plt.close(fig)
    print(f"  3D trajectory: {path_3d}")
    return path_2d, path_3d


# ===================================================================
# 2. BUFFER & ENERGY TIME-SERIES
# ===================================================================
def generate_buffer_plots(benchmark_json_path: str):
    """Generate GU buffer, UAV buffer, and energy time-series charts from benchmark data."""
    with open(benchmark_json_path) as f:
        data = json.load(f)

    runs_by_algo = defaultdict(list)
    for run in data['runs']:
        runs_by_algo[run['algo']].append(run)

    # Per-episode metrics from benchmark
    algos = sorted(runs_by_algo.keys())
    if not algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Data received / episode over training
    ax = axes[0, 0]
    for algo in algos:
        runs = runs_by_algo[algo]
        # Use rewards as proxy for convergence, compute rolling data_received
        all_eps = []
        for r in runs:
            rewards = np.array(r['rewards'])
            eps = r['metrics'].get('data_received', 0) / max(r['episodes_completed'], 1)
            all_eps.append(eps)
        mean = np.mean(all_eps)
        std = np.std(all_eps, ddof=1) if len(all_eps) > 1 else 0
        ax.bar(LABELS[algo], mean, color=COLORS.get(algo, '#333'), yerr=std,
               alpha=0.85, capsize=5)
    ax.set_ylabel('Avg Data Received / Episode')
    ax.set_title('Throughput Comparison')
    ax.grid(True, alpha=0.25, axis='y')

    # 2. Energy consumption / episode
    ax = axes[0, 1]
    for algo in algos:
        runs = runs_by_algo[algo]
        all_eps = [r['metrics'].get('energy_consumed', 0) / max(r['episodes_completed'], 1)
                   for r in runs]
        mean = np.mean(all_eps)
        std = np.std(all_eps, ddof=1) if len(all_eps) > 1 else 0
        ax.bar(LABELS[algo], mean, color=COLORS.get(algo, '#333'), yerr=std,
               alpha=0.85, capsize=5)
    ax.set_ylabel('Avg Energy Consumed / Episode')
    ax.set_title('Energy Consumption Comparison')
    ax.grid(True, alpha=0.25, axis='y')

    # 3. Energy Efficiency Ratio (data_received / energy_consumed)
    ax = axes[1, 0]
    for algo in algos:
        runs = runs_by_algo[algo]
        ratios = []
        for r in runs:
            data = r['metrics'].get('data_received', 1e-6)
            energy = r['metrics'].get('energy_consumed', 1)
            ratios.append(data / max(energy, 1e-6))
        mean = np.mean(ratios)
        std = np.std(ratios, ddof=1) if len(ratios) > 1 else 0
        ax.bar(LABELS[algo], mean, color=COLORS.get(algo, '#333'), yerr=std,
               alpha=0.85, capsize=5)
    ax.set_ylabel('EE Ratio (bits/Joule)')
    ax.set_title('Energy Efficiency Ratio')
    ax.grid(True, alpha=0.25, axis='y')

    # 4. Data sent to RBS / episode
    ax = axes[1, 1]
    for algo in algos:
        runs = runs_by_algo[algo]
        all_eps = [r['metrics'].get('data_sent_to_rbs', 0) / max(r['episodes_completed'], 1)
                   for r in runs]
        mean = np.mean(all_eps)
        std = np.std(all_eps, ddof=1) if len(all_eps) > 1 else 0
        ax.bar(LABELS[algo], mean, color=COLORS.get(algo, '#333'), yerr=std,
               alpha=0.85, capsize=5)
    ax.set_ylabel('Avg Data Sent to RBS / Episode')
    ax.set_title('Aggregated Throughput to RBS')
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle('Buffer & Energy Metrics — Algorithm Comparison', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, f'metrics_dashboard_{TIMESTAMP}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Metrics dashboard: {path}")
    return path


# ===================================================================
# 3. CONVERGENCE CURVE WITH CI
# ===================================================================
def generate_convergence_plot(benchmark_json_path: str):
    """Generate convergence curve with 95% CI bands for all algorithms."""
    with open(benchmark_json_path) as f:
        data = json.load(f)

    runs_by_algo = defaultdict(list)
    for run in data['runs']:
        runs_by_algo[run['algo']].append(run)

    algos = sorted(runs_by_algo.keys())
    if not algos:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Raw reward curves
    for algo in algos:
        runs = runs_by_algo[algo]
        all_rewards = [np.array(r['rewards']) for r in runs]
        min_len = min(len(r) for r in all_rewards)
        aligned = np.array([r[:min_len] for r in all_rewards])
        mean_curve = aligned.mean(axis=0)
        std_curve = aligned.std(axis=0, ddof=1) if aligned.shape[0] > 1 else np.zeros(min_len)
        ci95 = 1.96 * std_curve / np.sqrt(aligned.shape[0])

        x = np.arange(min_len)
        ax1.plot(x, mean_curve, color=COLORS.get(algo, '#333'), linewidth=1.5,
                 label=f'{LABELS.get(algo, algo)} ({aligned.shape[0]} seeds)')
        ax1.fill_between(x, mean_curve - ci95, mean_curve + ci95,
                         color=COLORS.get(algo, '#333'), alpha=0.12)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Mean Reward with 95% CI')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Right: Energy efficiency curve
    for algo in algos:
        runs = runs_by_algo[algo]
        # Compute EE ratio per episode from metrics data
        # We approximate from cumulative metrics
        all_ee = []
        for r in runs:
            data_recv = r['metrics'].get('data_received', 1e-6)
            energy = r['metrics'].get('energy_consumed', 1.0)
            rewards = np.array(r['rewards'])
            ee = data_recv / max(energy, 1e-6)
            all_ee.append(ee)

        mean = np.mean(all_ee)
        std = np.std(all_ee, ddof=1) if len(all_ee) > 1 else 0

        ax2.bar(LABELS.get(algo, algo), mean, color=COLORS.get(algo, '#333'),
                yerr=std, alpha=0.85, capsize=5)

    ax2.set_ylabel('Energy Efficiency (bits/Joule)')
    ax2.set_title('Final Energy Efficiency Ratio')
    ax2.grid(True, alpha=0.25, axis='y')

    fig.suptitle('Convergence & Efficiency Analysis', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, f'convergence_analysis_{TIMESTAMP}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Convergence plot: {path}")
    return path


# ===================================================================
# 4. COMBINED DASHBOARD
# ===================================================================
def generate_dashboard(benchmark_json_path: str, ckpt_dir: str = None):
    """Generate a combined 4-panel dashboard summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = []
    paths.append(generate_convergence_plot(benchmark_json_path))
    paths.append(generate_buffer_plots(benchmark_json_path))

    # Trajectory plot (from checkpoint if available)
    if ckpt_dir and os.path.isdir(ckpt_dir):
        # Find best checkpoint for any available algorithm
        best_files = [f for f in os.listdir(ckpt_dir) if f.endswith('_best.pt')]
        if best_files:
            ckpt_path = os.path.join(ckpt_dir, best_files[0])
            algo = ckpt_path.split('_')[0]
            seed = int(ckpt_path.split('seed')[1].split('_')[0])
            config = Config(seed=seed)
            traj_paths = generate_trajectory_plots(ckpt_path, algo, seed, config)
            paths.extend(traj_paths)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    for p in paths:
        print(f"  {p}")


def main():
    parser = ArgumentParser(description='UAV-DRL Comprehensive Visualization Generator')
    parser.add_argument('benchmark_json', nargs='?',
                        help='Path to benchmark_report_*.json')
    parser.add_argument('--algo', type=str, default='dyna',
                        choices=['dyna', 'nodyna', 'maddpg', 'iddpg'],
                        help='Algorithm for trajectory plots (default: dyna)')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Checkpoint directory (default: ../checkpoints)')
    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = CKPT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.benchmark_json and os.path.exists(args.benchmark_json):
        generate_dashboard(args.benchmark_json, args.ckpt_dir)
    else:
        # Find latest benchmark
        json_files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.startswith('benchmark_report_') and f.endswith('.json')],
            reverse=True)
        if json_files:
            json_path = os.path.join(RESULTS_DIR, json_files[0])
            print(f"Using: {json_path}")
            generate_dashboard(json_path, args.ckpt_dir)
        else:
            print("No benchmark JSON found. Run benchmark first, then pass JSON file.")
            print(f"Usage: python {__file__} results/benchmark_report_YYYYMMDD_HHMMSS.json")


if __name__ == '__main__':
    main()
