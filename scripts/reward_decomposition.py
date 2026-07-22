"""
Reward function decomposition experiment: compare EE ratio vs additive reward.

Compares two reward formulations at a fixed episode count to isolate the effect
of reward design on convergence speed and final performance.

Usage:
  python scripts/reward_decomposition.py
  python scripts/reward_decomposition.py --seeds 42,123 --eps 500 --algos maddpg,dyna
"""

import json
import os
import sys
import time
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from system_model import Config, Environment
from maddpg_agent import MADDPGAgent
from hierarchical_agent import HierarchicalAgent
from training_utils import get_state_action_dims, decay_epsilon, compose_full_actions, extract_lower_rewards

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DECOMP_DIR = os.path.join(RESULTS_DIR, 'reward_decomposition')
os.makedirs(DECOMP_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

COLORS = {'ee_ratio': '#1f77b4', 'additive': '#ff7f0e'}
LABELS = {'ee_ratio': 'EE Ratio (bits/J)', 'additive': 'Additive (data - energy)'}


# ---------------------------------------------------------------------------
# Config with reward mode switch
# ---------------------------------------------------------------------------
class AdditiveRewardConfig(Config):
    """Config that uses additive reward instead of EE ratio."""
    def __init__(self, seed=42):
        super().__init__(seed=seed)
        self.reward_mode = 'additive'  # 'ee_ratio' or 'additive'


def run_experiment(algo: str, seed: int, reward_mode: str, episodes: int, case: int = 1) -> dict:
    """Run training with a specific reward mode and return metrics."""
    config = Config(seed=seed)
    config.reward_mode = reward_mode  # 'ee_ratio' or 'additive'
    env = Environment(config)
    is_hier = algo in ('dyna', 'nodyna')

    if algo == 'maddpg':
        agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'dyna':
        agent = HierarchicalAgent(state_dim, action_dim, config.N, config, dyna_k=config.dyna_k)
    else:
        raise ValueError(f"Unsupported algo for reward decomp: {algo}")

    rewards_history = []
    metrics_accum = defaultdict(float)

    start = time.time()
    for ep in range(episodes):
        states = env.reset(case)
        ep_reward = 0.0

        while True:
            if is_hier:
                upper_actions = agent.upper_act(states)
                lower_actions = agent.lower_act(states)
                actions = compose_full_actions(upper_actions, lower_actions, config.N)
            else:
                actions = agent.act(states)

            next_states, rewards, done = env.step(actions)

            step_info = env.last_step_info or {}

            if is_hier:
                lower_rewards = extract_lower_rewards(step_info, rewards, config.N)
                agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
                agent.update_upper()
                for i in range(config.N):
                    agent.add_lower_memory(i, states[i], lower_actions[i], lower_rewards[i],
                                           next_states[i], done)
                    agent.update_lower(i)
                    if algo == 'dyna':
                        agent.update_model(i)
                        agent.dyna_plan(i)
            else:
                agent.add_memory(states, actions, rewards, next_states, done)
                agent.update()

            ep_reward += float(np.sum(rewards))
            for key in ['data_received', 'data_sent_to_rbs', 'energy_consumed',
                        'collision_events', 'flight_energy']:
                val = step_info.get('totals', {}).get(key, 0.0)
                metrics_accum[key] += float(val) if val is not None else 0.0

            states = next_states
            if done:
                break

        rewards_history.append(ep_reward)
        agent.step_episode_schedulers()
        if is_hier:
            decay_epsilon(agent)

    duration = time.time() - start
    return {
        'algo': algo,
        'reward_mode': reward_mode,
        'seed': seed,
        'episodes': episodes,
        'rewards': np.array(rewards_history),
        'metrics': {k: v for k, v in metrics_accum.items()},
        'final_reward': float(np.mean(rewards_history[-50:])) if len(rewards_history) >= 50 else float(np.mean(rewards_history)),
        'duration': duration,
    }


def plot_comparison(results: list):
    """Plot EE ratio vs additive reward comparison."""
    # Group by algo and reward mode
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        groups[r['algo']][r['reward_mode']].append(r)

    n_algos = len(groups)
    fig, axes = plt.subplots(1, n_algos + 1, figsize=(18, 6))
    if n_algos == 1:
        axes = [axes[0], axes[0]]

    for idx, (algo, mode_data) in enumerate(sorted(groups.items())):
        ax = axes[idx]
        for mode in ['ee_ratio', 'additive']:
            runs = mode_data.get(mode, [])
            if not runs:
                continue
            all_rewards = [r['rewards'] for r in runs]
            min_len = min(len(r) for r in all_rewards)
            aligned = np.array([r[:min_len] for r in all_rewards])
            mean_c = aligned.mean(axis=0)
            std_c = aligned.std(axis=0, ddof=1) if aligned.shape[0] > 1 else np.zeros(min_len)

            x = np.arange(min_len)
            ax.plot(x, mean_c, color=COLORS[mode], linewidth=1.5,
                    label=f'{LABELS[mode]} ({len(runs)} seeds)')
            ax.fill_between(x, mean_c - std_c, mean_c + std_c, color=COLORS[mode], alpha=0.12)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        algo_name = 'MADDPG' if algo == 'maddpg' else 'Hierarchical (Dyna-Q)'
        ax.set_title(f'{algo_name}')
        ax.legend()
        ax.grid(True, alpha=0.25)

    # Rightmost: EE ratio bar chart
    ax = axes[-1]
    x_pos = []
    bar_data = []
    bar_colors_list = []
    labels_list = []
    pos = 0
    for algo in sorted(groups.keys()):
        for mode in ['ee_ratio', 'additive']:
            runs = groups[algo].get(mode, [])
            if runs:
                ee_values = []
                for r in runs:
                    data = r['metrics'].get('data_received', 0)
                    energy = r['metrics'].get('energy_consumed', 1)
                    ee_values.append(data / max(energy, 1e-6))
                mean = np.mean(ee_values)
                std = np.std(ee_values, ddof=1) if len(ee_values) > 1 else 0
                x_pos.append(pos)
                bar_data.append(mean)
                bar_colors_list.append(COLORS[mode])
                algo_name = 'MADDPG' if algo == 'maddpg' else 'Dyna-Q'
                labels_list.append(f'{algo_name}\n{LABELS[mode].split(chr(32))[1]}')
                pos += 1

    ax.bar(x_pos, bar_data, color=bar_colors_list, alpha=0.85)
    ax.errorbar(x_pos, bar_data, yerr=[0] * len(bar_data), fmt='none', ecolor='black', capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, fontsize=8)
    ax.set_ylabel('EE Ratio (bits/Joule)')
    ax.set_title('Final Energy Efficiency')
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle('Reward Function Decomposition: EE Ratio vs Additive', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(DECOMP_DIR, f'reward_decomposition_{TIMESTAMP}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot: {path}")

    # JSON report
    report = {
        'timestamp': TIMESTAMP,
        'results': [
            {k: v for k, v in r.items() if k != 'rewards'}
            for r in results
        ],
    }
    json_path = os.path.join(DECOMP_DIR, f'reward_decomposition_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {json_path}")
    return path, json_path


def main():
    parser = ArgumentParser(description='Reward Function Decomposition Experiment')
    parser.add_argument('--seeds', type=str, default='42,123',
                        help='Comma-separated seeds (default: 42,123)')
    parser.add_argument('--eps', type=int, default=500,
                        help='Episode count per experiment')
    parser.add_argument('--algos', type=str, default='maddpg,dyna',
                        help='Algorithms to test')
    parser.add_argument('--modes', type=str, default='ee_ratio,additive',
                        help='Reward modes to compare')
    parser.add_argument('--case', type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    algos = [a.strip() for a in args.algos.split(',')]
    modes = [m.strip() for m in args.modes.split(',')]

    total = len(algos) * len(seeds) * len(modes)
    print(f"Reward Decomposition Experiment: {total} runs ({len(algos)} algos × {len(seeds)} seeds × {len(modes)} modes)")

    results = []
    for a in algos:
        for s in seeds:
            for m in modes:
                label = f"{a}/{m}/seed{s}"
                print(f"  [{len(results)+1}/{total}] {label} ...", end=' ', flush=True)
                r = run_experiment(a, s, m, args.eps, args.case)
                print(f"reward={r['final_reward']:.2f}, EE={r['metrics']['data_received']/max(r['metrics']['energy_consumed'],1e-6):.2f}")
                results.append(r)

    plot_comparison(results)
    print(f"\nResults saved to: {DECOMP_DIR}")


if __name__ == '__main__':
    main()
