import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_agent import HierarchicalAgent
from system_model import Config, Environment


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_EPISODES = 500
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_K_VALUES = (0, 1, 3)


@dataclass
class RunResult:
    k: int
    seed: int
    episodes: int
    rewards: np.ndarray
    metrics: dict
    duration: float


def summarize(values):
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
    else:
        std = 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 0 else 0.0
    return mean, std, ci95


def run_single_experiment(k, seed, episodes=DEFAULT_EPISODES, case=1):
    config = Config(seed=seed)
    config.dyna_k = k
    env = Environment(config)

    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config, dyna_k=k)

    rewards_history = []
    totals = {
        'collision_events': 0,
        'collision_penalty': 0.0,
        'data_received': 0.0,
        'data_sent_to_rbs': 0.0,
        'energy_consumed': 0.0,
        'harvested_energy': 0.0,
    }

    start_time = time.time()
    for episode in range(episodes):
        states = env.reset(case)
        episode_reward = 0.0

        while True:
            upper_actions = agent.upper_act(states)
            lower_actions = agent.lower_act(states)

            full_actions = np.array([
                np.concatenate([upper_actions[i], lower_actions[i]])
                for i in range(config.N)
            ])

            next_states, rewards, done = env.step(full_actions)
            step_info = env.last_step_info or {'totals': {k: 0.0 for k in totals}}

            agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
            agent.update_upper()

            for i in range(config.N):
                agent.add_lower_memory(i, states[i], lower_actions[i], rewards[i], next_states[i], done)
                agent.update_lower(i)
                if k > 0:
                    agent.update_model(i)
                    agent.dyna_plan(i)

            episode_reward += float(np.sum(rewards))
            for key in totals:
                totals[key] += float(step_info['totals'].get(key, 0.0))

            states = next_states
            if done:
                break

        rewards_history.append(episode_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        agent.step_episode_schedulers()

    duration = time.time() - start_time
    return RunResult(
        k=k,
        seed=seed,
        episodes=episodes,
        rewards=np.asarray(rewards_history, dtype=float),
        metrics={key: float(value) for key, value in totals.items()},
        duration=duration,
    )


def aggregate_results(run_results):
    rows = []
    for k in DEFAULT_K_VALUES:
        k_runs = [r for r in run_results if r.k == k]
        final_reward = [float(np.mean(r.rewards[-50:])) for r in k_runs]
        collision_events = [r.metrics['collision_events'] / r.episodes for r in k_runs]
        collision_penalty = [r.metrics['collision_penalty'] / r.episodes for r in k_runs]
        data_received = [r.metrics['data_received'] / r.episodes for r in k_runs]
        data_sent_to_rbs = [r.metrics['data_sent_to_rbs'] / r.episodes for r in k_runs]
        energy_consumed = [r.metrics['energy_consumed'] / r.episodes for r in k_runs]
        harvested_energy = [r.metrics['harvested_energy'] / r.episodes for r in k_runs]
        duration = [r.duration for r in k_runs]

        rows.append({
            'k': k,
            'final_reward_mean': summarize(final_reward),
            'collision_events_mean': summarize(collision_events),
            'collision_penalty_mean': summarize(collision_penalty),
            'data_received_mean': summarize(data_received),
            'data_sent_to_rbs_mean': summarize(data_sent_to_rbs),
            'energy_consumed_mean': summarize(energy_consumed),
            'harvested_energy_mean': summarize(harvested_energy),
            'duration_mean': summarize(duration),
            'reward_curve_mean': np.mean([r.rewards for r in k_runs], axis=0),
            'reward_curve_std': np.std([r.rewards for r in k_runs], axis=0, ddof=1) if len(k_runs) > 1 else np.zeros_like(k_runs[0].rewards),
        })

    return rows


def write_report(run_results, summary_rows):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(RESULTS_DIR, f'eta50_k_sweep_report_{timestamp}.txt')
    json_path = os.path.join(RESULTS_DIR, f'eta50_k_sweep_report_{timestamp}.json')

    lines = []
    lines.append('=' * 80)
    lines.append('ETA=50 UAV DRL Sweep Report')
    lines.append('=' * 80)
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Episodes per run: {DEFAULT_EPISODES}')
    lines.append(f'Seeds: {", ".join(map(str, DEFAULT_SEEDS))}')
    lines.append(f'K values: {", ".join(map(str, DEFAULT_K_VALUES))}')
    lines.append('')

    lines.append('Per-run final reward (last 50 episode mean):')
    lines.append(f"{'k':>3} {'seed':>8} {'final_mean':>14} {'duration_s':>12}")
    for result in run_results:
        final_mean = float(np.mean(result.rewards[-50:]))
        lines.append(f"{result.k:>3} {result.seed:>8} {final_mean:>14.2f} {result.duration:>12.1f}")

    lines.append('')
    lines.append('Aggregated statistics by k:')
    for row in summary_rows:
        lines.append(f"k={row['k']}")
        for label, key in [
            ('final_reward', 'final_reward_mean'),
            ('collision_events', 'collision_events_mean'),
            ('collision_penalty', 'collision_penalty_mean'),
            ('data_received', 'data_received_mean'),
            ('data_sent_to_rbs', 'data_sent_to_rbs_mean'),
            ('energy_consumed', 'energy_consumed_mean'),
            ('harvested_energy', 'harvested_energy_mean'),
            ('duration_s', 'duration_mean'),
        ]:
            mean, std, ci95 = row[key]
            lines.append(f"  {label}: mean={mean:.4f}, std={std:.4f}, ci95={ci95:.4f}")
        lines.append('')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    serializable = {
        'generated_at': datetime.now().isoformat(),
        'episodes': DEFAULT_EPISODES,
        'seeds': list(DEFAULT_SEEDS),
        'ks': list(DEFAULT_K_VALUES),
        'runs': [
            {
                'k': r.k,
                'seed': r.seed,
                'duration': r.duration,
                'metrics': r.metrics,
                'rewards': r.rewards.tolist(),
            }
            for r in run_results
        ],
        'summary': [
            {
                'k': row['k'],
                'final_reward_mean': row['final_reward_mean'],
                'collision_events_mean': row['collision_events_mean'],
                'collision_penalty_mean': row['collision_penalty_mean'],
                'data_received_mean': row['data_received_mean'],
                'data_sent_to_rbs_mean': row['data_sent_to_rbs_mean'],
                'energy_consumed_mean': row['energy_consumed_mean'],
                'harvested_energy_mean': row['harvested_energy_mean'],
                'duration_mean': row['duration_mean'],
            }
            for row in summary_rows
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return report_path, json_path


def plot_summary(summary_rows):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, axes = plt.subplots(1, len(summary_rows), figsize=(18, 5), sharey=True)
    if len(summary_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, summary_rows):
        curve = row['reward_curve_mean']
        std = row['reward_curve_std']
        ci95 = 1.96 * std / math.sqrt(len(DEFAULT_SEEDS))
        x = np.arange(len(curve))
        ax.plot(x, curve, color='#1f77b4', linewidth=1.6)
        ax.fill_between(x, curve - ci95, curve + ci95, color='#1f77b4', alpha=0.2)
        ax.set_title(f'k={row["k"]}')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('Episode reward')
    fig.suptitle('Eta=50 sweep: mean reward with 95% CI over seeds', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, f'eta50_k_sweep_reward_curves_{timestamp}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    print('Starting eta=50 sweep: k=0/1/3, seeds=42/123/2026, episodes=500')
    run_results = []
    for k in DEFAULT_K_VALUES:
        for seed in DEFAULT_SEEDS:
            print(f'Running k={k}, seed={seed}...')
            run_results.append(run_single_experiment(k=k, seed=seed, episodes=DEFAULT_EPISODES))

    summary_rows = aggregate_results(run_results)
    report_path, json_path = write_report(run_results, summary_rows)
    chart_path = plot_summary(summary_rows)

    print(f'Report saved: {report_path}')
    print(f'JSON saved: {json_path}')
    print(f'Chart saved: {chart_path}')

    for row in summary_rows:
        mean, std, ci95 = row['final_reward_mean']
        print(f"k={row['k']} final reward mean={mean:.2f}, std={std:.2f}, ci95={ci95:.2f}")


if __name__ == '__main__':
    main()
