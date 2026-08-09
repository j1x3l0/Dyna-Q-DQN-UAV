"""
Reward objective experiment: compare historical, paper-aligned, and additive rewards.

Compares reward formulations at a fixed episode count to isolate the effect
of reward design on convergence speed and final performance.

Usage:
  python scripts/reward_decomposition.py
  python scripts/reward_decomposition.py --seeds 42,123 --eps 500 --algos maddpg,dyna
  python scripts/reward_decomposition.py --modes paper_xi --eps 50
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
from hierarchical_agent import HierarchicalAgent, HierarchicalNoDynaAgent
from training_utils import get_state_action_dims, decay_epsilon, compose_full_actions, extract_lower_rewards

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DECOMP_DIR = os.path.join(RESULTS_DIR, 'reward_decomposition')
os.makedirs(DECOMP_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

REWARD_MODES = ('ee_ratio', 'paper_xi', 'additive')
COLORS = {
    'ee_ratio': '#1f77b4',
    'paper_xi': '#2ca02c',
    'additive': '#ff7f0e',
}
LABELS = {
    'ee_ratio': 'Historical EE Proxy',
    'paper_xi': 'Paper Xi (RBS data / energy)',
    'additive': 'Additive (data - energy)',
}
ALGO_LABELS = {
    'maddpg': 'MADDPG',
    'nodyna': 'Hierarchical (NoDyna)',
    'dyna': 'Hierarchical (Dyna-Q)',
}


# ---------------------------------------------------------------------------
def run_experiment(algo: str, seed: int, reward_mode: str, episodes: int, case: int = 1) -> dict:
    """Run training with a specific reward mode and return metrics."""
    config = Config(seed=seed)
    config.reward_mode = reward_mode
    env = Environment(config)
    is_hier = algo in ('dyna', 'nodyna')
    state_dim, action_dim = get_state_action_dims(config)

    if algo == 'maddpg':
        agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'dyna':
        agent = HierarchicalAgent(state_dim, action_dim, config.N, config, dyna_k=config.dyna_k)
    elif algo == 'nodyna':
        agent = HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config)
    else:
        raise ValueError(f"Unsupported algo for reward decomp: {algo}")

    rewards_history = []
    metrics_accum = defaultdict(float)
    reward_ee_steps = []
    paper_xi_steps = []
    episode_records = []

    start = time.time()
    for ep in range(episodes):
        states = env.reset(case)
        ep_reward = 0.0
        ep_metrics = defaultdict(float)
        ep_reward_ee_steps = []
        ep_paper_xi_steps = []

        while True:
            if is_hier:
                upper_actions = agent.upper_act(states)
                lower_actions = agent.lower_act(states)
                actions = compose_full_actions(upper_actions, lower_actions, config.N)
            else:
                actions = agent.act(states)

            next_states, rewards, done = env.step(actions)

            step_info = env.last_step_info or {}
            ep_metrics['scheduled_actions'] += sum(
                float(action[4] >= 0.5) for action in actions
            )
            ep_metrics['action_decisions'] += len(actions)

            if is_hier:
                _, executed_upper_actions = env.get_last_upper_actions()
                lower_rewards = extract_lower_rewards(step_info, rewards, config.N)
                agent.add_upper_memory(
                    states, upper_actions, rewards, next_states, done,
                    executed_actions=executed_upper_actions)
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
            for key in [
                'data_received', 'data_sent_to_rbs', 'energy_consumed',
                'collision_events', 'collision_penalty', 'flight_energy',
                'upper_reward', 'lower_reward',
            ]:
                val = step_info.get('totals', {}).get(key, 0.0)
                value = float(val) if val is not None else 0.0
                metrics_accum[key] += value
                ep_metrics[key] += value

            step_totals = step_info.get('totals', {})
            step_energy = (
                float(step_totals.get('energy_consumed', 0.0))
                + float(step_totals.get('flight_energy', 0.0))
            )
            step_reward_numerator = (
                float(step_totals.get('data_received', 0.0))
                + config.gamma_forward
                * float(step_totals.get('data_sent_to_rbs', 0.0))
            )
            step_reward_ee = (
                step_reward_numerator
                / max(step_energy, config.denom_epsilon)
            )
            step_paper_xi = (
                float(step_totals.get('data_sent_to_rbs', 0.0))
                / max(step_energy, config.denom_epsilon)
            )
            reward_ee_steps.append(step_reward_ee)
            paper_xi_steps.append(step_paper_xi)
            ep_reward_ee_steps.append(step_reward_ee)
            ep_paper_xi_steps.append(step_paper_xi)

            states = next_states
            if done:
                break

        rewards_history.append(ep_reward)
        ep_total_energy = (
            ep_metrics['energy_consumed'] + ep_metrics['flight_energy']
        )
        episode_records.append({
            'episode': ep,
            'reward': float(ep_reward),
            'reward_ee_proxy': float(np.mean(ep_reward_ee_steps)),
            'paper_xi': float(np.mean(ep_paper_xi_steps)),
            'paper_xi_ratio_of_sums': float(
                ep_metrics['data_sent_to_rbs']
                / max(ep_total_energy, config.denom_epsilon)
            ),
            'data_received': float(ep_metrics['data_received']),
            'data_sent_to_rbs': float(ep_metrics['data_sent_to_rbs']),
            'forwarding_ratio': float(
                ep_metrics['data_sent_to_rbs']
                / max(ep_metrics['data_received'], config.denom_epsilon)
            ),
            'total_energy': float(ep_total_energy),
            'collision_events': float(ep_metrics['collision_events']),
            'scheduling_rate': float(
                ep_metrics['scheduled_actions']
                / max(ep_metrics['action_decisions'], 1.0)
            ),
            'terminal_uav_buffer': float(sum(uav.buffer for uav in env.uavs)),
        })
        agent.step_episode_schedulers()
        if is_hier:
            decay_epsilon(agent)

    duration = time.time() - start
    total_energy = metrics_accum['energy_consumed'] + metrics_accum['flight_energy']
    # The environment reward uses received + weighted forwarded data, whereas
    # the paper objective Xi uses data delivered to the RBS. Keep both explicit.
    reward_ee_proxy = float(np.mean(reward_ee_steps)) if reward_ee_steps else 0.0
    paper_xi = float(np.mean(paper_xi_steps)) if paper_xi_steps else 0.0
    paper_xi_ratio_of_sums = (
        metrics_accum['data_sent_to_rbs']
        / max(total_energy, config.denom_epsilon)
    )
    episode_rewards = np.asarray([r['reward'] for r in episode_records])
    episode_xi = np.asarray([r['paper_xi'] for r in episode_records])
    reward_xi_correlation = None
    if np.std(episode_rewards) > 0 and np.std(episode_xi) > 0:
        reward_xi_correlation = float(
            np.corrcoef(episode_rewards, episode_xi)[0, 1]
        )
    forwarding_ratio = float(
        metrics_accum['data_sent_to_rbs']
        / max(metrics_accum['data_received'], config.denom_epsilon)
    )

    return {
        'algo': algo,
        'reward_mode': reward_mode,
        'seed': seed,
        'episodes': episodes,
        'rewards': np.array(rewards_history),
        'metrics': {k: v for k, v in metrics_accum.items()},
        'final_reward': float(np.mean(rewards_history[-50:])) if len(rewards_history) >= 50 else float(np.mean(rewards_history)),
        'reward_ee_proxy': float(reward_ee_proxy),
        'paper_xi': float(paper_xi),
        'paper_xi_ratio_of_sums': float(paper_xi_ratio_of_sums),
        'total_energy': float(total_energy),
        'episode_records': episode_records,
        'diagnostics': {
            'reward_xi_correlation': reward_xi_correlation,
            'forwarding_ratio': forwarding_ratio,
            'zero_reward_fraction': float(np.mean(episode_rewards == 0.0)),
            'mean_scheduling_rate': float(np.mean([
                r['scheduling_rate'] for r in episode_records
            ])),
            'mean_terminal_uav_buffer': float(np.mean([
                r['terminal_uav_buffer'] for r in episode_records
            ])),
        },
        'duration': duration,
    }


def plot_comparison(results: list):
    """Plot reward-objective comparison."""
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
        for mode in REWARD_MODES:
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
        ax.set_title(ALGO_LABELS[algo])
        ax.legend()
        ax.grid(True, alpha=0.25)

    # Rightmost: paper objective Xi (RBS throughput / total UAV energy)
    ax = axes[-1]
    x_pos = []
    bar_data = []
    bar_colors_list = []
    labels_list = []
    pos = 0
    for algo in sorted(groups.keys()):
        for mode in REWARD_MODES:
            runs = groups[algo].get(mode, [])
            if runs:
                xi_values = [r['paper_xi'] for r in runs]
                mean = np.mean(xi_values)
                x_pos.append(pos)
                bar_data.append(mean)
                bar_colors_list.append(COLORS[mode])
                labels_list.append(
                    f'{ALGO_LABELS[algo]}\n{LABELS[mode].split(chr(32))[1]}'
                )
                pos += 1

    ax.bar(x_pos, bar_data, color=bar_colors_list, alpha=0.85)
    ax.errorbar(
        x_pos,
        bar_data,
        yerr=[
            np.std(
                [r['paper_xi'] for r in groups[algo][mode]],
                ddof=1,
            ) if len(groups[algo][mode]) > 1 else 0.0
            for algo in sorted(groups)
            for mode in REWARD_MODES
            if groups[algo].get(mode)
        ],
        fmt='none',
        ecolor='black',
        capsize=5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, fontsize=8)
    ax.set_ylabel(r'Paper objective $\Xi$ (RBS data / total energy)')
    ax.set_title(r'System Energy Efficiency $\Xi$')
    ax.grid(True, alpha=0.25, axis='y')

    fig.suptitle('Reward Objective Comparison', fontsize=14)
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
    parser.add_argument('--algos', type=str, default='maddpg,nodyna,dyna',
                        help='Algorithms to test')
    parser.add_argument('--modes', type=str, default='ee_ratio,additive',
                        help='Reward modes to compare')
    parser.add_argument('--case', type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    algos = [a.strip() for a in args.algos.split(',')]
    modes = [m.strip() for m in args.modes.split(',')]
    invalid_modes = sorted(set(modes) - set(REWARD_MODES))
    if invalid_modes:
        parser.error(
            f"unsupported reward mode(s): {', '.join(invalid_modes)}; "
            f"choose from {', '.join(REWARD_MODES)}"
        )

    total = len(algos) * len(seeds) * len(modes)
    print(f"Reward Decomposition Experiment: {total} runs ({len(algos)} algos × {len(seeds)} seeds × {len(modes)} modes)")

    results = []
    for a in algos:
        for s in seeds:
            for m in modes:
                label = f"{a}/{m}/seed{s}"
                print(f"  [{len(results)+1}/{total}] {label} ...", end=' ', flush=True)
                r = run_experiment(a, s, m, args.eps, args.case)
                print(
                    f"reward={r['final_reward']:.2f}, "
                    f"reward_EE_proxy={r['reward_ee_proxy']:.4f}, "
                    f"paper_Xi={r['paper_xi']:.4f}, "
                    f"corr={r['diagnostics']['reward_xi_correlation']}, "
                    f"forward={r['diagnostics']['forwarding_ratio']:.4f}, "
                    f"buffer={r['diagnostics']['mean_terminal_uav_buffer']:.2f}"
                )
                results.append(r)

    plot_comparison(results)
    print(f"\nResults saved to: {DECOMP_DIR}")


if __name__ == '__main__':
    main()
