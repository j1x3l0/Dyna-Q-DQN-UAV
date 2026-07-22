"""Unified benchmark training script for all four algorithms.

Supports:
  - iDDPG, MADDPG, Hierarchical (NoDyna), Hierarchical (Dyna-Q)
  - Per-algorithm episode caps and early stopping
  - Periodic checkpoint saving
  - Multi-seed statistical validation
  - Comprehensive metrics and comparison reports

Usage:
  python scripts/run_full_benchmark.py                    # default: 3 seeds, all algorithms
  python scripts/run_full_benchmark.py --algo maddpg       # single algorithm
  python scripts/run_full_benchmark.py --seeds 42          # single seed, quick test
  python scripts/run_full_benchmark.py --algo dyna --seeds 42,123,2026 --episodes 10000
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from system_model import Config, Environment
from maddpg_agent import MADDPGAgent
from hierarchical_agent import HierarchicalAgent, HierarchicalNoDynaAgent
from iddpg_agent import iDDPGAgent
from cop_maddpg_agent import CoPMADDPGAgent
from matd3_agent import MATD3Agent
from training_utils import (
    get_state_action_dims,
    decay_epsilon,
    compose_full_actions,
    extract_lower_rewards,
    setup_training_logger,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Algorithm configurations
# ---------------------------------------------------------------------------
ALGO_CONFIGS = {
    'iddpg': {
        'name': 'iDDPG',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
    'maddpg': {
        'name': 'MADDPG',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
    'nodyna': {
        'name': 'Hierarchical (NoDyna)',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
    'dyna': {
        'name': 'Hierarchical (Dyna-Q)',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
    'cop_maddpg': {
        'name': 'CoP-MADDPG',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
    'matd3': {
        'name': 'MATD3',
        'max_episodes': 8000,
        'early_stop_patience': 1500,
        'early_stop_min_delta': 0.01,
        'convergence_window': 500,
        'checkpoint_every': 500,
        'use_degradation_detection': False,
    },
}

DEFAULT_SEEDS = (42, 123, 2026, 7, 2023)
DEFAULT_CASE = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    algo: str
    seed: int
    episodes_completed: int
    stopped_early: bool
    stop_reason: str
    rewards: np.ndarray
    metrics: dict
    duration: float
    checkpoint_episodes: List[int] = field(default_factory=list)


class EarlyStoppingTracker:
    """Track convergence and degradation for early stopping decisions."""

    def __init__(self, patience: int, min_delta: float, conv_window: int = 500,
                 use_degradation: bool = False, deg_window: int = 1000,
                 deg_threshold: float = 0.30):
        self.patience = patience
        self.min_delta = min_delta
        self.conv_window = conv_window
        self.use_degradation = use_degradation
        self.deg_window = deg_window
        self.deg_threshold = deg_threshold

        self.best_rolling_avg = -float('inf')
        self.last_best_episode = 0
        self.peak_rolling_avg = -float('inf')
        self.peak_episode = 0

    def check(self, episode: int, rewards: np.ndarray) -> Tuple[bool, str]:
        """Return (should_stop, reason)."""
        window = min(self.conv_window, len(rewards))
        if window < 10:
            return False, ''

        rolling_avg = float(np.mean(rewards[-window:]))

        # Convergence: track best rolling average by episode
        improvement = (rolling_avg - self.best_rolling_avg) / (abs(self.best_rolling_avg) + 1e-8)
        if improvement > self.min_delta or self.best_rolling_avg == -float('inf'):
            self.best_rolling_avg = rolling_avg
            self.last_best_episode = episode

        # Degradation detection (MADDPG-specific)
        if self.use_degradation:
            if rolling_avg > self.peak_rolling_avg:
                self.peak_rolling_avg = rolling_avg
                self.peak_episode = episode

            if self.peak_rolling_avg > 0:
                degradation = (self.peak_rolling_avg - rolling_avg) / (abs(self.peak_rolling_avg) + 1e-8)
                if degradation > self.deg_threshold and episode - self.peak_episode > self.deg_window:
                    return True, f'degradation: rolling_avg dropped {degradation*100:.1f}% from peak (ep {self.peak_episode})'

        # Patience exceeded
        episodes_since_best = episode - self.last_best_episode
        if episodes_since_best >= self.patience:
            return True, f'convergence: no improvement ({self.min_delta*100:.1f}%) in last {episodes_since_best} episodes'

        return False, ''


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def create_agent(algo: str, state_dim: int, action_dim: int, config: Config):
    """Create the appropriate agent for the given algorithm key."""
    if algo == 'iddpg':
        return iDDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'maddpg':
        return MADDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'nodyna':
        return HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config)
    elif algo == 'dyna':
        return HierarchicalAgent(state_dim, action_dim, config.N, config)
    elif algo == 'cop_maddpg':
        return CoPMADDPGAgent(state_dim, action_dim, config.N, config)
    elif algo == 'matd3':
        return MATD3Agent(state_dim, action_dim, config.N, config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def is_hierarchical(algo: str) -> bool:
    return algo in ('nodyna', 'dyna')


def needs_epsilon_decay(algo: str) -> bool:
    return algo in ('nodyna', 'dyna')


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_single_experiment(algo: str, seed: int, config_override: dict = None,
                          case: int = DEFAULT_CASE) -> RunResult:
    """Train one algorithm for one seed with early stopping and checkpointing.

    Args:
        algo: 'iddpg' | 'maddpg' | 'nodyna' | 'dyna'
        seed: random seed
        config_override: optional dict of Config attribute overrides (e.g. {'dyna_k': 3})
        case: environment initialisation case (1 or 2)

    Returns:
        RunResult with full metrics
    """
    algo_cfg = ALGO_CONFIGS[algo]
    logger, _ = setup_training_logger(f'{algo}_seed{seed}')

    # Setup config
    config = Config(seed=seed)
    if config_override:
        for key, val in config_override.items():
            setattr(config, key, val)
    env = Environment(config)

    state_dim, action_dim = get_state_action_dims(config)
    agent = create_agent(algo, state_dim, action_dim, config)

    # Early stopping tracker
    stopper = EarlyStoppingTracker(
        patience=algo_cfg['early_stop_patience'],
        min_delta=algo_cfg['early_stop_min_delta'],
        conv_window=algo_cfg['convergence_window'],
        use_degradation=algo_cfg['use_degradation_detection'],
        deg_window=algo_cfg.get('degradation_window', 1000),
        deg_threshold=algo_cfg.get('degradation_threshold', 0.30),
    )

    # Metrics accumulators
    rewards_history = []
    totals = {
        'collision_events': 0.0, 'collision_penalty': 0.0,
        'data_received': 0.0, 'data_sent_to_rbs': 0.0,
        'energy_consumed': 0.0, 'flight_energy': 0.0,
        'harvested_energy': 0.0,
    }
    checkpoint_episodes = []
    stopped_early = False
    stop_reason = f'reached max_episodes ({algo_cfg["max_episodes"]})'
    best_checkpoint = None
    best_rolling_avg = -float('inf')

    max_eps = algo_cfg['max_episodes']
    ckpt_every = algo_cfg['checkpoint_every']

    logger.info(f"Starting {algo_cfg['name']} | seed={seed} | max_episodes={max_eps} | "
                f"early_stop_patience={algo_cfg['early_stop_patience']}")

    start_time = time.time()

    for episode in range(max_eps):
        states = env.reset(case)
        episode_reward = 0.0

        while True:
            if is_hierarchical(algo):
                # Hierarchical action selection
                upper_actions = agent.upper_act(states)
                lower_actions = agent.lower_act(states)
                full_actions = compose_full_actions(upper_actions, lower_actions, config.N)
                next_states, rewards, done = env.step(full_actions)

                step_info = env.last_step_info or {}
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
                # Flat action selection (iDDPG / MADDPG)
                actions = agent.act(states)
                next_states, rewards, done = env.step(actions)
                step_info = env.last_step_info or {}

                agent.add_memory(states, actions, rewards, next_states, done)
                agent.update()

            episode_reward += float(np.sum(rewards))
            for key in totals:
                val = step_info.get('totals', {}).get(key, 0.0)
                totals[key] += float(val) if val is not None else 0.0

            states = next_states
            if done:
                break

        rewards_history.append(episode_reward)
        agent.step_episode_schedulers()

        if needs_epsilon_decay(algo):
            decay_epsilon(agent)

        # Early stopping check (start checking after convergence_window * 2)
        if episode >= algo_cfg['convergence_window'] * 2:
            should_stop, reason = stopper.check(episode, np.array(rewards_history))
            if should_stop:
                stopped_early = True
                stop_reason = reason
                logger.info(f"Early stop at episode {episode}: {reason}")
                break

        # Periodic checkpoint
        if (episode + 1) % ckpt_every == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'{algo}_seed{seed}_ep{episode+1}.pt')
            agent.save_checkpoint(ckpt_path, episode + 1)
            checkpoint_episodes.append(episode + 1)

            # Track best checkpoint
            window = min(algo_cfg['convergence_window'], len(rewards_history))
            rolling_avg = float(np.mean(rewards_history[-window:]))
            if rolling_avg > best_rolling_avg:
                best_rolling_avg = rolling_avg
                best_checkpoint = True
                # Save a "best" copy
                best_path = os.path.join(CKPT_DIR, f'{algo}_seed{seed}_best.pt')
                agent.save_checkpoint(best_path, episode + 1)

        # Periodic logging
        if episode % 100 == 0 or (episode < 50 and episode % 10 == 0):
            window = min(50, len(rewards_history))
            avg_r = np.mean(rewards_history[-window:]) if window > 0 else 0
            epsilon_str = f", eps={agent.epsilon:.4f}" if needs_epsilon_decay(algo) else ""
            logger.info(f"Ep {episode:5d} | avg_r(last_{window})={avg_r:10.2f} | "
                        f"best_ra={stopper.best_rolling_avg:10.2f}{epsilon_str}")

    duration = time.time() - start_time

    # Save final checkpoint
    if not stopped_early or episode not in checkpoint_episodes:
        final_ckpt = os.path.join(CKPT_DIR, f'{algo}_seed{seed}_final_ep{episode+1}.pt')
        agent.save_checkpoint(final_ckpt, episode + 1)

    logger.info(f"{algo_cfg['name']} seed={seed} completed: {episode+1} episodes, "
                f"duration={duration:.1f}s, stopped_early={stopped_early}, reason={stop_reason}")

    return RunResult(
        algo=algo,
        seed=seed,
        episodes_completed=episode + 1,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        rewards=np.array(rewards_history, dtype=float),
        metrics={k: float(v) for k, v in totals.items()},
        duration=duration,
        checkpoint_episodes=checkpoint_episodes,
    )


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------
def compute_convergence_episode(rewards: np.ndarray, window: int = 100, threshold: float = 0.9) -> float:
    """Return first episode where rolling avg reaches threshold * best_rolling_avg.

    Measures sample efficiency: how many real episodes needed to 'mostly converge'.
    'window' and 'threshold' should match EarlyStoppingTracker conventions.
    Returns np.nan if not reached.
    """
    if len(rewards) < window:
        return float(len(rewards))
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    best = float(np.max(rolling))
    target = threshold * best
    for i, val in enumerate(rolling):
        if val >= target:
            return float(i + window)
    return float('nan')


def summarize(values: list) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 0 else 0.0
    return mean, std, ci95


def aggregate_results(run_results: List[RunResult]) -> dict:
    """Aggregate multi-seed results per algorithm."""
    algos = sorted(set(r.algo for r in run_results))
    rows = []
    for algo in algos:
        algo_runs = [r for r in run_results if r.algo == algo]
        min_eps = min(r.episodes_completed for r in algo_runs)
        # Align all runs to the same episode count for comparison
        aligned_rewards = [r.rewards[:min_eps] for r in algo_runs]
        final_rewards = [float(np.mean(r[-50:])) if len(r) >= 50 else float(np.mean(r))
                        for r in aligned_rewards]

        conv_episodes = [compute_convergence_episode(r.rewards) for r in algo_runs]
        conv_valid = [c for c in conv_episodes if not np.isnan(c)]
        early_rewards = [
            float(np.mean(r.rewards[:min(500, len(r.rewards))])) if len(r.rewards) > 0 else 0.0
            for r in algo_runs
        ]

        rows.append({
            'algo': algo,
            'algo_name': ALGO_CONFIGS[algo]['name'],
            'num_runs': len(algo_runs),
            'min_episodes': min_eps,
            'stopped_early_count': sum(1 for r in algo_runs if r.stopped_early),
            'stop_reasons': [r.stop_reason for r in algo_runs],
            'final_reward': summarize(final_rewards),
            'conv_episode': summarize(conv_valid) if conv_valid else (float('nan'), 0.0, 0.0),
            'early_reward_500': summarize(early_rewards),
            'collision_events_avg': summarize([r.metrics['collision_events'] / r.episodes_completed for r in algo_runs]),
            'data_received_avg': summarize([r.metrics['data_received'] / r.episodes_completed for r in algo_runs]),
            'data_sent_avg': summarize([r.metrics['data_sent_to_rbs'] / r.episodes_completed for r in algo_runs]),
            'energy_avg': summarize([r.metrics['energy_consumed'] / r.episodes_completed for r in algo_runs]),
            'duration_avg': summarize([r.duration for r in algo_runs]),
            'reward_curve_mean': np.mean(aligned_rewards, axis=0),
            'reward_curve_std': np.std(aligned_rewards, axis=0, ddof=1) if len(algo_runs) > 1 else np.zeros(min_eps),
        })
    return rows


def write_report(run_results: List[RunResult], summary_rows: list,
                 timestamp: str = None) -> Tuple[str, str]:
    """Generate text and JSON reports."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    report_path = os.path.join(RESULTS_DIR, f'benchmark_report_{timestamp}.txt')
    json_path = os.path.join(RESULTS_DIR, f'benchmark_report_{timestamp}.json')

    # ---- Text report ----
    lines = []
    lines.append('=' * 80)
    lines.append('UAV DRL Full Benchmark Report')
    lines.append('=' * 80)
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Case: {DEFAULT_CASE}')
    lines.append('')

    lines.append('Algorithm configurations:')
    for algo_key, cfg in ALGO_CONFIGS.items():
        lines.append(f"  {cfg['name']}: max_eps={cfg['max_episodes']}, "
                     f"patience={cfg['early_stop_patience']}, "
                     f"degradation={cfg['use_degradation_detection']}")
    lines.append('')

    # Per-run summary
    lines.append('=' * 80)
    lines.append('Per-Run Results')
    lines.append('=' * 80)
    header = f"{'Algo':<8} {'Seed':>6} {'Eps':>6} {'EarlyStop':>10} {'FinalReward':>14} {'Duration(s)':>12} {'Reason'}"
    lines.append(header)
    lines.append('-' * len(header))
    for r in sorted(run_results, key=lambda x: (x.algo, x.seed)):
        final = float(np.mean(r.rewards[-50:])) if len(r.rewards) >= 50 else float(np.mean(r.rewards))
        lines.append(f"{r.algo:<8} {r.seed:>6} {r.episodes_completed:>6} "
                     f"{str(r.stopped_early):>10} {final:>14.2f} {r.duration:>12.1f} "
                     f"{r.stop_reason}")
    lines.append('')

    # Aggregated summary
    lines.append('=' * 80)
    lines.append('Aggregated Results (mean +- std over seeds)')
    lines.append('=' * 80)
    for row in summary_rows:
        lines.append(f"\n--- {row['algo_name']} ---")
        lines.append(f"  Runs: {row['num_runs']}, Early stopped: {row['stopped_early_count']}")
        lines.append(f"  Min episodes (aligned): {row['min_episodes']}")
        mean, std, ci95 = row['final_reward']
        lines.append(f"  Final reward (last 50): {mean:.2f} +- {std:.2f} (95% CI: +-{ci95:.2f})")
        mean, std, _ = row['collision_events_avg']
        lines.append(f"  Collision events/ep: {mean:.2f} +- {std:.2f}")
        mean, std, _ = row['data_received_avg']
        lines.append(f"  Data received/ep: {mean:.2f} +- {std:.2f}")
        mean, std, _ = row['data_sent_avg']
        lines.append(f"  Data sent to RBS/ep: {mean:.2f} +- {std:.2f}")
        mean, std, _ = row['energy_avg']
        lines.append(f"  Energy consumed/ep: {mean:.2f} +- {std:.2f}")
        mean, std, _ = row['duration_avg']
        lines.append(f"  Duration (s): {mean:.1f} +- {std:.1f}")
        conv_mean, conv_std, _ = row['conv_episode']
        conv_str = f"{conv_mean:.0f} +- {conv_std:.0f}" if not np.isnan(conv_mean) else "N/A"
        lines.append(f"  Convergence (eps to 90% peak): {conv_str}")
        early_mean, early_std, _ = row['early_reward_500']
        lines.append(f"  Early reward (first 500 eps): {early_mean:.2f} +- {early_std:.2f}")

    lines.append('')
    lines.append('=' * 80)
    lines.append('Cross-Algorithm Comparison')
    lines.append('=' * 80)

    # Pairwise improvements (final reward)
    if len(summary_rows) >= 2:
        lines.append('\n--- Final Reward Comparison ---')
        for i in range(len(summary_rows)):
            for j in range(i + 1, len(summary_rows)):
                a_name = summary_rows[i]['algo_name']
                b_name = summary_rows[j]['algo_name']
                a_mean = summary_rows[i]['final_reward'][0]
                b_mean = summary_rows[j]['final_reward'][0]
                better = a_name if a_mean > b_mean else b_name
                diff = abs(a_mean - b_mean)
                base = max(abs(a_mean), abs(b_mean), 1e-8)
                pct = diff / base * 100
                lines.append(f"  {a_name} vs {b_name}: {better} better by {diff:.2f} ({pct:.1f}%)")

        # Sample efficiency comparison (convergence speed)
        lines.append('\n--- Sample Efficiency (Convergence Speed) ---')
        for i in range(len(summary_rows)):
            for j in range(i + 1, len(summary_rows)):
                a_name = summary_rows[i]['algo_name']
                b_name = summary_rows[j]['algo_name']
                a_conv, _, _ = summary_rows[i]['conv_episode']
                b_conv, _, _ = summary_rows[j]['conv_episode']
                if not np.isnan(a_conv) and not np.isnan(b_conv) and a_conv > 0 and b_conv > 0:
                    faster_name = a_name if a_conv < b_conv else b_name
                    slower_name = b_name if faster_name == a_name else a_name
                    speedup = max(a_conv, b_conv) / min(a_conv, b_conv)
                    lines.append(f"  {a_name} ({a_conv:.0f} eps) vs {b_name} ({b_conv:.0f} eps): "
                                 f"{faster_name} converges {speedup:.2f}x faster than {slower_name}")

        # Early reward comparison (first 500 eps)
        lines.append('\n--- Early Reward (first 500 eps) ---')
        for i in range(len(summary_rows)):
            for j in range(i + 1, len(summary_rows)):
                a_name = summary_rows[i]['algo_name']
                b_name = summary_rows[j]['algo_name']
                a_early, _, _ = summary_rows[i]['early_reward_500']
                b_early, _, _ = summary_rows[j]['early_reward_500']
                better = a_name if a_early > b_early else b_name
                diff = abs(a_early - b_early)
                lines.append(f"  {a_name} ({a_early:.2f}) vs {b_name} ({b_early:.2f}): "
                             f"{better} better by {diff:.2f}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # ---- JSON report ----
    serializable = {
        'generated_at': datetime.now().isoformat(),
        'case': DEFAULT_CASE,
        'algo_configs': ALGO_CONFIGS,
        'runs': [
            {
                'algo': r.algo, 'seed': r.seed,
                'episodes_completed': r.episodes_completed,
                'stopped_early': r.stopped_early,
                'stop_reason': r.stop_reason,
                'duration': r.duration,
                'metrics': r.metrics,
                'rewards': r.rewards.tolist(),
            }
            for r in run_results
        ],
        'summary': [
            {k: (v if k not in ('reward_curve_mean', 'reward_curve_std')
                  else v.tolist())
             for k, v in row.items()}
            for row in summary_rows
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return report_path, json_path


def plot_comparison(summary_rows: list, timestamp: str = None) -> str:
    """Generate multi-panel comparison plot."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    n_algos = len(summary_rows)
    colors = {'iddpg': '#d62728', 'maddpg': '#ff7f0e',
              'nodyna': '#2ca02c', 'dyna': '#1f77b4'}

    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    for ax, row in zip(axes, summary_rows):
        algo = row['algo']
        curve = row['reward_curve_mean']
        std = row['reward_curve_std']
        ci95 = 1.96 * std / math.sqrt(row['num_runs'])
        x = np.arange(len(curve))

        color = colors.get(algo, '#333333')
        ax.plot(x, curve, color=color, linewidth=1.6, label=row['algo_name'])
        ax.fill_between(x, curve - ci95, curve + ci95, color=color, alpha=0.15)
        ax.set_title(f"{row['algo_name']}\n({row['num_runs']} seeds, max {len(curve)} eps)")
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel('Episode Reward')
    fig.suptitle('UAV DRL Algorithm Benchmark — Mean Reward with 95% CI', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(RESULTS_DIR, f'benchmark_comparison_{timestamp}.png')
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # ---- Overlay plot ----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for row in summary_rows:
        algo = row['algo']
        curve = row['reward_curve_mean']
        x = np.arange(len(curve))
        color = colors.get(algo, '#333333')
        ax2.plot(x, curve, color=color, linewidth=1.8, label=row['algo_name'], alpha=0.85)

    ax2.set_title('Algorithm Comparison — Mean Reward Overlay', fontsize=14)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()

    overlay_path = os.path.join(RESULTS_DIR, f'benchmark_overlay_{timestamp}.png')
    fig2.savefig(overlay_path, dpi=150)
    plt.close(fig2)

    # ---- Sample efficiency overlay (x-axis = episodes, annotations for conv speed) ----
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for row in summary_rows:
        algo = row['algo']
        curve = row['reward_curve_mean']
        x = np.arange(len(curve))
        color = colors.get(algo, '#333333')
        ax3.plot(x, curve, color=color, linewidth=1.8, label=row['algo_name'], alpha=0.85)
        conv_mean, conv_std, _ = row['conv_episode']
        if not np.isnan(conv_mean) and conv_mean < len(curve):
            y_at_conv = curve[min(int(conv_mean), len(curve)-1)]
            ax3.axvline(x=conv_mean, color=color, linestyle='--', alpha=0.4, linewidth=1)
            ax3.annotate(f'{row["algo_name"]}\n{conv_mean:.0f} eps',
                         xy=(conv_mean, y_at_conv),
                         xytext=(conv_mean + len(curve)*0.02, y_at_conv),
                         fontsize=8, color=color,
                         arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))

    ax3.set_title('Sample Efficiency — Convergence Speed (90% peak)', fontsize=14)
    ax3.set_xlabel('Episode (real environment interactions)')
    ax3.set_ylabel('Episode Reward')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.25)
    fig3.tight_layout()
    sample_eff_path = os.path.join(RESULTS_DIR, f'sample_efficiency_{timestamp}.png')
    fig3.savefig(sample_eff_path, dpi=150)
    plt.close(fig3)

    return chart_path, overlay_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='UAV DRL Full Benchmark')
    parser.add_argument('--algos', type=str, default='iddpg,maddpg,nodyna,dyna',
                        help='Comma-separated algorithm keys (default: all four)')
    parser.add_argument('--seeds', type=str, default='42,123,2026',
                        help='Comma-separated seeds (default: 42,123,2026)')
    parser.add_argument('--case', type=int, default=DEFAULT_CASE,
                        help='Environment case (1 or 2)')
    parser.add_argument('--dyna-k', type=str, default=None,
                        help='Comma-separated Dyna-K values for sweep, e.g. "1,5,10,20" (default: from Config)')
    return parser.parse_args()


def main():
    args = parse_args()
    algos = [a.strip() for a in args.algos.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    dyna_k_values = [int(k) for k in args.dyna_k.split(',')] if args.dyna_k else [None]

    print(f"{'='*60}")
    print(f"UAV DRL Full Benchmark")
    print(f"{'='*60}")
    print(f"Algorithms: {[ALGO_CONFIGS[a]['name'] for a in algos]}")
    print(f"Seeds: {seeds}")
    print(f"Case: {args.case}")
    if args.dyna_k:
        print(f"Dyna-K sweep: {dyna_k_values}")
    print()

    total_runs = len(algos) * len(seeds) * len(dyna_k_values)
    run_results = []

    run_idx = 0
    for k in dyna_k_values:
        co = {}
        k_tag = ''
        if k is not None:
            co['dyna_k'] = k
            k_tag = f'k{k}'
        for algo in algos:
            for seed in seeds:
                run_idx += 1
                print(f"\n[Run {run_idx}/{total_runs}] {ALGO_CONFIGS[algo]['name']} seed={seed} {k_tag}")
                print(f"  Max episodes: {ALGO_CONFIGS[algo]['max_episodes']}")
                result = run_single_experiment(algo, seed, co, case=args.case)
                run_results.append(result)
                print(f"  Completed: {result.episodes_completed} eps, "
                      f"early_stop={result.stopped_early}, duration={result.duration:.1f}s")
                final = float(np.mean(result.rewards[-50:])) if len(result.rewards) >= 50 else float(np.mean(result.rewards))
                print(f"  Final reward (last 50): {final:.2f}")

    # Sanity check: verify data collection is functioning
    print(f"\n{'='*60}")
    print("Sanity check...")
    for r in run_results:
        eps = r.episodes_completed
        data_recv = r.metrics.get('data_received', 0.0)
        avg_data = data_recv / max(eps, 1)
        if avg_data < 1.0:
            print(f"  ⚠ WARNING: {r.algo} seed={r.seed}: avg data_received/ep={avg_data:.4f} "
                  f"(total={data_recv:.2f} over {eps} eps) — reward signal may be degenerate")
    print("Aggregating results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_rows = aggregate_results(run_results)

    report_path, json_path = write_report(run_results, summary_rows, timestamp)
    print(f"Report: {report_path}")
    print(f"JSON:   {json_path}")

    chart_path, overlay_path = plot_comparison(summary_rows, timestamp)
    print(f"Charts: {chart_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Sample efficiency: {os.path.join(RESULTS_DIR, f'sample_efficiency_{timestamp}.png')}")

    print(f"\n{'='*60}")
    print("Final Results Summary:")
    print(f"{'='*60}")
    for row in summary_rows:
        mean, std, ci95 = row['final_reward']
        print(f"  {row['algo_name']:30s}: {mean:10.2f} +- {std:.2f} "
              f"({row['min_episodes']} eps, {row['stopped_early_count']}/{row['num_runs']} early-stopped)")


if __name__ == '__main__':
    main()
