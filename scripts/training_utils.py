"""Shared training utilities for UAV DRL project.

Centralizes functions that were duplicated across training scripts:
- State/action dimension calculation
- Epsilon decay (fixes bug in train_hierarchical.py)
- Full action composition (upper + lower)
- Logging setup
- Result saving (rewards array + plot)
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

def get_state_action_dims(config):
    """Return (state_dim, action_dim) for the given Config."""
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    return state_dim, action_dim


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def compose_full_actions(upper_actions, lower_actions, N):
    """Concatenate upper and lower actions into full action vectors.

    Args:
        upper_actions: (N, 5) array from upper_act()
        lower_actions: (N, 2*M) array from lower_act()
        N: number of agents

    Returns:
        (N, 5 + 2*M) array
    """
    full_actions = []
    for i in range(N):
        full_action = np.concatenate([upper_actions[i], lower_actions[i]])
        full_actions.append(full_action)
    return np.array(full_actions)


def extract_lower_rewards(step_info, rewards, N):
    """Extract per-agent lower rewards from environment step_info.

    Args:
        step_info: dict from env.last_step_info
        rewards: upper rewards array (N,)
        N: number of agents

    Returns:
        (N,) array of lower rewards
    """
    per_agent_info = step_info.get('per_agent', [])
    lower_rewards = np.array([
        per_agent_info[i].get('lower_reward', float(rewards[i]))
        if i < len(per_agent_info) else float(rewards[i])
        for i in range(N)
    ], dtype=float)
    return lower_rewards


# ---------------------------------------------------------------------------
# Epsilon helpers
# ---------------------------------------------------------------------------

def decay_epsilon(agent):
    """Decay epsilon for agents that support epsilon-greedy exploration.

    Call once per episode. Handles HierarchicalAgent, HierarchicalNoDynaAgent.
    Safe no-op for MADDPGAgent and iDDPGAgent (which lack epsilon attributes).
    """
    if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay') and hasattr(agent, 'epsilon_min'):
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_training_logger(name, log_dir=None):
    """Create a logger with rotating file + stream handlers.

    Args:
        name: logger name (e.g. 'maddpg_training')
        log_dir: directory for log files (default: ../logs)

    Returns:
        (logger, timestamp_string)
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers
    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, timestamp


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_rewards(rewards_history, filename, title='Training Rewards', results_dir=None):
    """Save rewards history as .npy and plot as .png.

    Args:
        rewards_history: list/array of episode rewards
        filename: base filename (without extension), e.g. 'maddpg_case1'
        title: plot title
        results_dir: directory for results (default: ../results)

    Returns:
        (npy_path, png_path)
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    npy_path = os.path.join(results_dir, f'{filename}.npy')
    png_path = os.path.join(results_dir, f'{filename}.png')

    np.save(npy_path, rewards_history)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(png_path, dpi=150)
    plt.close()

    return npy_path, png_path
