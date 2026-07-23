"""
Independent DDPG (iDDPG) Agent for UAV-Assisted Wireless Sensor Networks.

Each UAV learns independently using its own actor-critic pair. Unlike MADDPG,
the critic only sees that agent's own state and action — no centralized
training. This serves as the weakest baseline in the paper's three-way
comparison (iDDPG vs MADDPG vs HMADDPG).

Reuses Actor and Critic network classes from maddpg_agent.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
from logging.handlers import RotatingFileHandler

from maddpg_agent import Actor, Critic

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'iddpg_agent.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class iDDPGAgent:
    """Independent DDPG — each UAV learns from its own observations only.

    Key difference from MADDPGAgent:
      - Critic(input_dim) = state_dim + action_dim  (single-agent)
        vs MADDPG's Critic(state_dim*N + action_dim*N)  (centralized)
      - update() only uses agent i's own slice of each transition.
    """

    def __init__(self, state_dim, action_dim, num_agents, config):
        logger.info("=" * 60)
        logger.info("Initializing iDDPGAgent (Independent DDPG)...")
        logger.info("=" * 60)

        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        self.action_rng = config.rngs['action']
        self.replay_rng = config.rngs['replay']

        logger.info(f"iDDPG params: num_agents={num_agents}, state_dim={state_dim}, "
                     f"action_dim={action_dim}, device={self.device}")

        logger.info("Creating actor networks (per-agent)...")
        # M9: 前4维tanh，剩余维sigmoid
        self.actors = [Actor(state_dim, 4, action_dim - 4).to(self.device) for _ in range(num_agents)]

        logger.info("Creating critic networks (independent, per-agent)...")
        self.critics = [Critic(state_dim, action_dim).to(self.device) for _ in range(num_agents)]

        logger.info("Creating target networks...")
        self.target_actors = [Actor(state_dim, 4, action_dim - 4).to(self.device) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim, action_dim).to(self.device) for _ in range(num_agents)]

        logger.info("Copying weights to target networks...")
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        logger.info("Creating optimizers...")
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=1e-4) for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=1e-3) for i in range(num_agents)]

        logger.info("Creating learning rate schedulers...")
        self.actor_schedulers = [optim.lr_scheduler.StepLR(self.actor_optimizers[i], step_size=500, gamma=0.9)
                                 for i in range(num_agents)]
        self.critic_schedulers = [optim.lr_scheduler.StepLR(self.critic_optimizers[i], step_size=500, gamma=0.9)
                                  for i in range(num_agents)]

        logger.info("Initializing replay memory...")
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.clip_norm = 1.0
        # M7: 目标网络更新模式开关
        self.target_update_mode = getattr(config, 'target_update_mode', 'hard')
        self.hard_update_every = getattr(config, 'hard_update_every', 100)
        self._hard_counter = 0

        logger.info(f"Memory capacity: {self.memory.maxlen}, gamma={self.gamma}, tau={self.tau}, "
                     f"batch_size={self.batch_size}, clip_norm={self.clip_norm}")
        logger.info("=" * 60)
        logger.info("iDDPGAgent initialization complete!")
        logger.info("=" * 60)

    def act(self, states, noise=True):
        """Select actions for all agents independently."""
        logger.debug(f"act() called: states shape={states.shape}, noise={noise}")
        actions = []

        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.actors[i](state).detach().cpu().numpy()[0]

            if noise:
                noise_val = self.action_rng.normal(0, 0.1, size=action.shape)
                action += noise_val
                logger.debug(f"Agent {i} action with noise: noise_norm={np.linalg.norm(noise_val):.4f}")

            action[:4] = np.clip(action[:4], -1, 1)
            action[4:] = np.clip(action[4:], 0, 1)

            logger.debug(f"Agent {i} action: speed={action[3]:.4f}, scheduled={bool(action[-1])}")
            actions.append(action)

        actions_array = np.array(actions)
        logger.debug(f"act() completed: actions shape={actions_array.shape}")
        return actions_array

    def add_memory(self, states, actions, rewards, next_states, dones):
        """Store a transition in the shared replay buffer."""
        logger.debug(f"add_memory() called: states shape={states.shape}, rewards={rewards}, dones={dones}")
        self.memory.append((states, actions, rewards, next_states, dones))
        logger.debug(f"Memory size: {len(self.memory)}/{self.memory.maxlen}")

    def update(self):
        """Independent update — each agent learns from its own (s_i, a_i, r_i, s'_i)."""
        if len(self.memory) < self.batch_size:
            logger.debug(f"update() skipped: memory size {len(self.memory)} < batch size {self.batch_size}")
            return

        logger.debug(f"update() called: memory size={len(self.memory)}, batch_size={self.batch_size}")

        batch = self.replay_rng.choice(len(self.memory), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []

        for idx in batch:
            s, a, r, ns, d = self.memory[idx]
            states_batch.append(s)
            actions_batch.append(a)
            rewards_batch.append(r)
            next_states_batch.append(ns)
            dones_batch.append(d)

        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        dones_batch = torch.FloatTensor(np.array(dones_batch)).to(self.device)

        logger.debug(f"Batch tensors created: states={states_batch.shape}, actions={actions_batch.shape}")

        for i in range(self.num_agents):
            logger.debug(f"Updating agent {i} networks (independent)...")

            # Target Q: Q'(s'_i, actor'(s'_i))
            target_next_action = self.target_actors[i](next_states_batch[:, i])
            target_q = self.target_critics[i](next_states_batch[:, i], target_next_action)

            # TD target: r_i + gamma * Q' * (1 - done)
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))

            # Current Q: Q(s_i, a_i)
            q_i = self.critics[i](states_batch[:, i], actions_batch[:, i])

            # Critic loss
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            logger.debug(f"Agent {i} critic loss: {critic_loss.item():.6f}")

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.clip_norm)
            self.critic_optimizers[i].step()

            # Actor loss: -Q(s_i, actor(s_i))
            self.actor_optimizers[i].zero_grad()
            actor_loss = -self.critics[i](states_batch[:, i], self.actors[i](states_batch[:, i])).mean()
            logger.debug(f"Agent {i} actor loss: {actor_loss.item():.6f}")

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_norm)
            self.actor_optimizers[i].step()

            # Soft-update targets
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])

        logger.debug("update() completed successfully!")

    def step_episode_schedulers(self):
        for i in range(self.num_agents):
            self.actor_schedulers[i].step()
            self.critic_schedulers[i].step()
        if self.target_update_mode == 'hard':
            self._hard_counter += 1
            if self._hard_counter >= self.hard_update_every:
                self._hard_counter = 0
                for i in range(self.num_agents):
                    self.target_actors[i].load_state_dict(self.actors[i].state_dict())
                    self.target_critics[i].load_state_dict(self.critics[i].state_dict())
                logger.debug(f"iDDPG hard target sync at episode boundary (every {self.hard_update_every})")

    def soft_update(self, source, target):
        if getattr(self, 'target_update_mode', 'soft') == 'hard':
            return
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, filepath, episode):
        checkpoint = {
            'episode': episode,
            'actors': {i: self.actors[i].state_dict() for i in range(self.num_agents)},
            'target_actors': {i: self.target_actors[i].state_dict() for i in range(self.num_agents)},
            'critics': {i: self.critics[i].state_dict() for i in range(self.num_agents)},
            'target_critics': {i: self.target_critics[i].state_dict() for i in range(self.num_agents)},
            'actor_optimizers': {i: self.actor_optimizers[i].state_dict() for i in range(self.num_agents)},
            'critic_optimizers': {i: self.critic_optimizers[i].state_dict() for i in range(self.num_agents)},
            'actor_schedulers': {i: self.actor_schedulers[i].state_dict() for i in range(self.num_agents)},
            'critic_schedulers': {i: self.critic_schedulers[i].state_dict() for i in range(self.num_agents)},
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} (episode {episode})")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.target_actors[i].load_state_dict(checkpoint['target_actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.target_critics[i].load_state_dict(checkpoint['target_critics'][i])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizers'][i])
            self.actor_schedulers[i].load_state_dict(checkpoint['actor_schedulers'][i])
            self.critic_schedulers[i].load_state_dict(checkpoint['critic_schedulers'][i])
        logger.info(f"Checkpoint loaded from {filepath} (episode {checkpoint['episode']})")
        return checkpoint['episode']
