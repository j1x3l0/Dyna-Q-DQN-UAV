"""
MATD3: Twin-Delayed Multi-Agent Deep Deterministic Policy Gradient.

Extends MADDPG with TD3 improvements:
  1. Twin critics (two Q-networks per agent, take minimum to reduce overestimation)
  2. Delayed policy updates (update actor every d steps)
  3. Target policy smoothing (add clipped noise to target actions)

Usage: same interface as MADDPGAgent — drop-in replacement.

Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (TD3, 2018)
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'matd3_agent.log'),
    maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


class Critic(nn.Module):
    """Single critic network (used as one of the twin pair)."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MATD3Agent:
    """Multi-Agent Twin Delayed DDPG (MATD3).

    Key hyperparameters (from TD3 paper):
      - policy_delay: 2 (update actor every d critic updates)
      - target_noise: 0.2 (std of noise added to target action)
      - noise_clip: 0.5 (clip range for target noise)
    """

    def __init__(self, state_dim, action_dim, num_agents, config):
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

        # TD3-specific parameters
        self.policy_delay = 2
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self._update_counter = 0

        # Actors
        self.actors = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]

        # Twin critics per agent (Q1, Q2)
        self.critics_A = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                          for _ in range(num_agents)]
        self.critics_B = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                          for _ in range(num_agents)]
        self.target_critics_A = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                                 for _ in range(num_agents)]
        self.target_critics_B = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                                 for _ in range(num_agents)]

        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics_A[i].load_state_dict(self.critics_A[i].state_dict())
            self.target_critics_B[i].load_state_dict(self.critics_B[i].state_dict())

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=1e-3)
                                 for i in range(num_agents)]
        self.critic_A_optimizers = [optim.Adam(self.critics_A[i].parameters(), lr=1e-4)
                                    for i in range(num_agents)]
        self.critic_B_optimizers = [optim.Adam(self.critics_B[i].parameters(), lr=1e-4)
                                    for i in range(num_agents)]
        self.actor_schedulers = [optim.lr_scheduler.StepLR(self.actor_optimizers[i], step_size=500, gamma=0.9)
                                 for i in range(num_agents)]

        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.tau = 0.005  # Slower soft update (TD3 uses slower tau)
        self.batch_size = 32
        self.clip_norm = 1.0

        logger.info(f"MATD3 initialized: policy_delay={self.policy_delay}, "
                    f"target_noise={self.target_noise}, noise_clip={self.noise_clip}, tau={self.tau}")

    def act(self, states, noise=True):
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.actors[i](state).detach().cpu().numpy()[0]
            if noise:
                action += self.action_rng.normal(0, 0.1, size=action.shape)
            action[:4] = np.clip(action[:4], -1, 1)
            action[4:] = np.clip(action[4:], 0, 1)
            actions.append(action)
        return np.array(actions)

    def add_memory(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def _compute_target(self, next_states_batch, rewards_batch, dones_batch, agent_idx):
        """Compute TD target using twin critics and target smoothing."""
        with torch.no_grad():
            # Target actions with smoothing noise
            next_actions = []
            for j in range(self.num_agents):
                ns_j = next_states_batch[:, j]
                act = self.target_actors[j](ns_j)
                noise = torch.randn_like(act) * self.target_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                act = torch.clamp(act + noise, -1, 1)
                next_actions.append(act)
            next_actions = torch.cat(next_actions, dim=1)

            ns_cat = next_states_batch.view(self.batch_size, -1)
            q1 = self.target_critics_A[agent_idx](ns_cat, next_actions)
            q2 = self.target_critics_B[agent_idx](ns_cat, next_actions)
            min_q = torch.min(q1, q2)  # Twin critic: take minimum
            return rewards_batch[:, agent_idx].unsqueeze(1) + \
                self.gamma * min_q * (1 - dones_batch.unsqueeze(1))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self._update_counter += 1

        batch = self.replay_rng.choice(len(self.memory), self.batch_size, replace=False)
        s, a, r, ns, d = [], [], [], [], []
        for idx in batch:
            st, ac, rw, nst, dn = self.memory[idx]
            s.append(st); a.append(ac); r.append(rw); ns.append(nst); d.append(dn)

        states_b = torch.FloatTensor(np.array(s)).to(self.device)
        actions_b = torch.FloatTensor(np.array(a)).to(self.device)
        rewards_b = torch.FloatTensor(np.array(r)).to(self.device)
        next_states_b = torch.FloatTensor(np.array(ns)).to(self.device)
        dones_b = torch.FloatTensor(np.array(d)).to(self.device)

        s_cat = states_b.view(self.batch_size, -1)
        a_cat = actions_b.view(self.batch_size, -1)

        for i in range(self.num_agents):
            # Critic update (always)
            target = self._compute_target(next_states_b, rewards_b, dones_b, i)
            q1 = self.critics_A[i](s_cat, a_cat)
            q2 = self.critics_B[i](s_cat, a_cat)
            loss_A = nn.MSELoss()(q1, target.detach())
            loss_B = nn.MSELoss()(q2, target.detach())

            self.critic_A_optimizers[i].zero_grad()
            loss_A.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_A[i].parameters(), self.clip_norm)
            self.critic_A_optimizers[i].step()

            self.critic_B_optimizers[i].zero_grad()
            loss_B.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_B[i].parameters(), self.clip_norm)
            self.critic_B_optimizers[i].step()

            # Actor update (delayed)
            if self._update_counter % self.policy_delay == 0:
                self.actor_optimizers[i].zero_grad()
                cur_actions = []
                for j in range(self.num_agents):
                    if j == i:
                        cur_actions.append(self.actors[j](states_b[:, j]))
                    else:
                        cur_actions.append(actions_b[:, j].detach())
                cur_actions = torch.cat(cur_actions, dim=1)
                actor_loss = -self.critics_A[i](s_cat, cur_actions).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_norm)
                self.actor_optimizers[i].step()

                # Soft update targets (only when actor updates)
                self.soft_update(self.actors[i], self.target_actors[i])
                self.soft_update(self.critics_A[i], self.target_critics_A[i])
                self.soft_update(self.critics_B[i], self.target_critics_B[i])

    def step_episode_schedulers(self):
        for i in range(self.num_agents):
            self.actor_schedulers[i].step()

    def save_checkpoint(self, filepath, episode):
        ckpt = {
            'episode': episode,
            'actors': {i: self.actors[i].state_dict() for i in range(self.num_agents)},
            'target_actors': {i: self.target_actors[i].state_dict() for i in range(self.num_agents)},
            'critics_A': {i: self.critics_A[i].state_dict() for i in range(self.num_agents)},
            'critics_B': {i: self.critics_B[i].state_dict() for i in range(self.num_agents)},
            'target_critics_A': {i: self.target_critics_A[i].state_dict() for i in range(self.num_agents)},
            'target_critics_B': {i: self.target_critics_B[i].state_dict() for i in range(self.num_agents)},
            'actor_optimizers': {i: self.actor_optimizers[i].state_dict() for i in range(self.num_agents)},
            'critic_A_optimizers': {i: self.critic_A_optimizers[i].state_dict() for i in range(self.num_agents)},
            'critic_B_optimizers': {i: self.critic_B_optimizers[i].state_dict() for i in range(self.num_agents)},
        }
        torch.save(ckpt, filepath)
        logger.info(f"MATD3 checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.target_actors[i].load_state_dict(ckpt['target_actors'][i])
            self.critics_A[i].load_state_dict(ckpt['critics_A'][i])
            self.critics_B[i].load_state_dict(ckpt['critics_B'][i])
            self.target_critics_A[i].load_state_dict(ckpt['target_critics_A'][i])
            self.target_critics_B[i].load_state_dict(ckpt['target_critics_B'][i])
            self.actor_optimizers[i].load_state_dict(ckpt['actor_optimizers'][i])
            self.critic_A_optimizers[i].load_state_dict(ckpt['critic_A_optimizers'][i])
            self.critic_B_optimizers[i].load_state_dict(ckpt['critic_B_optimizers'][i])
        logger.info(f"MATD3 checkpoint loaded from {filepath}")
        return ckpt['episode']

    def soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
