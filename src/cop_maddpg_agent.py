"""
CoP-MADDPG: Communication Protocol MADDPG agent.

Key differences from standard MADDPG:
  1. Communication module (MessageEncoder / MessageDecoder) for inter-agent messaging
  2. Agents share encoded observations before acting — learned end-to-end
  3. Communication is deterministic during execution, noisy during training

Reference: "CoP-MADDPG: Communication Protocol for Multi-Agent DRL" (2023)

Usage: same interface as MADDPGAgent — drop-in replacement in benchmark.
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
    os.path.join(log_dir, 'cop_maddpg_agent.log'),
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


class MessageEncoder(nn.Module):
    """Encode local observation into a message for other agents."""
    def __init__(self, state_dim, msg_dim=32, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, msg_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


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


class CoPMADDPGAgent:
    """MADDPG with learned inter-agent communication protocol."""

    def __init__(self, state_dim, action_dim, num_agents, config):
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.msg_dim = 32
        self.aug_dim = state_dim + self.msg_dim * (num_agents - 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        self.action_rng = config.rngs['action']
        self.replay_rng = config.rngs['replay']

        # Communication encoders: state → message
        self.encoders = [MessageEncoder(state_dim, self.msg_dim).to(self.device)
                         for _ in range(num_agents)]
        self.encoder_optimizers = [optim.Adam(self.encoders[i].parameters(), lr=1e-4)
                                   for i in range(num_agents)]
        self.target_encoders = [MessageEncoder(state_dim, self.msg_dim).to(self.device)
                                for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_encoders[i].load_state_dict(self.encoders[i].state_dict())

        # Actors receive augmented state (own state + others' messages)
        self.actors = [Actor(self.aug_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.target_actors = [Actor(self.aug_dim, action_dim).to(self.device) for _ in range(num_agents)]

        # Critics unchanged: full centralized state-action input
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                        for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
                               for _ in range(num_agents)]

        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=1e-3)
                                 for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=1e-4)
                                  for i in range(num_agents)]
        self.actor_schedulers = [optim.lr_scheduler.StepLR(self.actor_optimizers[i], step_size=500, gamma=0.9)
                                 for i in range(num_agents)]
        self.critic_schedulers = [optim.lr_scheduler.StepLR(self.critic_optimizers[i], step_size=500, gamma=0.9)
                                  for i in range(num_agents)]

        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.clip_norm = 1.0
        logger.info(f"CoP-MADDPG initialized: msg_dim={self.msg_dim}")

    def communicate(self, states_tensor):
        """Encode states and exchange messages. Returns augmented states."""
        # states_tensor: (N, state_dim)
        msgs = [self.encoders[i](states_tensor[i].unsqueeze(0)) for i in range(self.num_agents)]
        augmented = []
        for i in range(self.num_agents):
            other_msgs = [self.target_encoders[j](states_tensor[j].unsqueeze(0)).detach()
                          for j in range(self.num_agents) if j != i]
            aug = torch.cat([states_tensor[i].unsqueeze(0)] + other_msgs, dim=1)
            augmented.append(aug)
        return augmented

    def act(self, states, noise=True):
        states_tensor = torch.FloatTensor(states).to(self.device)  # (N, state_dim)
        augmented = self.communicate(states_tensor)
        actions = []
        for i in range(self.num_agents):
            action = self.actors[i](augmented[i]).detach().cpu().numpy()[0]
            if noise:
                action += self.action_rng.normal(0, 0.1, size=action.shape)
            action[:4] = np.clip(action[:4], -1, 1)
            action[4:] = np.clip(action[4:], 0, 1)
            actions.append(action)
        return np.array(actions)

    def add_memory(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def _batch_augment(self, batch_states, use_target=False):
        """Compute augmented states for a batch. batch_states: (B, N, state_dim).

        Returns list of N tensors, each (B, input_dim_for_actor) where
        input_dim = state_dim + (N-1)*msg_dim.
        """
        B, N, _ = batch_states.shape
        encs = self.target_encoders if use_target else self.encoders
        # msg[j]: (B, msg_dim)
        msgs = [encs[j](batch_states[:, j]) for j in range(N)]
        augmented = []
        for i in range(N):
            other = [msgs[j].detach() for j in range(N) if j != i]
            # own state: (B, state_dim), other msgs: N-1 × (B, msg_dim)
            aug = torch.cat([batch_states[:, i]] + other, dim=1)
            augmented.append(aug)
        return augmented

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.replay_rng.choice(len(self.memory), self.batch_size, replace=False)
        s_list, a_list, r_list, ns_list, d_list = [], [], [], [], []
        for idx in batch:
            st, ac, rw, nst, dn = self.memory[idx]
            s_list.append(st); a_list.append(ac)
            r_list.append(rw); ns_list.append(nst); d_list.append(dn)

        states_b = torch.FloatTensor(np.array(s_list)).to(self.device)       # (B, N, sd)
        actions_b = torch.FloatTensor(np.array(a_list)).to(self.device)     # (B, N, ad)
        rewards_b = torch.FloatTensor(np.array(r_list)).to(self.device)     # (B, N)
        next_states_b = torch.FloatTensor(np.array(ns_list)).to(self.device) # (B, N, sd)
        dones_b = torch.FloatTensor(np.array(d_list)).to(self.device)       # (B,)

        s_cat = states_b.reshape(self.batch_size, -1)        # (B, N*sd)
        a_cat = actions_b.reshape(self.batch_size, -1)       # (B, N*ad)
        ns_cat = next_states_b.reshape(self.batch_size, -1)  # (B, N*sd)

        # Compute target augmented states (using target encoders)
        ns_aug = self._batch_augment(next_states_b, use_target=True)
        # Compute current augmented states (using online encoders, for actor update)
        s_aug = self._batch_augment(states_b, use_target=False)

        for i in range(self.num_agents):
            # --- Twin-target critic update (clipped double-Q style) ---
            with torch.no_grad():
                tgt_actions = []
                for j in range(self.num_agents):
                    # Use target actor with augmented next-state for agent j
                    tgt_act = self.target_actors[j](ns_aug[j])
                    tgt_actions.append(tgt_act)
                tgt_actions = torch.cat(tgt_actions, dim=1)  # (B, N*ad)
                tgt_q = self.target_critics[i](ns_cat, tgt_actions)
                y = rewards_b[:, i].unsqueeze(1) + self.gamma * tgt_q * (1 - dones_b.unsqueeze(1))

            # Critic loss (online)
            cr_loss = nn.MSELoss()(self.critics[i](s_cat, a_cat), y.detach())
            self.critic_optimizers[i].zero_grad()
            cr_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.clip_norm)
            self.critic_optimizers[i].step()

            # --- Actor + Encoder update ---
            self.actor_optimizers[i].zero_grad()
            self.encoder_optimizers[i].zero_grad()

            cur_actions = []
            for j in range(self.num_agents):
                if j == i:
                    cur_actions.append(self.actors[j](s_aug[j]))
                else:
                    cur_actions.append(actions_b[:, j].detach())
            cur_actions = torch.cat(cur_actions, dim=1)  # (B, N*ad)

            actor_loss = -self.critics[i](s_cat, cur_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_norm)
            torch.nn.utils.clip_grad_norm_(self.encoders[i].parameters(), self.clip_norm)
            self.actor_optimizers[i].step()
            self.encoder_optimizers[i].step()

            # Soft update targets
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])
            self.soft_update(self.encoders[i], self.target_encoders[i])

    def step_episode_schedulers(self):
        for i in range(self.num_agents):
            self.actor_schedulers[i].step()
            self.critic_schedulers[i].step()

    def save_checkpoint(self, filepath, episode):
        ckpt = {
            'episode': episode,
            'actors': {i: self.actors[i].state_dict() for i in range(self.num_agents)},
            'critics': {i: self.critics[i].state_dict() for i in range(self.num_agents)},
            'target_actors': {i: self.target_actors[i].state_dict() for i in range(self.num_agents)},
            'target_critics': {i: self.target_critics[i].state_dict() for i in range(self.num_agents)},
            'encoders': {i: self.encoders[i].state_dict() for i in range(self.num_agents)},
            'target_encoders': {i: self.target_encoders[i].state_dict() for i in range(self.num_agents)},
            'actor_optimizers': {i: self.actor_optimizers[i].state_dict() for i in range(self.num_agents)},
            'critic_optimizers': {i: self.critic_optimizers[i].state_dict() for i in range(self.num_agents)},
            'encoder_optimizers': {i: self.encoder_optimizers[i].state_dict() for i in range(self.num_agents)},
        }
        torch.save(ckpt, filepath)
        logger.info(f"CoP-MADDPG checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.critics[i].load_state_dict(ckpt['critics'][i])
            self.target_actors[i].load_state_dict(ckpt['target_actors'][i])
            self.target_critics[i].load_state_dict(ckpt['target_critics'][i])
            self.encoders[i].load_state_dict(ckpt['encoders'][i])
            self.target_encoders[i].load_state_dict(ckpt['target_encoders'][i])
            self.actor_optimizers[i].load_state_dict(ckpt['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(ckpt['critic_optimizers'][i])
            if 'encoder_optimizers' in ckpt:
                self.encoder_optimizers[i].load_state_dict(ckpt['encoder_optimizers'][i])
        logger.info(f"CoP-MADDPG checkpoint loaded from {filepath}")
        return ckpt['episode']

    def soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
