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
    os.path.join(log_dir, 'maddpg_agent.log'),
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        logger.info(f"Creating Actor: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        logger.info(f"Creating Critic: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, config):
        logger.info("=" * 60)
        logger.info("Initializing MADDPGAgent...")
        logger.info("=" * 60)
        
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"MADDPG params: num_agents={num_agents}, state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
        
        logger.info("Creating actor networks...")
        self.actors = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating critic networks...")
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating target actor networks...")
        self.target_actors = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating target critic networks...")
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Copying weights to target networks...")
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        logger.info("Creating optimizers...")
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=1e-3) for i in range(num_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating learning rate schedulers...")
        self.actor_schedulers = [optim.lr_scheduler.StepLR(self.actor_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.critic_schedulers = [optim.lr_scheduler.StepLR(self.critic_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        
        logger.info("Initializing replay memory...")
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.clip_norm = 1.0
        
        logger.info(f"Memory capacity: {self.memory.maxlen}, gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, clip_norm={self.clip_norm}")
        logger.info("=" * 60)
        logger.info("MADDPGAgent initialization complete!")
        logger.info("=" * 60)
    
    def act(self, states, noise=True):
        logger.debug(f"act() called: states shape={states.shape}, noise={noise}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.actors[i](state).detach().cpu().numpy()[0]
            
            if noise:
                noise_val = np.random.normal(0, 0.1, size=action.shape)
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
        logger.debug(f"add_memory() called: states shape={states.shape}, rewards={rewards}, dones={dones}")
        self.memory.append((states, actions, rewards, next_states, dones))
        logger.debug(f"Memory size: {len(self.memory)}/{self.memory.maxlen}")
    
    def update(self):
        if len(self.memory) < self.batch_size:
            logger.debug(f"update() skipped: memory size {len(self.memory)} < batch size {self.batch_size}")
            return
        
        logger.debug(f"update() called: memory size={len(self.memory)}, batch_size={self.batch_size}")
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
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
        
        logger.debug(f"Batch tensors created: states={states_batch.shape}, actions={actions_batch.shape}, rewards={rewards_batch.shape}")
        
        for i in range(self.num_agents):
            logger.debug(f"Updating agent {i} networks...")
            
            target_next_actions = []
            for j in range(self.num_agents):
                target_next_actions.append(self.target_actors[j](next_states_batch[:, j]))
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            next_states_cat = next_states_batch.view(self.batch_size, -1)
            target_q = self.target_critics[i](next_states_cat, target_next_actions)
            
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))
            
            states_cat = states_batch.view(self.batch_size, -1)
            actions_cat = actions_batch.view(self.batch_size, -1)
            q_i = self.critics[i](states_cat, actions_cat)
            
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            logger.debug(f"Agent {i} critic loss: {critic_loss.item():.6f}")
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.clip_norm)
            self.critic_optimizers[i].step()
            
            self.actor_optimizers[i].zero_grad()
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_actions.append(self.actors[j](states_batch[:, j]))
                else:
                    current_actions.append(actions_batch[:, j].detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            actor_loss = -self.critics[i](states_cat, current_actions).mean()
            logger.debug(f"Agent {i} actor loss: {actor_loss.item():.6f}")
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_norm)
            self.actor_optimizers[i].step()
            
            self.actor_schedulers[i].step()
            self.critic_schedulers[i].step()
            
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])
        
        logger.debug("update() completed successfully!")
    
    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)