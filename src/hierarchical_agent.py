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
    os.path.join(log_dir, 'hierarchical_agent.log'),
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

class UpperActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(UpperActor, self).__init__()
        logger.info(f"Creating UpperActor: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class UpperCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(UpperCritic, self).__init__()
        logger.info(f"Creating UpperCritic: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
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

class LowerDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(LowerDQN, self).__init__()
        logger.info(f"Creating LowerDQN: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Model, self).__init__()
        logger.info(f"Creating Model: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)
        self.fc_next_state = nn.Linear(hidden_dim, state_dim)
        self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        reward = self.fc_reward(x)
        next_state = self.fc_next_state(x)
        return reward, next_state

class HierarchicalAgent:
    def __init__(self, state_dim, action_dim, num_agents, config):
        logger.info("=" * 60)
        logger.info("Initializing HierarchicalAgent...")
        logger.info("=" * 60)
        
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Hierarchical params: num_agents={num_agents}, state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
        
        logger.info("Creating upper-layer actor networks...")
        self.upper_actors = [UpperActor(state_dim, 5).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer critic networks...")
        self.upper_critics = [UpperCritic(state_dim * num_agents, 5 * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer target actor networks...")
        self.target_upper_actors = [UpperActor(state_dim, 5).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer target critic networks...")
        self.target_upper_critics = [UpperCritic(state_dim * num_agents, 5 * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Copying upper-layer weights to target networks...")
        for i in range(num_agents):
            self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
            self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())
        
        logger.info("Creating upper-layer optimizers...")
        self.upper_actor_optimizers = [optim.Adam(self.upper_actors[i].parameters(), lr=1e-3) for i in range(num_agents)]
        self.upper_critic_optimizers = [optim.Adam(self.upper_critics[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating lower-layer DQN networks...")
        self.lower_dqns = [LowerDQN(state_dim, 2 * config.M).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating lower-layer target DQN networks...")
        self.target_lower_dqns = [LowerDQN(state_dim, 2 * config.M).to(self.device) for _ in range(num_agents)]
        
        logger.info("Copying lower-layer weights to target networks...")
        for i in range(num_agents):
            self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
        
        logger.info("Creating lower-layer optimizers...")
        self.lower_optimizers = [optim.Adam(self.lower_dqns[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating Dyna-Q model networks...")
        self.models = [Model(state_dim, 2 * config.M).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating model optimizers...")
        self.model_optimizers = [optim.Adam(self.models[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating learning rate schedulers...")
        self.upper_actor_schedulers = [optim.lr_scheduler.StepLR(self.upper_actor_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.upper_critic_schedulers = [optim.lr_scheduler.StepLR(self.upper_critic_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.lower_schedulers = [optim.lr_scheduler.StepLR(self.lower_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.model_schedulers = [optim.lr_scheduler.StepLR(self.model_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        
        logger.info("Initializing upper-layer replay memory...")
        self.upper_memory = deque(maxlen=10000)
        
        logger.info("Initializing lower-layer replay memories...")
        self.lower_memory = [deque(maxlen=10000) for _ in range(num_agents)]
        
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.dyna_k = 5
        self.clip_norm = 1.0
        
        logger.info(f"Upper memory capacity: {self.upper_memory.maxlen}")
        logger.info(f"Lower memory capacity: {self.lower_memory[0].maxlen} per agent")
        logger.info(f"gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, epsilon={self.epsilon}")
        logger.info(f"epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, clip_norm={self.clip_norm}")
        logger.info("=" * 60)
        logger.info("HierarchicalAgent initialization complete!")
        logger.info("=" * 60)
    
    def upper_act(self, states, noise=True):
        logger.debug(f"upper_act() called: states shape={states.shape}, noise={noise}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.upper_actors[i](state).detach().cpu().numpy()[0]
            
            if noise:
                noise_val = np.random.normal(0, 0.1, size=action.shape)
                action += noise_val
                logger.debug(f"Upper Agent {i} action with noise: noise_norm={np.linalg.norm(noise_val):.4f}")
            
            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], 0, 1)
            
            logger.debug(f"Upper Agent {i} action: direction={action[:3]}, speed={action[3]:.4f}, scheduled={bool(action[4])}")
            actions.append(action)
        
        actions_array = np.array(actions)
        logger.debug(f"upper_act() completed: actions shape={actions_array.shape}")
        return actions_array
    
    def lower_act(self, states):
        logger.debug(f"lower_act() called: states shape={states.shape}, epsilon={self.epsilon}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            
            if np.random.random() < self.epsilon:
                action = np.random.rand(2 * self.config.M)
                logger.debug(f"Lower Agent {i} exploration action: random choice")
            else:
                q_values = self.lower_dqns[i](state).detach().cpu().numpy()[0]
                action = np.zeros(2 * self.config.M)
                for j in range(self.config.M):
                    action[j] = 1 if q_values[j] > q_values[j + self.config.M] else 0
                    action[j + self.config.M] = 1 - action[j]
                logger.debug(f"Lower Agent {i} exploitation action: Q-values computed")
            
            logger.debug(f"Lower Agent {i} action: access={action[:self.config.M]}, mode={action[self.config.M:]}")
            actions.append(action)
        
        actions_array = np.array(actions)
        logger.debug(f"lower_act() completed: actions shape={actions_array.shape}")
        return actions_array
    
    def add_upper_memory(self, states, actions, rewards, next_states, dones):
        logger.debug(f"add_upper_memory() called: states shape={states.shape}, rewards={rewards}, dones={dones}")
        self.upper_memory.append((states, actions, rewards, next_states, dones))
        logger.debug(f"Upper memory size: {len(self.upper_memory)}/{self.upper_memory.maxlen}")
    
    def add_lower_memory(self, agent_idx, state, action, reward, next_state, done):
        logger.debug(f"add_lower_memory() called: agent_idx={agent_idx}, reward={reward:.4f}, done={done}")
        self.lower_memory[agent_idx].append((state, action, reward, next_state, done))
        logger.debug(f"Lower memory[{agent_idx}] size: {len(self.lower_memory[agent_idx])}/{self.lower_memory[agent_idx].maxlen}")
    
    def update_upper(self):
        if len(self.upper_memory) < self.batch_size:
            logger.debug(f"update_upper() skipped: memory size {len(self.upper_memory)} < batch size {self.batch_size}")
            return
        
        logger.debug(f"update_upper() called: memory size={len(self.upper_memory)}, batch_size={self.batch_size}")
        
        batch = np.random.choice(len(self.upper_memory), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.upper_memory[idx]
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
        
        logger.debug(f"Upper batch tensors created: states={states_batch.shape}, actions={actions_batch.shape}")
        
        for i in range(self.num_agents):
            logger.debug(f"Updating upper-layer agent {i} networks...")
            
            target_next_actions = []
            for j in range(self.num_agents):
                target_next_actions.append(self.target_upper_actors[j](next_states_batch[:, j]))
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            next_states_cat = next_states_batch.view(self.batch_size, -1)
            target_q = self.target_upper_critics[i](next_states_cat, target_next_actions)
            
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))
            
            states_cat = states_batch.view(self.batch_size, -1)
            actions_cat = actions_batch.view(self.batch_size, -1)
            q_i = self.upper_critics[i](states_cat, actions_cat)
            
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            logger.debug(f"Upper Agent {i} critic loss: {critic_loss.item():.6f}")
            
            self.upper_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_critics[i].parameters(), self.clip_norm)
            self.upper_critic_optimizers[i].step()
            
            self.upper_actor_optimizers[i].zero_grad()
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_actions.append(self.upper_actors[j](states_batch[:, j]))
                else:
                    current_actions.append(actions_batch[:, j].detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            actor_loss = -self.upper_critics[i](states_cat, current_actions).mean()
            logger.debug(f"Upper Agent {i} actor loss: {actor_loss.item():.6f}")
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_actors[i].parameters(), self.clip_norm)
            self.upper_actor_optimizers[i].step()
            
            self.upper_actor_schedulers[i].step()
            self.upper_critic_schedulers[i].step()
            
            self.soft_update(self.upper_actors[i], self.target_upper_actors[i])
            self.soft_update(self.upper_critics[i], self.target_upper_critics[i])
        
        logger.debug("update_upper() completed successfully!")
    
    def update_lower(self, agent_idx):
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            logger.debug(f"update_lower({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} < batch size {self.batch_size}")
            return
        
        logger.debug(f"update_lower({agent_idx}) called: memory size={len(self.lower_memory[agent_idx])}, batch_size={self.batch_size}")
        
        batch = np.random.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
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
        
        logger.debug(f"Lower batch tensors created: states={states_batch.shape}, actions={actions_batch.shape}")
        
        q_values = self.lower_dqns[agent_idx](states_batch)
        action_indices = actions_batch.argmax(dim=1)
        q_i = q_values.gather(1, action_indices.unsqueeze(1))
        
        next_q_values = self.target_lower_dqns[agent_idx](next_states_batch)
        next_max_q = next_q_values.max(dim=1)[0].unsqueeze(1)
        
        y_i = rewards_batch.unsqueeze(1) + self.gamma * next_max_q * (1 - dones_batch.unsqueeze(1))
        
        loss = nn.MSELoss()(q_i, y_i.detach())
        logger.debug(f"Lower Agent {agent_idx} DQN loss: {loss.item():.6f}")
        
        self.lower_optimizers[agent_idx].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_dqns[agent_idx].parameters(), self.clip_norm)
        self.lower_optimizers[agent_idx].step()
        self.lower_schedulers[agent_idx].step()
        
        self.soft_update(self.lower_dqns[agent_idx], self.target_lower_dqns[agent_idx])
        
        logger.debug(f"update_lower({agent_idx}) completed successfully!")
    
    def model_predict(self, agent_idx, state, action):
        logger.debug(f"model_predict({agent_idx}) called: state shape={state.shape}, action shape={action.shape}")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        self.models[agent_idx].eval()
        with torch.no_grad():
            reward_pred, next_state_pred = self.models[agent_idx](state_tensor, action_tensor)
        reward_pred = reward_pred.detach().cpu().numpy()[0, 0]
        next_state_pred = next_state_pred.detach().cpu().numpy()[0]
        logger.debug(f"model_predict({agent_idx}) completed: reward_pred={reward_pred:.4f}, next_state_pred_norm={np.linalg.norm(next_state_pred):.4f}")
        return reward_pred, next_state_pred
    
    def update_model(self, agent_idx):
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            logger.debug(f"update_model({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} < batch size {self.batch_size}")
            return
        
        logger.debug(f"update_model({agent_idx}) called: memory size={len(self.lower_memory[agent_idx])}, batch_size={self.batch_size}")
        
        batch = np.random.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
            states_batch.append(s)
            actions_batch.append(a)
            rewards_batch.append(r)
            next_states_batch.append(ns)
        
        states_batch = torch.FloatTensor(np.array(states_batch)).to(self.device)
        actions_batch = torch.FloatTensor(np.array(actions_batch)).to(self.device)
        rewards_batch = torch.FloatTensor(np.array(rewards_batch)).unsqueeze(1).to(self.device)
        next_states_batch = torch.FloatTensor(np.array(next_states_batch)).to(self.device)
        
        self.models[agent_idx].train()
        reward_pred, next_state_pred = self.models[agent_idx](states_batch, actions_batch)
        
        reward_loss = nn.MSELoss()(reward_pred, rewards_batch)
        state_loss = nn.MSELoss()(next_state_pred, next_states_batch)
        total_loss = reward_loss + state_loss
        
        logger.debug(f"Model {agent_idx} loss: reward_loss={reward_loss.item():.6f}, state_loss={state_loss.item():.6f}, total_loss={total_loss.item():.6f}")
        
        self.model_optimizers[agent_idx].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.models[agent_idx].parameters(), self.clip_norm)
        self.model_optimizers[agent_idx].step()
        self.model_schedulers[agent_idx].step()
    
    def dyna_plan(self, agent_idx, k=None):
        if k is None:
            k = self.dyna_k
        
        if len(self.lower_memory[agent_idx]) < k:
            logger.debug(f"dyna_plan({agent_idx}) skipped: memory size {len(self.lower_memory[agent_idx])} < k={k}")
            return
        
        logger.debug(f"dyna_plan({agent_idx}) called: k={k}, memory size={len(self.lower_memory[agent_idx])}")
        
        batch = np.random.choice(len(self.lower_memory[agent_idx]), k, replace=False)
        
        self.lower_optimizers[agent_idx].zero_grad()
        
        total_loss = 0.0
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
            
            r_pred, s_pred = self.model_predict(agent_idx, s, a)
            
            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(a).unsqueeze(0).to(self.device)
            next_state_pred_tensor = torch.FloatTensor(s_pred).unsqueeze(0).to(self.device)
            reward_pred_tensor = torch.FloatTensor([[r_pred]]).to(self.device)
            
            self.lower_dqns[agent_idx].train()
            
            q_values = self.lower_dqns[agent_idx](state_tensor)
            action_index = action_tensor.argmax().unsqueeze(0).unsqueeze(0)
            q_i = q_values.gather(1, action_index)
            
            next_q_values = self.target_lower_dqns[agent_idx](next_state_pred_tensor)
            next_max_q = next_q_values.max(dim=1)[0].unsqueeze(1)
            
            y_i = reward_pred_tensor + self.gamma * next_max_q * (1 - torch.FloatTensor([[d]]).to(self.device))
            
            loss = nn.MSELoss()(q_i, y_i.detach())
            total_loss += loss.item()
            
            loss.backward()
        
        self.lower_optimizers[agent_idx].step()
        
        logger.debug(f"dyna_plan({agent_idx}) completed: avg_loss={total_loss/k:.6f}, k={k}")
    
    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


class HierarchicalNoDynaAgent:
    def __init__(self, state_dim, action_dim, num_agents, config):
        logger.info("=" * 60)
        logger.info("Initializing HierarchicalNoDynaAgent (without Dyna-Q)...")
        logger.info("=" * 60)
        
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"HierarchicalNoDyna params: num_agents={num_agents}, state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
        
        logger.info("Creating upper-layer actor networks...")
        self.upper_actors = [UpperActor(state_dim, 5).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer critic networks...")
        self.upper_critics = [UpperCritic(state_dim * num_agents, 5 * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer target actor networks...")
        self.target_upper_actors = [UpperActor(state_dim, 5).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating upper-layer target critic networks...")
        self.target_upper_critics = [UpperCritic(state_dim * num_agents, 5 * num_agents).to(self.device) for _ in range(num_agents)]
        
        logger.info("Copying upper-layer weights to target networks...")
        for i in range(num_agents):
            self.target_upper_actors[i].load_state_dict(self.upper_actors[i].state_dict())
            self.target_upper_critics[i].load_state_dict(self.upper_critics[i].state_dict())
        
        logger.info("Creating upper-layer optimizers...")
        self.upper_actor_optimizers = [optim.Adam(self.upper_actors[i].parameters(), lr=1e-3) for i in range(num_agents)]
        self.upper_critic_optimizers = [optim.Adam(self.upper_critics[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating lower-layer DQN networks...")
        self.lower_dqns = [LowerDQN(state_dim, 2 * config.M).to(self.device) for _ in range(num_agents)]
        
        logger.info("Creating lower-layer target DQN networks...")
        self.target_lower_dqns = [LowerDQN(state_dim, 2 * config.M).to(self.device) for _ in range(num_agents)]
        
        logger.info("Copying lower-layer weights to target networks...")
        for i in range(num_agents):
            self.target_lower_dqns[i].load_state_dict(self.lower_dqns[i].state_dict())
        
        logger.info("Creating lower-layer optimizers...")
        self.lower_optimizers = [optim.Adam(self.lower_dqns[i].parameters(), lr=1e-4) for i in range(num_agents)]
        
        logger.info("Creating learning rate schedulers...")
        self.upper_actor_schedulers = [optim.lr_scheduler.StepLR(self.upper_actor_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.upper_critic_schedulers = [optim.lr_scheduler.StepLR(self.upper_critic_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        self.lower_schedulers = [optim.lr_scheduler.StepLR(self.lower_optimizers[i], step_size=500, gamma=0.9) for i in range(num_agents)]
        
        logger.info("Initializing upper-layer replay memory...")
        self.upper_memory = deque(maxlen=10000)
        
        logger.info("Initializing lower-layer replay memories...")
        self.lower_memory = [deque(maxlen=10000) for _ in range(num_agents)]
        
        self.gamma = 0.95
        self.tau = 0.01
        self.batch_size = 32
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.clip_norm = 1.0
        
        logger.info(f"Upper memory capacity: {self.upper_memory.maxlen}")
        logger.info(f"Lower memory capacity: {self.lower_memory[0].maxlen} per agent")
        logger.info(f"gamma={self.gamma}, tau={self.tau}, batch_size={self.batch_size}, epsilon={self.epsilon}")
        logger.info(f"epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, clip_norm={self.clip_norm}")
        logger.info("=" * 60)
        logger.info("HierarchicalNoDynaAgent initialization complete!")
        logger.info("=" * 60)
    
    def upper_act(self, states, noise=True):
        logger.debug(f"upper_act() called: states shape={states.shape}, noise={noise}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            action = self.upper_actors[i](state).detach().cpu().numpy()[0]
            
            if noise:
                noise_val = np.random.normal(0, 0.1, size=action.shape)
                action += noise_val
            
            action[:4] = np.clip(action[:4], -1, 1)
            action[4] = np.clip(action[4], 0, 1)
            actions.append(action)
        
        actions_array = np.array(actions)
        return actions_array
    
    def lower_act(self, states):
        logger.debug(f"lower_act() called: states shape={states.shape}, epsilon={self.epsilon}")
        actions = []
        
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            
            if np.random.random() < self.epsilon:
                action = np.random.rand(2 * self.config.M)
            else:
                q_values = self.lower_dqns[i](state).detach().cpu().numpy()[0]
                action = np.zeros(2 * self.config.M)
                for j in range(self.config.M):
                    action[j] = 1 if q_values[j] > q_values[j + self.config.M] else 0
                    action[j + self.config.M] = 1 - action[j]
            
            actions.append(action)
        
        actions_array = np.array(actions)
        return actions_array
    
    def add_upper_memory(self, states, actions, rewards, next_states, dones):
        self.upper_memory.append((states, actions, rewards, next_states, dones))
    
    def add_lower_memory(self, agent_idx, state, action, reward, next_state, done):
        self.lower_memory[agent_idx].append((state, action, reward, next_state, done))
    
    def update_upper(self):
        if len(self.upper_memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.upper_memory), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.upper_memory[idx]
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
        
        for i in range(self.num_agents):
            target_next_actions = []
            for j in range(self.num_agents):
                target_next_actions.append(self.target_upper_actors[j](next_states_batch[:, j]))
            target_next_actions = torch.cat(target_next_actions, dim=1)
            
            next_states_cat = next_states_batch.view(self.batch_size, -1)
            target_q = self.target_upper_critics[i](next_states_cat, target_next_actions)
            
            y_i = rewards_batch[:, i].unsqueeze(1) + self.gamma * target_q * (1 - dones_batch.unsqueeze(1))
            
            states_cat = states_batch.view(self.batch_size, -1)
            actions_cat = actions_batch.view(self.batch_size, -1)
            q_i = self.upper_critics[i](states_cat, actions_cat)
            
            critic_loss = nn.MSELoss()(q_i, y_i.detach())
            
            self.upper_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_critics[i].parameters(), self.clip_norm)
            self.upper_critic_optimizers[i].step()
            
            self.upper_actor_optimizers[i].zero_grad()
            current_actions = []
            for j in range(self.num_agents):
                if j == i:
                    current_actions.append(self.upper_actors[j](states_batch[:, j]))
                else:
                    current_actions.append(actions_batch[:, j].detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            actor_loss = -self.upper_critics[i](states_cat, current_actions).mean()
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.upper_actors[i].parameters(), self.clip_norm)
            self.upper_actor_optimizers[i].step()
            
            self.upper_actor_schedulers[i].step()
            self.upper_critic_schedulers[i].step()
            
            self.soft_update(self.upper_actors[i], self.target_upper_actors[i])
            self.soft_update(self.upper_critics[i], self.target_upper_critics[i])
    
    def update_lower(self, agent_idx):
        if len(self.lower_memory[agent_idx]) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.lower_memory[agent_idx]), self.batch_size, replace=False)
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for idx in batch:
            s, a, r, ns, d = self.lower_memory[agent_idx][idx]
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
        
        q_values = self.lower_dqns[agent_idx](states_batch)
        action_indices = actions_batch.argmax(dim=1)
        q_i = q_values.gather(1, action_indices.unsqueeze(1))
        
        next_q_values = self.target_lower_dqns[agent_idx](next_states_batch)
        next_max_q = next_q_values.max(dim=1)[0].unsqueeze(1)
        
        y_i = rewards_batch.unsqueeze(1) + self.gamma * next_max_q * (1 - dones_batch.unsqueeze(1))
        
        loss = nn.MSELoss()(q_i, y_i.detach())
        
        self.lower_optimizers[agent_idx].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lower_dqns[agent_idx].parameters(), self.clip_norm)
        self.lower_optimizers[agent_idx].step()
        self.lower_schedulers[agent_idx].step()
        
        self.soft_update(self.lower_dqns[agent_idx], self.target_lower_dqns[agent_idx])
    
    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)