import os
import sys
import logging
import torch
from datetime import datetime
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_model import Config, Environment
from hierarchical_agent import HierarchicalAgent

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

training_logger = logging.getLogger('training')
training_logger.setLevel(logging.DEBUG)
training_logger.propagate = False

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'training_hierarchical_{timestamp}.log'),
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

training_logger.addHandler(file_handler)
training_logger.addHandler(stream_handler)

def train_hierarchical(case=1, episodes=500):
    training_logger.info(f"Starting Hierarchical training - Case: {case}, Episodes: {episodes}")
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    training_logger.info(f"Agent config: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset(case)
        total_reward = 0.0
        
        while True:
            upper_actions = agent.upper_act(states)
            lower_actions = agent.lower_act(states)
            
            full_actions = []
            for i in range(config.N):
                full_action = np.concatenate([upper_actions[i], lower_actions[i]])
                full_actions.append(full_action)
            
            full_actions = np.array(full_actions)
            next_states, rewards, done = env.step(full_actions)
            step_info = env.last_step_info or {}
            per_agent_info = step_info.get('per_agent', [])
            lower_rewards = np.array([
                per_agent_info[i].get('lower_reward', float(rewards[i])) if i < len(per_agent_info) else float(rewards[i])
                for i in range(config.N)
            ], dtype=float)

            agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
            agent.update_upper()

            for i in range(config.N):
                agent.add_lower_memory(i, states[i], lower_actions[i], lower_rewards[i], next_states[i], done)
                agent.update_lower(i)
                agent.update_model(i)
                agent.dyna_plan(i)
            
            total_reward += np.sum(rewards)
            states = next_states
            
            if done:
                break
        
        rewards_history.append(total_reward)
        agent.step_episode_schedulers()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if episode % 50 == 0:
            training_logger.info(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, f'hierarchical_rewards_case{case}.npy'), rewards_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title(f'Hierarchical Learning Reward History (Case {case})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(results_dir, f'hierarchical_rewards_case{case}.png'))
    plt.close()
    
    training_logger.info(f"Hierarchical Training Case {case} completed! Final reward: {rewards_history[-1]:.2f}")

if __name__ == '__main__':
    train_hierarchical(case=1, episodes=500)
