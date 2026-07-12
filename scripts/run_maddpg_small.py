import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from system_model import Config, Environment
from maddpg_agent import MADDPGAgent

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

training_logger = logging.getLogger('training')
training_logger.setLevel(logging.DEBUG)
training_logger.propagate = False

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'training_{timestamp}.log'),
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

def train_maddpg(case=1, episodes=100):
    training_logger.info(f"Starting MADDPG training - Case: {case}, Episodes: {episodes}")
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    training_logger.info(f"Agent config: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset(case)
        total_reward = 0.0
        
        t = 0
        while t < 10:
            actions = agent.act(states)
            next_states, rewards, done = env.step(actions)
            
            agent.add_memory(states, actions, rewards, next_states, done)
            agent.update()
            
            total_reward += np.sum(rewards)
            states = next_states
            t += 1
        
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            training_logger.info(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, f'maddpg_rewards_case{case}_small.npy'), rewards_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title(f'MADDPG Reward History (Case {case})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(results_dir, f'maddpg_rewards_case{case}_small.png'))
    plt.close()
    
    training_logger.info(f"\nTraining completed! Final reward: {rewards_history[-1]:.2f}")
    training_logger.info(f"Average reward (last 10 episodes): {np.mean(rewards_history[-10:]):.2f}")

if __name__ == '__main__':
    train_maddpg(case=1, episodes=100)