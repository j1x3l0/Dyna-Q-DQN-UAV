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
from hierarchical_agent import HierarchicalNoDynaAgent

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

logger = logging.getLogger('hierarchical_no_dyna')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'hierarchical_no_dyna_{timestamp}.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def train_hierarchical_no_dyna(case=1, episodes=500):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Hierarchical (MADDPG+DQN, no Dyna-Q) Training - Episodes: {episodes}")
    logger.info(f"{'='*60}")
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    agent = HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config)
    
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
            
            agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
            agent.update_upper()
            
            for i in range(config.N):
                agent.add_lower_memory(i, states[i], lower_actions[i], rewards[i], next_states[i], done)
                agent.update_lower(i)
            
            total_reward += np.sum(rewards)
            states = next_states
            
            if done:
                break
        
        rewards_history.append(total_reward)
        
        if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay') and hasattr(agent, 'epsilon_min'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f} | Epsilon: {agent.epsilon:.4f}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nHierarchical (MADDPG+DQN, no Dyna-Q) Training Completed!")
    logger.info(f"Final Average Reward (last 50): {final_avg:.2f}")
    logger.info(f"Best Reward: {np.max(rewards_history):.2f}")
    logger.info(f"Worst Reward: {np.min(rewards_history):.2f}")
    
    return rewards_history


def generate_report(rewards_history, episodes):
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Report for {episodes}-episode training")
    logger.info(f"{'='*60}")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    smooth = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
    plt.plot(smooth, label=f'Hierarchical (MADDPG+DQN) - {episodes} episodes', linewidth=2, color='#ff7f0e')
    
    plt.title(f'Reward History (Smoothed, {episodes} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10 episodes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, f'hierarchical_no_dyna_{episodes}_{timestamp}.png'), dpi=150)
    plt.close()
    
    np.save(os.path.join(results_dir, f'hierarchical_no_dyna_rewards_{episodes}_{timestamp}.npy'), rewards_history)
    
    report_path = os.path.join(results_dir, f'hierarchical_no_dyna_report_{episodes}_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Hierarchical (MADDPG+DQN, no Dyna-Q) Training Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nExperiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Episodes: {episodes}\n")
        f.write(f"Case: 1\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"RESULTS SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nFinal Average Reward (last 50): {np.mean(rewards_history[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(rewards_history):.2f}\n")
        f.write(f"Worst Reward: {np.min(rewards_history):.2f}\n")
        f.write(f"Standard Deviation: {np.std(rewards_history):.2f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CONFIGURATION\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nSystem Parameters:\n")
        f.write(f"  - Number of UAVs (N): 3\n")
        f.write(f"  - Number of Ground Users (M): 6\n")
        f.write(f"  - Number of Resource Blocks (F): 4\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"  - Learning Rate: 1e-4\n")
        f.write(f"  - Gamma: 0.95\n")
        f.write(f"  - Tau: 0.01\n")
        f.write(f"  - Batch Size: 32\n")
        f.write(f"  - Memory Size: 2000\n")
        f.write(f"  - Dyna-Q: NOT USED\n")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Chart saved to results/ directory")
    
    return report_path


def main():
    logger.info(f"\n{'='*60}")
    logger.info(f"Hierarchical (MADDPG+DQN, no Dyna-Q) Training")
    logger.info(f"{'='*60}")
    
    for episodes in [500, 5000]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {episodes}-episode training...")
        logger.info(f"{'='*60}")
        
        rewards = train_hierarchical_no_dyna(case=1, episodes=episodes)
        generate_report(rewards, episodes)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{episodes}-episode training completed!")
        logger.info(f"{'='*60}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All training completed successfully!")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
