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
from hierarchical_agent import HierarchicalAgent

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

logger = logging.getLogger('test_training')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'test_training_{timestamp}.log'),
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

def train_maddpg_test(case=1, episodes=500, seed=42):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting MADDPG Test Training - Case: {case}, Episodes: {episodes}, Seed: {seed}")
    logger.info(f"{'='*60}")

    config = Config(seed=seed)
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    logger.info(f"Configuration: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    
    rewards_history = []
    avg_rewards = []
    
    for episode in range(episodes):
        states = env.reset(case)
        total_reward = 0.0
        
        while True:
            actions = agent.act(states)
            next_states, rewards, done = env.step(actions)
            
            agent.add_memory(states, actions, rewards, next_states, done)
            agent.update()
            
            total_reward += np.sum(rewards)
            states = next_states
            
            if done:
                break
        
        rewards_history.append(total_reward)
        agent.step_episode_schedulers()
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            avg_rewards.append(avg_reward)
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nMADDPG Training Completed!")
    logger.info(f"Final Average Reward (last 50 episodes): {final_avg:.2f}")
    logger.info(f"Best Reward: {np.max(rewards_history):.2f}")
    logger.info(f"Worst Reward: {np.min(rewards_history):.2f}")
    
    return rewards_history, avg_rewards

def train_hierarchical_test(case=1, episodes=500, seed=42):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Hierarchical (Dyna-Q) Test Training - Case: {case}, Episodes: {episodes}, Seed: {seed}")
    logger.info(f"{'='*60}")

    config = Config(seed=seed)
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    logger.info(f"Configuration: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    
    rewards_history = []
    avg_rewards = []
    
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
        
        if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay') and hasattr(agent, 'epsilon_min'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            avg_rewards.append(avg_reward)
            epsilon_str = f" | Epsilon: {agent.epsilon:.4f}" if hasattr(agent, 'epsilon') else ""
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f}{epsilon_str}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nHierarchical (Dyna-Q) Training Completed!")
    logger.info(f"Final Average Reward (last 50 episodes): {final_avg:.2f}")
    logger.info(f"Best Reward: {np.max(rewards_history):.2f}")
    logger.info(f"Worst Reward: {np.min(rewards_history):.2f}")
    
    return rewards_history, avg_rewards

def generate_report(maddpg_rewards, hierarchical_rewards):
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Experiment Report")
    logger.info(f"{'='*60}")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(maddpg_rewards, label='MADDPG', alpha=0.7)
    plt.title('MADDPG Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(hierarchical_rewards, label='Hierarchical (Dyna-Q)', alpha=0.7)
    plt.title('Hierarchical (Dyna-Q) Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'reward_comparison_{timestamp}.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    maddpg_smooth = np.convolve(maddpg_rewards, np.ones(10)/10, mode='valid')
    hierarchical_smooth = np.convolve(hierarchical_rewards, np.ones(10)/10, mode='valid')
    
    plt.plot(maddpg_smooth, label='MADDPG (smoothed)', linewidth=2)
    plt.plot(hierarchical_smooth, label='Hierarchical (Dyna-Q) (smoothed)', linewidth=2)
    plt.title('Reward Comparison (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10 episodes)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, f'reward_comparison_smooth_{timestamp}.png'), dpi=150)
    plt.close()
    
    np.save(os.path.join(results_dir, f'maddpg_rewards_test_{timestamp}.npy'), maddpg_rewards)
    np.save(os.path.join(results_dir, f'hierarchical_rewards_test_{timestamp}.npy'), hierarchical_rewards)
    
    report_path = os.path.join(results_dir, f'experiment_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"UAV-Assisted Wireless Sensor Networks\n")
        f.write(f"Deep Reinforcement Learning Experiment Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nExperiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Episodes: 5000\n")
        f.write(f"Case: 1\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"RESULTS SUMMARY\n")
        f.write(f"{'='*60}\n")
        
        f.write(f"\n--- MADDPG ---\n")
        f.write(f"Final Average Reward (last 50): {np.mean(maddpg_rewards[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(maddpg_rewards):.2f}\n")
        f.write(f"Worst Reward: {np.min(maddpg_rewards):.2f}\n")
        f.write(f"Standard Deviation: {np.std(maddpg_rewards):.2f}\n")
        
        f.write(f"\n--- Hierarchical (Dyna-Q) ---\n")
        f.write(f"Final Average Reward (last 50): {np.mean(hierarchical_rewards[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(hierarchical_rewards):.2f}\n")
        f.write(f"Worst Reward: {np.min(hierarchical_rewards):.2f}\n")
        f.write(f"Standard Deviation: {np.std(hierarchical_rewards):.2f}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f"COMPARISON\n")
        f.write(f"{'='*60}\n")
        
        maddpg_final = np.mean(maddpg_rewards[-50:])
        hierarchical_final = np.mean(hierarchical_rewards[-50:])
        diff = hierarchical_final - maddpg_final
        improvement = (diff / abs(maddpg_final) * 100) if maddpg_final != 0 else 0
        
        f.write(f"Final Reward Difference: {diff:.2f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f"CONFIGURATION\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nSystem Parameters:\n")
        f.write(f"  - Number of UAVs (N): 3\n")
        f.write(f"  - Number of Ground Users (M): 6\n")
        f.write(f"  - Number of Resource Blocks (F): 4\n")
        f.write(f"  - Simulation Area: 100x100 m\n")
        f.write(f"  - UAV Altitude Range: 50-100 m\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"  - Learning Rate: 1e-4\n")
        f.write(f"  - Gamma: 0.95\n")
        f.write(f"  - Tau: 0.01\n")
        f.write(f"  - Batch Size: 32\n")
        f.write(f"  - Memory Size: 2000\n")
        f.write(f"  - Dyna-K: 5\n")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Reward comparison charts saved to results/ directory")

def main():
    logger.info(f"\n{'='*60}")
    logger.info(f"UAV-Assisted Wireless Sensor Networks")
    logger.info(f"Deep Reinforcement Learning Test Training")
    logger.info(f"{'='*60}")
    
    maddpg_rewards, _ = train_maddpg_test(case=1, episodes=5000)
    
    hierarchical_rewards, _ = train_hierarchical_test(case=1, episodes=5000)
    
    generate_report(maddpg_rewards, hierarchical_rewards)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Completed Successfully!")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()
