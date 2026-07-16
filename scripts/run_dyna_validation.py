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
from hierarchical_agent import HierarchicalAgent, HierarchicalNoDynaAgent

log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

logger = logging.getLogger('dyna_validation')
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'dyna_validation_{timestamp}.log'),
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


def train_maddpg_test(case=1, episodes=500):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting MADDPG Training - Episodes: {episodes}")
    logger.info(f"{'='*60}")
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    
    rewards_history = []
    
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
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nMADDPG Training Completed!")
    logger.info(f"Final Average Reward (last 50): {final_avg:.2f}")
    
    return rewards_history


def train_hierarchical_no_dyna_test(case=1, episodes=500):
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
            epsilon_str = f" | Epsilon: {agent.epsilon:.4f}" if hasattr(agent, 'epsilon') else ""
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f}{epsilon_str}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nHierarchical (no Dyna-Q) Training Completed!")
    logger.info(f"Final Average Reward (last 50): {final_avg:.2f}")
    
    return rewards_history


def train_hierarchical_dyna_test(case=1, episodes=500):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Hierarchical (Dyna-Q) Training - Episodes: {episodes}")
    logger.info(f"{'='*60}")
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
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
        
        if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay') and hasattr(agent, 'epsilon_min'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else total_reward
            epsilon_str = f" | Epsilon: {agent.epsilon:.4f}" if hasattr(agent, 'epsilon') else ""
            logger.info(f"Episode {episode:4d} | Total Reward: {total_reward:10.2f} | Avg Reward: {avg_reward:10.2f}{epsilon_str}")
    
    final_avg = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
    logger.info(f"\nHierarchical (Dyna-Q) Training Completed!")
    logger.info(f"Final Average Reward (last 50): {final_avg:.2f}")
    
    return rewards_history


def generate_report(maddpg_rewards, hierarchical_no_dyna_rewards, hierarchical_dyna_rewards, episodes):
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Dyna-Q Validation Report")
    logger.info(f"{'='*60}")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    maddpg_smooth = np.convolve(maddpg_rewards, np.ones(10)/10, mode='valid')
    no_dyna_smooth = np.convolve(hierarchical_no_dyna_rewards, np.ones(10)/10, mode='valid')
    dyna_smooth = np.convolve(hierarchical_dyna_rewards, np.ones(10)/10, mode='valid')
    
    plt.plot(maddpg_smooth, label='MADDPG', linewidth=2, color='#d62728')
    plt.plot(no_dyna_smooth, label='Hierarchical (MADDPG+DQN)', linewidth=2, color='#ff7f0e')
    plt.plot(dyna_smooth, label='Hierarchical (Dyna-Q)', linewidth=2, color='#1f77b4')
    
    plt.title(f'Reward Comparison (Smoothed, {episodes} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10 episodes)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.savefig(os.path.join(results_dir, f'dyna_validation_comparison_{episodes}_{timestamp}.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    labels = ['MADDPG', 'MADDPG+DQN', 'Dyna-Q']
    final_avgs = [
        np.mean(maddpg_rewards[-50:]),
        np.mean(hierarchical_no_dyna_rewards[-50:]),
        np.mean(hierarchical_dyna_rewards[-50:])
    ]
    stds = [
        np.std(maddpg_rewards[-50:]),
        np.std(hierarchical_no_dyna_rewards[-50:]),
        np.std(hierarchical_dyna_rewards[-50:])
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x, final_avgs, width, yerr=stds, capsize=5, color=['#d62728', '#ff7f0e', '#1f77b4'])
    plt.title(f'Final Average Reward Comparison ({episodes} Episodes)')
    plt.xlabel('Algorithm')
    plt.ylabel('Final Average Reward (last 50)')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(final_avgs):
        plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(results_dir, f'dyna_validation_bar_{episodes}_{timestamp}.png'), dpi=150)
    plt.close()
    
    np.save(os.path.join(results_dir, f'maddpg_rewards_dyna_val_{episodes}_{timestamp}.npy'), maddpg_rewards)
    np.save(os.path.join(results_dir, f'hierarchical_no_dyna_rewards_{episodes}_{timestamp}.npy'), hierarchical_no_dyna_rewards)
    np.save(os.path.join(results_dir, f'hierarchical_dyna_rewards_{episodes}_{timestamp}.npy'), hierarchical_dyna_rewards)
    
    report_path = os.path.join(results_dir, f'dyna_validation_report_{episodes}_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"Dyna-Q Acceleration Validation Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"\nExperiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Episodes: {episodes}\n")
        f.write(f"Case: 1\n")
        f.write(f"\n{'='*70}\n")
        f.write(f"RESULTS SUMMARY\n")
        f.write(f"{'='*70}\n")
        
        f.write(f"\n--- MADDPG ---\n")
        f.write(f"Final Average Reward (last 50): {np.mean(maddpg_rewards[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(maddpg_rewards):.2f}\n")
        f.write(f"Worst Reward: {np.min(maddpg_rewards):.2f}\n")
        f.write(f"Standard Deviation: {np.std(maddpg_rewards):.2f}\n")
        
        f.write(f"\n--- Hierarchical (MADDPG+DQN, no Dyna-Q) ---\n")
        f.write(f"Final Average Reward (last 50): {np.mean(hierarchical_no_dyna_rewards[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(hierarchical_no_dyna_rewards):.2f}\n")
        f.write(f"Worst Reward: {np.min(hierarchical_no_dyna_rewards):.2f}\n")
        f.write(f"Standard Deviation: {np.std(hierarchical_no_dyna_rewards):.2f}\n")
        
        f.write(f"\n--- Hierarchical (Dyna-Q) ---\n")
        f.write(f"Final Average Reward (last 50): {np.mean(hierarchical_dyna_rewards[-50:]):.2f}\n")
        f.write(f"Best Reward: {np.max(hierarchical_dyna_rewards):.2f}\n")
        f.write(f"Worst Reward: {np.min(hierarchical_dyna_rewards):.2f}\n")
        f.write(f"Standard Deviation: {np.std(hierarchical_dyna_rewards):.2f}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"DYNA-Q ACCELERATION EFFECT ANALYSIS\n")
        f.write(f"{'='*70}\n")
        
        maddpg_final = np.mean(maddpg_rewards[-50:])
        no_dyna_final = np.mean(hierarchical_no_dyna_rewards[-50:])
        dyna_final = np.mean(hierarchical_dyna_rewards[-50:])
        
        f.write(f"\nPerformance Improvement:\n")
        f.write(f"  MADDPG -> MADDPG+DQN: {((no_dyna_final - maddpg_final) / abs(maddpg_final) * 100):.2f}%\n")
        f.write(f"  MADDPG+DQN -> Dyna-Q: {((dyna_final - no_dyna_final) / abs(no_dyna_final) * 100):.2f}%\n")
        f.write(f"  MADDPG -> Dyna-Q: {((dyna_final - maddpg_final) / abs(maddpg_final) * 100):.2f}%\n")
        
        f.write(f"\nDyna-Q Contribution:\n")
        f.write(f"  Pure Hierarchical (MADDPG+DQN) Final Reward: {no_dyna_final:.2f}\n")
        f.write(f"  With Dyna-Q Model Planning Final Reward: {dyna_final:.2f}\n")
        f.write(f"  Dyna-Q Improvement: {((dyna_final - no_dyna_final) / abs(no_dyna_final) * 100):.2f}%\n")
        
        if dyna_final > no_dyna_final:
            f.write(f"\nCONCLUSION: Dyna-Q framework provides acceleration effect!\n")
            f.write(f"The model-based planning component helps the lower-layer DQN\n")
            f.write(f"learn more efficiently from virtual experiences.\n")
        else:
            f.write(f"\nCONCLUSION: Dyna-Q framework does not show clear acceleration effect.\n")
            f.write(f"Possible reasons: model prediction error, insufficient training,\n")
            f.write(f"or Dyna-K parameter needs tuning.\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"CONFIGURATION\n")
        f.write(f"{'='*70}\n")
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
        f.write(f"  - Dyna-K: 5\n")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Charts saved to results/ directory")
    
    return report_path


def main():
    logger.info(f"\n{'='*70}")
    logger.info(f"Dyna-Q Acceleration Validation Experiment")
    logger.info(f"{'='*70}")
    
    for episodes in [500, 5000]:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {episodes}-episode validation...")
        logger.info(f"{'='*70}")
        
        maddpg_rewards = train_maddpg_test(case=1, episodes=episodes)
        
        hierarchical_no_dyna_rewards = train_hierarchical_no_dyna_test(case=1, episodes=episodes)
        
        hierarchical_dyna_rewards = train_hierarchical_dyna_test(case=1, episodes=episodes)
        
        generate_report(maddpg_rewards, hierarchical_no_dyna_rewards, hierarchical_dyna_rewards, episodes)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"{episodes}-episode validation completed!")
        logger.info(f"{'='*70}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"All Dyna-Q Validation Experiments Completed!")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
