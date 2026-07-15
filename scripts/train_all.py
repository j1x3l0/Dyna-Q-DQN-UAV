import os
import sys
import time
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
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
training_logger = logging.getLogger('train_all')
training_logger.setLevel(logging.DEBUG)
training_logger.propagate = False

file_handler = RotatingFileHandler(
    os.path.join(log_dir, f'train_all_{timestamp}.log'),
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

def train_maddpg(episodes=5000):
    start_time = time.time()
    training_logger.info("=" * 70)
    training_logger.info(f"Starting Pure MADDPG Training - Episodes: {episodes}")
    training_logger.info("=" * 70)
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    training_logger.info(f"Agent config: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = MADDPGAgent(state_dim, action_dim, config.N, config)
    training_logger.info(f"Using device: {agent.device}")
    
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset(1)
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
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else total_reward
            training_logger.info(f"MADDPG - Episode {episode:5d}, Total Reward: {total_reward:10.2f}, Avg (last 100): {avg_reward:10.2f}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    np.save(os.path.join(results_dir, f'maddpg_rewards_5000_{timestamp}.npy'), rewards_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('MADDPG Reward History (5000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(results_dir, f'maddpg_rewards_5000_{timestamp}.png'))
    plt.close()
    
    training_logger.info("=" * 70)
    training_logger.info(f"MADDPG Training completed!")
    training_logger.info(f"Final reward: {rewards_history[-1]:.2f}")
    training_logger.info(f"Average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")
    training_logger.info(f"Training duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    training_logger.info("=" * 70)
    
    return {
        'algorithm': 'MADDPG',
        'final_reward': rewards_history[-1],
        'avg_reward_last_100': np.mean(rewards_history[-100:]),
        'avg_reward_last_500': np.mean(rewards_history[-500:]),
        'max_reward': np.max(rewards_history),
        'min_reward': np.min(rewards_history),
        'duration': duration,
        'rewards_history': rewards_history
    }

def train_hierarchical_no_dyna(episodes=5000):
    start_time = time.time()
    training_logger.info("=" * 70)
    training_logger.info(f"Starting Hierarchical (MADDPG+DQN, no Dyna-Q) Training - Episodes: {episodes}")
    training_logger.info("=" * 70)
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    training_logger.info(f"Agent config: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = HierarchicalNoDynaAgent(state_dim, action_dim, config.N, config)
    training_logger.info(f"Using device: {agent.device}")
    
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset(1)
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
        
        if episode % 1000 == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else total_reward
            training_logger.info(f"Hierarchical(no Dyna) - Episode {episode:5d}, Total Reward: {total_reward:10.2f}, Avg (last 100): {avg_reward:10.2f}, Epsilon: {agent.epsilon:.4f}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    np.save(os.path.join(results_dir, f'hierarchical_no_dyna_rewards_5000_{timestamp}.npy'), rewards_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Hierarchical (MADDPG+DQN, no Dyna-Q) Reward History (5000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(results_dir, f'hierarchical_no_dyna_rewards_5000_{timestamp}.png'))
    plt.close()
    
    training_logger.info("=" * 70)
    training_logger.info(f"Hierarchical (no Dyna-Q) Training completed!")
    training_logger.info(f"Final reward: {rewards_history[-1]:.2f}")
    training_logger.info(f"Average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")
    training_logger.info(f"Training duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    training_logger.info("=" * 70)
    
    return {
        'algorithm': 'Hierarchical (MADDPG+DQN)',
        'final_reward': rewards_history[-1],
        'avg_reward_last_100': np.mean(rewards_history[-100:]),
        'avg_reward_last_500': np.mean(rewards_history[-500:]),
        'max_reward': np.max(rewards_history),
        'min_reward': np.min(rewards_history),
        'duration': duration,
        'rewards_history': rewards_history
    }

def train_hierarchical_dyna(episodes=5000):
    start_time = time.time()
    training_logger.info("=" * 70)
    training_logger.info(f"Starting Hierarchical (MADDPG+Dyna-Q) Training - Episodes: {episodes}")
    training_logger.info("=" * 70)
    
    config = Config()
    env = Environment(config)
    
    state_dim = 30
    action_dim = 4 + 2 * config.M + 1
    
    training_logger.info(f"Agent config: state_dim={state_dim}, action_dim={action_dim}, num_agents={config.N}")
    
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    training_logger.info(f"Using device: {agent.device}")
    
    rewards_history = []
    
    for episode in range(episodes):
        states = env.reset(1)
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
                agent.update_model(i)
                agent.dyna_plan(i)
            
            total_reward += np.sum(rewards)
            states = next_states
            
            if done:
                break
        
        if episode % 1000 == 0:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else total_reward
            training_logger.info(f"Hierarchical(Dyna-Q) - Episode {episode:5d}, Total Reward: {total_reward:10.2f}, Avg (last 100): {avg_reward:10.2f}, Epsilon: {agent.epsilon:.4f}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    np.save(os.path.join(results_dir, f'hierarchical_dyna_rewards_5000_{timestamp}.npy'), rewards_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Hierarchical (MADDPG+Dyna-Q) Reward History (5000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(results_dir, f'hierarchical_dyna_rewards_5000_{timestamp}.png'))
    plt.close()
    
    training_logger.info("=" * 70)
    training_logger.info(f"Hierarchical (Dyna-Q) Training completed!")
    training_logger.info(f"Final reward: {rewards_history[-1]:.2f}")
    training_logger.info(f"Average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")
    training_logger.info(f"Training duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    training_logger.info("=" * 70)
    
    return {
        'algorithm': 'Hierarchical (MADDPG+Dyna-Q)',
        'final_reward': rewards_history[-1],
        'avg_reward_last_100': np.mean(rewards_history[-100:]),
        'avg_reward_last_500': np.mean(rewards_history[-500:]),
        'max_reward': np.max(rewards_history),
        'min_reward': np.min(rewards_history),
        'duration': duration,
        'rewards_history': rewards_history
    }

def generate_comparison_report(results_list):
    training_logger.info("=" * 70)
    training_logger.info("Generating Comparison Report...")
    training_logger.info("=" * 70)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("    UAV-Assisted Wireless Sensor Networks - Training Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Training Episodes: 5000")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    report_lines.append("1. Training Results Summary")
    report_lines.append("-" * 40)
    
    header = f"{'Algorithm':<35} {'Final Reward':>12} {'Avg(last100)':>12} {'Avg(last500)':>12} {'Duration(h)':>12}"
    report_lines.append(header)
    report_lines.append("-" * 80)
    
    for result in results_list:
        line = f"{result['algorithm']:<35} {result['final_reward']:>12.2f} {result['avg_reward_last_100']:>12.2f} {result['avg_reward_last_500']:>12.2f} {result['duration']/3600:>12.2f}"
        report_lines.append(line)
    
    report_lines.append("")
    report_lines.append("2. Performance Comparison")
    report_lines.append("-" * 40)
    
    best_final = max(results_list, key=lambda x: x['final_reward'])
    best_avg100 = max(results_list, key=lambda x: x['avg_reward_last_100'])
    fastest = min(results_list, key=lambda x: x['duration'])
    
    report_lines.append(f"- Best Final Reward: {best_final['algorithm']} ({best_final['final_reward']:.2f})")
    report_lines.append(f"- Best Average Reward (last 100): {best_avg100['algorithm']} ({best_avg100['avg_reward_last_100']:.2f})")
    report_lines.append(f"- Fastest Training: {fastest['algorithm']} ({fastest['duration']/3600:.2f} hours)")
    report_lines.append("")
    
    for i, result in enumerate(results_list):
        if result['algorithm'] == best_final['algorithm']:
            continue
        improvement = (best_final['final_reward'] - result['final_reward']) / abs(result['final_reward']) * 100 if result['final_reward'] != 0 else float('inf')
        report_lines.append(f"- {best_final['algorithm']} outperforms {result['algorithm']} by {improvement:.2f}% in final reward")
    
    report_lines.append("")
    report_lines.append("3. Stability Analysis")
    report_lines.append("-" * 40)
    
    for result in results_list:
        rewards = result['rewards_history']
        volatility = np.std(rewards[-500:]) / abs(np.mean(rewards[-500:])) * 100 if np.mean(rewards[-500:]) != 0 else float('inf')
        report_lines.append(f"- {result['algorithm']}:")
        report_lines.append(f"   Max Reward: {result['max_reward']:.2f}")
        report_lines.append(f"   Min Reward: {result['min_reward']:.2f}")
        report_lines.append(f"   Volatility (last 500): {volatility:.2f}%")
        report_lines.append("")
    
    report_lines.append("4. Convergence Analysis")
    report_lines.append("-" * 40)
    
    for result in results_list:
        rewards = result['rewards_history']
        early_avg = np.mean(rewards[:100])
        late_avg = np.mean(rewards[-100:])
        improvement_ratio = (late_avg - early_avg) / abs(early_avg) * 100 if early_avg != 0 else float('inf')
        report_lines.append(f"- {result['algorithm']}:")
        report_lines.append(f"   Early Reward (episodes 1-100): {early_avg:.2f}")
        report_lines.append(f"   Late Reward (episodes {len(rewards)-100}-{len(rewards)}): {late_avg:.2f}")
        report_lines.append(f"   Improvement: {improvement_ratio:.2f}%")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_content = "\n".join(report_lines)
    
    report_path = os.path.join(results_dir, f'training_comparison_report_5000_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    training_logger.info(f"Report saved to: {report_path}")
    
    plt.figure(figsize=(12, 7))
    for result in results_list:
        plt.plot(result['rewards_history'], label=result['algorithm'], alpha=0.8)
    
    plt.title('Reward Comparison - Three Training Approaches (5000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, f'reward_comparison_5000_{timestamp}.png'))
    plt.close()
    
    training_logger.info(f"Comparison chart saved to results/")
    training_logger.info("=" * 70)
    
    print("\n" + report_content + "\n")

if __name__ == '__main__':
    training_logger.info("=" * 70)
    training_logger.info("Starting All Training Sessions")
    training_logger.info("=" * 70)
    
    results = []
    
    results.append(train_maddpg(episodes=5000))
    results.append(train_hierarchical_no_dyna(episodes=5000))
    results.append(train_hierarchical_dyna(episodes=5000))
    
    generate_comparison_report(results)
    
    training_logger.info("=" * 70)
    training_logger.info("All Training Sessions Completed!")
    training_logger.info("=" * 70)