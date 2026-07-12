import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def smooth_rewards(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def compare_results():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    maddpg_case1 = np.load(os.path.join(results_dir, 'maddpg_rewards_case1.npy'))
    maddpg_case2 = np.load(os.path.join(results_dir, 'maddpg_rewards_case2.npy'))
    hierarchical_case1 = np.load(os.path.join(results_dir, 'hierarchical_rewards_case1.npy'))
    hierarchical_case2 = np.load(os.path.join(results_dir, 'hierarchical_rewards_case2.npy'))
    
    maddpg_case1_smooth = smooth_rewards(maddpg_case1)
    maddpg_case2_smooth = smooth_rewards(maddpg_case2)
    hierarchical_case1_smooth = smooth_rewards(hierarchical_case1)
    hierarchical_case2_smooth = smooth_rewards(hierarchical_case2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(maddpg_case1_smooth, label='MADDPG Case I', alpha=0.7)
    plt.plot(maddpg_case2_smooth, label='MADDPG Case II', alpha=0.7)
    plt.plot(hierarchical_case1_smooth, label='Hierarchical Case I', alpha=0.7)
    plt.plot(hierarchical_case2_smooth, label='Hierarchical Case II', alpha=0.7)
    plt.title('Reward Comparison: MADDPG vs Hierarchical Learning')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'reward_comparison.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(maddpg_case1_smooth, label='MADDPG', alpha=0.7)
    plt.plot(hierarchical_case1_smooth, label='Hierarchical', alpha=0.7)
    plt.title('Case I: Different Starting Points')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'case1_comparison.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(maddpg_case2_smooth, label='MADDPG', alpha=0.7)
    plt.plot(hierarchical_case2_smooth, label='Hierarchical', alpha=0.7)
    plt.title('Case II: Same Starting Point')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'case2_comparison.png'))
    plt.close()
    
    print("Comparison plots saved successfully!")
    print(f"\nMADDPG Case I Final Reward: {maddpg_case1[-1]:.2f}")
    print(f"MADDPG Case II Final Reward: {maddpg_case2[-1]:.2f}")
    print(f"Hierarchical Case I Final Reward: {hierarchical_case1[-1]:.2f}")
    print(f"Hierarchical Case II Final Reward: {hierarchical_case2[-1]:.2f}")
    
    print(f"\nMADDPG Case I Average (last 1000): {np.mean(maddpg_case1[-1000:]):.2f}")
    print(f"MADDPG Case II Average (last 1000): {np.mean(maddpg_case2[-1000:]):.2f}")
    print(f"Hierarchical Case I Average (last 1000): {np.mean(hierarchical_case1[-1000:]):.2f}")
    print(f"Hierarchical Case II Average (last 1000): {np.mean(hierarchical_case2[-1000:]):.2f}")

if __name__ == '__main__':
    compare_results()