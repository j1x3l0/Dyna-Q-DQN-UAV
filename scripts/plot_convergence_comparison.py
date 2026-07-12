import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

maddpg_500 = np.load(os.path.join(results_dir, 'maddpg_rewards_test_20260710_223148.npy'))
hierarchical_500 = np.load(os.path.join(results_dir, 'hierarchical_rewards_test_20260710_223148.npy'))

maddpg_5000 = np.load(os.path.join(results_dir, 'maddpg_rewards_test_20260711_001504.npy'))
hierarchical_5000 = np.load(os.path.join(results_dir, 'hierarchical_rewards_test_20260711_001504.npy'))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

window = 20

maddpg_500_smooth = np.convolve(maddpg_500, np.ones(window)/window, mode='valid')
hierarchical_500_smooth = np.convolve(hierarchical_500, np.ones(window)/window, mode='valid')

axes[0, 0].plot(maddpg_500_smooth, label='MADDPG', linewidth=2, color='#d62728', alpha=0.8)
axes[0, 0].plot(hierarchical_500_smooth, label='Hierarchical (Dyna-Q)', linewidth=2, color='#1f77b4', alpha=0.8)
axes[0, 0].set_title('500 Episodes - Smoothed Reward (window=20)', fontsize=14)
axes[0, 0].set_xlabel('Episode', fontsize=12)
axes[0, 0].set_ylabel('Average Reward', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

axes[0, 1].plot(maddpg_500, label='MADDPG', linewidth=1, color='#d62728', alpha=0.6)
axes[0, 1].plot(hierarchical_500, label='Hierarchical (Dyna-Q)', linewidth=1, color='#1f77b4', alpha=0.6)
axes[0, 1].set_title('500 Episodes - Raw Reward', fontsize=14)
axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('Reward', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

maddpg_5000_smooth = np.convolve(maddpg_5000, np.ones(window)/window, mode='valid')
hierarchical_5000_smooth = np.convolve(hierarchical_5000, np.ones(window)/window, mode='valid')

axes[1, 0].plot(maddpg_5000_smooth, label='MADDPG', linewidth=2, color='#d62728', alpha=0.8)
axes[1, 0].plot(hierarchical_5000_smooth, label='Hierarchical (Dyna-Q)', linewidth=2, color='#1f77b4', alpha=0.8)
axes[1, 0].set_title('5000 Episodes - Smoothed Reward (window=20)', fontsize=14)
axes[1, 0].set_xlabel('Episode', fontsize=12)
axes[1, 0].set_ylabel('Average Reward', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

axes[1, 1].plot(maddpg_5000, label='MADDPG', linewidth=1, color='#d62728', alpha=0.6)
axes[1, 1].plot(hierarchical_5000, label='Hierarchical (Dyna-Q)', linewidth=1, color='#1f77b4', alpha=0.6)
axes[1, 1].set_title('5000 Episodes - Raw Reward', fontsize=14)
axes[1, 1].set_xlabel('Episode', fontsize=12)
axes[1, 1].set_ylabel('Reward', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'convergence_comparison_500_vs_5000.png'), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(maddpg_500_smooth, label='MADDPG (500)', linewidth=2, color='#d62728', alpha=0.8)
axes[0].plot(hierarchical_500_smooth, label='Hierarchical (500)', linewidth=2, color='#1f77b4', alpha=0.8)
axes[0].set_title('MADDPG vs Hierarchical - 500 Episodes', fontsize=14)
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Smoothed Reward', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

axes[1].plot(maddpg_5000_smooth, label='MADDPG (5000)', linewidth=2, color='#d62728', alpha=0.8)
axes[1].plot(hierarchical_5000_smooth, label='Hierarchical (5000)', linewidth=2, color='#1f77b4', alpha=0.8)
axes[1].set_title('MADDPG vs Hierarchical - 5000 Episodes', fontsize=14)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Smoothed Reward', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'algorithm_comparison_500_vs_5000.png'), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

maddpg_5000_500_window = np.convolve(maddpg_5000, np.ones(50)/50, mode='valid')
hierarchical_5000_500_window = np.convolve(hierarchical_5000, np.ones(50)/50, mode='valid')

axes[0].plot(maddpg_5000_500_window, label='MADDPG', linewidth=2, color='#d62728')
axes[0].set_title('MADDPG - 5000 Episodes (50-episode average)', fontsize=14)
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('50-episode Rolling Average', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

axes[1].plot(hierarchical_5000_500_window, label='Hierarchical (Dyna-Q)', linewidth=2, color='#1f77b4')
axes[1].set_title('Hierarchical (Dyna-Q) - 5000 Episodes (50-episode average)', fontsize=14)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('50-episode Rolling Average', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'rolling_average_comparison.png'), dpi=150)
plt.close()

print("=" * 70)
print("Convergence Comparison Results")
print("=" * 70)

print("\n--- 500 Episodes ---")
print(f"MADDPG:")
print(f"  Final 50 avg: {np.mean(maddpg_500[-50:]):.2f}")
print(f"  Best reward: {np.max(maddpg_500):.2f}")
print(f"  Worst reward: {np.min(maddpg_500):.2f}")
print(f"  Std: {np.std(maddpg_500):.2f}")
print(f"\nHierarchical (Dyna-Q):")
print(f"  Final 50 avg: {np.mean(hierarchical_500[-50:]):.2f}")
print(f"  Best reward: {np.max(hierarchical_500):.2f}")
print(f"  Worst reward: {np.min(hierarchical_500):.2f}")
print(f"  Std: {np.std(hierarchical_500):.2f}")

print("\n--- 5000 Episodes ---")
print(f"MADDPG:")
print(f"  Final 50 avg: {np.mean(maddpg_5000[-50:]):.2f}")
print(f"  Best reward: {np.max(maddpg_5000):.2f}")
print(f"  Worst reward: {np.min(maddpg_5000):.2f}")
print(f"  Std: {np.std(maddpg_5000):.2f}")
print(f"\nHierarchical (Dyna-Q):")
print(f"  Final 50 avg: {np.mean(hierarchical_5000[-50:]):.2f}")
print(f"  Best reward: {np.max(hierarchical_5000):.2f}")
print(f"  Worst reward: {np.min(hierarchical_5000):.2f}")
print(f"  Std: {np.std(hierarchical_5000):.2f}")

print("\n--- Improvement Analysis ---")
improvement_500 = ((np.mean(hierarchical_500[-50:]) - np.mean(maddpg_500[-50:])) / abs(np.mean(maddpg_500[-50:])) * 100)
improvement_5000 = ((np.mean(hierarchical_5000[-50:]) - np.mean(maddpg_5000[-50:])) / abs(np.mean(maddpg_5000[-50:])) * 100)

print(f"500 Episodes - Hierarchical improvement over MADDPG: {improvement_500:.2f}%")
print(f"5000 Episodes - Hierarchical improvement over MADDPG: {improvement_5000:.2f}%")

print("\n--- Convergence Analysis ---")
print(f"\nMADDPG convergence:")
print(f"  Early (0-100): {np.mean(maddpg_5000[:100]):.2f}")
print(f"  Mid (2000-2500): {np.mean(maddpg_5000[2000:2500]):.2f}")
print(f"  Late (4500-5000): {np.mean(maddpg_5000[4500:]):.2f}")

print(f"\nHierarchical (Dyna-Q) convergence:")
print(f"  Early (0-100): {np.mean(hierarchical_5000[:100]):.2f}")
print(f"  Mid (2000-2500): {np.mean(hierarchical_5000[2000:2500]):.2f}")
print(f"  Late (4500-5000): {np.mean(hierarchical_5000[4500:]):.2f}")

print("\n" + "=" * 70)
print("Charts saved to results/ directory:")
print("  - convergence_comparison_500_vs_5000.png")
print("  - algorithm_comparison_500_vs_5000.png")
print("  - rolling_average_comparison.png")
print("=" * 70)
