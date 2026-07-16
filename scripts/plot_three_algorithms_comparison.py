import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

maddpg_rewards = np.load(os.path.join(results_dir, 'maddpg_rewards_case1.npy'))
hierarchical_dyna_rewards = np.load(os.path.join(results_dir, 'hierarchical_rewards_case1.npy'))
hierarchical_no_dyna_rewards = np.load(os.path.join(results_dir, 'hierarchical_no_dyna_rewards_500_20260715_121635.npy'))

algorithms = [
    {'name': 'MADDPG', 'rewards': maddpg_rewards, 'color': '#d62728', 'label': 'MADDPG'},
    {'name': 'Hierarchical (Dyna-Q)', 'rewards': hierarchical_dyna_rewards, 'color': '#1f77b4', 'label': 'Hierarchical + Dyna-Q'},
    {'name': 'Hierarchical (no Dyna-Q)', 'rewards': hierarchical_no_dyna_rewards, 'color': '#ff7f0e', 'label': 'Hierarchical (no Dyna-Q)'},
]

window = 10
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for algo in algorithms:
    smooth = np.convolve(algo['rewards'], np.ones(window)/window, mode='valid')
    axes[0].plot(smooth, label=algo['label'], linewidth=2, color=algo['color'], alpha=0.8)

axes[0].set_title('Three Algorithms Comparison - Smoothed Reward (window=10)', fontsize=14)
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Average Reward', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

for algo in algorithms:
    axes[1].plot(algo['rewards'], label=algo['label'], linewidth=1.5, color=algo['color'], alpha=0.6)

axes[1].set_title('Three Algorithms Comparison - Raw Reward', fontsize=14)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Reward', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'three_algorithms_comparison_500.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

window = 50
for algo in algorithms:
    smooth = np.convolve(algo['rewards'], np.ones(window)/window, mode='valid')
    ax.plot(smooth, label=algo['label'], linewidth=2.5, color=algo['color'], alpha=0.85)

ax.set_title('Three Algorithms Comparison - 50-Episode Rolling Average', fontsize=16)
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('50-Episode Average Reward', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'three_algorithms_rolling_avg_500.png'), dpi=150)
plt.close()

print("=" * 70)
print("Three Algorithms Comparison Results (500 Episodes)")
print("=" * 70)

for algo in algorithms:
    rewards = algo['rewards']
    print(f"\n{algo['name']}:")
    print(f"  Final 50 avg: {np.mean(rewards[-50:]):.2f}")
    print(f"  Best reward: {np.max(rewards):.2f}")
    print(f"  Worst reward: {np.min(rewards):.2f}")
    print(f"  Std: {np.std(rewards):.2f}")

print("\n--- Improvement Analysis ---")
maddpg_final = np.mean(maddpg_rewards[-50:])
hierarchical_dyna_final = np.mean(hierarchical_dyna_rewards[-50:])
hierarchical_no_dyna_final = np.mean(hierarchical_no_dyna_rewards[-50:])

improvement_dyna = ((hierarchical_dyna_final - maddpg_final) / abs(maddpg_final) * 100)
improvement_no_dyna = ((hierarchical_no_dyna_final - maddpg_final) / abs(maddpg_final) * 100)
dyna_vs_no_dyna = ((hierarchical_dyna_final - hierarchical_no_dyna_final) / abs(hierarchical_no_dyna_final) * 100)

print(f"Hierarchical (Dyna-Q) improvement over MADDPG: {improvement_dyna:.2f}%")
print(f"Hierarchical (no Dyna-Q) improvement over MADDPG: {improvement_no_dyna:.2f}%")
print(f"Dyna-Q vs no Dyna-Q improvement: {dyna_vs_no_dyna:.2f}%")

print("\n" + "=" * 70)
print("Charts saved to results/ directory:")
print("  - three_algorithms_comparison_500.png")
print("  - three_algorithms_rolling_avg_500.png")
print("=" * 70)