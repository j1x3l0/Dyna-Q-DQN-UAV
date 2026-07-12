"""Convergence analysis for 5000-episode training results.

Generates detailed convergence trend chart and statistical report
comparing MADDPG vs Hierarchical (Dyna-Q) learning.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
TIMESTAMP = "20260711_001504"

MADDPG_FILE = os.path.join(RESULTS_DIR, f'maddpg_rewards_test_{TIMESTAMP}.npy')
HIER_FILE = os.path.join(RESULTS_DIR, f'hierarchical_rewards_test_{TIMESTAMP}.npy')


def sliding_window_mean(data, window=100):
    """Compute sliding window mean."""
    if len(data) < window:
        return np.array([np.mean(data)])
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def compute_phase_stats(rewards, phase_ranges):
    """Compute mean and std for each training phase."""
    stats = []
    for label, (start, end) in phase_ranges.items():
        phase = rewards[start:end]
        stats.append({
            'phase': label,
            'range': f"[{start},{end})",
            'mean': float(np.mean(phase)),
            'std': float(np.std(phase)),
            'min': float(np.min(phase)),
            'max': float(np.max(phase))
        })
    return stats


def main():
    maddpg_rewards = np.load(MADDPG_FILE)
    hier_rewards = np.load(HIER_FILE)

    phase_ranges = {
        'Early (0-500)': (0, 500),
        'Mid-Early (500-1000)': (500, 1000),
        'Mid (1000-2000)': (1000, 2000),
        'Mid-Late (2000-3000)': (2000, 3000),
        'Late (3000-4000)': (3000, 4000),
        'Final (4000-5000)': (4000, 5000),
    }

    maddpg_stats = compute_phase_stats(maddpg_rewards, phase_ranges)
    hier_stats = compute_phase_stats(hier_rewards, phase_ranges)

    # ===== Convergence trend chart =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5000-Episode Convergence Analysis: MADDPG vs Hierarchical (Dyna-Q)',
                 fontsize=15, fontweight='bold')

    # Subplot 1: Raw rewards
    ax1 = axes[0, 0]
    ax1.plot(maddpg_rewards, alpha=0.3, color='#d62728', label='MADDPG (raw)')
    ax1.plot(hier_rewards, alpha=0.3, color='#1f77b4', label='Hierarchical (raw)')
    ax1.set_title('Raw Episode Rewards', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Smoothed (sliding window 100)
    ax2 = axes[0, 1]
    m_smooth = sliding_window_mean(maddpg_rewards, 100)
    h_smooth = sliding_window_mean(hier_rewards, 100)
    x_smooth = np.arange(99, len(maddpg_rewards))
    ax2.plot(x_smooth, m_smooth, color='#d62728', linewidth=2, label='MADDPG (100-ep MA)')
    ax2.plot(x_smooth, h_smooth, color='#1f77b4', linewidth=2, label='Hierarchical (100-ep MA)')
    ax2.axhline(y=np.mean(hier_rewards[-100:]), color='#1f77b4',
                linestyle='--', alpha=0.5, label=f'Hier converged: {np.mean(hier_rewards[-100:]):.1f}')
    ax2.axhline(y=np.mean(maddpg_rewards[-100:]), color='#d62728',
                linestyle='--', alpha=0.5, label=f'MADDPG final: {np.mean(maddpg_rewards[-100:]):.1f}')
    ax2.set_title('Smoothed Rewards (100-episode Moving Average)', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smoothed Reward')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Phase-wise mean comparison (bar chart)
    ax3 = axes[1, 0]
    phase_labels = [s['phase'] for s in maddpg_stats]
    m_means = [s['mean'] for s in maddpg_stats]
    h_means = [s['mean'] for s in hier_stats]
    x = np.arange(len(phase_labels))
    width = 0.35
    ax3.bar(x - width / 2, m_means, width, color='#d62728', alpha=0.8, label='MADDPG')
    ax3.bar(x + width / 2, h_means, width, color='#1f77b4', alpha=0.8, label='Hierarchical')
    ax3.set_title('Mean Reward per Training Phase', fontsize=12)
    ax3.set_xlabel('Training Phase')
    ax3.set_ylabel('Mean Reward')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phase_labels, rotation=20, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Standard deviation per phase (stability)
    ax4 = axes[1, 1]
    m_stds = [s['std'] for s in maddpg_stats]
    h_stds = [s['std'] for s in hier_stats]
    ax4.bar(x - width / 2, m_stds, width, color='#d62728', alpha=0.8, label='MADDPG')
    ax4.bar(x + width / 2, h_stds, width, color='#1f77b4', alpha=0.8, label='Hierarchical')
    ax4.set_title('Stability (Std Dev per Training Phase)', fontsize=12)
    ax4.set_xlabel('Training Phase')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phase_labels, rotation=20, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    chart_path = os.path.join(RESULTS_DIR, f'convergence_analysis_{TIMESTAMP}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence chart saved: {chart_path}")

    # ===== Text report =====
    report_path = os.path.join(RESULTS_DIR, f'convergence_report_{TIMESTAMP}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("5000-Episode Convergence Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Start: 2026-07-11 00:15\n")
        f.write(f"Training End:   2026-07-11 15:26\n")
        f.write(f"Duration: ~15 hours\n")
        f.write(f"Episodes: 5000\n\n")

        f.write("=" * 70 + "\n")
        f.write("PHASE-WISE STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- MADDPG ---\n")
        f.write(f"{'Phase':<25} {'Mean':>12} {'Std':>10} {'Min':>12} {'Max':>10}\n")
        f.write("-" * 70 + "\n")
        for s in maddpg_stats:
            f.write(f"{s['phase']:<25} {s['mean']:>12.2f} {s['std']:>10.2f} "
                    f"{s['min']:>12.2f} {s['max']:>10.2f}\n")

        f.write("\n--- Hierarchical (Dyna-Q) ---\n")
        f.write(f"{'Phase':<25} {'Mean':>12} {'Std':>10} {'Min':>12} {'Max':>10}\n")
        f.write("-" * 70 + "\n")
        for s in hier_stats:
            f.write(f"{s['phase']:<25} {s['mean']:>12.2f} {s['std']:>10.2f} "
                    f"{s['min']:>12.2f} {s['max']:>10.2f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. MADDPG Convergence Behavior:\n")
        f.write(f"   - Initial (0-500): {maddpg_stats[0]['mean']:.2f}\n")
        f.write(f"   - Mid (1000-2000): {maddpg_stats[2]['mean']:.2f}\n")
        f.write(f"   - Final (4000-5000): {maddpg_stats[5]['mean']:.2f}\n")
        m_trend = maddpg_stats[5]['mean'] - maddpg_stats[0]['mean']
        f.write(f"   - Trend (Final - Initial): {m_trend:.2f}\n")
        if m_trend < 0:
            f.write(f"   - CONCLUSION: MADDPG shows IMPROVEMENT of {abs(m_trend):.2f}\n")
        else:
            f.write(f"   - CONCLUSION: MADDPG shows DEGRADATION of {m_trend:.2f}\n")
        f.write(f"   - Std change: {maddpg_stats[0]['std']:.2f} -> {maddpg_stats[5]['std']:.2f}\n\n")

        f.write("2. Hierarchical (Dyna-Q) Convergence Behavior:\n")
        f.write(f"   - Initial (0-500): {hier_stats[0]['mean']:.2f}\n")
        f.write(f"   - Mid (1000-2000): {hier_stats[2]['mean']:.2f}\n")
        f.write(f"   - Final (4000-5000): {hier_stats[5]['mean']:.2f}\n")
        h_trend = hier_stats[5]['mean'] - hier_stats[0]['mean']
        f.write(f"   - Trend (Final - Initial): {h_trend:.2f}\n")
        if h_trend > 0:
            f.write(f"   - CONCLUSION: Hierarchical shows DEGRADATION of {h_trend:.2f}\n")
        else:
            f.write(f"   - CONCLUSION: Hierarchical shows IMPROVEMENT of {abs(h_trend):.2f}\n")
        f.write(f"   - Std change: {hier_stats[0]['std']:.2f} -> {hier_stats[5]['std']:.2f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("STABILITY COMPARISON (Lower std = More stable)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Phase':<25} {'MADDPG Std':>15} {'Hier Std':>15} {'Ratio (M/H)':>15}\n")
        f.write("-" * 70 + "\n")
        for m, h in zip(maddpg_stats, hier_stats):
            ratio = m['std'] / h['std'] if h['std'] > 0 else float('inf')
            f.write(f"{m['phase']:<25} {m['std']:>15.2f} {h['std']:>15.2f} {ratio:>15.2f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("=" * 70 + "\n\n")
        final_m = np.mean(maddpg_rewards[-100:])
        final_h = np.mean(hier_rewards[-100:])
        improvement = (final_m - final_h) / abs(final_m) * 100 if final_m != 0 else 0

        f.write(f"Final 100-episode Average Reward:\n")
        f.write(f"  MADDPG:        {final_m:.2f}\n")
        f.write(f"  Hierarchical:  {final_h:.2f}\n")
        f.write(f"  Improvement:   {improvement:.2f}%\n\n")

        f.write(f"Stability (Final phase std):\n")
        f.write(f"  MADDPG:        {maddpg_stats[5]['std']:.2f}\n")
        f.write(f"  Hierarchical:  {hier_stats[5]['std']:.2f}\n")
        f.write(f"  Stability Ratio (M/H): {maddpg_stats[5]['std']/hier_stats[5]['std']:.2f}x\n\n")

        if final_h > final_m:
            f.write("VERDICT: Hierarchical (Dyna-Q) outperforms MADDPG.\n")
            f.write("This is consistent with the original paper's conclusion that\n")
            f.write("the hierarchical learning framework achieves better performance\n")
            f.write("through model-based planning (Dyna-Q) combined with MADDPG.\n")
        else:
            f.write("VERDICT: MADDPG outperforms Hierarchical in this run.\n")

    print(f"Convergence report saved: {report_path}")
    print("\n===== KEY FINDINGS =====")
    print(f"MADDPG:      Initial={maddpg_stats[0]['mean']:.2f} -> Final={maddpg_stats[5]['mean']:.2f}")
    print(f"Hierarchical: Initial={hier_stats[0]['mean']:.2f} -> Final={hier_stats[5]['mean']:.2f}")
    print(f"Improvement: {improvement:.2f}%")


if __name__ == '__main__':
    main()
