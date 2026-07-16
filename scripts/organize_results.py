import os
import shutil
import sys

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

file_groups = {
    '20260711_test': [
        'maddpg_rewards_test_20260711_001504.npy',
        'hierarchical_rewards_test_20260711_001504.npy',
        'experiment_report_20260711_001504.txt',
        'convergence_report_20260711_001504.txt',
    ],
    '20260711_dyna_validation': [
        'dyna_validation_comparison_500_20260711_172547.png',
        'maddpg_rewards_dyna_val_500_20260711_172547.npy',
        'hierarchical_no_dyna_rewards_500_20260711_172547.npy',
        'dyna_validation_report_500_20260711_172547.txt',
    ],
    '20260711_hierarchical_no_dyna_500': [
        'hierarchical_no_dyna_500_20260711_190829.png',
        'hierarchical_no_dyna_report_500_20260711_190829.txt',
    ],
    '20260711_rolling_avg': [
        'rolling_average_comparison.png',
    ],
    '20260712_hierarchical_no_dyna_5000': [
        'hierarchical_no_dyna_5000_20260712_085614.png',
        'hierarchical_no_dyna_report_5000_20260712_085614.txt',
        'hierarchical_no_dyna_rewards_5000_20260712_085614.npy',
    ],
    '20260713_training_5000': [
        'hierarchical_dyna_rewards_5000_20260713_122952.npy',
        'hierarchical_dyna_rewards_5000_20260713_122952.png',
        'hierarchical_no_dyna_rewards_5000_20260713_122952.npy',
        'hierarchical_no_dyna_rewards_5000_20260713_122952.png',
        'maddpg_rewards_5000_20260713_122952.npy',
        'maddpg_rewards_5000_20260713_122952.png',
        'reward_comparison_5000_20260713_122952.png',
        'training_comparison_report_5000_20260713_122952.txt',
    ],
    '20260714_analysis': [
        'algorithm_comparison_500_vs_5000.png',
        'convergence_comparison_500_vs_5000.png',
        'training_analysis_report.md',
    ],
    '20260715_training_500': [
        'maddpg_rewards_case1.npy',
        'maddpg_rewards_case1.png',
        'hierarchical_no_dyna_500_20260715_121635.png',
        'hierarchical_no_dyna_rewards_500_20260715_121635.npy',
        'hierarchical_no_dyna_report_500_20260715_121635.txt',
        'hierarchical_rewards_case1.npy',
        'hierarchical_rewards_case1.png',
        'three_algorithms_comparison_500.png',
        'three_algorithms_rolling_avg_500.png',
    ],
}

print("=" * 70)
print("Organizing results directory...")
print(f"Results dir: {results_dir}")
print("=" * 70)

for group_name, files in file_groups.items():
    group_dir = os.path.join(results_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    print(f"\nProcessing group: {group_name}")
    print(f"  Target directory: {group_dir}")
    
    moved_count = 0
    for filename in files:
        src_path = os.path.join(results_dir, filename)
        dst_path = os.path.join(group_dir, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"  ✓ Moved: {filename}")
            moved_count += 1
        else:
            print(f"  ✗ Not found: {filename}")
    
    print(f"  Total: {moved_count}/{len(files)} files")

print("\n" + "=" * 70)
print("Organization complete!")
print("=" * 70)

print("\nRemaining files in root results/ directory:")
remaining = [f for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
if remaining:
    for f in remaining:
        print(f"  {f}")
else:
    print("  (empty)")

print("\nResult directories created:")
for dir_name in sorted(os.listdir(results_dir)):
    dir_path = os.path.join(results_dir, dir_name)
    if os.path.isdir(dir_path):
        file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        print(f"  {dir_name}/ ({file_count} files)")