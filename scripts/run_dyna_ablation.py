"""Stage-D Dyna-Q ablation runner.

Sweeps planning steps and world-model warm-up while reusing the frozen
benchmark training loop, reward objective, early stopping, and metrics.

Example:
  python scripts/run_dyna_ablation.py --episodes 5 --seeds 42 --gpu-ids 1
  python scripts/run_dyna_ablation.py --k-values 1,3,5,10 \
      --warmups 32,64,128 --seeds 42,123,2026,3407,8888 --gpu-ids 1
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_full_benchmark import (  # noqa: E402
    _run_single_worker,
    compute_convergence_episode,
)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def summarize(values):
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(array))) if len(array) else 0.0
    return {'mean': mean, 'std': std, 'ci95': ci95}


def serialize_run(result, dyna_k, warmup):
    final_window = result.rewards[-50:]
    xi = [row['paper_xi_ratio_of_sums'] for row in result.episode_metrics]
    model_loss = [
        row['model_total_loss']
        for row in result.episode_metrics
        if row.get('model_total_loss') is not None
    ]
    return {
        'variant': f'k{dyna_k}_w{warmup}',
        'dyna_k': dyna_k,
        'dyna_warmup': warmup,
        'seed': result.seed,
        'reward_mode': result.reward_mode,
        'episodes_completed': result.episodes_completed,
        'stopped_early': result.stopped_early,
        'stop_reason': result.stop_reason,
        'duration': result.duration,
        'final_reward': float(np.mean(final_window)),
        'early_reward_500': float(np.mean(result.rewards[:500])),
        'convergence_episode': compute_convergence_episode(result.rewards),
        'final_xi_50': float(np.mean(xi[-50:])),
        'final_model_loss_50': (
            float(np.mean(model_loss[-50:])) if model_loss else None
        ),
        'rewards': result.rewards.tolist(),
        'episode_metrics': result.episode_metrics,
    }


def aggregate(runs):
    summaries = []
    variants = sorted({run['variant'] for run in runs})
    for variant in variants:
        group = [run for run in runs if run['variant'] == variant]
        valid_conv = [
            run['convergence_episode']
            for run in group
            if not np.isnan(run['convergence_episode'])
        ]
        model_losses = [
            run['final_model_loss_50']
            for run in group
            if run['final_model_loss_50'] is not None
        ]
        summaries.append({
            'variant': variant,
            'dyna_k': group[0]['dyna_k'],
            'dyna_warmup': group[0]['dyna_warmup'],
            'num_runs': len(group),
            'final_reward': summarize([run['final_reward'] for run in group]),
            'early_reward_500': summarize([
                run['early_reward_500'] for run in group
            ]),
            'final_xi_50': summarize([run['final_xi_50'] for run in group]),
            'convergence_episode': summarize(valid_conv) if valid_conv else None,
            'final_model_loss_50': (
                summarize(model_losses) if model_losses else None
            ),
            'duration': summarize([run['duration'] for run in group]),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description='Stage-D Dyna-Q ablation')
    parser.add_argument('--k-values', default='1,3,5,10')
    parser.add_argument('--warmups', default='32,64,128')
    parser.add_argument('--seeds', default='42,123,2026,3407,8888')
    parser.add_argument('--episodes', type=int, default=8000)
    parser.add_argument('--reward-mode', choices=('paper_xi', 'ee_ratio'),
                        default='paper_xi')
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--gpu-ids', default='1')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    warmups = parse_int_list(args.warmups)
    seeds = parse_int_list(args.seeds)
    gpu_ids = parse_int_list(args.gpu_ids)
    if not k_values or not warmups or not seeds or not gpu_ids:
        parser.error('K, warm-up, seed, and GPU lists must not be empty')

    tasks = [
        (k, warmup, seed)
        for k in k_values for warmup in warmups for seed in seeds
    ]
    print(
        f'Stage-D: {len(tasks)} runs | K={k_values} | warmups={warmups} | '
        f'seeds={seeds} | reward={args.reward_mode} | GPUs={gpu_ids}'
    )

    runs = []
    workers = min(args.workers, len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, (k, warmup, seed) in enumerate(tasks):
            config = {
                'dyna_k': k,
                'dyna_warmup': warmup,
                'reward_mode': args.reward_mode,
                '_max_episodes': args.episodes,
            }
            gpu_id = gpu_ids[index % len(gpu_ids)]
            future = executor.submit(
                _run_single_worker, gpu_id, 'dyna', seed, config, args.case
            )
            futures[future] = (k, warmup, seed)

        for future in as_completed(futures):
            k, warmup, seed = futures[future]
            try:
                run = serialize_run(future.result(), k, warmup)
                runs.append(run)
                print(
                    f"{run['variant']} seed={seed}: "
                    f"eps={run['episodes_completed']}, "
                    f"reward={run['final_reward']:.3f}, "
                    f"Xi={run['final_xi_50']:.6f}"
                )
            except Exception as error:
                print(f'FAILED k{k}_w{warmup} seed={seed}: {error}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.environ.get(
        'UAV_STORAGE',
        '/mnt/UAV' if os.path.isdir('/mnt/UAV') else
        os.path.join(os.path.dirname(__file__), '..'),
    )
    output_dir = os.path.join(output_dir, 'results', 'dyna_ablation')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'dyna_ablation_{timestamp}.json'
    )
    report = {
        'generated_at': datetime.now().isoformat(),
        'settings': vars(args),
        'runs': sorted(runs, key=lambda x: (x['dyna_k'], x['dyna_warmup'], x['seed'])),
        'summary': aggregate(runs),
    }
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f'Report: {output_path}')

    if len(runs) != len(tasks):
        raise RuntimeError(f'{len(tasks) - len(runs)} ablation run(s) failed')


if __name__ == '__main__':
    main()
