#!/usr/bin/env python3
"""Run the Stage-E MATD3/CoP-MADDPG short hyperparameter gate."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_full_benchmark import run_single_experiment


VARIANTS = {
    'matd3_default': ('matd3', {}),
    'matd3_balanced': ('matd3', {
        'matd3_actor_lr': 3e-4,
        'matd3_critic_lr': 3e-4,
        'matd3_target_noise': 0.1,
    }),
    'matd3_conservative': ('matd3', {
        'matd3_actor_lr': 1e-4,
        'matd3_critic_lr': 1e-4,
        'matd3_target_noise': 0.1,
    }),
    'cop_default': ('cop_maddpg', {}),
    'cop_balanced': ('cop_maddpg', {
        'cop_actor_lr': 3e-4,
        'cop_critic_lr': 3e-4,
        'cop_encoder_lr': 3e-4,
    }),
    'cop_conservative': ('cop_maddpg', {
        'cop_actor_lr': 1e-4,
        'cop_critic_lr': 1e-4,
        'cop_encoder_lr': 1e-4,
    }),
}


def worker(gpu_id, name, seed, episodes, reward_mode, case):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    algo, params = VARIANTS[name]
    override = {
        **params,
        '_max_episodes': episodes,
        '_run_tag_suffix': f'_stagee_{name}',
        'reward_mode': reward_mode,
    }
    return name, params, run_single_experiment(algo, seed, override, case)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def summarize(name, params, result):
    rewards = np.asarray(result.rewards, dtype=float)
    return {
        'variant': name,
        'algo': result.algo,
        'seed': result.seed,
        'params': params,
        'episodes_completed': result.episodes_completed,
        'stopped_early': result.stopped_early,
        'stop_reason': result.stop_reason,
        'duration_seconds': result.duration,
        'reward_mean_50': float(np.mean(rewards[-50:])),
        'reward_auc': float(np.mean(rewards)),
        'rewards': rewards,
        'metrics': result.metrics,
        'episode_metrics': result.episode_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--reward-mode', default='paper_xi', choices=('paper_xi', 'ee_ratio'))
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--gpu-id', type=int, default=1)
    parser.add_argument('--workers', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    names = [name.strip() for name in args.variants.split(',') if name.strip()]
    unknown = set(names) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Unknown variants: {sorted(unknown)}')

    results, errors = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(names))) as executor:
        futures = {
            executor.submit(
                worker, args.gpu_id, name, args.seed, args.episodes,
                args.reward_mode, args.case,
            ): name
            for name in names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                variant, params, result = future.result()
                row = summarize(variant, params, result)
                results.append(row)
                print(f"{name}: reward50={row['reward_mean_50']:.3f}, "
                      f"auc={row['reward_auc']:.3f}, time={row['duration_seconds']:.0f}s")
            except Exception as exc:
                errors.append({'variant': name, 'error': repr(exc)})
                print(f'{name}: FAILED: {exc}', file=sys.stderr)

    base = os.environ.get('UAV_STORAGE', '/mnt/UAV' if os.path.isdir('/mnt/UAV') else '.')
    output_dir = os.path.join(base, 'results', 'stage_e_gate')
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = os.path.join(output_dir, f'stage_e_gate_{stamp}.json')
    payload = {
        'metadata': vars(args),
        'results': sorted(results, key=lambda row: row['variant']),
        'errors': errors,
    }
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(serializable(payload), handle, ensure_ascii=False)
    print(f'Report: {output}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
