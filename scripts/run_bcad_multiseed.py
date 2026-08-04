"""Paired 200-episode screen for the Bellman-Consistent Adaptive Dyna method.

The default experiment compares NoDyna, the repaired state-MAE-gated Dyna
baseline, and BCAD on five paired seeds.  The manifest is rewritten after every
completed task so a pre-empted server retains all finished runs.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

from run_full_benchmark import run_single_experiment


VARIANTS = {
    'nodyna': ('nodyna', {}),
    'dyna_current': ('dyna', {
        'dyna_k': 1,
        'dyna_warmup_steps': 25_600,
        'dyna_planning_strategy': 'state_gate',
        'dyna_model_error_threshold': 0.20,
        'dyna_model_fraction': 0.25,
        'dyna_counterfactual_fraction': 0.25,
    }),
    'bcad': ('dyna', {
        'dyna_k': 1,
        'dyna_warmup_steps': 25_600,
        'dyna_planning_strategy': 'bcad',
        'dyna_model_fraction': 0.25,
        'dyna_counterfactual_fraction': 0.0,
        'dyna_bellman_beta': 2.0,
        'dyna_bellman_denom_floor': 1.0,
        'dyna_plan_probability_max': 0.25,
        'dyna_plan_ramp_steps': 8_000,
        'dyna_model_update_lr_scale': 0.10,
    }),
}


def _worker(gpu_id, variant, seed, episodes, reward_mode, case):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    algo, override = VARIANTS[variant]
    override = dict(override)
    override.update({
        'reward_mode': reward_mode,
        '_max_episodes': episodes,
        '_run_tag_suffix': f'_bcad_screen_{variant}',
    })
    return variant, seed, run_single_experiment(algo, seed, override, case)


def _serialize(result):
    data = asdict(result)
    data['rewards'] = result.rewards.tolist()
    return data


def _write_manifest(path, manifest):
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42,123,2026,3407,888')
    parser.add_argument('--variants', default='nodyna,dyna_current,bcad')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--reward-mode', default='paper_xi',
                        choices=('paper_xi', 'ee_ratio', 'additive'))
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    variants = [x.strip() for x in args.variants.split(',') if x.strip()]
    unknown = [x for x in variants if x not in VARIANTS]
    if unknown:
        raise ValueError(f'Unknown variants: {unknown}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', f'bcad_multiseed_{timestamp}'))
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    manifest = {
        'started_at': datetime.now().isoformat(),
        'seeds': seeds,
        'variants_requested': variants,
        'episodes': args.episodes,
        'reward_mode': args.reward_mode,
        'case': args.case,
        'variant_configs': {name: VARIANTS[name][1] for name in variants},
        'runs': {},
    }
    _write_manifest(manifest_path, manifest)

    tasks = [(variant, seed) for seed in seeds for variant in variants]
    workers = min(max(1, args.workers), len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _worker, args.gpu_id, variant, seed, args.episodes,
                args.reward_mode, args.case): (variant, seed)
            for variant, seed in tasks
        }
        for future in as_completed(futures):
            variant, seed = futures[future]
            key = f'{variant}_seed{seed}'
            try:
                returned_variant, returned_seed, result = future.result()
                if returned_variant != variant or returned_seed != seed:
                    raise RuntimeError('worker identity mismatch')
                manifest['runs'][key] = {
                    'status': 'completed',
                    'result': _serialize(result),
                }
                print(
                    f'{key}: completed, episodes={result.episodes_completed}, '
                    f'last50={sum(result.rewards[-50:]) / min(50, len(result.rewards)):.3f}',
                    flush=True)
            except Exception as exc:
                manifest['runs'][key] = {
                    'status': 'failed',
                    'error': repr(exc),
                }
                print(f'{key}: FAILED: {exc!r}', flush=True)
            _write_manifest(manifest_path, manifest)

    manifest['completed_at'] = datetime.now().isoformat()
    _write_manifest(manifest_path, manifest)
    print(f'manifest={manifest_path}', flush=True)


if __name__ == '__main__':
    main()
