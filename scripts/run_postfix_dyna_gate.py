"""Small post-fix attribution gate for the corrected environment semantics.

Runs three one-seed arms with isolated checkpoint tags:
  1. Hierarchical NoDyna;
  2. HierarchicalAgent with Dyna disabled (K=0 wrapper control);
  3. HierarchicalAgent with validation-gated Dyna planning.

The JSON manifest is rewritten after every completed arm so partial results survive.
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
    'dyna_off': ('dyna', {'dyna_k': 0, 'dyna_warmup_steps': 0}),
    'dyna_gated': ('dyna', {
        'dyna_k': 1,
        'dyna_warmup_steps': 25_600,
        'dyna_model_error_threshold': 0.20,
        'dyna_model_fraction': 0.25,
        'dyna_counterfactual_fraction': 0.25,
    }),
}


def _worker(gpu_id, variant, seed, episodes, reward_mode, case):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    algo, override = VARIANTS[variant]
    override = dict(override)
    override.update({
        'reward_mode': reward_mode,
        '_max_episodes': episodes,
        '_run_tag_suffix': f'_postfix_{variant}',
    })
    return variant, run_single_experiment(algo, seed, override, case)


def _serialize(result):
    data = asdict(result)
    data['rewards'] = result.rewards.tolist()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--reward-mode', default='paper_xi',
                        choices=('paper_xi', 'ee_ratio', 'additive'))
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--gpu-id', default='1')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', f'postfix_dyna_gate_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    manifest = {
        'started_at': datetime.now().isoformat(),
        'seed': args.seed,
        'episodes': args.episodes,
        'reward_mode': args.reward_mode,
        'case': args.case,
        'variants': {},
    }

    workers = min(args.workers, len(VARIANTS))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _worker, args.gpu_id, variant, args.seed, args.episodes,
                args.reward_mode, args.case): variant
            for variant in VARIANTS
        }
        for future in as_completed(futures):
            variant = futures[future]
            try:
                returned_variant, result = future.result()
                manifest['variants'][returned_variant] = {
                    'status': 'completed',
                    'result': _serialize(result),
                }
                print(f'{returned_variant}: completed, episodes={result.episodes_completed}, '
                      f'final_reward={result.rewards[-1]:.3f}', flush=True)
            except Exception as exc:
                manifest['variants'][variant] = {
                    'status': 'failed',
                    'error': repr(exc),
                }
                print(f'{variant}: FAILED: {exc!r}', flush=True)
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

    manifest['completed_at'] = datetime.now().isoformat()
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f'manifest={os.path.abspath(manifest_path)}', flush=True)


if __name__ == '__main__':
    main()
