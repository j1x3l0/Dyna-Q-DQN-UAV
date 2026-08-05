"""Re-run failed / missing arms in a scenario gate manifest (post-fix impl).

Reads a scenario manifest produced by run_scenario_gate.py, finds (seed, variant)
entries that are missing or whose status is not 'completed', and re-runs ONLY
those arms via run_single_experiment (same scenario overrides, same seeds).
Rewrites the manifest after every arm so partial re-runs survive.

Usage:
  python scripts/run_scenario_rerun.py --manifest <manifest.json> --gpu-id 0
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from run_postfix_dyna_gate import VARIANTS
from run_full_benchmark import run_single_experiment


def _worker(gpu_id, variant, seed, episodes, reward_mode, scenario_overrides):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    algo, override = VARIANTS[variant]
    override = dict(override)
    scenario_name = scenario_overrides.get('_name', 'scn')
    case = scenario_overrides.get('case', 1)
    for key, val in scenario_overrides.items():
        if key not in ('case', '_name'):
            override[key] = val
    override.update({
        'reward_mode': reward_mode,
        '_max_episodes': episodes,
        '_run_tag_suffix': f'_{scenario_name}_postfix_{variant}',
    })
    return variant, run_single_experiment(algo, seed, override, case)


def _serialize(result):
    data = asdict(result)
    data['rewards'] = result.rewards.tolist()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)

    scenario_overrides = manifest.get('scenario_overrides', {})
    seeds = manifest.get('seeds', [])
    episodes = manifest.get('episodes', 200)
    reward_mode = manifest.get('reward_mode', 'paper_xi')
    arms = manifest.get('arms') or list(VARIANTS)

    # collect (seed, variant) that are failed/missing
    todo = []
    for sd in seeds:
        for variant in arms:
            rec = manifest.get('runs', {}).get(str(sd), {}).get(variant, {})
            if rec.get('status') != 'completed':
                todo.append((sd, variant))

    if not todo:
        print('no failed/missing arms to re-run')
        return

    print(f're-running {len(todo)} arms: {todo}')
    for sd, variant in todo:
        workers = min(args.workers, 1 if len(todo) == 1 else args.workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, args.gpu_id, variant, int(sd), episodes,
                                reward_mode, scenario_overrides): variant
            }
            for future in as_completed(futures):
                v = futures[future]
                try:
                    returned_variant, result = future.result()
                    manifest.setdefault('runs', {}).setdefault(str(sd), {})[returned_variant] = {
                        'status': 'completed',
                        'episodes_completed': result.episodes_completed,
                        'duration_s': result.duration,
                        'stopped_early': result.stopped_early,
                        'final_reward': float(result.rewards[-1]) if len(result.rewards) else None,
                        'result': _serialize(result),
                    }
                    print(f'{scenario_overrides.get("_name")} seed={sd} {v}: completed, '
                          f'final_reward={result.rewards[-1]:.3f}', flush=True)
                except Exception as exc:
                    manifest.setdefault('runs', {}).setdefault(str(sd), {})[v] = {
                        'status': 'failed', 'error': repr(exc)}
                    print(f'{v}: re-run FAILED: {exc!r}', flush=True)
                with open(manifest_path, 'w', encoding='utf-8') as handle:
                    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f'manifest updated: {manifest_path}')


if __name__ == '__main__':
    main()
