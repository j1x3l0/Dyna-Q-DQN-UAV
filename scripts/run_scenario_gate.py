"""Multi-seed, multi-scenario post-fix Dyna gate (200-episode comparisons).

Runs NoDyna / Dyna-off (K=0) / Dyna-gated for multiple seeds across scenario
complexity configs on the fixed (dyna-core-fix) implementation.

Scenarios (complexity / robustness probes):
  base   : case 1, boundary=500   (baseline; same config as the seed-42 gate)
  case2  : case 2, boundary=500   (all UAVs same start -> harder initial deployment)
  sparse : case 1, boundary=800   (larger area -> sparser coverage -> higher real-interaction cost)

Purpose: probe whether Dyna's sample-efficiency / reward gap vs NoDyna changes
with scenario complexity after the core semantic fixes. Primary metric for the
analysis pass is episodes-to-threshold (prospective) and mean-reward gap.

One scenario per invocation (so scenarios can run in parallel across GPUs):
  python scripts/run_scenario_gate.py --scenario base   --gpu-id 0 &
  python scripts/run_scenario_gate.py --scenario case2  --gpu-id 1 &
  python scripts/run_scenario_gate.py --scenario sparse --gpu-id 0 &

A per-scenario manifest (manifest.json) is rewritten after every completed arm
so partial results survive.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from run_postfix_dyna_gate import VARIANTS  # reuse the three arm configs
from run_full_benchmark import run_single_experiment


SCENARIOS = {
    # name: {config attribute overrides, 'case': env case, '_name': run-tag suffix}
    'base':   {'boundary': 500.0, 'case': 1, '_name': 'b500'},
    'case2':  {'boundary': 500.0, 'case': 2, '_name': 'c2'},
    'sparse': {'boundary': 800.0, 'case': 1, '_name': 'b800'},
}


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
    parser.add_argument('--scenario', required=True,
                        choices=sorted(SCENARIOS))
    parser.add_argument('--seeds', default='42,123,2026,3407,8888')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--reward-mode', default='paper_xi',
                        choices=('paper_xi', 'ee_ratio', 'additive'))
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    scenario_overrides = SCENARIOS[args.scenario]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results',
        f'scenario_gate_{args.scenario}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.json')
    manifest = {
        'started_at': datetime.now().isoformat(),
        'scenario': args.scenario,
        'scenario_overrides': scenario_overrides,
        'seeds': seeds,
        'episodes': args.episodes,
        'reward_mode': args.reward_mode,
        'gpu_id': args.gpu_id,
        'arms': list(VARIANTS),
        'runs': {},
    }

    for seed in seeds:
        print(f'--- scenario={args.scenario} seed={seed} ---', flush=True)
        manifest['runs'][str(seed)] = {}
        workers = min(args.workers, len(VARIANTS))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, args.gpu_id, variant, seed,
                                args.episodes, args.reward_mode,
                                scenario_overrides): variant
                for variant in VARIANTS
            }
            for future in as_completed(futures):
                variant = futures[future]
                try:
                    returned_variant, result = future.result()
                    manifest['runs'][str(seed)][returned_variant] = {
                        'status': 'completed',
                        'episodes_completed': result.episodes_completed,
                        'duration_s': result.duration,
                        'stopped_early': result.stopped_early,
                        'final_reward': float(result.rewards[-1]) if len(result.rewards) else None,
                        'result': _serialize(result),
                    }
                    print(f'{variant}: completed, episodes={result.episodes_completed}, '
                          f'final_reward={result.rewards[-1]:.3f}', flush=True)
                except Exception as exc:
                    manifest['runs'][str(seed)][variant] = {
                        'status': 'failed', 'error': repr(exc)}
                    print(f'{variant}: FAILED: {exc!r}', flush=True)
                with open(manifest_path, 'w', encoding='utf-8') as handle:
                    json.dump(manifest, handle, ensure_ascii=False, indent=2)

    manifest['completed_at'] = datetime.now().isoformat()
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f'manifest={os.path.abspath(manifest_path)}', flush=True)


if __name__ == '__main__':
    main()
