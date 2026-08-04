#!/usr/bin/env python3
"""Run the Stage-E MATD3 / CoP-MADDPG 5-seed formal training.

Uses the gate-selected conservative configurations:
  - MATD3:      actor/critic lr=1e-4, target noise=0.1
  - CoP-MADDPG: actor/critic/encoder lr=1e-4

Runs with the same seeds / early-stopping / case / reward-mode as Stage B so the
results are comparable to the core-three (MADDPG, NoDyna, DynaQ) benchmark.
Writes the standard benchmark report plus a Stage-E manifest recording commit,
configs, seeds and output paths.

Usage:
  python scripts/run_stage_e_formal.py                    # matd3,cop_maddpg x 5 seeds
  python scripts/run_stage_e_formal.py --seeds 42 --episodes 1   # CPU smoke test
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from run_full_benchmark import (
    ALGO_CONFIGS,
    aggregate_results,
    run_single_experiment,
    write_report,
)

# Gate-selected conservative hyperparameters (Stage E gate, 2026-08-03)
CONSERVATIVE_OVERRIDES = {
    'matd3': {
        'matd3_actor_lr': 1e-4,
        'matd3_critic_lr': 1e-4,
        'matd3_target_noise': 0.1,
    },
    'cop_maddpg': {
        'cop_actor_lr': 1e-4,
        'cop_critic_lr': 1e-4,
        'cop_encoder_lr': 1e-4,
    },
}

DEFAULT_ALGOS = ('matd3', 'cop_maddpg')
DEFAULT_SEEDS = (42, 123, 2026, 3407, 8888)
RUN_TAG_SUFFIX = '_stagee_conservative'


def worker(gpu_id, algo, seed, case, reward_mode, episodes):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    override = {
        **CONSERVATIVE_OVERRIDES[algo],
        '_run_tag_suffix': RUN_TAG_SUFFIX,
        'reward_mode': reward_mode,
    }
    if episodes is not None:
        override['_max_episodes'] = episodes
    return algo, seed, run_single_experiment(algo, seed, override, case)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algos', default=','.join(DEFAULT_ALGOS))
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)))
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--reward-mode', default='paper_xi',
                        choices=('paper_xi', 'ee_ratio'))
    parser.add_argument('--gpu-ids', type=str, default='1')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override max episodes (smoke tests only)')
    return parser.parse_args()


def git_head():
    try:
        out = os.popen('git -C %s rev-parse HEAD 2>/dev/null' %
                       os.path.join(os.path.dirname(__file__), '..')).read().strip()
        return out or 'unknown'
    except Exception:
        return 'unknown'


def main():
    args = parse_args()
    algos = [a.strip() for a in args.algos.split(',') if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
    unknown = set(algos) - set(CONSERVATIVE_OVERRIDES)
    if unknown:
        raise ValueError(f'No conservative config for: {sorted(unknown)}')

    tasks = [(gpu_ids[i % len(gpu_ids)], algo, seed, args.case, args.reward_mode,
              args.episodes)
             for i, (algo, seed) in enumerate(
                 (a, s) for a in algos for s in seeds)]
    total = len(tasks)
    workers = min(args.workers, total)

    print(f'Stage-E formal training: algos={algos} seeds={seeds} case={args.case} '
          f'reward={args.reward_mode} gpu_ids={gpu_ids} workers={workers} '
          f'total={total} runs')

    run_results = {}
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, *t): t for t in tasks}
        for fut in as_completed(futures):
            algo, seed = futures[fut][1], futures[fut][2]
            try:
                algo_done, seed_done, result = fut.result()
                run_results[(algo_done, seed_done)] = result
                final = (float(result.rewards[-50:].mean())
                         if len(result.rewards) >= 50
                         else float(result.rewards.mean()))
                print(f'[{algo_done} seed={seed_done}] done: '
                      f'{result.episodes_completed} eps, final_r={final:.2f}, '
                      f'stop={result.stopped_early} ({result.stop_reason})')
            except Exception as exc:
                failures.append({'algo': algo, 'seed': seed, 'error': repr(exc)})
                print(f'[{algo} seed={seed}] FAILED: {exc}', file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    if len(run_results) != total:
        print(f'\nWARNING: {len(run_results)}/{total} runs completed', file=sys.stderr)
        if not run_results:
            raise SystemExit(1)

    ordered = [run_results[(algo, seed)] for algo in algos for seed in seeds
               if (algo, seed) in run_results]
    summary_rows = aggregate_results(ordered)
    report_path, json_path = write_report(ordered, summary_rows)
    print(f'Report: {report_path}')
    print(f'JSON:   {json_path}')

    base = os.environ.get('UAV_STORAGE', '/mnt/UAV' if os.path.isdir('/mnt/UAV') else '.')
    manifest_dir = os.path.join(base, 'results', 'stage_e_formal')
    os.makedirs(manifest_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'commit': git_head(),
        'configs': CONSERVATIVE_OVERRIDES,
        'algos': algos,
        'seeds': seeds,
        'case': args.case,
        'reward_mode': args.reward_mode,
        'gpu_ids': gpu_ids,
        'workers': workers,
        'run_tag_suffix': RUN_TAG_SUFFIX,
        'early_stop': {
            algo: {k: v for k, v in ALGO_CONFIGS[algo].items()
                   if k in ('max_episodes', 'early_stop_patience',
                            'early_stop_min_delta', 'convergence_window')}
            for algo in algos
        },
        'episodes_override': args.episodes,
        'report_path': report_path,
        'json_path': json_path,
        'failures': failures,
        'completed_runs': len(run_results),
    }
    manifest_path = os.path.join(manifest_dir, f'stage_e_formal_manifest_{stamp}.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f'Manifest: {manifest_path}')

    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
