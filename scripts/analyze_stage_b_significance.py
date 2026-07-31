"""Exact paired significance analysis for the five-seed Stage-B benchmark."""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def convergence_episode(rewards, window=100, threshold=0.9):
    rewards = np.asarray(rewards, dtype=float)
    if len(rewards) < window:
        return float(len(rewards))
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    target = threshold * float(np.max(rolling))
    return float(np.argmax(rolling >= target) + window)


def run_metrics(run):
    rewards = np.asarray(run['rewards'], dtype=float)
    episodes = run.get('episode_metrics', [])
    xi = [row['paper_xi_ratio_of_sums'] for row in episodes]
    return {
        'final_reward': float(np.mean(rewards[-50:])),
        'early_reward_500': float(np.mean(rewards[:500])),
        'convergence_episode': convergence_episode(rewards),
        'final_xi_50': float(np.mean(xi[-50:])),
        'duration_hours': float(run['duration'] / 3600),
    }


def exact_sign_flip(diff):
    diff = np.asarray(diff, dtype=float)
    observed = float(np.mean(diff))
    permuted = np.array([
        np.mean(diff * signs)
        for signs in itertools.product((-1, 1), repeat=len(diff))
    ])
    tolerance = 1e-12
    two_sided = float(np.mean(
        np.abs(permuted) >= abs(observed) - tolerance
    ))
    greater = float(np.mean(permuted >= observed - tolerance))
    return observed, two_sided, greater


def exact_bootstrap_ci(diff):
    diff = np.asarray(diff, dtype=float)
    means = np.array([
        np.mean(diff[list(indices)])
        for indices in itertools.product(range(len(diff)), repeat=len(diff))
    ])
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def paired_test(left, right):
    diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    mean_diff, p_two, p_greater = exact_sign_flip(diff)
    std = float(np.std(diff, ddof=1))
    return {
        'differences': diff.tolist(),
        'mean_difference': mean_diff,
        'bootstrap_ci95': exact_bootstrap_ci(diff),
        'cohens_dz': mean_diff / std if std > 0 else None,
        'p_exact_two_sided': p_two,
        'p_exact_directional': p_greater,
    }


def holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=Path)
    parser.add_argument('output_json', type=Path)
    args = parser.parse_args()
    assert exact_sign_flip([1] * 5)[1:] == (0.0625, 0.03125)
    assert holm_adjust([0.01, 0.03, 0.04]) == [0.03, 0.06, 0.06]

    source = json.loads(args.input_json.read_text())
    runs = {
        algo: {
            int(run['seed']): run_metrics(run)
            for run in source['runs'] if run['algo'] == algo
        }
        for algo in ('maddpg', 'nodyna', 'dyna')
    }
    seeds = sorted(set.intersection(*(set(group) for group in runs.values())))
    if len(seeds) != 5:
        raise ValueError(f'expected 5 paired seeds, found {seeds}')

    def values(algo, metric):
        return [runs[algo][seed][metric] for seed in seeds]

    tests = {
        'nodyna_vs_maddpg_final_reward': paired_test(
            values('nodyna', 'final_reward'),
            values('maddpg', 'final_reward'),
        ),
        'dyna_vs_nodyna_final_reward': paired_test(
            values('dyna', 'final_reward'),
            values('nodyna', 'final_reward'),
        ),
        'dyna_vs_nodyna_early_reward_500': paired_test(
            values('dyna', 'early_reward_500'),
            values('nodyna', 'early_reward_500'),
        ),
        'nodyna_minus_dyna_convergence_episode': paired_test(
            values('nodyna', 'convergence_episode'),
            values('dyna', 'convergence_episode'),
        ),
        'dyna_vs_nodyna_final_xi_50': paired_test(
            values('dyna', 'final_xi_50'),
            values('nodyna', 'final_xi_50'),
        ),
        'dyna_vs_nodyna_duration_hours': paired_test(
            values('dyna', 'duration_hours'),
            values('nodyna', 'duration_hours'),
        ),
    }

    primary = [
        'nodyna_vs_maddpg_final_reward',
        'dyna_vs_nodyna_early_reward_500',
        'nodyna_minus_dyna_convergence_episode',
    ]
    adjusted = holm_adjust([
        tests[name]['p_exact_directional'] for name in primary
    ])
    for name, p_adjusted in zip(primary, adjusted):
        tests[name]['p_holm_directional'] = p_adjusted

    report = {
        'source': str(args.input_json),
        'seeds': seeds,
        'alpha': 0.05,
        'method': {
            'test': 'exact paired sign-flip permutation on mean difference',
            'permutations': int(2 ** len(seeds)),
            'ci': 'exact paired bootstrap percentile over all n^n resamples',
            'multiplicity': 'Holm correction over three directional primary tests',
        },
        'per_seed_metrics': runs,
        'tests': tests,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(args.output_json)


if __name__ == '__main__':
    main()
