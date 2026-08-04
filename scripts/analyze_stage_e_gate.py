#!/usr/bin/env python3
"""Analyze the Stage-E single-seed short parameter gate."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def row_for(run):
    rewards = np.asarray(run['rewards'], dtype=float)
    episodes = run['episode_metrics']

    def values(key):
        return np.asarray([
            np.nan if item.get(key) is None else item[key] for item in episodes
        ], dtype=float)

    energy = values('energy_consumed') + values('flight_energy')
    return {
        'variant': run['variant'],
        'algo': run['algo'],
        'episodes': run['episodes_completed'],
        'reward_auc': float(np.mean(rewards)),
        'reward_last_50': float(np.mean(rewards[-50:])),
        'xi_last_50': float(np.nanmean(values('paper_xi_ratio_of_sums')[-50:])),
        'data_sent_last_50': float(np.mean(values('data_sent_to_rbs')[-50:])),
        'data_received_last_50': float(np.mean(values('data_received')[-50:])),
        'energy_last_50': float(np.mean(energy[-50:])),
        'collisions_per_episode': float(np.sum(values('collision_events')) / len(episodes)),
        'duration_seconds': float(run['duration_seconds']),
        'params': run['params'],
        'rewards': rewards,
    }


def choose(rows, algo):
    candidates = [row for row in rows if row['algo'] == algo]
    valid = [
        row for row in candidates
        if row['episodes'] == 200
        and np.isfinite(row['reward_auc'])
        and np.isfinite(row['xi_last_50'])
    ]
    if not valid:
        raise ValueError(f'No valid candidates for {algo}')
    return max(valid, key=lambda row: (row['reward_auc'], row['xi_last_50']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_json')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.input_json, encoding='utf-8'))
    if payload['errors'] or len(payload['results']) != 6:
        raise ValueError(f"Expected 6 successful runs; errors={payload['errors']}")
    rows = sorted((row_for(run) for run in payload['results']),
                  key=lambda row: (row['algo'], row['variant']))
    selected = {
        algo: choose(rows, algo)
        for algo in ('matd3', 'cop_maddpg')
    }

    for row in rows:
        default = next(
            item for item in rows
            if item['algo'] == row['algo'] and item['variant'].endswith('_default')
        )
        for key in ('reward_auc', 'reward_last_50', 'xi_last_50',
                    'data_sent_last_50', 'energy_last_50', 'duration_seconds'):
            row[f'{key}_vs_default_pct'] = (
                100 * (row[key] - default[key]) / max(abs(default[key]), 1e-12)
            )

    columns = [key for key in rows[0] if key not in ('rewards', 'params')]
    with (output / 'stage_e_gate_metrics_20260803.csv').open(
        'w', newline='', encoding='utf-8'
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])

    analysis = {
        'input': args.input_json,
        'scope': 'single-seed 200-episode engineering gate; not statistical evidence',
        'selection_rule': 'highest reward AUC among complete finite runs; final Xi breaks ties',
        'selected': {
            algo: {
                key: value for key, value in row.items()
                if key not in ('rewards',)
            }
            for algo, row in selected.items()
        },
        'rows': [
            {key: value for key, value in row.items() if key != 'rewards'}
            for row in rows
        ],
    }
    with (output / 'stage_e_gate_analysis_20260803.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, algo in enumerate(('matd3', 'cop_maddpg')):
        group = [row for row in rows if row['algo'] == algo]
        labels = [row['variant'].replace('matd3_', '').replace('cop_', '') for row in group]
        winner = selected[algo]['variant']
        colors = ['#2a6fbb' if row['variant'] == winner else '0.7' for row in group]
        axes[0, col].bar(labels, [row['xi_last_50'] for row in group], color=colors)
        axes[0, col].set(title=f'{algo}: final Xi', ylabel='Xi (last 50)')
        axes[0, col].tick_params(axis='x', rotation=15)
        for row in group:
            smooth = np.convolve(row['rewards'], np.ones(20) / 20, mode='valid')
            axes[1, col].plot(np.arange(20, len(row['rewards']) + 1), smooth,
                              label=row['variant'].split('_', 1)[1])
        axes[1, col].set(title=f'{algo}: reward learning curve',
                         xlabel='Episode', ylabel='20-episode mean')
        axes[1, col].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'stage_e_gate_20260803.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

