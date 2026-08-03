#!/usr/bin/env python3
"""Analyze the Stage-D warm-up gate against the warm-up=32 anchor."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def metrics(run):
    rewards = np.asarray(run['rewards'][:200], dtype=float)
    episodes = run['episode_metrics'][:200]

    def values(key):
        return np.asarray([
            np.nan if row[key] is None else row[key] for row in episodes
        ], dtype=float)

    energy = values('energy_consumed') + values('flight_energy')
    return {
        'warmup': run['dyna_warmup'],
        'reward_auc_200': float(np.mean(rewards)),
        'reward_last_50': float(np.mean(rewards[-50:])),
        'xi_last_50': float(np.nanmean(values('paper_xi_ratio_of_sums')[-50:])),
        'model_loss_last_50': float(np.nanmean(values('model_total_loss')[-50:])),
        'data_sent_last_50': float(np.mean(values('data_sent_to_rbs')[-50:])),
        'data_received_last_50': float(np.mean(values('data_received')[-50:])),
        'energy_last_50': float(np.mean(energy[-50:])),
        'collisions_per_episode': float(np.sum(values('collision_events')) / 200),
        'seconds_per_episode': float(run['duration'] / run['episodes_completed']),
        'rewards': rewards,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate_json')
    parser.add_argument('anchor_json')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gate = json.load(open(args.gate_json, encoding='utf-8'))
    anchor = json.load(open(args.anchor_json, encoding='utf-8'))
    runs = list(gate['runs'])
    runs += [
        run for run in anchor['runs']
        if run['dyna_k'] == 1 and run['dyna_warmup'] == 32 and run['seed'] == 42
    ]
    rows = sorted((metrics(run) for run in runs), key=lambda row: row['warmup'])
    if [row['warmup'] for row in rows] != [0, 32, 64, 128]:
        raise ValueError('Expected exactly warm-up 0/32/64/128 for K=1, seed42')

    anchor_row = next(row for row in rows if row['warmup'] == 32)
    for row in rows:
        for key in ('reward_auc_200', 'reward_last_50', 'xi_last_50',
                    'model_loss_last_50', 'seconds_per_episode'):
            base = anchor_row[key]
            row[f'{key}_change_pct'] = 100 * (row[key] - base) / abs(base)

    csv_path = output / 'stage_d_warmup_gate_metrics_20260803.csv'
    columns = [key for key in rows[0] if key != 'rewards']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])

    payload = {
        'inputs': {'gate': args.gate_json, 'anchor': args.anchor_json},
        'selection': {
            'dyna_k': 1,
            'dyna_warmup': 128,
            'scope': 'single-seed engineering gate; not statistical evidence',
            'reason': 'best trade-off of AUC, final Xi, model loss, and wall time',
        },
        'rows': [{key: value for key, value in row.items() if key != 'rewards'} for row in rows],
    }
    with (output / 'stage_d_warmup_gate_analysis_20260803.json').open(
        'w', encoding='utf-8'
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    warmups = [row['warmup'] for row in rows]
    colors = ['0.65', '0.65', '0.65', '#2a6fbb']
    axes[0].bar([str(x) for x in warmups], [row['reward_auc_200'] for row in rows], color=colors)
    axes[0].set(title='Reward AUC (episodes 1–200)', xlabel='Warm-up', ylabel='Mean reward')
    axes[1].bar([str(x) for x in warmups], [row['xi_last_50'] for row in rows], color=colors)
    axes[1].set(title='Final system efficiency', xlabel='Warm-up', ylabel='Xi (last 50)')
    window = 20
    for row in rows:
        smooth = np.convolve(row['rewards'], np.ones(window) / window, mode='valid')
        axes[2].plot(np.arange(window, 201), smooth)
    axes[2].set(title='Reward learning curve (20-episode mean)', xlabel='Episode', ylabel='Reward')
    axes[2].legend([f'w={x}' for x in warmups], frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'stage_d_warmup_gate_20260803.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
