"""Analyze post-fix scenario-complexity gate manifests.

For each scenario manifest (base/case2/sparse x 5 seeds x 3 arms), compute:
  - per-arm mean final-reward / last-50 reward over seeds
  - Dyna-gated vs NoDyna paired gap (mean, % episodes won)
  - prospective episodes-to-threshold (fixed per-scenario threshold =
    0.9 x that scenario's NoDyna mean final Xi), using the 100-episode rolling Xi
  - model error EMA / planning enable ratio (if recorded) to contextualize

Usage:
  python scripts/analyze_scenario_gate.py <manifest1.json> [manifest2.json ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def rolling(x, w=100):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode='valid')


def episodes_to_threshold(xi, thr, w=100):
    r = rolling(xi, w)
    idx = np.argmax(r >= thr)
    return float(idx + w) if r[idx] >= thr else None


def analyze(path):
    m = json.loads(Path(path).read_text())
    scenario = m.get('scenario', path)
    seeds = m.get('seeds', [])
    runs = m.get('runs', {})
    print(f'\n===== scenario: {scenario} =====')
    print(f'overrides: {m.get("scenario_overrides")}  seeds: {seeds}')

    # threshold from this scenario's NoDyna final Xi
    nodyna_final_xi = []
    for sd in seeds:
        arm = runs.get(str(sd), {}).get('nodyna', {})
        res = arm.get('result')
        if res and res.get('episode_metrics'):
            xi = [row.get('paper_xi_ratio_of_sums') for row in res['episode_metrics']]
            nodyna_final_xi.append(np.mean(xi[-50:]))
    thr = 0.9 * np.mean(nodyna_final_xi) if nodyna_final_xi else None
    if thr:
        print(f'NoDyna mean final Xi = {np.mean(nodyna_final_xi):.5f} -> threshold = {thr:.5f}')

    arms = m.get('arms') or list(next(iter(runs.values()), {}).keys())
    for arm in arms:
        finals, lasts, convs, reached = [], [], [], 0
        for sd in seeds:
            rec = runs.get(str(sd), {}).get(arm, {})
            res = rec.get('result')
            if not res:
                continue
            rw = np.asarray(res['rewards'], float)
            finals.append(rw[-1])
            lasts.append(np.mean(rw[-50:]) if len(rw) >= 50 else rw.mean())
            xi = [row.get('paper_xi_ratio_of_sums') for row in res['episode_metrics']]
            e = episodes_to_threshold(xi, thr) if thr else None
            if e is not None:
                convs.append(e); reached += 1
        if finals:
            print(f'  {arm:11s} final={np.mean(finals):7.2f}±{np.std(finals,ddof=1):6.2f} '
                  f'last50={np.mean(lasts):7.2f}±{np.std(lasts,ddof=1):6.2f} '
                  f'episodes_to_thr={np.mean(convs):6.0f} (reached {reached}/{len(seeds)})'
                  if convs else
                  f'  {arm:11s} final={np.mean(finals):7.2f}±{np.std(finals,ddof=1):6.2f} last50={np.mean(lasts):7.2f}')

    # Dyna-gated vs NoDyna paired gap on last-50
    if 'dyna_gated' in arms:
        gaps = []
        for sd in seeds:
            g = runs.get(str(sd), {}).get('dyna_gated', {}).get('result')
            n = runs.get(str(sd), {}).get('nodyna', {}).get('result')
            if g and n:
                gaps.append(np.mean(g['rewards'][-50:]) - np.mean(n['rewards'][-50:]))
        if gaps:
            print(f'  Dyna-gated - NoDyna last50 gap: {np.mean(gaps):+7.2f} per-seed {[f"{g:+.1f}" for g in gaps]}')
    return m


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)
