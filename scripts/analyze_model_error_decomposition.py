"""Phase 0d: offline world-model error decomposition.

Loads a converged Stage-D K=1 checkpoint (server: dyna_paper_xi_k1_w32_seed123_final),
freezes the policy (epsilon=0, no action noise), collects (s, a, r, ns, d)
transitions, and decomposes the world-model next-state prediction error by
dimension group, normalized by per-dimension standard deviation.

Dimension layout (system_model.get_uav_state):
  [0:3]   UAV position                    -> deterministic (upper-action driven)
  [3:5]   UAV buffer, energy              -> deterministic
  [5:6]   distance to RBS                 -> deterministic
  [6:6+F] RBS channel gains (Rician)      -> stochastic
  active GU block (len(coverage) x (2+F)) -> per GU: energy, buffer (det), channel (stoch)

Answer: is error concentrated in inherently-unpredictable (channel) dims
(H-smoothing: model ~ Bayes-optimal) or also large in deterministic dims
(H-error: underfit)?
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from system_model import Config, Environment
from hierarchical_agent import HierarchicalAgent
from training_utils import compose_full_actions, extract_lower_rewards


CKPT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dyna_k1_w32_seed123_final.pt'
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = 123
F = None  # set from config


def collect(agent, env, config, n_episodes):
    """Collect (s, a_lower, r_lower, ns) transitions under the frozen policy."""
    agent.epsilon = 0.0  # greedy: converged policy, no exploration
    records = {i: [] for i in range(config.N)}
    for _ in range(n_episodes):
        states = env.reset(1)
        while True:
            upper_actions = agent.upper_act(states, noise=False)
            lower_actions = agent.lower_act(states)
            full_actions = compose_full_actions(upper_actions, lower_actions, config.N)
            next_states, rewards, done = env.step(full_actions)
            step_info = env.last_step_info or {}
            lower_rewards = extract_lower_rewards(step_info, rewards, config.N)
            for i in range(config.N):
                records[i].append((states[i], lower_actions[i], lower_rewards[i], next_states[i]))
            states = next_states
            if done:
                break
    return records


def decompose(records, config, agent, env):
    F = config.F
    M = config.M
    state_dim = config.state_dim
    # per-dimension accumulators (mean abs error, count) for non-padding dims
    n = len(next(iter(records.values())))  # steps per agent
    err_sum = np.zeros((config.N, state_dim))
    cnt = np.zeros((config.N, state_dim))
    # per-dimension std of the true next states (for normalization)
    ns_stack = {i: [] for i in range(config.N)}

    for i in range(config.N):
        for s, a, r, ns in records[i]:
            r_pred, s_pred = agent.model_predict(i, s, a)
            ns = np.asarray(ns, dtype=float)
            s_pred = np.asarray(s_pred, dtype=float)
            err = np.abs(s_pred - ns)
            err_sum[i] += err
            cnt[i] += 1.0
            ns_stack[i].append(ns)

    # per-dimension std over the true next states (global across agents)
    all_ns = np.concatenate(list(ns_stack.values()), axis=0)
    dim_std = all_ns.std(axis=0)
    dim_std = np.where(dim_std < 1e-12, 1.0, dim_std)  # avoid div-by-zero

    mean_err = err_sum / np.maximum(cnt, 1)
    # normalized error: mean |err| / std of that dimension
    norm_err = mean_err / dim_std.reshape(1, -1)

    groups = {
        'uav_pos (det)': (0, 3),
        'uav_buffer_energy (det)': (3, 5),
        'd_i0 (det)': (5, 6),
        'rbs_channel (stoch)': (6, 6 + F),
    }
    print(f'\ncollected {n} transitions/agent x {config.N} agents, state_dim={state_dim}, F={F}, M={M}')
    print('per-agent mean ABS error (unnormalized) by group:')
    for name, (lo, hi) in groups.items():
        vals = mean_err[:, lo:hi].mean(axis=1)
        print(f'  {name:26s}: {np.mean(vals):10.3f}  (per-agent {[f"{v:.1f}" for v in vals]})')

    print('\nper-agent mean NORMALIZED error (|err| / dim-std) by group:')
    print('  (normalized error >1 means the model is worse than predicting the mean)')
    for name, (lo, hi) in groups.items():
        vals = norm_err[:, lo:hi].mean(axis=1)
        print(f'  {name:26s}: {np.mean(vals):10.3f}  (per-agent {[f"{v:.2f}" for v in vals]})')

    # GU block: split deterministic (energy/buffer) vs stochastic (channel) per GU
    base = 6 + F
    gu_det_errs = []
    gu_ch_errs = []
    for i in range(config.N):
        for s, a, r, ns in records[i]:
            ns = np.asarray(ns, dtype=float)
            cov = env.get_coverage(env.uavs[i])
            n_gu = len(cov)
            if n_gu == 0:
                continue
            r_pred, s_pred = agent.model_predict(i, s, a)
            s_pred = np.asarray(s_pred, dtype=float)
            for g in range(n_gu):
                chunk = base + g * (2 + F)
                e = np.abs(s_pred[chunk:chunk + 2 + F] - ns[chunk:chunk + 2 + F])
                # normalize by dim std
                e_norm = e / dim_std[chunk:chunk + 2 + F]
                gu_det_errs.append(e_norm[:2])   # energy, buffer (deterministic)
                gu_ch_errs.append(e_norm[2:])    # channel (stochastic)
    print(f'\nGU features (deterministic energy/buffer vs stochastic channel):')
    print(f'  GU buffer/energy (det)  mean normalized err: {np.mean(np.concatenate(gu_det_errs)):.3f}')
    print(f'  GU channel     (stoch)  mean normalized err: {np.mean(np.concatenate(gu_ch_errs)):.3f}')


def main():
    global F
    config = Config(seed=SEED)
    F = config.F
    env = Environment(config)
    state_dim, action_dim = config.state_dim, 4 + 2 * config.M + 1
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    ep = agent.load_checkpoint(CKPT)
    print(f'loaded {CKPT} (episode {ep}, epsilon={agent.epsilon:.4f})')
    records = collect(agent, env, config, EPISODES)
    decompose(records, config, agent, env)


if __name__ == '__main__':
    main()
