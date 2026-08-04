"""Smoke test for the Replay-1 attribution arm (Phase 0e).

Verifies that HierarchicalAgent supports planning_mode in {'model', 'replay'}:
  - 'model'  (default, unchanged): dyna_plan predicts via the learned world model
    and records zero-cost online model-error instrumentation.
  - 'replay' (perfect-oracle control): dyna_plan uses the true (r, s') stored in
    the replay tuple; no model_predict call, no instrumentation.

Runs one episode per mode on CPU. No physics formula is touched (red line).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from system_model import Config, Environment
from hierarchical_agent import HierarchicalAgent
from training_utils import (
    get_state_action_dims,
    compose_full_actions,
    extract_lower_rewards,
)


def run_episode(planning_mode, seed=42, episodes=2):
    config = Config(seed=seed)
    config.planning_mode = planning_mode
    # This attribution smoke test intentionally bypasses the production gate.
    config.dyna_warmup_steps = 0
    config.dyna_model_error_threshold = float('inf')
    env = Environment(config)
    state_dim, action_dim = get_state_action_dims(config)
    agent = HierarchicalAgent(state_dim, action_dim, config.N, config)
    assert agent.planning_mode == planning_mode, (agent.planning_mode, planning_mode)
    if planning_mode == 'model':
        # constructor fallback via config attribute
        agent2 = HierarchicalAgent(state_dim, action_dim, config.N, config, planning_mode=None)
        assert agent2.planning_mode == 'model'

    total_reward = 0.0
    for ep in range(episodes):
        states = env.reset(1)
        agent.reset_planning_errors()
        while True:
            upper_actions = agent.upper_act(states)
            lower_states = env.prepare_step(upper_actions)
            lower_actions = agent.lower_act(
                lower_states, action_masks=env.get_lower_action_masks())
            next_states, rewards, done = env.complete_step(lower_actions)
            step_info = env.last_step_info or {}
            lower_rewards = extract_lower_rewards(step_info, rewards, config.N)
            agent.add_upper_memory(states, upper_actions, rewards, next_states, done)
            agent.update_upper()
            for i in range(config.N):
                agent.add_lower_memory(i, lower_states[i], lower_actions[i], lower_rewards[i],
                                       next_states[i], done)
                agent.update_lower(i)
                if agent.dyna_k > 0:
                    agent.update_model(i)
                    agent.dyna_plan(i)
            total_reward += float(sum(rewards))
            states = next_states
            if done:
                break
        pe = agent.planning_error_summary()
        if planning_mode == 'replay':
            assert all(v['count'] == 0 for v in pe.values()), \
                f'replay mode must not record instrumentation: {pe}'
        else:
            assert any(v['count'] > 0 for v in pe.values()), \
                f'model mode must record instrumentation: {pe}'
        agent.step_episode_schedulers()
    return total_reward, agent


def main():
    for mode in ('model', 'replay'):
        total, agent = run_episode(mode)
        pe = agent.planning_error_summary()
        print(f'[{mode}] total_reward={total:.2f} | planning_error_summary: '
              f'counts={[v["count"] for v in pe.values()]}')
        if mode == 'model':
            mr = [v['mean_reward_err'] for v in pe.values() if v['mean_reward_err'] is not None]
            ms = [v['mean_state_err'] for v in pe.values() if v['mean_state_err'] is not None]
            print(f'  model mode mean|r_hat-r|={np.mean(mr):.4f} '
                  f'(n={len(mr)} agents)  mean||s_hat\'-s\'||={np.mean(ms):.3f}')
    # Invalid mode must raise
    cfg = Config(seed=1)
    cfg.planning_mode = 'bogus'
    try:
        HierarchicalAgent(*get_state_action_dims(cfg), cfg.N, cfg)
        raise SystemExit('FAIL: bogus planning_mode did not raise')
    except ValueError:
        print('OK: bogus planning_mode raises ValueError')
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
