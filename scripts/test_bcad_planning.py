"""Deterministic smoke test for Bellman-Consistent Adaptive Dyna planning."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_agent import HierarchicalAgent
from system_model import Config, Environment


def main():
    config = Config(seed=17)
    config.dyna_k = 1
    config.dyna_warmup_steps = 0
    config.dyna_planning_strategy = 'bcad'
    config.dyna_counterfactual_fraction = 0.0
    config.dyna_bellman_beta = 0.0
    config.dyna_plan_probability_max = 1.0
    config.dyna_plan_ramp_steps = 1
    config.dyna_model_update_lr_scale = 0.1

    env = Environment(config)
    agent = HierarchicalAgent(config.state_dim, 4 + 2 * config.M + 1,
                              config.N, config)
    states = env.reset(case=1)

    planned = 0
    for _ in range(48):
        upper_actions = agent.upper_act(states)
        lower_states = env.prepare_step(upper_actions)
        lower_actions = agent.lower_act(
            lower_states, action_masks=env.get_lower_action_masks())
        next_states, rewards, done = env.complete_step(lower_actions)
        lower_rewards = np.asarray(
            (env.last_step_info or {}).get('lower_rewards', rewards), dtype=float)
        for i in range(config.N):
            agent.add_lower_memory(
                i, lower_states[i], lower_actions[i], lower_rewards[i],
                next_states[i], done)
            agent.update_lower(i)
            stats = agent.update_model(i)
            if stats is not None:
                assert np.isfinite(stats['bellman_error_ema'])
            loss = agent.dyna_plan(i)
            if loss is not None:
                info = agent.last_plan_info[i]
                assert info['planned'] == 1.0
                assert np.isfinite(info['sample_bellman_error'])
                assert 0.0 < info['sample_weight'] <= 1.0
                assert np.isfinite(loss)
                planned += 1
        states = env.reset(case=1) if done else next_states

    assert planned > 0
    print(f'BCAD PLANNING TEST PASSED: planned_updates={planned}')


if __name__ == '__main__':
    main()
