"""Deterministic checks for decision-consistent transitions and Dyna variants."""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, SCRIPT_DIR)

from hierarchical_agent import HierarchicalAgent
from system_model import Config, Environment
from training_utils import extract_lower_rewards


def exercise(strategy):
    config = Config(seed=29)
    config.lower_transition_mode = 'decision_point'
    config.dyna_k = 1
    config.dyna_warmup_steps = 0
    config.dyna_planning_strategy = strategy
    config.dyna_counterfactual_fraction = 0.0
    config.dyna_plan_probability_min = 1.0
    config.dyna_plan_probability_max = 1.0
    config.dyna_plan_ramp_steps = 1
    config.dyna_plan_keep_fraction = 0.5
    config.dyna_priority_candidate_multiplier = 2

    env = Environment(config)
    agent = HierarchicalAgent(
        config.state_dim, 4 + 2 * config.M + 1, config.N, config)
    states = env.reset(case=1)
    pending = None
    planned = 0
    checked_successors = 0

    for _ in range(48):
        upper_actions = agent.upper_act(states)
        lower_states = env.prepare_step(upper_actions)
        if pending is not None:
            for i in range(config.N):
                prev_state, prev_action, prev_reward = pending[i]
                agent.add_lower_memory(
                    i, prev_state, prev_action, prev_reward,
                    lower_states[i], False)
                assert np.array_equal(
                    agent.lower_memory[i][-1][3], lower_states[i])
                checked_successors += 1
                agent.update_lower(i)
                stats = agent.update_model(i)
                if stats is not None:
                    assert np.isfinite(stats['bellman_error_ema'])
                loss = agent.dyna_plan(i)
                if loss is not None:
                    info = agent.last_plan_info[i]
                    assert np.isfinite(loss)
                    assert np.isfinite(info['sample_bellman_error'])
                    assert np.isfinite(info['robust_scale'])
                    assert 0.0 < info['sample_weight'] <= 1.0
                    assert 0.0 < info['selected_fraction'] <= 1.0
                    if strategy == 'utility_priority':
                        assert np.isfinite(info['mean_utility'])
                    planned += 1

        lower_actions = agent.lower_act(
            lower_states, action_masks=env.get_lower_action_masks())
        next_states, rewards, done = env.complete_step(lower_actions)
        lower_rewards = extract_lower_rewards(
            env.last_step_info or {}, rewards, config.N)
        pending = [
            (lower_states[i].copy(), lower_actions[i].copy(),
             float(lower_rewards[i]))
            for i in range(config.N)
        ]
        states = next_states
        assert not done

    assert checked_successors > 0
    assert planned > 0
    return planned


def main():
    total = 0
    for strategy in ('robust_budget', 'real_anchor', 'utility_priority'):
        planned = exercise(strategy)
        print(f'{strategy}: planned_updates={planned}')
        total += planned
    print(f'DYNA REDESIGN TEST PASSED: total_planned_updates={total}')


if __name__ == '__main__':
    main()
