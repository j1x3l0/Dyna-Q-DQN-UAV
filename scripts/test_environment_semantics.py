"""Regression checks for the corrected slot and action semantics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from system_model import Config, Environment


def main():
    config = Config(seed=7)
    env = Environment(config)
    states = env.reset(1)
    assert states.shape == (config.N, config.state_dim) == (3, 53)
    assert np.array_equal(states, env.get_state()), "get_state must not redraw fading"

    for i, uav in enumerate(env.uavs):
        uav.pos = np.array([50.0 * i, 0.0, 75.0])
    start = env.uavs[0].pos.copy()
    upper = np.zeros((config.N, 5), dtype=float)
    upper[:, 0] = 1.0
    upper[:, 3] = 1.0
    upper[:, 4] = 1.0
    lower_states = env.prepare_step(upper)
    displacement = np.linalg.norm(env.uavs[0].pos - start)
    assert np.isclose(displacement, config.v_max * config.tau_f, atol=1e-4)
    assert sum(uav.scheduled for uav in env.uavs) == 1
    assert np.array_equal(lower_states, env.get_state())
    assert not np.any(env._pending_step['safety_interventions'])

    base, stride = 7 + config.F, 3 + config.F
    masks = env.get_lower_action_masks()
    for i in range(config.N):
        for m in range(config.M):
            covered = lower_states[i, base + m * stride] >= 0.5
            assert covered == bool(masks[i, m, 2])

    before = np.array([gu.buffer for gu in env.gus])
    next_states, rewards, done = env.complete_step(
        np.zeros((config.N, 2 * config.M), dtype=float))
    after = np.array([gu.buffer for gu in env.gus])
    assert np.all(after > before), "all GUs must receive one arrival per slot"
    assert next_states.shape == (config.N, config.state_dim)
    assert rewards.shape == (config.N,)
    assert not done

    # Coincident commanded endpoints must be projected to the hard safe set.
    env.reset(1)
    for uav in env.uavs:
        uav.pos = np.array([0.0, 0.0, 75.0])
    upper = np.zeros((config.N, 5), dtype=float)
    origins = np.array([uav.pos.copy() for uav in env.uavs])
    env.prepare_step(upper)
    commanded, executed = env.get_last_upper_actions()
    distances = [
        np.linalg.norm(env.uavs[i].pos - env.uavs[j].pos)
        for i in range(config.N)
        for j in range(i + 1, config.N)
    ]
    assert min(distances) >= config.d_min - config.safety_projection_tolerance
    assert np.any(env._pending_step['safety_interventions'])
    assert np.array_equal(commanded, upper)
    assert np.any(np.linalg.norm(executed[:, :4] - commanded[:, :4], axis=1) > 0)
    assert np.array_equal(executed[:, 4], commanded[:, 4])
    replayed_positions = []
    for i in range(config.N):
        direction = executed[i, :3]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        speed = abs(executed[i, 3]) * config.v_max
        replayed_positions.append(origins[i] + direction * speed * config.tau_f)
    assert np.allclose(
        replayed_positions, [uav.pos for uav in env.uavs], atol=1e-5)
    assert np.all(np.linalg.norm(
        np.array([uav.velocity for uav in env.uavs]), axis=1) <= config.v_max + 1e-6)
    _, rewards, _ = env.complete_step(
        np.zeros((config.N, 2 * config.M), dtype=float))
    assert env.last_step_info['totals']['collision_events'] == 0
    assert env.last_step_info['totals']['collision_penalty'] == 0.0
    assert env.last_step_info['totals']['safety_interventions'] > 0
    assert env.last_step_info['totals']['min_uav_distance'] >= config.d_min - 1e-6
    assert np.array_equal(env.last_step_info['commanded_upper_actions'], commanded)
    assert np.array_equal(env.last_step_info['executed_upper_actions'], executed)

    # Legacy penalty mode remains available for paired U0/U1 experiments.
    legacy_config = Config(seed=7)
    legacy_config.collision_constraint_mode = 'penalty'
    legacy_env = Environment(legacy_config)
    legacy_env.reset(1)
    for uav in legacy_env.uavs:
        uav.pos = np.array([0.0, 0.0, 75.0])
    legacy_env.prepare_step(np.zeros((legacy_config.N, 5), dtype=float))
    legacy_env.complete_step(
        np.zeros((legacy_config.N, 2 * legacy_config.M), dtype=float))
    assert legacy_env.last_step_info['totals']['collision_events'] > 0
    assert legacy_env.last_step_info['totals']['collision_penalty'] > 0.0
    assert legacy_env.last_step_info['totals']['safety_interventions'] == 0
    print("ENVIRONMENT SEMANTICS TEST PASSED")


if __name__ == '__main__':
    main()
