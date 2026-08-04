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

    for uav in env.uavs:
        uav.pos = np.array([0.0, 0.0, 75.0])
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
    print("ENVIRONMENT SEMANTICS TEST PASSED")


if __name__ == '__main__':
    main()
