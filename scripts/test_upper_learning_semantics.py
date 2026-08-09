"""Regression checks for normalized networks and executed-action replay."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_agent import (
    HierarchicalNoDynaAgent,
    LowerDQN,
    UpperActor,
    UpperCritic,
    _upper_noise_stds,
    _projection_residual_loss,
)
from system_model import Config


def assert_state_normalization():
    scale = np.array([10.0, 100.0], dtype=np.float32)
    raw = torch.tensor([[10.0, 100.0]])
    normalized = raw / torch.tensor(scale)

    torch.manual_seed(11)
    actor_scaled = UpperActor(2, 1, 1, hidden_dim=8, state_scale=scale)
    actor_unit = UpperActor(2, 1, 1, hidden_dim=8)
    actor_unit.load_state_dict(actor_scaled.state_dict())
    assert 'state_scale' not in actor_scaled.state_dict()
    assert torch.allclose(actor_scaled(raw), actor_unit(normalized))

    critic_scaled = UpperCritic(2, 2, hidden_dim=8, state_scale=scale)
    critic_unit = UpperCritic(2, 2, hidden_dim=8)
    critic_unit.load_state_dict(critic_scaled.state_dict())
    action = torch.tensor([[0.2, 0.8]])
    assert torch.allclose(
        critic_scaled(raw, action), critic_unit(normalized, action))

    dqn_scaled = LowerDQN(2, 3, hidden_dim=8, state_scale=scale)
    dqn_unit = LowerDQN(2, 3, hidden_dim=8)
    dqn_unit.load_state_dict(dqn_scaled.state_dict())
    assert torch.allclose(dqn_scaled(raw), dqn_unit(normalized))


def assert_executed_action_replay():
    config = Config(seed=19)
    agent = HierarchicalNoDynaAgent(
        config.state_dim, 4 + 2 * config.M + 1, config.N, config)
    agent.batch_size = 1

    states = np.ones((config.N, config.state_dim), dtype=float)
    commanded = np.zeros((config.N, 5), dtype=float)
    executed = commanded.copy()
    executed[0, 0] = 0.75
    rewards = np.zeros(config.N, dtype=float)
    captured_actions = []

    def capture_critic_action(_module, inputs, _output):
        captured_actions.append(inputs[1].detach().cpu().numpy().copy())

    hook = agent.upper_critics[0].register_forward_hook(capture_critic_action)
    agent.add_upper_memory(
        states, commanded, rewards, states.copy(), True,
        executed_actions=executed)
    agent.update_upper()
    hook.remove()

    assert len(agent.upper_memory[-1]) == 6
    assert np.array_equal(agent.upper_memory[-1][1], commanded)
    assert np.array_equal(agent.upper_memory[-1][2], executed)
    assert np.array_equal(captured_actions[0], executed.reshape(1, -1))
    assert agent.last_upper_update_info[0]['projection_loss'] >= 0.0

    current = torch.zeros((1, 5), requires_grad=True)
    no_projection_loss = _projection_residual_loss(
        current, torch.zeros((1, 5)), torch.zeros((1, 5)), 1e-6)
    assert no_projection_loss.item() == 0.0
    projected = torch.zeros((1, 5))
    projected[0, 0] = 0.75
    projection_loss = _projection_residual_loss(
        current, torch.zeros((1, 5)), projected, 1e-6)
    assert projection_loss.item() > 0.0


def assert_exploration_schedule():
    config = Config(seed=23)
    initial = _upper_noise_stds(config, 0)
    final = _upper_noise_stds(config, config.upper_noise_decay_episodes)
    assert np.allclose(
        list(initial.values()), [0.15, 0.10, 0.05])
    assert np.allclose(
        list(final.values()), [0.02, 0.02, 0.01])
    assert np.allclose(
        list(_upper_noise_stds(config, 10_000).values()),
        list(final.values()),
    )

    agent = HierarchicalNoDynaAgent(
        config.state_dim, 4 + 2 * config.M + 1, config.N, config)
    states = np.ones((config.N, config.state_dim), dtype=float)
    assert np.array_equal(
        agent.upper_act(states, noise=False),
        agent.upper_act(states, noise=False),
    )
    masks = np.ones((config.N, config.M, 3), dtype=bool)
    assert np.array_equal(
        agent.lower_act(states, action_masks=masks, explore=False),
        agent.lower_act(states, action_masks=masks, explore=False),
    )
    agent.step_episode_schedulers()
    assert agent.upper_noise_episode == 1


def main():
    assert_state_normalization()
    assert_executed_action_replay()
    assert_exploration_schedule()
    print("UPPER LEARNING SEMANTICS TEST PASSED")


if __name__ == '__main__':
    main()
