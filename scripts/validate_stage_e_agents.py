#!/usr/bin/env python3
"""CPU validation for Stage-E MATD3 and CoP-MADDPG fixes."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cop_maddpg_agent import CoPMADDPGAgent
from matd3_agent import MATD3Agent


def config(seed=42):
    return SimpleNamespace(
        torch_seed=seed,
        rngs={
            'action': np.random.default_rng(seed + 1),
            'replay': np.random.default_rng(seed + 2),
        },
    )


def transitions(agent, state_dim, action_dim, count=40):
    rng = np.random.default_rng(123)
    for _ in range(count):
        states = rng.normal(size=(agent.num_agents, state_dim)).astype(np.float32)
        next_states = rng.normal(size=(agent.num_agents, state_dim)).astype(np.float32)
        actions = rng.uniform(-1, 1, size=(agent.num_agents, action_dim)).astype(np.float32)
        actions[:, 4:] = rng.uniform(0, 1, size=(agent.num_agents, action_dim - 4))
        rewards = rng.normal(size=agent.num_agents).astype(np.float32)
        agent.add_memory(states, actions, rewards, next_states, False)


def assert_bounds(agent, state_dim):
    states = np.random.default_rng(7).normal(size=(agent.num_agents, state_dim))
    actions = agent.act(states, noise=False)
    assert np.all(actions[:, :4] >= -1) and np.all(actions[:, :4] <= 1)
    assert np.all(actions[:, 4:] >= 0) and np.all(actions[:, 4:] <= 1)


def changed(before, module):
    after = list(module.parameters())
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, after))


def main():
    torch.set_num_threads(1)
    state_dim, action_dim, num_agents = 12, 5, 3

    matd3 = MATD3Agent(state_dim, action_dim, num_agents, config())
    assert_bounds(matd3, state_dim)
    transitions(matd3, state_dim, action_dim)
    actor_before = [p.detach().clone() for p in matd3.actors[0].parameters()]
    matd3.update()
    matd3.update()
    assert changed(actor_before, matd3.actors[0]), 'MATD3 delayed actor did not update'

    cop = CoPMADDPGAgent(state_dim, action_dim, num_agents, config())
    assert_bounds(cop, state_dim)
    transitions(cop, state_dim, action_dim)
    encoder_before = [
        [p.detach().clone() for p in encoder.parameters()] for encoder in cop.encoders
    ]
    cop.update()
    assert all(
        changed(before, encoder)
        for before, encoder in zip(encoder_before, cop.encoders)
    ), 'One or more CoP sender encoders did not receive a policy update'

    print('PASS: mixed action bounds are valid')
    print('PASS: MATD3 delayed actor updates')
    print('PASS: all CoP sender encoders receive policy gradients')


if __name__ == '__main__':
    main()
