import sys
import unittest
import warnings
import logging
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hierarchical_agent import HierarchicalAgent
from system_model import Config, Environment

logging.getLogger("system_model").setLevel(logging.ERROR)


class KExperimentSupportTests(unittest.TestCase):
    def make_agent(self, seed=456, dyna_k=None):
        config = Config()
        torch.manual_seed(123)
        return HierarchicalAgent(
            30,
            4 + 2 * config.M + 1,
            config.N,
            config,
            seed=seed,
            dyna_k=dyna_k,
        )

    def test_environment_seed_is_reproducible(self):
        config_a = Config()
        config_b = Config()
        env_a = Environment(config_a, seed=123)
        env_b = Environment(config_b, seed=123)
        np.testing.assert_allclose(env_a.reset(1), env_b.reset(1))

    def test_dyna_rng_does_not_change_action_rng(self):
        agent_a = self.make_agent()
        agent_b = self.make_agent()
        agent_b.dyna_rng.random(1000)
        states = np.zeros((agent_a.num_agents, agent_a.state_dim), dtype=np.float32)
        np.testing.assert_allclose(agent_a.upper_act(states), agent_b.upper_act(states))
        np.testing.assert_allclose(agent_a.lower_act(states), agent_b.lower_act(states))

    def test_k_zero_is_a_no_op(self):
        agent = self.make_agent(dyna_k=0)
        self.assertEqual(agent.models, [])
        self.assertIsNone(agent.dyna_plan(0, k=0))
        self.assertIsNone(agent.update_model(0))

    def test_scheduler_advances_once_per_episode(self):
        agent = self.make_agent()
        initial_lr = agent.upper_actor_optimizers[0].param_groups[0]["lr"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(499):
                agent.end_episode()
        self.assertEqual(agent.upper_actor_optimizers[0].param_groups[0]["lr"], initial_lr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            agent.end_episode()
        self.assertAlmostEqual(
            agent.upper_actor_optimizers[0].param_groups[0]["lr"],
            initial_lr * 0.9,
        )

    def test_environment_exposes_step_metrics(self):
        config = Config()
        env = Environment(config, seed=123)
        env.reset(1)
        action_dim = 4 + 2 * config.M + 1
        env.step(np.zeros((config.N, action_dim), dtype=np.float32))
        self.assertEqual(
            set(env.last_step_metrics),
            {
                "collision_events",
                "collision_penalty",
                "data_collected",
                "data_delivered",
                "energy_consumed",
                "energy_harvested",
            },
        )

    def test_data_rates_enable_collection_and_delivery_without_reward_change(self):
        config = Config()
        env = Environment(config, seed=321)
        env.reset(1)

        for gu in env.gus:
            gu.pos = np.array([0.0, 0.0])
            gu.buffer = 10.0
            gu.energy = config.E_max

        env.uavs[0].pos = np.array([0.0, 0.0, 50.0])
        env.uavs[1].pos = np.array([200.0, 200.0, 50.0])
        env.uavs[2].pos = np.array([-200.0, -200.0, 50.0])

        action_dim = 4 + 2 * config.M + 1
        actions = np.zeros((config.N, action_dim), dtype=np.float32)
        actions[0, 4:4 + config.M] = 1.0
        actions[0, 4 + config.M:4 + 2 * config.M] = 1.0
        actions[0, -1] = 1.0

        _, rewards, _ = env.step(actions)

        self.assertGreater(env.last_step_metrics["data_collected"], 0.0)
        self.assertGreater(env.last_step_metrics["data_delivered"], 0.0)
        self.assertTrue(all(gu.data_rate_a > 0.0 for gu in env.gus))
        self.assertEqual(env.last_step_metrics["collision_events"], 0)

        expected_total_reward = (
            env.last_step_metrics["data_delivered"]
            - config.eta1 * env.last_step_metrics["energy_consumed"]
        )
        self.assertAlmostEqual(float(np.sum(rewards)), expected_total_reward)

    def test_collision_penalty_uses_eta_ten(self):
        config = Config()
        self.assertEqual(config.eta, 10.0)
        env = Environment(config, seed=654)
        env.reset(1)

        for gu in env.gus:
            gu.pos = np.array([1000.0, 1000.0])

        env.uavs[0].pos = np.array([0.0, 0.0, 50.0])
        env.uavs[1].pos = np.array([0.0, 0.0, 50.0])
        env.uavs[2].pos = np.array([200.0, 200.0, 50.0])

        action_dim = 4 + 2 * config.M + 1
        actions = np.zeros((config.N, action_dim), dtype=np.float32)
        _, rewards, _ = env.step(actions)

        self.assertEqual(env.last_step_metrics["collision_events"], 2)
        self.assertEqual(env.last_step_metrics["collision_penalty"], 20.0)
        self.assertAlmostEqual(float(np.sum(rewards)), -20.0)


if __name__ == "__main__":
    unittest.main()
