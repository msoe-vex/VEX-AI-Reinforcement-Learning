import numpy as np
import torch
import unittest
from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from vex_core.base_env import (
    MESSAGE_SIZE,
    OPTION_TERMINATION_THRESHOLD,
    VexMultiAgentEnv,
)
from vex_core.base_game import ActionStep
from vex_custom_model import VexCustomPPO
from vex_model_training import _option_critic_termination_loss, env_creator
from vex_model_run import VexModelRunner


def _attention_action(action: int, should_stop: int = 0):
    return action, np.zeros(MESSAGE_SIZE, dtype=np.float32), should_stop


def _make_env() -> VexMultiAgentEnv:
    return env_creator(
        {
            "game": "override",
            "communication_mode": "attention",
            "randomize": False,
        }
    )


def _install_timed_action(env: VexMultiAgentEnv, duration: float) -> None:
    def execute_action(agent: str, action: int):
        state = env.environment_state["agents"][agent]
        return [
            ActionStep(
                duration,
                state["position"].copy(),
                state["orientation"].copy(),
            )
        ], 0.0

    env.game.execute_action = execute_action


class _ConstantOptionModel(nn.Module):
    def __init__(self, action_count: int, selected_action: int, beta: float):
        super().__init__()
        self.action_count = action_count
        self.selected_action = selected_action
        self.beta = beta

    def forward(self, observation, action_mask):
        logits = torch.full(
            (observation.shape[0], self.action_count),
            -10.0,
            device=observation.device,
        )
        logits[:, self.selected_action] = 10.0
        logits += torch.clamp(torch.log(action_mask + 1e-10), min=-1e8)
        message = torch.zeros(
            observation.shape[0],
            MESSAGE_SIZE * 2,
            device=observation.device,
        )
        beta = torch.full(
            (observation.shape[0], self.action_count),
            self.beta,
            device=observation.device,
        )
        return torch.cat([logits, message, beta], dim=-1)


class OptionCriticTests(unittest.TestCase):
    def test_termination_loss_uses_active_option_advantage_and_switching_cost(self):
        beta = torch.tensor([[0.2, 0.7], [0.3, 0.8]], requires_grad=True)
        active_action = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        advantages = torch.tensor([-2.0, 4.0])

        loss = _option_critic_termination_loss(beta, active_action, advantages)

        self.assertTrue(torch.isclose(loss, torch.tensor(-0.2)))
        loss.backward()
        self.assertTrue(torch.isclose(beta.grad[0, 0], torch.tensor(-1.0)))
        self.assertEqual(torch.count_nonzero(beta.grad[1]).item(), 0)

    def test_busy_action_checks_follow_duration_cadence(self):
        for duration, expected_ticks in ((3.0, 10), (10.0, 10), (11.0, 50)):
            with self.subTest(duration=duration):
                env = _make_env()
                try:
                    env.reset(seed=1)
                    _install_timed_action(env, duration)
                    actions = {agent: _attention_action(0) for agent in env.agents}

                    env.step(actions)

                    self.assertEqual(env.num_ticks, expected_ticks)
                    self.assertEqual(set(env.agents), set(env.possible_agents))
                    self.assertTrue(all(agent in env.busy_state for agent in env.possible_agents))
                finally:
                    env.close()

    def test_composite_action_distributions_include_termination(self):
        for mode, expected_components in (("none", 2), ("copy", 2), ("attention", 3)):
            with self.subTest(mode=mode):
                env = env_creator(
                    {
                        "game": "override",
                        "communication_mode": mode,
                        "randomize": False,
                    }
                )
                try:
                    observations, _ = env.reset(seed=2)
                    agent = env.possible_agents[0]
                    core_dim = env.game.get_game_observation_space(agent).shape[0]
                    comm_dim = (
                        MESSAGE_SIZE
                        if mode == "attention"
                        else env.observation_space(agent)["observations"].shape[0] - core_dim
                    )
                    module = RLModuleSpec(
                        module_class=VexCustomPPO,
                        observation_space=env.observation_space(agent),
                        action_space=env.action_space(agent),
                        model_config={
                            "communication_mode": mode,
                            "communication_dim": comm_dim,
                        },
                    ).build()
                    batch = {
                        Columns.OBS: {
                            key: torch.as_tensor(
                                np.stack([value]),
                                dtype=torch.float32,
                            )
                            for key, value in observations[agent].items()
                        }
                    }
                    output = module._forward_inference(batch)
                    distribution_cls = module.get_exploration_action_dist_cls()
                    distribution = distribution_cls.from_logits(
                        output[Columns.ACTION_DIST_INPUTS]
                    )
                    sampled_action = distribution.sample()

                    self.assertEqual(len(sampled_action), expected_components)
                    self.assertEqual(
                        output["termination_probabilities"].shape,
                        (1, env.game.num_actions),
                    )
                    self.assertTrue(
                        torch.all(
                            (sampled_action[-1] == 0) | (sampled_action[-1] == 1)
                        )
                    )
                finally:
                    env.close()

    def test_same_action_continues_but_different_action_is_interrupted_with_penalty(self):
        env = _make_env()
        try:
            env.reset(seed=1)
            _install_timed_action(env, 3.0)
            agents = list(env.agents)
            env.step({agent: _attention_action(0) for agent in agents})

            same_action_infos = {
                agent: _attention_action(0, should_stop=1)
                for agent in env.agents
            }
            _, _, _, _, infos = env.step(same_action_infos)
            self.assertTrue(
                all(not info.get("action_interrupted", False) for info in infos.values())
            )
            self.assertTrue(
                all(env.environment_state["agents"][agent]["current_action"] == 0 for agent in agents)
            )

            interrupting_agent = agents[0]
            replacement_actions = {
                agent: _attention_action(
                    1 if agent == interrupting_agent else 0,
                    should_stop=int(agent == interrupting_agent),
                )
                for agent in env.agents
            }
            _, rewards, _, _, infos = env.step(replacement_actions)

            self.assertTrue(infos[interrupting_agent]["action_interrupted"])
            self.assertEqual(infos[interrupting_agent]["interruption_penalty"], 10.0)
            self.assertEqual(rewards[interrupting_agent], -1.0)
            self.assertEqual(
                env.environment_state["agents"][interrupting_agent]["current_action"],
                1,
            )
        finally:
            env.close()

    def test_inference_preempts_only_above_threshold_for_a_different_action(self):
        env = _make_env()
        try:
            observations, _ = env.reset(seed=1)
            agent = env.possible_agents[0]
            selected_action = env.game.fallback_action
            runner = VexModelRunner(
                model_path=None,
                game=env.game,
                agent_name=agent,
                model=_ConstantOptionModel(
                    env.game.num_actions,
                    selected_action,
                    OPTION_TERMINATION_THRESHOLD + 0.1,
                ),
            )
            observation = env.game.get_game_observation(agent, game_time=0.0)
            active_action = (selected_action + 1) % env.game.num_actions

            result = runner.get_inference(
                observation,
                active_action=active_action,
            )
            self.assertGreaterEqual(result[3], OPTION_TERMINATION_THRESHOLD)
            self.assertTrue(result[4])

            same_action_result = runner.get_inference(
                observation,
                active_action=selected_action,
            )
            self.assertFalse(same_action_result[4])
            self.assertEqual(same_action_result[1], [])

            low_probability_runner = VexModelRunner(
                model_path=None,
                game=env.game,
                agent_name=agent,
                model=_ConstantOptionModel(
                    env.game.num_actions,
                    selected_action,
                    OPTION_TERMINATION_THRESHOLD - 0.1,
                ),
            )
            below_threshold_result = low_probability_runner.get_inference(
                observation,
                active_action=active_action,
            )
            self.assertLess(below_threshold_result[3], OPTION_TERMINATION_THRESHOLD)
            self.assertFalse(below_threshold_result[4])
            self.assertEqual(below_threshold_result[1], [])
        finally:
            env.close()
