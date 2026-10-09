import argparse
import csv
import os

import numpy as np
from gymnasium import spaces

from vex_core.base_env import MESSAGE_SIZE, VexMultiAgentEnv
from vex_core.config import CommunicationOption, VexEnvConfig
from override import OverrideGame
from pushback import PushBackGame


VEX_GAMES = {"vexu_skills", "vexu_comp", "vexai_skills", "vexai_comp"}


def get_game_class(game_name):
    return PushBackGame if game_name.lower() in VEX_GAMES else OverrideGame


def choose_random_action(env, agent, observation, rng, communication_mode):
    action_space = env.action_space(agent)
    discrete_space = action_space[0] if isinstance(action_space, spaces.Tuple) else action_space
    action_mask = observation.get("action_mask") if isinstance(observation, dict) else None

    if action_mask is None:
        valid_actions = np.arange(discrete_space.n)
    else:
        valid_actions = np.flatnonzero(action_mask > 0.5)

    if valid_actions.size == 0:
        action = int(env.game.fallback_action)
    else:
        action = int(rng.choice(valid_actions))

    if not isinstance(action_space, spaces.Tuple):
        return action

    if communication_mode == CommunicationOption.NONE:
        message = np.zeros(MESSAGE_SIZE, dtype=np.float32)
    else:
        message = rng.uniform(-1.0, 1.0, size=MESSAGE_SIZE).astype(np.float32)
    return action, message


def run_environment_test(config, iterations=1, output_dir="vex_environment_test", seed=None,
                         test_communication_mode=None):
    if test_communication_mode is None:
        test_communication_mode = config.communication_mode

    rng = np.random.default_rng(seed)
    game_class = get_game_class(config.game_name)
    game = game_class.get_game(
        config.game_name,
        communication_mode=config.communication_mode,
        deterministic=config.deterministic,
    )
    env = VexMultiAgentEnv(game=game, config=config)
    env.output_directory = output_dir
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for iteration in range(1, iterations + 1):
        print(f"\nRunning random-action simulation {iteration}/{iterations}...")
        episode_seed = None if seed is None else seed + iteration - 1
        observations, _ = env.reset(seed=episode_seed)

        if config.render_mode == "image":
            env.clearTicksDirectory()
            env.render()

        done = False
        total_reward = 0.0
        while not done:
            actions = {}
            for agent in list(env.agents):
                observation = observations.get(agent)
                if observation is None:
                    continue
                actions[agent] = choose_random_action(
                    env, agent, observation, rng, test_communication_mode
                )

            if not actions:
                break

            observations, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += sum(float(reward) for reward in rewards.values())
            done = terminations.get("__all__", False) or truncations.get("__all__", False)

        print(
            f"Simulation ended after {env.num_steps} environment steps "
            f"({env.num_ticks} internal ticks). Final score: {env.score}. "
            f"Total reward: {total_reward:.3f}"
        )
        row = {
            "iteration": iteration,
            "env_steps": env.num_steps,
            "internal_ticks": env.num_ticks,
            "total_reward": total_reward,
        }
        if isinstance(env.score, dict):
            row.update({f"score_{team}": float(score) for team, score in env.score.items()})
        else:
            row["score"] = env.score
        results.append(row)

        if config.render_mode == "image":
            print("Creating GIF of the simulation...")
            env.createGIF()

    if results:
        fieldnames = list(dict.fromkeys(key for row in results for key in row))
        results_path = os.path.join(output_dir, "results.csv")
        with open(results_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved per-iteration results to: {results_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run VEX environment simulations using random valid actions."
    )
    parser.add_argument("--game", default="override", help="Game variant, such as override or vexu_skills")
    parser.add_argument(
        "--render-mode", choices=["terminal", "image", "none"], default="image",
        help="Rendering mode: image saves tick frames and a GIF; none runs silently",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Number of simulation episodes")
    parser.add_argument("--output-dir", default="vex_environment_test", help="Output directory for GIFs and results.csv")
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable random actions and episodes")
    parser.add_argument(
        "--randomize", action=argparse.BooleanOptionalAction, default=False,
        help="Randomize initial game state",
    )
    parser.add_argument(
        "--communication-mode", choices=[mode.value for mode in CommunicationOption], default="none",
        help="Environment communication mode",
    )
    parser.add_argument(
        "--test-communication-mode", choices=[mode.value for mode in CommunicationOption], default=None,
        help="Use random messages or zero messages when communication is enabled",
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False,
        help="Use deterministic environment mechanics",
    )
    parser.add_argument("--copy-message-dropout-prob", type=float, default=0.0)
    args = parser.parse_args()

    config = VexEnvConfig(
        game_name=args.game,
        render_mode="image" if args.render_mode == "none" else args.render_mode,
        experiment_path=args.output_dir,
        randomize=args.randomize,
        communication_mode=CommunicationOption(args.communication_mode),
        deterministic=args.deterministic,
        copy_message_dropout_prob=float(np.clip(args.copy_message_dropout_prob, 0.0, 1.0)),
    )
    test_mode = (
        None if args.test_communication_mode is None
        else CommunicationOption(args.test_communication_mode)
    )
    run_environment_test(
        config,
        iterations=args.iterations,
        output_dir=args.output_dir,
        seed=args.seed,
        test_communication_mode=test_mode,
    )


if __name__ == "__main__":
    main()