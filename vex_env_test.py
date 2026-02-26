"""
VEX Push Back Environment Test Script

A test/demo script for the VEX Push Back reinforcement learning environment.
Uses the new modular architecture from vex_core and pushback modules.

Usage:
    python vex_env_test.py --mode vex_u_skills --steps 20
    python vex_env_test.py --mode vex_ai_skills --steps 30 --no-render
"""

import argparse

import numpy as np

from vex_core.base_env import VexMultiAgentEnv
from vex_core.config import VexEnvConfig
from pushback import PushBackGame, Actions


def main():
    parser = argparse.ArgumentParser(description="VEX Push Back Environment Test")
    VexEnvConfig.add_cli_args(parser, experiment_path="vex_env_test")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of steps to run"
    )
    args = parser.parse_args()
    config = VexEnvConfig.from_args(args)
    
    # Create game instance using factory method
    game = PushBackGame.get_game(
        config.game_name,
        enable_communication=config.enable_communication,
        deterministic=config.deterministic,
    )
    
    print(f"Testing VEX Push Back environment...")
    print(f"Game: {config.game_name}")
    print(f"Game class: {game.__class__.__name__}")
    print(f"Communication: {config.enable_communication}")
    
    # Create environment
    env = VexMultiAgentEnv(
        game=game,
        config=config,
    )
    
    # Reset environment
    observations, infos = env.reset()
    
    if args.render_mode == "image":
        env.clearTicksDirectory()
    
    print(f"Agents: {env.agents}")
    print(f"Time limit: {game.total_time}s")
    print()
    
    # Initial state logging
    if args.render_mode in ["image", "terminal"]:
        print("Step 0: Initial positions")
    if args.render_mode == "image":
        env.render()
    
    done = False
    step_count = 0
    
    while not done and step_count < args.steps:        
        # Sample random actions from action space
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step the environment
        step_count += 1
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations.get("__all__", False) or truncations.get("__all__", False)
        if args.render_mode:
            env.render()
    
    print(
        f"\nSimulation complete after {step_count} steps "
        f"(env steps: {env.num_steps}, internal ticks: {env.num_ticks})."
    )
    print(f"Final score: {env.score}")
    
    if args.render_mode == "image":
        env.createGIF()
        print(f"GIF saved to {config.experiment_path}/simulation.gif")


if __name__ == "__main__":
    main()
