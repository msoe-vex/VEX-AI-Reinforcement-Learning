"""
VEX Push Back Environment Test Script

A test/demo script for the VEX Push Back reinforcement learning environment.
Uses the new modular architecture from vex_core and pushback modules.

Usage:
    python vex_env_test.py --mode vex_u_skills --steps 20
    python vex_env_test.py --mode vex_ai_skills --steps 30 --no-render
"""

import argparse
import random

import numpy as np
from gymnasium import spaces

from vex_core.base_env import VexMultiAgentEnv
from vex_core.config import VexEnvConfig
from pushback import PushBackGame, Actions


def main():
    parser = argparse.ArgumentParser(description="VEX Push Back Environment Test")
    VexEnvConfig.add_cli_args(parser, experiment_path="vex_env_test", render_mode="terminal")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of steps to run"
    )
    args = parser.parse_args()
    config = VexEnvConfig.from_args(args)
    
    # Create game instance using factory method
    game = PushBackGame.get_game(
        config.game_name,
        communication_mode=config.communication_mode,
        deterministic=config.deterministic,
    )
    
    print(f"Testing VEX Push Back environment...")
    print(f"Game: {config.game_name}")
    print(f"Game class: {game.__class__.__name__}")
    print(f"Communication: {config.communication_mode.value}")
    
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
    
    while not done and (args.steps == 0 or step_count < args.steps):        
        # Sample random valid actions
        actions = {}
        for agent in env.agents:
            obs = observations[agent]
            action_space = env.action_space(agent)
            
            # Use action mask from observation dict
            action_mask = obs["action_mask"] if isinstance(obs, dict) and "action_mask" in obs else None
            
            if action_mask is not None:
                valid_indices = np.where(action_mask > 0)[0]
                action = random.choice(valid_indices) if len(valid_indices) > 0 else env.game.fallback_action()
            else:
                action = env.game.fallback_action()
             
            # Wrap to match space if needed
            if isinstance(action_space, spaces.Tuple):
                msg_sample = action_space[1].sample()
                actions[agent] = (action, msg_sample)
            else:
                actions[agent] = action
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
