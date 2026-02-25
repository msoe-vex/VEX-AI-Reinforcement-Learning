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
from pushback import PushBackGame, Actions


def main():
    parser = argparse.ArgumentParser(description="VEX Push Back Environment Test")
    parser.add_argument(
        "--game", 
        type=str, 
        default="vexai_skills",
        help="Game variant to test"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of steps to run"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["terminal", "image", "none"],
        default="image",
        help="Rendering mode: 'image' (saves frames & GIF), 'terminal' (prints text only), 'none' (silent)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vex_env_test",
        help="Output directory for renders"
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Disable randomization of initial agent positions and orientations"
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic environment mechanics (use --no-deterministic for stochastic outcomes)"
    )
    parser.add_argument(
        "--communication",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable agent communication (use --communication to enable)"
    )
    args = parser.parse_args()
    
    # Create game instance using factory method
    game = PushBackGame.get_game(
        args.game,
        enable_communication=args.communication,
        deterministic=args.deterministic,
    )
    
    print(f"Testing VEX Push Back environment...")
    print(f"Game: {args.game}")
    print(f"Game class: {game.__class__.__name__}")
    print(f"Communication: {args.communication}")
    
    # Create environment
    env = VexMultiAgentEnv(
        game=game,
        render_mode=args.render_mode if args.render_mode != "none" else None,
        output_directory=args.output_dir,
        randomize=not args.no_randomize,
        enable_communication=args.communication,
        deterministic=args.deterministic,
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
        print(f"GIF saved to {args.output_dir}/simulation.gif")


if __name__ == "__main__":
    main()
