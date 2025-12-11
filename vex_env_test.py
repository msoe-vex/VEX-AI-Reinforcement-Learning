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
        default=100,
        help="Number of steps to run"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vex_env_test",
        help="Output directory for renders"
    )
    args = parser.parse_args()
    
    # Create game instance using factory method
    game = PushBackGame.get_game(args.game)
    
    # Determine render mode
    render_mode = None if args.no_render else "all"
    
    print(f"Testing VEX Push Back environment...")
    print(f"Game: {args.game}")
    print(f"Game class: {game.__class__.__name__}")
    
    # Create environment
    env = VexMultiAgentEnv(
        game=game,
        render_mode=render_mode,
        output_directory=args.output_dir,
        randomize=False,
    )
    
    # Reset environment
    observations, infos = env.reset()
    
    if render_mode:
        env.clearStepsDirectory()
    
    print(f"Agents: {env.agents}")
    print(f"Time limit: {game.total_time}s")
    print()
    
    # Render initial state
    if render_mode:
        print("Step 0: Initial positions")
        env.render()
    
    done = False
    step_count = 0
    
    while not done and step_count < args.steps:
        # Random actions for testing
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        step_count += 1
        print(f"\nStep {step_count}:")
        for agent, action in actions.items():
            action_name = Actions(action).name
            print(f"  {agent}: {action_name}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = terminations.get("__all__", False) or truncations.get("__all__", False)
        
        if render_mode:
            # Convert actions to names for rendering
            named_actions = {agent: Actions(a).name for agent, a in actions.items()}
            env.render(actions=named_actions, rewards=rewards)
    
    print(f"\nSimulation complete after {step_count} steps.")
    print(f"Final score: {env.score}")
    
    if render_mode:
        env.createGIF()
        print(f"GIF saved to {args.output_dir}/simulation.gif")


if __name__ == "__main__":
    main()
