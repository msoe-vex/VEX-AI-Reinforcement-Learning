import os
import argparse
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_environment import VEXHighStakesEnv

# -----------------------------------------------------------------------------
# Description: Run a PPO agent on the VEXHighStakesEnv.
# -----------------------------------------------------------------------------
def run_agent(model_path, save_path, randomize_positions, realistic_pathing, realistic_vision, robot_num):
    # Check if the environment follows Gymnasium API
    env = VEXHighStakesEnv(save_path=save_path, randomize_positions=randomize_positions, realistic_pathing=realistic_pathing, realistic_vision=realistic_vision, robot_num=robot_num)
    check_env(env, warn=True)

    model = PPO.load(model_path)

    done = False
    obs, _ = env.reset()
    step_num = 0
    images = []

    # -----------------------------------------------------------------------------
    # Description: Run one episode and collect rendered frames.
    # -----------------------------------------------------------------------------
    env.clearAuton()
    env.clearPNGs()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        env.render(step_num=step_num, action=action, reward=reward)
        images.append(imageio.imread(f"{env.steps_save_path}/step_{step_num}.png"))
        step_num += 1

    print(f"Total score: {env.total_score}")
    print("Creating GIF...")
    imageio.mimsave(f'{save_path}/simulation.gif', images, fps=3)
    env.close()

# -----------------------------------------------------------------------------
# Description: Parse arguments and run the agent.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a PPO agent on the VEXHighStakesEnv.")
    parser.add_argument('--model-path', type=str, default=None, help='Path to an existing model to load and run')
    parser.add_argument('--timesteps', type=int, default=10000, help='Total timesteps for training the model')
    parser.add_argument('--train', action='store_true', help='Train a new model if specified')
    parser.add_argument('--randomize', action='store_true', help='Randomize positions in the environment')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions in the environment')
    parser.add_argument('--realistic-pathing', action='store_true', help='Use realistic pathing')
    parser.add_argument('--no-realistic-pathing', action='store_false', dest='realistic_pathing', help='Do not use realistic pathing')
    parser.add_argument('--realistic-vision', action='store_true', help='Use realistic vision')
    parser.add_argument('--no-realistic-vision', action='store_false', dest='realistic_vision', help='Do not use realistic vision')
    parser.add_argument('--robot-num', type=int, choices=[0, 1, 2], default=0, help='Specify which robot to use (0-2)')
    parser.set_defaults(realistic_pathing=False, realistic_vision=True, randomize=True)
    args = parser.parse_args()

    save_path = "run_agent_results"

    if args.train:
        # Check if the environment follows Gymnasium API
        env = VEXHighStakesEnv(save_path=save_path, randomize_positions=args.randomize, realistic_pathing=args.realistic_pathing, realistic_vision=args.realistic_vision, robot_num=args.robot_num)
        check_env(env, warn=True)

        # Train a PPO agent on the environment
        if args.model_path:
            print("Loading existing model...")
            model = PPO.load(args.model_path, env=env, verbose=1)
        else:
            print("Creating new model...")
            model = PPO("MultiInputPolicy", env, verbose=1)

        print("Training model...")
        model.learn(total_timesteps=args.timesteps)
        print("Training complete.")

        # Save the trained model
        model_save_path = os.path.join(save_path, "model")
        model.save(model_save_path)

        env.close()

        run_agent(model_save_path, save_path, args.randomize, args.realistic_pathing, args.realistic_vision, args.robot_num)
    elif args.model_path:
        run_agent(args.model_path, save_path, args.randomize, args.realistic_pathing, args.realistic_vision, args.robot_num)
    else:
        print("Please specify --train to train a new model or provide a --model-path to run an existing model.")