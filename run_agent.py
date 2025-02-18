from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import imageio.v2 as imageio
import os
import argparse
from rl_environment import VEXHighStakesEnv

def run_agent(model_path, randomize_positions):
    # Check if the environment follows Gymnasium API
    env = VEXHighStakesEnv(randomize_positions=randomize_positions)
    check_env(env, warn=True)

    model = PPO.load(model_path)

    # Create a directory to save the images
    save_path = "simulation_steps"
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.makedirs(save_path, exist_ok=True)

    done = False
    obs, _ = env.reset()
    step_num = 0
    images = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        env.render(save_path=save_path, step_num=step_num, action=action, reward=reward)
        images.append(imageio.imread(f"{save_path}/step_{step_num}.png"))
        step_num += 1
    print(f"Total score: {env.total_score}")

    print("Creating GIF...")
    imageio.mimsave('simulation.gif', images, fps=10)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run a PPO agent on the VEXHighStakesEnv.")
    parser.add_argument('--model_path', type=str, default=None, help='Path to an existing model to load and run')
    parser.add_argument('--timesteps', type=int, default=10000, help='Total timesteps for training the model')
    parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train a new model')
    parser.add_argument('--randomize', action='store_true', help='Randomize positions in the environment')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions in the environment')
    parser.set_defaults(randomize=True)
    args = parser.parse_args()

    print(args.randomize)

    if args.train:
        # Check if the environment follows Gymnasium API
        env = VEXHighStakesEnv(randomize_positions=args.randomize)
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
        model.save("vex_high_stakes_ppo")

        env.close()

        run_agent("vex_high_stakes_ppo", args.randomize)
    elif args.model_path:
        run_agent(args.model_path, args.randomize)
    else:
        print("Please specify either --train to train a new model, or --model_path to run an existing model.")