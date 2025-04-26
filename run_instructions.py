import argparse
import imageio.v2 as imageio
from rl_environment import VEXHighStakesEnv, Actions

# -----------------------------------------------------------------------------
# Description: Run the VEXHighStakesEnv based on a list of custom actions.
# -----------------------------------------------------------------------------
def run_instructions(instructions_path, save_path, randomize_positions, realistic_pathing, realistic_vision, robot_num):
    # Check if the environment follows Gymnasium API
    env = VEXHighStakesEnv(save_path=save_path, randomize_positions=randomize_positions, realistic_pathing=realistic_pathing, realistic_vision=realistic_vision, robot_num=robot_num)

    done = False
    obs, _ = env.reset()
    step_num = 0
    images = []

    # Read custom actions from the instruction file
    with open(instructions_path, 'r') as file:
        actions = [line.strip() for line in file.readlines() if line.strip()]

    # -----------------------------------------------------------------------------
    # Description: Run one episode based on custom actions and collect rendered frames.
    # -----------------------------------------------------------------------------
    env.clearAuton()
    env.clearPNGs()
    for action_str in actions:
        if done:
            break
        action = Actions[action_str].value
        obs, reward, done, truncated, _ = env.step(action)
        env.render(step_num=step_num, action=action, reward=reward)
        images.append(imageio.imread(f"{env.steps_save_path}/step_{step_num}.png"))
        step_num += 1

    print(f"Total score: {env.total_score}")
    print("Creating GIF...")
    imageio.mimsave(f'{save_path}/simulation.gif', images, fps=5)
    env.close()

# -----------------------------------------------------------------------------
# Description: Parse arguments and run the custom actions.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VEXHighStakesEnv based on a list of custom actions.")
    parser.add_argument('--instructions-path', type=str, required=True, help='Path to the instruction file containing custom actions')
    parser.add_argument('--randomize', action='store_true', help='Randomize positions in the environment')
    parser.add_argument('--no-randomize', action='store_false', dest='randomize', help='Do not randomize positions in the environment')
    parser.add_argument('--realistic-pathing', action='store_true', help='Use realistic pathing')
    parser.add_argument('--no-realistic-pathing', action='store_false', dest='realistic_pathing', help='Do not use realistic pathing')
    parser.add_argument('--realistic-vision', action='store_true', help='Use realistic vision')
    parser.add_argument('--no-realistic-vision', action='store_false', dest='realistic_vision', help='Do not use realistic vision')
    parser.add_argument('--robot-num', type=int, choices=[0, 1, 2], default=0, help='Specify which robot to use (0-2)')
    parser.set_defaults(realistic_pathing=False, realistic_vision=True, randomize=True)
    args = parser.parse_args()

    save_path = "run_instructions_results"

    run_instructions(args.instructions_path, save_path, args.randomize, args.realistic_pathing, args.realistic_vision, args.robot_num)
