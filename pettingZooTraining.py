import ray
import argparse
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.rllib.utils import check_env
from ray.train import CheckpointConfig
from ray import tune
import warnings

from pettingZooEnv import POSSIBLE_AGENTS, env_creator

from pettingZooCompile import compile_checkpoint_to_torchscript

# Policy mapping function to assign agents to policies.
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id

if __name__ == "__main__":
    # Suppress all deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Get parameters from command line arguments
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')  # Default: 0.0005
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor')  # Default: 0.99
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')  # Default: 0.01
    parser.add_argument('--num-iters', type=int, default=1, help='Number of training iterations')  # Default: 10

    args = parser.parse_args()

    # Register your environment with RLlib so it can be created by name
    register_env("High_Stakes_Multi_Agent_Env", env_creator)

    # Create a temporary instance to retrieve observation and action spaces for a sample agent.
    temp_env = env_creator(None)
    check_env(temp_env)  # Check if the environment is compatible with RLlib
    obs_space = temp_env.observation_space(POSSIBLE_AGENTS[0])
    act_space = temp_env.action_space(POSSIBLE_AGENTS[0])

    # Define a policy for each agent.
    policies = {
        agent_id: (None, obs_space, act_space, {})
        for agent_id in temp_env.possible_agents
    }

    # Configure the RLlib Trainer using PPO (you can switch to another algorithm if desired)
    config = (
        PPOConfig()
        .environment(env="High_Stakes_Multi_Agent_Env")
        .framework("torch")  # change to "tf" if you prefer TensorFlow
        .rollouts(num_rollout_workers=1)  # adjust number of workers as needed
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            lr=args.learning_rate,
            gamma=args.discount_factor,
            entropy_coeff=args.entropy,
        )
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run the training process with logger callbacks
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=1,  # Save a checkpoint after every iteration
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            num_to_keep=5,  # Keep the last 5 checkpoints
        ),
        stop={"training_iteration": args.num_iters},
        callbacks=[
            JsonLoggerCallback(),
            CSVLoggerCallback(),
            TBXLoggerCallback(),
        ],
        metric="episode_reward_mean",  # Metric to optimize
        mode="max"  # Maximize the metric
    )

    # Use the best checkpoint directly from the analysis
    best_checkpoint = analysis.best_checkpoint
    best_checkpoint_path = best_checkpoint.path

    compile_checkpoint_to_torchscript(best_checkpoint_path)

    # Shutdown Ray
    ray.shutdown()
