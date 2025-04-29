import ray
import argparse
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from pettingZooEnv import parallel_env

# Environment creator function that returns the raw PettingZoo parallel env
def env_creator(_):
    # Wrap the PettingZoo parallel environment in an RLlib-compatible multi-agent environment
    return (parallel_env(render_mode=None))

# Register your environment with RLlib so it can be created by name
register_env("custom_pettingzoo_env", env_creator)

# Create a temporary instance to retrieve observation and action spaces for a sample agent.
temp_env = env_creator(None)
obs_space = temp_env.observation_space("robot_0")
act_space = temp_env.action_space("robot_0")

# Define a policy for each agent.
policies = {
    agent_id: (None, obs_space, act_space, {})
    for agent_id in temp_env.possible_agents
}

# Policy mapping function to assign agents to policies.
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id

# Configure the RLlib Trainer using PPO (you can switch to another algorithm if desired)
config = (
    PPOConfig()
    .environment(env="custom_pettingzoo_env")
    .framework("torch")  # change to "tf" if you prefer TensorFlow
    .rollouts(num_rollout_workers=1)  # adjust number of workers as needed
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
)

if __name__ == "__main__":
    #Get parameters from command line arguments
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')  # Default: 0.0005
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor')  # Default: 0.99
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')  # Default: 0.01

    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Set the configuration for the training process.
    config.training(
        lr=args.learning_rate,
        gamma=args.discount_factor,
        entropy_coeff=args.entropy,
    )
    
    
    # Build the trainer from the configuration.
    trainer = config.build()

    # Training loop
    NUM_ITERS = 3  # Set the number of iterations for training
    for i in range(NUM_ITERS):
        result = trainer.train()
        print(f"Iteration {i}: mean episode reward = {result['episode_reward_mean']}")

    # Save the trained policy checkpoint.
    checkpoint = trainer.save()
    print(f"Checkpoint saved at {checkpoint}")

    # Shutdown Ray
    ray.shutdown()
