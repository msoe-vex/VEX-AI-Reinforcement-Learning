import ray
import argparse
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray import tune
import warnings
import time
import os

# Import from new modular architecture
from vex_core import VexMultiAgentEnv
from pushback import PushBackGame

from vexModelCompile import compile_checkpoint_to_torchscript
import sys
import json


def env_creator(config=None):
    """Create environment instance for RLlib registration."""
    config = config or {}
    game = PushBackGame.get_game(config.get("game", "vexu_skills"))
    return VexMultiAgentEnv(
        game=game,
        render_mode=None,
        randomize=config.get("randomize", True),
    )


# Policy mapping function to assign agents to policies.
def policy_mapping_fn(agent_id, episode):
    return "shared_policy" # Use the same policy for all agents
    # return agent_id # Change to agent_id if you want to use different policies for each agent

if __name__ == "__main__":
    # Suppress excessive experiment checkpoint warnings completely
    # os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    
    # Suppress all deprecation warnings
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Suppress gymnasium precision/space warnings (RLlib internal env creation)
    warnings.filterwarnings("ignore", module="gymnasium")
    
    # Get parameters from command line arguments
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')  # Default: 0.0005
    parser.add_argument('--discount-factor', type=float, default=0.99, help='Discount factor')  # Default: 0.99
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')  # Default: 0.01
    parser.add_argument('--num-iters', type=int, default=10, help='Number of training iterations')  # Default: 10
    parser.add_argument('--cpus-per-task', type=int, default=1, help='Number of CPUs per task')  # Default: 1
    parser.add_argument('--job-id', type=str, default="", help='SLURM job ID')  # Job ID for logging
    parser.add_argument('--model-path', type=str, default="", help='Path to save/load the model')
    parser.add_argument('--randomize', type=bool, default=True, help='Enable or disable randomization (True or False)')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--num-nodes', type=int, default=64, help='Number of nodes per layer in the model')
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--partition', type=str, default="teaching", help='SLURM partition to use')
    parser.add_argument('--algorithm', type=str, default="PPO", help='Algorithm to use for training')
    parser.add_argument('--checkpoint-path', type=str, default="", help='Path to the checkpoint directory')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode: 0 = silent, 1 = default, 2 = verbose')
    parser.add_argument('--game', type=str, required=True, help='Game variant to train')

    args = parser.parse_args()

    print("Training parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print("Initializing training configuration...")
    # Register your environment with RLlib so it can be created by name
    register_env("VEX_Multi_Agent_Env", env_creator)

    # Create a temporary instance to retrieve observation and action spaces for a sample agent.
    temp_env = env_creator({"game": args.game, "randomize": args.randomize})

    # Get observation and action spaces for module spec
    sample_agent = temp_env.possible_agents[0]
    obs_space = temp_env.observation_space(sample_agent)
    act_space = temp_env.action_space(sample_agent)
    
    # Initialize Ray with GPU detection and runtime environment config
    ray.init(
        ignore_reinit_error=True, 
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                "RAY_max_restarts": "0"  # Disable actor restarts
            }
        }
    )

    # Import RLModule specs for new API
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    
    # Configure the RLlib Trainer using PPO with new API stack
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="VEX_Multi_Agent_Env",
            env_config={"randomize": args.randomize, "game": args.game}
        )
        .framework("torch")  # change to "tf" for TensorFlow
        .resources(
            num_gpus=ray.available_resources().get("GPU", 0)  # Use available GPUs
        )
        .env_runners(
            num_env_runners=args.cpus_per_task-1,  # Use 1 runner for each CPU core plus 1 for the main process
        )
        .multi_agent(
            policies={"shared_policy"},  # Define policy IDs
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                observation_space=obs_space,
                action_space=act_space,
                model_config={
                    "fcnet_hiddens": [args.num_nodes] * args.num_layers,
                    "fcnet_activation": "relu"
                }
            )
        )
        .fault_tolerance(
            max_num_env_runner_restarts=0,  # Disable env runner restarts to avoid object store issues
        )
        .training(
            lr=args.learning_rate,
            gamma=args.discount_factor,
            entropy_coeff=args.entropy,
        )
        .debugging(log_level="ERROR")  # Reduce logging verbosity
    )

    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    sys.stdout.flush()
    # Determine the output directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    if args.job_id:
        output_directory = os.path.join(script_directory, "job_results", f"job_{args.job_id}")
    else:
        output_directory = os.path.join(script_directory, "vexEnvTraining")  # Default directory if no job ID is provided

    # Run the training process with logger callbacks
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        storage_path=output_directory,
        checkpoint_freq=5,  # Checkpoint every 5 iterations to reduce overhead
        keep_checkpoints_num=2,  # Keep 2 best checkpoints
        checkpoint_score_attr="env_runners/episode_return_mean",  # Use this metric for best checkpoint
        sync_config=tune.SyncConfig(
            sync_period=300,  # Sync every 5 minutes instead of constantly
        ),
        stop={"training_iteration": args.num_iters},
        callbacks=[
            JsonLoggerCallback(),
            CSVLoggerCallback(),
            #TBXLoggerCallback(),
        ],
        log_to_file=False,
        metric="env_runners/episode_return_mean",
        mode="max",
        verbose=args.verbose,  # Use the verbosity level from the argument
    )

    print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Use the best checkpoint directly from the analysis
    best_checkpoint = analysis.best_checkpoint
    best_checkpoint_path = best_checkpoint.path
    
    # Get the actual trial directory (PPO_2025-...) where results are stored
    # The checkpoint path typically looks like: .../PPO_2025-12-11_09-08-15/PPO_VEX_Multi_Agent_Env_xxxxx/checkpoint_xxx
    # We want the PPO_2025-... directory
    trial_dir = best_checkpoint_path
    while not os.path.basename(trial_dir).startswith('PPO_'):
        parent = os.path.dirname(trial_dir)
        if parent == trial_dir:  # Reached root
            trial_dir = output_directory
            break
        trial_dir = parent
    
    print(f"Saving results to: {trial_dir}")
    
    # Save game metadata to the trial directory
    metadata = {
        "game": args.game,
        "learning_rate": args.learning_rate,
        "discount_factor": args.discount_factor,
        "entropy": args.entropy,
        "num_layers": args.num_layers,
        "num_nodes": args.num_nodes,
        "randomize": args.randomize,
    }
    metadata_path = os.path.join(trial_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to {metadata_path}")

    compile_checkpoint_to_torchscript(temp_env.game, best_checkpoint_path, trial_dir)

    # Shutdown Ray
    ray.shutdown()
