import ray
import argparse
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.logger.json import JsonLoggerCallback
from ray.tune.logger.csv import CSVLoggerCallback
# from ray.tune.logger.tensorboardx import TBXLoggerCallback  # Uncomment if you want TensorBoard
from ray import tune
import warnings
import time
import os
import glob

# Import from new modular architecture
from vex_core.base_env import VexMultiAgentEnv
from pushback import PushBackGame
from vex_custom_model import VexCustomPPO

from vex_model_compile import compile_checkpoint_to_torchscript
import sys
import json


class VexScoreCallback(RLlibCallback):
    """Custom callback to track team scores at the end of each episode."""
    
    def on_episode_end(self, *, episode, env_runner, metrics_logger, env, env_index, rl_module, **kwargs):
        """Called at the end of each episode to log team scores."""
        try:
            # Unwrap the OrderEnforcing wrapper to get the actual VexMultiAgentEnv
            wrapped_env = env.envs[env_index]
            actual_env = wrapped_env.env if hasattr(wrapped_env, 'env') else wrapped_env
            
            # Use the already-computed score stored in the env (updated each step)
            if hasattr(actual_env, 'score') and actual_env.score:
                red = actual_env.score.get('red', 0)
                blue = actual_env.score.get('blue', 0)
                
                metrics_logger.log_value("red_team_score_mean", red)
                metrics_logger.log_value("blue_team_score_mean", blue)
        except Exception:
            pass  # Silently ignore errors

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        """Called after each training iteration to print scores to stdout."""
        env_runner_results = result.get("env_runners", {})
        episode_return = env_runner_results.get("episode_return_mean")
        iteration = result.get("training_iteration", 0)
        
        red_mean = env_runner_results.get("red_team_score_mean")
        blue_mean = env_runner_results.get("blue_team_score_mean")
        
        if red_mean is not None and episode_return is not None:
            print(f"[Iter {iteration}] Red Score Mean: {red_mean:.1f}, Blue Score Mean: {blue_mean:.1f}, Episode Return Mean: {episode_return:.1f}")
        elif episode_return is not None:
            print(f"[Iter {iteration}] Episode Return Mean: {episode_return:.1f}")
        sys.stdout.flush()


def env_creator(config=None):
    """Create environment instance for RLlib registration."""
    config = config or {}
    enable_communication = config.get("enable_communication", True)
    deterministic = config.get("deterministic", True)
    game = PushBackGame.get_game(
        config.get("game", "vexai_skills"),
        enable_communication=enable_communication,
        deterministic=deterministic,
    )
    return VexMultiAgentEnv(
        game=game,
        render_mode=None,
        randomize=config.get("randomize", True),
        enable_communication=enable_communication,
        deterministic=deterministic,
    )


# Policy mapping function to assign agents to policies.
def policy_mapping_fn(agent_id, episode):
    # return "shared_policy" # Use the same policy for all agents
    return agent_id # Change to agent_id if you want to use different policies for each agent


def find_latest_checkpoint(experiment_directory: str):
    """Find the latest RLlib checkpoint directory under experiment_directory."""
    checkpoint_candidates = []

    primary_pattern = os.path.join(experiment_directory, "PPO_VEX*", "checkpoint_*")
    checkpoint_candidates.extend(glob.glob(primary_pattern))

    if not checkpoint_candidates:
        fallback_pattern = os.path.join(experiment_directory, "**", "checkpoint_*")
        checkpoint_candidates.extend(glob.glob(fallback_pattern, recursive=True))

    checkpoint_dirs = [p for p in checkpoint_candidates if os.path.isdir(p)]
    if not checkpoint_dirs:
        return None

    def checkpoint_sort_key(path):
        base = os.path.basename(path)
        try:
            idx = int(base.split("_")[-1])
        except (ValueError, IndexError):
            idx = -1
        return (idx, os.path.getmtime(path))

    return max(checkpoint_dirs, key=checkpoint_sort_key)


def apply_training_metadata_overrides(args, metadata):
    """Override CLI args with values from training_metadata.json."""
    metadata_to_arg = {
        "game": "game",
        "learning_rate": "learning_rate",
        "learning_rate_final": "learning_rate_final",
        "discount_factor": "discount_factor",
        "entropy": "entropy",
        "entropy_final": "entropy_final",
        "randomize": "randomize",
        "num_iters": "num_iters",
        "enable_communication": "enable_communication",
        "deterministic": "deterministic",
    }

    for metadata_key, arg_name in metadata_to_arg.items():
        if metadata_key in metadata:
            setattr(args, arg_name, metadata[metadata_key])

if __name__ == "__main__":
    # Suppress excessive experiment checkpoint warnings completely
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    
    # Suppress deprecation warnings from RLlib internal code
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    
    # Suppress gymnasium precision/space warnings (RLlib internal env creation)
    warnings.filterwarnings("ignore", module="gymnasium")


    
    # Get parameters from command line arguments
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')  # Default: 0.0005
    parser.add_argument('--learning-rate-final', type=float, default=0.00005,
                        help='Final learning rate for linear schedule (if not set, learning rate is constant)')
    parser.add_argument('--discount-factor', type=float, default=0.98, help='Discount factor')  # Default: 0.98
    parser.add_argument('--entropy', type=float, default=0.05, help='Entropy coefficient')  # Default: 0.05
    parser.add_argument('--entropy-final', type=float, default=0.005,
                        help='Final entropy coefficient for linear schedule (if not set, entropy is constant)')
    parser.add_argument('--num-iters', type=int, default=10, help='Number of training iterations')  # Default: 10
    parser.add_argument('--cpus-per-task', type=int, default=1, help='Number of CPUs per task')  # Default: 1
    parser.add_argument('--job-id', type=str, default="", help='SLURM job ID')  # Job ID for logging
    parser.add_argument('--randomize', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable or disable environment randomization (use --no-randomize to explicitly disable)')
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--partition', type=str, default="teaching", help='SLURM partition to use')
    parser.add_argument('--algorithm', type=str, default="PPO", help='Algorithm to use for training')
    parser.add_argument('--experiment-path', type=str, default="", help='Path to an existing experiment directory to restore from')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode: 0 = silent, 1 = default, 2 = verbose')
    parser.add_argument('--game', type=str, default="vexai_skills", help='Game variant to train')
    parser.add_argument('--communication', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable or disable agent communication (use --no-communication to explicitly disable)')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable deterministic environment mechanics (use --no-deterministic for stochastic outcomes)')

    args = parser.parse_args()

    restore_path = None
    restore_experiment_directory = args.experiment_path
    if restore_experiment_directory:
        restore_experiment_directory = os.path.abspath(restore_experiment_directory)
        metadata_path = os.path.join(restore_experiment_directory, "training_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            apply_training_metadata_overrides(args, metadata)
            print(f"Loaded and applied metadata overrides from: {metadata_path}")
        else:
            print(f"Warning: No training metadata found at {metadata_path}; using CLI/default parameters.")

        restore_path = find_latest_checkpoint(restore_experiment_directory)
        if restore_path:
            print(f"Found latest checkpoint: {restore_path}")
        else:
            print(f"Warning: No checkpoint directories found under {restore_experiment_directory}; training will start fresh.")

    print("Training parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print("Initializing training configuration...")
    # Register your environment with RLlib so it can be created by name
    register_env("VEX_Multi_Agent_Env", env_creator)

    # Create a temporary instance to retrieve observation and action spaces for a sample agent.
    temp_env = env_creator({
        "game": args.game,
        "randomize": args.randomize,
        "enable_communication": args.communication,
        "deterministic": args.deterministic,
    })

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
    
    # Detect available GPUs
    num_gpus_available = int(ray.available_resources().get("GPU", 0))
    print(f"GPUs detected by Ray: {num_gpus_available}")

    all_policies = temp_env.possible_agents
    train_batch_size_per_learner = 2400

    # Optional schedules for lr and entropy over estimated total timesteps.
    # Behavior: hold initial value for first 20%, then linearly move to final value.
    estimated_total_timesteps = max(1, args.num_iters * train_batch_size_per_learner)
    hold_fraction = 0.2
    hold_timesteps = int(estimated_total_timesteps * hold_fraction)
    lr_schedule = None
    entropy_schedule = None

    if args.learning_rate_final is not None and args.learning_rate_final != args.learning_rate:
        lr_schedule = [
            [0, float(args.learning_rate)],
            [hold_timesteps, float(args.learning_rate)],
            [estimated_total_timesteps, float(args.learning_rate_final)],
        ]

    if args.entropy_final is not None and args.entropy_final != args.entropy:
        entropy_schedule = [
            [0, float(args.entropy)],
            [hold_timesteps, float(args.entropy)],
            [estimated_total_timesteps, float(args.entropy_final)],
        ]

    if lr_schedule:
        print(f"Using learning rate schedule: {lr_schedule}")
    else:
        print(f"Using constant learning rate: {args.learning_rate}")

    if entropy_schedule:
        print(f"Using entropy schedule: {entropy_schedule}")
    else:
        print(f"Using constant entropy coefficient: {args.entropy}")

    # RLlib new API: schedules must be provided directly via `lr` and
    # `entropy_coeff` (not via deprecated *_schedule fields).
    lr_config_value = lr_schedule if lr_schedule is not None else float(args.learning_rate)
    entropy_config_value = entropy_schedule if entropy_schedule is not None else float(args.entropy)
    
    # Configure the RLlib Trainer using PPO with new API stack
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env="VEX_Multi_Agent_Env",
            env_config={
                "randomize": args.randomize,
                "game": args.game,
                "enable_communication": args.communication,
                "deterministic": args.deterministic,
            }
        )
        .framework("torch")  # change to "tf" for TensorFlow
        .resources(
            num_gpus=0,  # Don't assign GPUs at top level with new API
        )
        .learners(
            num_learners=1 if num_gpus_available > 0 else 0,  # Use 1 GPU learner if available
            num_gpus_per_learner=1 if num_gpus_available > 0 else 0,  # Assign GPU to learner
        )
        .env_runners(
            num_env_runners=args.cpus_per_task-1,  # Use 1 runner for each CPU core plus 1 for the main process
            batch_mode="complete_episodes",  # Collect complete episodes, not fragments
            rollout_fragment_length="auto",  # Let RLlib calculate the optimal fragment length
        )
        .multi_agent(
            policies=set(all_policies),  # Define policy IDs
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(all_policies),
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=VexCustomPPO,  # Use custom model with clean architecture
                observation_space=obs_space,
                action_space=act_space,
                model_config={"enable_communication": args.communication}  # Model architecture defined in vex_custom_model.py
            )
        )
        .fault_tolerance(
            max_num_env_runner_restarts=0,  # Disable env runner restarts to avoid object store issues
        )
        .training(
            lr=lr_config_value,
            gamma=args.discount_factor,
            entropy_coeff=entropy_config_value,
            train_batch_size_per_learner=train_batch_size_per_learner,  # 4x episode length (~600) to ensure episodes complete
        )
        .callbacks(VexScoreCallback)  # Track team scores
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
        output_directory = os.path.join(script_directory, "vex_model_training")  # Default directory if no job ID is provided

    # Generate experiment name with timestamp so we know the trial directory upfront
    experiment_name = f"PPO_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}"
    experiment_dir = os.path.join(output_directory, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Save game metadata early (before training starts)
    metadata = {
        "game": args.game,
        "learning_rate": args.learning_rate,
        "learning_rate_final": args.learning_rate_final,
        "learning_rate_schedule": lr_schedule,
        "discount_factor": args.discount_factor,
        "entropy": args.entropy,
        "entropy_final": args.entropy_final,
        "entropy_schedule": entropy_schedule,
        "randomize": args.randomize,
        "num_iters": args.num_iters,
        "estimated_total_timesteps": estimated_total_timesteps,
        "schedule_hold_fraction": hold_fraction,
        "schedule_hold_timesteps": hold_timesteps,
        "source_experiment_directory": restore_experiment_directory,
        "restored_checkpoint_path": restore_path,
        "enable_communication": args.communication,
        "deterministic": args.deterministic,
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    metadata_path = os.path.join(experiment_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to {metadata_path}")

    # Calculate checkpoint frequency - ensure at least one checkpoint is created
    # Checkpoint at the end and at reasonable intervals
    checkpoint_freq = max(1, min(5, args.num_iters))  # Every 5 iters, but at least once
    print(f"Checkpoint frequency: every {checkpoint_freq} iterations")

    # Prepare restore parameter if an experiment directory was provided
    if restore_path:
        print(f"Restoring from checkpoint: {restore_path}")

    # Run the training process with logger callbacks
    analysis = tune.run(
        "PPO",
        name=experiment_name,  # Use our pre-generated experiment name
        config=config.to_dict(),
        storage_path=output_directory,
        checkpoint_freq=checkpoint_freq,
        keep_checkpoints_num=1,  # Keep 1 best checkpoint
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
        restore=restore_path,  # Resume from checkpoint if provided
    )

    print(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Use the best checkpoint directly from the analysis
    best_checkpoint = analysis.best_checkpoint
    best_checkpoint_path = best_checkpoint.path
    
    print(f"Saving results to: {experiment_dir}")

    compile_checkpoint_to_torchscript(temp_env.game, best_checkpoint_path, experiment_dir)

    # Shutdown Ray
    ray.shutdown()
