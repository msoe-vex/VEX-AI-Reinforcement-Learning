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
from vex_core.config import VexEnvConfig
from pushback import PushBackGame
from vex_custom_model import VexCustomPPO

from vex_model_compile import compile_checkpoint_to_torchscript
from vex_model_test import run_simulation
import sys
import json

TRAINING_PHASES = [
    # Phase 1: BOOTSTRAP (The "Silent" Era)
    # Goal: Learn basic robot movement and game mechanics without message noise.
    {"iterations": 100,  "train_encoder": True,  "train_action": True,  "train_message": False, "lr": 0.0005,  "entropy": 0.05},

    # Phase 2: TALK ONLY (The "Encoding" Era)
    # Goal: Freeze the good behavior. Now, force the message head to explain 
    # what the robot is doing/seeing.
    {"iterations": 90,  "train_encoder": False, "train_action": False, "train_message": True,  "lr": 0.0005,  "entropy": 0.05},

    # Fine tune
    {"iterations": 10,  "train_encoder": True, "train_action": True,  "train_message": True, "lr": 0.00005,  "entropy": 0.005},

    # Phase 3: LISTEN ONLY (The "Coordination" Era)
    # Goal: Keep messages stable. Now, teach the robots to use those messages 
    # to improve their existing policy.
    {"iterations": 90,  "train_encoder": False, "train_action": True,  "train_message": False, "lr": 0.0005,  "entropy": 0.03},

    # Fine tune
    {"iterations": 10,  "train_encoder": True, "train_action": True,  "train_message": True, "lr": 0.00005,  "entropy": 0.005},

    # Phase 4: CO-EVOLUTION (The "Bilingual" Era)
    # Goal: Small tweaks to both heads to align the "language" with the "actions."
    {"iterations": 100,  "train_encoder": False, "train_action": True,  "train_message": True,  "lr": 0.0001,  "entropy": 0.02},

    # Phase 5: FINAL POLISH
    # Goal: Train all heads to optimize the policy and communication.
    {"iterations": 100, "train_encoder": True,  "train_action": True,  "train_message": True,  "lr": 0.00005, "entropy": 0.005},
]

def get_training_phase(iteration, phases):
    """Determine the current training phase settings based on the iteration."""
    iters_accumulated = 0
    for phase in phases:
        if iteration < iters_accumulated + phase["iterations"]:
            return phase
        iters_accumulated += phase["iterations"]
    return phases[-1]

def apply_head_training_status_on_learner(learner, phase):
    """Helper function to set specific learning rates for encoder, action and message heads."""
    train_encoder = phase.get("train_encoder", True)
    train_action = phase.get("train_action", True)
    train_message = phase.get("train_message", True)
    base_lr = phase.get("lr", 0.0005)
    new_entropy = phase.get("entropy", 0.05)
    
    # Very small learning rate for "frozen" components to keep momentum alive but prevent large updates
    frozen_lr = 1e-8 
    
    enc_lr = base_lr if train_encoder else frozen_lr
    act_lr = base_lr if train_action else frozen_lr
    msg_lr = base_lr if train_message else frozen_lr
    
    # Update Entropy coefficient
    if hasattr(learner, "entropy_coeff_schedule"):
        from ray.rllib.utils.schedules.constant_schedule import ConstantSchedule
        learner.entropy_coeff_schedule = ConstantSchedule(new_entropy, framework="torch")
    
    import torch
    for attr in ["entropy_coeff", "_entropy_coeff"]:
        if hasattr(learner, attr):
            val = getattr(learner, attr)
            if isinstance(val, torch.Tensor):
                val.data = torch.tensor(new_entropy, dtype=torch.float32, device=val.device)
            else:
                setattr(learner, attr, new_entropy)

    enc_params = set()
    act_params = set()
    msg_params = set()

    for module_id, module in learner.module.items():
        target_module = module.unwrapped() if hasattr(module, "unwrapped") else module
        
        if hasattr(target_module, "_encoder_net"):
            enc_params.update(target_module._encoder_net.parameters())
            
        if hasattr(target_module, "pi"):
            act_params.update(target_module.pi.parameters())
            
        if hasattr(target_module, "message_head") and target_module.message_head is not None:
            msg_params.update(target_module.message_head.parameters())
            msg_params.update(target_module.attention_unit.parameters())
            msg_params.add(target_module.msg_log_std)

    if hasattr(learner, "_optimizers"):
        for opt in learner._optimizers.values():
            all_params = []
            for g in opt.param_groups:
                all_params.extend(g['params'])
            
            new_groups = []
            enc_group = {'params': [], 'lr': enc_lr}
            act_group = {'params': [], 'lr': act_lr}
            msg_group = {'params': [], 'lr': msg_lr}
            other_group = {'params': [], 'lr': base_lr}
            
            for p in all_params:
                p.requires_grad = True
                if p in enc_params:
                    enc_group['params'].append(p)
                elif p in act_params:
                    act_group['params'].append(p)
                elif p in msg_params:
                    msg_group['params'].append(p)
                else:
                    other_group['params'].append(p)
                    
            if opt.param_groups:
                base_kwargs = {k: v for k, v in opt.param_groups[0].items() if k not in ('params', 'lr')}
                
                for g in [enc_group, act_group, msg_group, other_group]:
                    if g['params']:
                        g.update(base_kwargs)
                        new_groups.append(g)
                        
                opt.param_groups = new_groups

class VexScoreCallback(RLlibCallback):
    """Custom callback to track team scores at the end of each episode and toggle frozen layers."""
    
    def on_algorithm_init(self, *, algorithm, **kwargs):
        """Called when a new algorithm instance has been created."""
        try:
            phase = get_training_phase(0, TRAINING_PHASES)
            algorithm.learner_group.foreach_learner(
                lambda learner: apply_head_training_status_on_learner(learner, phase)
            )
            
            enc_status = "TRAINING" if phase.get("train_encoder", True) else "FROZEN"
            action_status = "TRAINING" if phase.get("train_action", True) else "FROZEN"
            msg_status = "TRAINING" if phase.get("train_message", True) else "FROZEN"
            print(f"  [Init] Phase Config: Encoder [{enc_status}] | Action [{action_status}] | Message [{msg_status}] | LR: {phase.get('lr', 0.0005)} | Ent: {phase.get('entropy', 0.05)}")
        except getattr(Exception, "dummy", Exception):
            pass

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
        """Called after each training iteration to print scores and toggle frozen layers."""
        env_runner_results = result.get("env_runners", {})
        episode_return = env_runner_results.get("episode_return_mean")
        iteration = result.get("training_iteration", 0)
        
        red_mean = env_runner_results.get("red_team_score_mean")
        blue_mean = env_runner_results.get("blue_team_score_mean")
        
        if red_mean is not None and episode_return is not None:
            print(f"[Iter {iteration}] Red Score Mean: {red_mean:.1f}, Blue Score Mean: {blue_mean:.1f}, Episode Return Mean: {episode_return:.1f}")
        elif episode_return is not None:
            print(f"[Iter {iteration}] Episode Return Mean: {episode_return:.1f}")
            
        # Toggle frozen heads
        try:
            phase = get_training_phase(iteration, TRAINING_PHASES)
            algorithm.learner_group.foreach_learner(
                lambda learner: apply_head_training_status_on_learner(learner, phase)
            )
            
            enc_status = "TRAINING" if phase.get("train_encoder", True) else "FROZEN"
            action_status = "TRAINING" if phase.get("train_action", True) else "FROZEN"
            msg_status = "TRAINING" if phase.get("train_message", True) else "FROZEN"
            print(f"  -> Phase Config: Encoder [{enc_status}] | Action [{action_status}] | Message [{msg_status}] | LR: {phase.get('lr', 0.0005)} | Ent: {phase.get('entropy', 0.05)}")
        except getattr(Exception, "dummy", Exception):
            pass
            
        sys.stdout.flush()


def env_creator(config=None):
    """Create environment instance for RLlib registration."""
    config = config or {}
    enable_communication = config.get("enable_communication", True)
    deterministic = config.get("deterministic", True)
    game_name = config.get("game", "vexai_skills")
    
    env_config = VexEnvConfig(
        game_name=game_name,
        render_mode=None,
        experiment_path=config.get("experiment_path", "vex_model_training"),
        randomize=config.get("randomize", True),
        enable_communication=enable_communication,
        deterministic=deterministic
    )
    
    game = PushBackGame.get_game(
        game_name,
        enable_communication=enable_communication,
        deterministic=deterministic,
    )
    return VexMultiAgentEnv(
        game=game,
        config=env_config,
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


def apply_training_metadata_overrides(args, metadata, explicit_cli_flags):
    """Override CLI args with values from training_metadata.json unless explicitly set via CLI."""
    metadata_to_arg = {
        "learning_rate": ("learning_rate", ["--learning-rate"]),
        "learning_rate_final": ("learning_rate_final", ["--learning-rate-final"]),
        "discount_factor": ("discount_factor", ["--discount-factor"]),
        "entropy": ("entropy", ["--entropy"]),
        "entropy_final": ("entropy_final", ["--entropy-final"]),
        "num_iters": ("num_iters", ["--num-iters"]),
    }

    for metadata_key, (arg_name, cli_flags) in metadata_to_arg.items():
        if metadata_key in metadata and not any(flag in explicit_cli_flags for flag in cli_flags):
            setattr(args, arg_name, metadata[metadata_key])


def get_explicit_cli_flags(raw_argv):
    """Return set of explicitly provided long-form CLI flags from argv."""
    explicit_flags = set()
    for token in raw_argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        explicit_flags.add(flag)
    return explicit_flags

if __name__ == "__main__":
    # Suppress excessive experiment checkpoint warnings completely
    os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
    
    # Suppress deprecation warnings from RLlib internal code
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    
    # Suppress gymnasium precision/space warnings (RLlib internal env creation)
    warnings.filterwarnings("ignore", module="gymnasium")


    
    # Get parameters from command line arguments
    parser = argparse.ArgumentParser(description="Train multiple agents concurrently.")
    VexEnvConfig.add_cli_args(
        parser,
        communication=True,
        experiment_path=""
    )
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
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--partition', type=str, default="teaching", help='SLURM partition to use')
    parser.add_argument('--algorithm', type=str, default="PPO", help='Algorithm to use for training')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity mode: 0 = silent, 1 = default, 2 = verbose')

    explicit_cli_flags = get_explicit_cli_flags(sys.argv[1:])
    args = parser.parse_args()
    env_config_obj = VexEnvConfig.from_args(args)

    restore_path = None
    restore_experiment_directory = env_config_obj.experiment_path
    if restore_experiment_directory:
        restore_experiment_directory = os.path.abspath(restore_experiment_directory)
        metadata_path = os.path.join(restore_experiment_directory, "training_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            apply_training_metadata_overrides(args, metadata, explicit_cli_flags)
            # Recreate config obj in case args were updated
            env_config_obj = VexEnvConfig.from_args(args)
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
        "game": env_config_obj.game_name,
        "randomize": env_config_obj.randomize,
        "enable_communication": env_config_obj.enable_communication,
        "deterministic": env_config_obj.deterministic,
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

    all_agents = temp_env.possible_agents
    configured_policy_ids = sorted({policy_mapping_fn(agent_id, None) for agent_id in all_agents})
    if not configured_policy_ids:
        raise ValueError("No policies were produced by policy_mapping_fn.")
    print(f"Configured policy IDs: {configured_policy_ids}")
    train_batch_size_per_learner = 4096

    # Set number of iterations manually according to phase list
    args.num_iters = sum(phase["iterations"] for phase in TRAINING_PHASES)
    print(f"Total training iterations set to {args.num_iters} based on TRAINING_PHASES.")

    print("Using custom callback for LR and entropy scheduling based on iterations.")

    # RLlib new API: schedules must be provided directly via `lr` and
    # `entropy_coeff` (not via deprecated *_schedule fields). Pass the initial values.
    lr_config_value = TRAINING_PHASES[0]["lr"]
    entropy_config_value = TRAINING_PHASES[0]["entropy"]
    
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
                "randomize": env_config_obj.randomize,
                "game": env_config_obj.game_name,
                "enable_communication": env_config_obj.enable_communication,
                "deterministic": env_config_obj.deterministic,
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
            policies=set(configured_policy_ids),  # Define policy IDs from mapping function
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(configured_policy_ids),
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=VexCustomPPO,  # Use custom model with clean architecture
                observation_space=obs_space,
                action_space=act_space,
                model_config={"enable_communication": env_config_obj.enable_communication}  # Model architecture defined in vex_custom_model.py
            )
        )
        .fault_tolerance(
            max_num_env_runner_restarts=0,  # Disable env runner restarts to avoid object store issues
        )
        .training(
            lr=lr_config_value,
            gamma=args.discount_factor,
            entropy_coeff=entropy_config_value,
            train_batch_size_per_learner=train_batch_size_per_learner,
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
        "game": env_config_obj.game_name,
        "training_phases": TRAINING_PHASES,
        "discount_factor": args.discount_factor,
        "randomize": env_config_obj.randomize,
        "num_iters": args.num_iters,
        "source_experiment_directory": restore_experiment_directory,
        "restored_checkpoint_path": restore_path,
        "enable_communication": env_config_obj.enable_communication,
        "deterministic": env_config_obj.deterministic,
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
    best_checkpoint_path = best_checkpoint.path if best_checkpoint is not None else find_latest_checkpoint(experiment_dir)
    if best_checkpoint_path is None:
        raise RuntimeError(
            f"No checkpoint was found in {experiment_dir}. "
            "Check checkpoint settings or trial logs before compilation."
        )
    
    print(f"Saving results to: {experiment_dir}")

    compile_checkpoint_to_torchscript(temp_env.game, best_checkpoint_path, experiment_dir, env_config=env_config_obj)

    # Shutdown Ray
    ray.shutdown()
    
    # Run the test script automatically via imported function
    print(f"Running automated testing for 1000 iterations...")
    try:
        test_config = VexEnvConfig(
            game_name=env_config_obj.game_name,
            render_mode=None,
            experiment_path=experiment_dir,
            randomize=env_config_obj.randomize,
            enable_communication=env_config_obj.enable_communication,
            deterministic=env_config_obj.deterministic
        )
        run_simulation(
            config=test_config,
            iterations=1000,
            export_gif=False,
            communication_override=env_config_obj.enable_communication
        )
        print("Automated testing complete.")
    except Exception as e:
        print(f"Automated testing failed: {e}")

