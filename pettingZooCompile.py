import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
import warnings

# Ensure pettingZooEnv.py is accessible
from pettingZooEnv import env_creator

# Policy mapping function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id

def compile_checkpoint_to_torchscript(checkpoint_path: str, output_path: str = None):
    """
    Loads a PPO agent from a checkpoint and saves its models as TorchScript files.

    Args:
        checkpoint_path (str): Path to the RLLib checkpoint directory.
    """
    if not os.path.isdir(checkpoint_path):
        print(f"Error: Checkpoint path {checkpoint_path} not found or not a directory.")
        return

    torch, nn = try_import_torch()
    if not torch:
        print("Error: PyTorch not found. Please install PyTorch.")
        return

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("High_Stakes_Multi_Agent_Env", env_creator)

    # Restore the trainer from the checkpoint
    try:
        # Use PPO.from_checkpoint to load the trainer and its config
        trainer = PPO.from_checkpoint(checkpoint_path)
        print(f"Successfully restored trainer from checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error restoring trainer from checkpoint: {e}")
        ray.shutdown()
        return

    for policy_id in trainer.config.policies:
        print(f"Saving policy: {policy_id}")
        try:
            policy = trainer.get_policy(policy_id)
            if policy is None:
                print(f"Error: Could not get policy for agent {policy_id}")
                trainer.stop()
                ray.shutdown()
                return
        except Exception as e:
            print(f"Error getting policy: {e}")
            trainer.stop()
            ray.shutdown()
            return
        
        # Create a temporary environment to get observation and action spaces
        # This is still needed for the dummy_input_dict for tracing.
        temp_env = env_creator(None)
        obs_space = temp_env.observation_space(policy_id)
        temp_env.close()

        # Extract the model from the policy
        model = policy.model
        model.eval()  # Set the model to evaluation mode

        # Create a dummy input_dict with the expected structure (only observation)
        dummy_input_dict = {
            "obs": torch.randn(1, *obs_space.shape).clone().detach(),
        }

        # Wrap the model for tracing
        class TracedModel(torch.nn.Module):
            def __init__(self, original_model_ref):
                super(TracedModel, self).__init__()
                self.original_model_ref = original_model_ref

            def forward(self, obs):
                input_dict_local = {"obs": obs}
                # The model's forward pass returns (output_logits, state_list)
                # We are interested in the logits for action selection.
                return self.original_model_ref(input_dict_local)[0]

        traced_wrapper = TracedModel(model)

        # Trace the model
        try:
            final_traced_model = torch.jit.trace(traced_wrapper, (dummy_input_dict["obs"]))
        except Exception as e:
            print(f"Error during model tracing: {e}")
            trainer.stop()
            ray.shutdown()
            return

        # Determine the path to save the traced model
        if output_path is not None:
            # If output_path is a directory, save as <output_path>/<policy_id>.pt
            os.makedirs(output_path, exist_ok=True)
            traced_model_path = os.path.join(output_path, f"{policy_id}.pt")
        else:
            traced_model_path = os.path.join(checkpoint_path, f"{policy_id}.pt")
        try:
            final_traced_model.save(traced_model_path)
            print(f"Traced TorchScript model saved to: {traced_model_path}")
        except Exception as e:
            print(f"Error saving traced model: {e}")

    # Clean up
    trainer.stop()
    ray.shutdown()

if __name__ == "__main__":
    # Suppress all deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(description="Compile a PPO checkpoint to a TorchScript model.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the RLLib checkpoint directory (e.g., /path/to/PPO_High_Stakes_Multi_Agent_Env_.../checkpoint_000005).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        default=None,
        help="Optional output path for the TorchScript model(s). If a directory, models are saved as <policy_id>.pt inside it.",
    )
    args = parser.parse_args()

    compile_checkpoint_to_torchscript(args.checkpoint_path, args.output_path)
