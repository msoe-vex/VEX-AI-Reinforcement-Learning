import argparse
import os
import json
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
import warnings

# Import from new modular architecture
from vex_core.base_game import VexGame
from vex_core.base_env import VexMultiAgentEnv
from pushback import PushBackGame


def compile_checkpoint_to_torchscript(game: VexGame, checkpoint_path: str, output_path: str = None):
    """
    Loads an agent from a checkpoint and saves its RL Modules as TorchScript files.

    Args:
        game (VexGame): The environment to use for the game.
        checkpoint_path (str): Path to the RLLib checkpoint directory.
    """
    def env_creator(config=None):
        """Create environment instance for RLlib registration."""
        return VexMultiAgentEnv(game=game, render_mode=None)

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
    register_env("VEX_Multi_Agent_Env", env_creator)

    # Restore the algorithm from the checkpoint
    try:
        # Use Algorithm.from_checkpoint to be generic for any algorithm
        algo = Algorithm.from_checkpoint(checkpoint_path)
        print(f"Successfully restored algorithm from checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error restoring algorithm from checkpoint: {e}")
        ray.shutdown()
        return

    # Iterate over all policies/modules in the algorithm
    for policy_id in algo.config.policies:
        print(f"Saving module for policy: {policy_id}")
        
        # This is the modern way to access the underlying neural network (RLModule).
        try:
            module = algo.get_module(policy_id)
            module.eval()  # Set the module to evaluation mode
        except Exception as e:
            print(f"Error getting RLModule: {e}")
            algo.stop()
            ray.shutdown()
            return
        
        # Use static method to get observation shape
        obs_shape = game.observation_space(game.possible_agents[0]).shape

        # Create a dummy observation tensor
        dummy_obs = torch.randn(1, *obs_shape).clone().detach()

        # The new RLModule uses a dictionary for input and `forward_inference` for output.
        class TracedModel(torch.nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self.original_module = original_module

            def forward(self, obs):
                # The RLModule's forward_inference expects a batched dictionary.
                input_dict = {"obs": obs}
                # The output is also a dictionary. We extract the action logits.
                output_dict = self.original_module.forward_inference(input_dict)
                # 'action_dist_inputs' is the standard key for action logits.
                return output_dict['action_dist_inputs']

        traced_wrapper = TracedModel(module)

        # Trace the model
        try:
            final_traced_model = torch.jit.trace(traced_wrapper, (dummy_obs))
        except Exception as e:
            print(f"Error during model tracing: {e}")
            algo.stop()
            ray.shutdown()
            return

        # Determine the path to save the traced model
        os.makedirs(output_path, exist_ok=True)
        traced_model_path = os.path.join(output_path, f"{policy_id}.pt")
        
        try:
            final_traced_model.save(traced_model_path)
            print(f"Traced TorchScript model saved to: {traced_model_path}")
        except Exception as e:
            print(f"Error saving traced model: {e}")

    # Clean up
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(description="Compile an RLLib checkpoint to a TorchScript model.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the RLLib checkpoint directory.")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for the TorchScript model(s).")
    parser.add_argument("--game", type=str, default=None, help="Game variant (if not provided, reads from training_metadata.json)")
    args = parser.parse_args()
    
    # Try to read game from metadata if not provided
    game_name = args.game
    if game_name is None:
        metadata_path = os.path.join(args.output_path, "training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                game_name = metadata.get("game", "vexai_skills")
            print(f"Read game variant from metadata: {game_name}")
        else:
            game_name = "vexai_skills"
            print(f"No metadata found, using default game: {game_name}")
    
    game = PushBackGame.get_game(game_name)
    compile_checkpoint_to_torchscript(game, args.checkpoint_path, args.output_path)