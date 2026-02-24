import argparse
import os
import json
import glob
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import torch
import torch.nn as nn

# Import from new modular architecture
from vex_core.base_game import VexGame
from vex_core.base_env import VexMultiAgentEnv
from pushback import PushBackGame
# Import your custom model class to ensure pickle can find it
from vex_custom_model import VexCustomPPO


def find_latest_checkpoint(experiment_path: str):
    """Find the latest RLlib checkpoint directory under an experiment path."""
    checkpoint_candidates = []

    primary_pattern = os.path.join(experiment_path, "PPO_VEX*", "checkpoint_*")
    checkpoint_candidates.extend(glob.glob(primary_pattern))

    if not checkpoint_candidates:
        fallback_pattern = os.path.join(experiment_path, "**", "checkpoint_*")
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

def compile_checkpoint_to_torchscript(game: VexGame, checkpoint_path: str, output_path: str = None, enable_communication: bool = True):
    def env_creator(config=None):
        config = config or {}
        return VexMultiAgentEnv(
            game=game, 
            render_mode=None,
            enable_communication=config.get("enable_communication", enable_communication)
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    register_env("VEX_Multi_Agent_Env", env_creator)

    # Normalize the checkpoint URI to an absolute path (Ray may expect a file:// or absolute path)
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"Resolved checkpoint to absolute path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path does not exist: {checkpoint_path}")
        return

    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        # Provide clearer guidance for common URI errors
        err_msg = str(e)
        print(f"Error restoring algorithm: {err_msg}")
        if "URI has empty scheme" in err_msg:
            print("Retrying with file:// scheme...")
            try:
                algo = Algorithm.from_checkpoint(f"file://{checkpoint_path}")
            except Exception as e2:
                print(f"Retry failed: {e2}")
                return
        else:
            return

    for policy_id in algo.config.policies:
        print(f"--- Exporting Policy: {policy_id} ---")
        
        rl_module = algo.get_module(policy_id)
        rl_module.eval()
        
        if hasattr(rl_module, 'clean_encoder'):
            clean_encoder = rl_module.clean_encoder
            # Export head: include message head + log-std when available so
            # the exported TorchScript matches RLModule ACTION_DIST_INPUTS.
            pi_head = rl_module.pi
            message_head = getattr(rl_module, "message_head", None)
            msg_log_std = getattr(rl_module, "msg_log_std", None)

            class CombinedHead(nn.Module):
                """Wrapper that returns concatenated [logits, msg_mean, msg_log_std]
                when the ATOC message head exists; otherwise returns logits only.
                """
                def __init__(self, pi, msg_head=None, msg_log_std=None):
                    super().__init__()
                    self.pi = pi
                    self.msg_head = msg_head
                    self.msg_log_std = msg_log_std

                def forward(self, feats: torch.Tensor) -> torch.Tensor:
                    logits = self.pi(feats)
                    if (self.msg_head is None) or (self.msg_log_std is None):
                        return logits
                    batch_size = feats.shape[0]
                    msg_mean = self.msg_head(feats)
                    msg_log_std_exp = self.msg_log_std.expand(batch_size, -1)
                    return torch.cat([logits, msg_mean, msg_log_std_exp], dim=1)

            clean_head = CombinedHead(pi_head, message_head, msg_log_std)
            
            # CRITICAL: Convert any numpy types to Python native types
            # This is necessary for torch.jit.script compatibility
            def convert_module_to_native_types(module):
                """Recursively convert numpy types in module attributes to Python types."""
                import numpy as np
                for name, value in module.__dict__.items():
                    if isinstance(value, np.integer):
                        setattr(module, name, int(value))
                    elif isinstance(value, np.floating):
                        setattr(module, name, float(value))
                # Also check submodules
                for submodule in module.children():
                    convert_module_to_native_types(submodule)
            
            # Apply conversion to both encoder and head
            convert_module_to_native_types(clean_encoder)
            convert_module_to_native_types(clean_head)
            
            # Combine them into a simple container for export
            class ExportModel(nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head
                
                def forward(self, obs):
                    # Pure Tensor operations!
                    feats = self.encoder(obs)
                    return self.head(feats)
            
            export_model = ExportModel(clean_encoder, clean_head)
            export_model.eval()
            
            # Save using JIT SCRIPT (Best for C++)
            save_path = os.path.join(output_path, f"{policy_id}.pt")
            temp_env = env_creator()
            obs_shape = temp_env.observation_space(temp_env.possible_agents[0]).shape
            dummy_obs = torch.randn(1, *obs_shape)
            
            try:
                # We can script this directly because it has no dicts/kwargs!
                scripted = torch.jit.script(export_model)
                scripted.save(save_path)
                print(f"✓ SUCCESS: Saved fully JIT Scripted model to {save_path}")
            except Exception as e:
                print(f"Scripting failed ({e}), trying trace...")
                # Fallback to trace if needed
                traced = torch.jit.trace(export_model, dummy_obs)
                traced.save(save_path)
                print(f"✓ SUCCESS: Saved JIT Traced model to {save_path}")

        else:
            print("Could not find clean encoder in custom model.")

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-path", type=str, required=True)
    args = parser.parse_args()

    experiment_path = os.path.abspath(args.experiment_path)
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")

    metadata_path = os.path.join(experiment_path, "training_metadata.json")
    game_name = "vexai_skills"
    enable_communication = False

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                game_name = metadata.get("game", game_name)
                enable_communication = metadata.get("enable_communication", False)
            print(f"Read metadata: game={game_name}, enable_communication={enable_communication}")
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")
    else:
        print(f"Warning: No training metadata found at {metadata_path}. Using defaults.")

    checkpoint_path = find_latest_checkpoint(experiment_path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint directories found under: {experiment_path}")

    print(f"Using checkpoint: {checkpoint_path}")
    game = PushBackGame.get_game(game_name, enable_communication=enable_communication)
    compile_checkpoint_to_torchscript(game, checkpoint_path, experiment_path, enable_communication)