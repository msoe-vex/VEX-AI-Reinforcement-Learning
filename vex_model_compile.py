import argparse
import os
import json
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

def compile_checkpoint_to_torchscript(game: VexGame, checkpoint_path: str, output_path: str = None):
    def env_creator(config=None):
        return VexMultiAgentEnv(game=game, render_mode=None)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    register_env("VEX_Multi_Agent_Env", env_creator)

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error restoring algorithm: {e}")
        return

    for policy_id in algo.config.policies:
        print(f"--- Exporting Policy: {policy_id} ---")
        
        rl_module = algo.get_module(policy_id)
        rl_module.eval()
        
        if hasattr(rl_module, 'clean_encoder'):
            clean_encoder = rl_module.clean_encoder
            clean_head = rl_module.pi
            
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
            obs_shape = game.observation_space(game.possible_agents[0]).shape
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
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--game", type=str, default="vexai_skills")
    args = parser.parse_args()
    
    game = PushBackGame.get_game(args.game)
    compile_checkpoint_to_torchscript(game, args.checkpoint_path, args.output_path)