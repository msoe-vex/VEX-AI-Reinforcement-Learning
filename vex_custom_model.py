"""
Custom RLModule for VEX robotics with clean, exportable architecture.
"""

import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from gymnasium.spaces import Discrete, Box

# Import internal constants required for the output dictionary
from ray.rllib.core.models.base import ENCODER_OUT, CRITIC, ACTOR


# ============================================================================
# SIMPLE MODEL DEFINITION - Easy to modify!
# ============================================================================

def build_vex_model(obs_dim, action_dim):
    """
    Build a simple feedforward neural network for VEX robotics.
    
    This is a PURE PyTorch model - no RLlib dependencies!
    Modify this function to change your model architecture.
    
    Args:
        obs_dim: Input observation dimension
        action_dim: Output action dimension
    
    Returns:
        encoder: nn.Sequential feature extractor
        policy_head: nn.Linear for action logits
        value_head: nn.Linear for value predictions
    """
    # Build encoder layers
    encoder_layers = []
    prev_dim = obs_dim

    # Normalize inputs for better training stability
    encoder_layers.append(nn.LayerNorm(obs_dim))

    hidden_layers = [256, 512, 256]
    dropout_rate = 0.1  # Light dropout for regularization
    
    for hidden_size in hidden_layers:
        encoder_layers.append(nn.Linear(prev_dim, hidden_size))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Dropout(dropout_rate))
        
        prev_dim = hidden_size
    
    encoder = nn.Sequential(*encoder_layers)
    policy_head = nn.Linear(prev_dim, action_dim)
    value_head = nn.Linear(prev_dim, 1)
    
    return encoder, policy_head, value_head


# ============================================================================
# RLLIB COMPATIBILITY WRAPPER - Don't modify unless you know what you're doing
# ============================================================================

class VexCustomPPO(DefaultPPOTorchRLModule):
    """
    RLlib-compatible wrapper for the VEX model.
    Uses build_vex_model() for the actual neural network.
    """
    
    def setup(self):
        # Get dimensions from environment spaces
        obs_dim = self.observation_space.shape[0]
        
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
            is_continuous = False
        elif isinstance(self.action_space, Box):
            action_dim = self.action_space.shape[0]
            is_continuous = True
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        # For continuous actions, double the output size (mean + log_std)
        policy_output_dim = action_dim * 2 if is_continuous else action_dim
        
        # Build the actual model using our simple function
        self._encoder_net, self.pi, self.value_head = build_vex_model(
            obs_dim=obs_dim,
            action_dim=policy_output_dim
        )
        
        # Create RLlib bridge for encoder
        class EncoderBridge(nn.Module):
            def __init__(self, clean_net):
                super().__init__()
                self.net = clean_net
            
            def forward(self, x):
                if isinstance(x, dict):
                    x = x[Columns.OBS]
                features = self.net(x)
                return {ENCODER_OUT: features}
        
        self.encoder_bridge = EncoderBridge(self._encoder_net)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        output = {}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT] 
        output[Columns.ACTION_DIST_INPUTS] = self.pi(features)
        output[Columns.VF_PREDS] = self.value_head(features).squeeze(-1)
        output[Columns.EMBEDDINGS] = features
        return output

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        output = {}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT]
        output[Columns.ACTION_DIST_INPUTS] = self.pi(features)
        return output

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    def compute_values(self, batch, embeddings=None):
        obs = batch[Columns.OBS]
        features = self._encoder_net(obs)
        return self.value_head(features).squeeze(-1)
    
    @property
    def encoder(self):
        return self.encoder_bridge

    @property
    def clean_encoder(self):
        """Returns the pure PyTorch encoder for TorchScript export."""
        return self._encoder_net