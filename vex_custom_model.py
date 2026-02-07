"""
Custom RLModule for VEX robotics with clean, exportable architecture.
"""

import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from gymnasium.spaces import Discrete, Box, Tuple

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
    
    # ATOC Heads
    # 1. Intention Head (The Policy)
    intention_head = nn.Linear(prev_dim, action_dim)
    
    # 2. Value Head
    value_head = nn.Linear(prev_dim, 1)
    
    # 3. Attention Unit (Should I communicate? / Communication Weight)
    # Output: 1 scalar (logit)
    attention_unit = nn.Linear(prev_dim, 1)
    
    # 4. Message Encoder (What to say)
    # Output: 8 dimensions (Mean of the logical message)
    message_encoder = nn.Linear(prev_dim, 8)
    
    # Message LogStd (Learnable parameter)
    message_log_std = nn.Parameter(torch.zeros(1, 8))
    
    return encoder, intention_head, value_head, attention_unit, message_encoder, message_log_std



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
        elif isinstance(self.action_space, Tuple):
            # Assumes structure: (Discrete(Action), Box(Message))
            # We focus on the first part for the Intention Head (Action Logic)
            # The message part is handled by the explicit Message Head
            first_space = self.action_space[0]
            if isinstance(first_space, Discrete):
                action_dim = first_space.n
                is_continuous = False
            else:
                 raise ValueError(f"Unsupported first element in Tuple action space: {type(first_space)}")
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        # For continuous actions, double the output size (mean + log_std)
        policy_output_dim = action_dim * 2 if is_continuous else action_dim
        
        # Build the actual model using our simple function
        self._encoder_net, self.pi, self.value_head, self.attention_unit, self.message_head, self.msg_log_std = build_vex_model(
            obs_dim=obs_dim,
            action_dim=action_dim if not isinstance(self.action_space, Tuple) else 
                       (self.action_space[0].n if isinstance(self.action_space[0], Discrete) else 0)
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
        
        # Heads
        intention_logits = self.pi(features)
        
        # ATOC Outputs
        attention_logits = self.attention_unit(features)
        msg_mean = self.message_head(features)
        
        # Concat outputs for Tuple Distribution: [DiscreteLogits, BoxMean, BoxLogStd]
        # DiscreteLogits: (Batch, N)
        # BoxMean: (Batch, 8)
        # BoxLogStd: (Batch, 8) - Expand parameter
        batch_size = features.shape[0]
        msg_log_std_exp = self.msg_log_std.expand(batch_size, -1)
        
        dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp], dim=1)
        
        output[Columns.ACTION_DIST_INPUTS] = dist_inputs
        output[Columns.VF_PREDS] = self.value_head(features).squeeze(-1)
        output[Columns.EMBEDDINGS] = features
        
        # Pass extra outputs
        output["attention_logits"] = attention_logits
        output["message"] = msg_mean
        
        return output


    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        output = {}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT]
        
        intention_logits = self.pi(features)
        
        # ATOC Outputs
        attention_logits = self.attention_unit(features)
        msg_mean = self.message_head(features)
        
        # Concat outputs for Tuple Distribution: [DiscreteLogits, BoxMean, BoxLogStd]
        batch_size = features.shape[0]
        msg_log_std_exp = self.msg_log_std.expand(batch_size, -1)
        
        dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp], dim=1)
        
        output[Columns.ACTION_DIST_INPUTS] = dist_inputs
        
        # Also compute messages for the environment to use next step
        output["attention_logits"] = attention_logits
        output["message"] = msg_mean
        
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