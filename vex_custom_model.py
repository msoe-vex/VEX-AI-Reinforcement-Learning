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

def build_vex_model(obs_dim, action_dim, enable_communication=False):
    """
    Build a simple feedforward neural network for VEX robotics.
    
    This is a PURE PyTorch model - no RLlib dependencies!
    Modify this function to change your model architecture.
    
    Args:
        obs_dim: Input observation dimension
        action_dim: Output action dimension
        enable_communication: Whether to enable ATOC communication heads
    
    Returns:
        encoder: nn.Sequential feature extractor
        policy_head: nn.Linear for action logits
        value_head: nn.Linear for value predictions
        attention_unit: nn.Linear for communication attention (or None if disabled)
        message_encoder: nn.Linear for message encoding (or None if disabled)
        message_log_std: nn.Parameter for message log std (or None if disabled)
    """
    # Build encoder layers
    encoder_layers = []
    prev_dim = obs_dim

    # Normalize inputs for better training stability
    encoder_layers.append(nn.LayerNorm(obs_dim))

    hidden_layers = [512, 1024, 1024, 512]
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
    
    # 3. Termination Head (Option-Critic β)
    # Output: 1 scalar (mean for Box(1)), trained via PPO policy gradient
    termination_head = nn.Linear(prev_dim, 1)
    termination_log_std = nn.Parameter(torch.zeros(1, 1))
    
    # 4. Attention Unit (Should I communicate? / Communication Weight)
    # Output: 1 scalar (logit)
    # Only create if communication is enabled
    if enable_communication:
        attention_unit = nn.Linear(prev_dim, 1)
        # 5. Message Encoder (What to say)
        # Output: 8 dimensions (Mean of the logical message)
        message_encoder = nn.Linear(prev_dim, 8)
        # Message LogStd (Learnable parameter)
        message_log_std = nn.Parameter(torch.zeros(1, 8))
    else:
        attention_unit = None
        message_encoder = None
        message_log_std = None
    
    return encoder, intention_head, value_head, termination_head, termination_log_std, attention_unit, message_encoder, message_log_std



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
        
        # Determine enable_communication from ACTION SPACE TYPE (primary source of truth)
        # This ensures the model outputs match what RLlib expects
        enable_communication = isinstance(self.action_space, Tuple)
        
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, Tuple):
            # Assumes structure: (Discrete(Action), Box(Message), Box(Beta))
            first_space = self.action_space[0]
            if isinstance(first_space, Discrete):
                action_dim = first_space.n
            else:
                 raise ValueError(f"Unsupported first element in Tuple action space: {type(first_space)}")
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        # Build the actual model using our simple function
        (
            self._encoder_net, self.pi, self.value_head,
            self.termination_head, self.beta_log_std,
            self.attention_unit, self.message_head, self.msg_log_std
        ) = build_vex_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            enable_communication=enable_communication
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
        batch_size = features.shape[0]
        
        # Termination head (option-critic β) — always present
        beta_mean = self.termination_head(features)  # (batch, 1)
        beta_log_std_exp = self.beta_log_std.expand(batch_size, -1)  # (batch, 1)
        
        if self.attention_unit is not None and self.message_head is not None:
            # ATOC Outputs - Communication enabled
            attention_logits = self.attention_unit(features)
            attention_weight = torch.sigmoid(attention_logits)  # (batch, 1) ∈ [0, 1]
            msg_mean_raw = self.message_head(features)
            msg_mean = msg_mean_raw * attention_weight  # Gate message by attention
            
            # Concat for Tuple Distribution: [DiscreteLogits, MsgMean(8), MsgLogStd(8), BetaMean(1), BetaLogStd(1)]
            msg_log_std_exp = self.msg_log_std.expand(batch_size, -1)
            
            dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp, beta_mean, beta_log_std_exp], dim=1)
            
            output["attention_logits"] = attention_logits
            output["attention_weight"] = attention_weight
            output["message"] = msg_mean
        else:
            # No communication — still include β
            dist_inputs = torch.cat([intention_logits, beta_mean, beta_log_std_exp], dim=1)
        
        output[Columns.ACTION_DIST_INPUTS] = dist_inputs
        output[Columns.VF_PREDS] = self.value_head(features).squeeze(-1)
        output[Columns.EMBEDDINGS] = features
        
        return output


    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        output = {}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT]
        
        intention_logits = self.pi(features)
        batch_size = features.shape[0]
        
        # Termination head (option-critic β) — always present
        beta_mean = self.termination_head(features)
        beta_log_std_exp = self.beta_log_std.expand(batch_size, -1)
        
        if self.attention_unit is not None and self.message_head is not None:
            # ATOC Outputs - Communication enabled
            attention_logits = self.attention_unit(features)
            attention_weight = torch.sigmoid(attention_logits)
            msg_mean_raw = self.message_head(features)
            msg_mean = msg_mean_raw * attention_weight
            
            msg_log_std_exp = self.msg_log_std.expand(batch_size, -1)
            dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp, beta_mean, beta_log_std_exp], dim=1)
            
            output["attention_logits"] = attention_logits
            output["attention_weight"] = attention_weight
            output["message"] = msg_mean
        else:
            dist_inputs = torch.cat([intention_logits, beta_mean, beta_log_std_exp], dim=1)
        
        output[Columns.ACTION_DIST_INPUTS] = dist_inputs
        
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