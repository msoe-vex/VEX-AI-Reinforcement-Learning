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

from vex_core.base_env import MESSAGE_SIZE


# ============================================================================
# SIMPLE MODEL DEFINITION - Easy to modify!
# ============================================================================

class FusionEncoder(nn.Module):
    def __init__(self, core_dim, comm_dim):
        super().__init__()
        self.core_dim = core_dim
        self.comm_dim = comm_dim
        
        # 1. Core Feature Extractor
        self.core_net = nn.Sequential(
            nn.LayerNorm(core_dim),
            nn.Linear(core_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Communication Feature Extractor (only if comm_dim > 0)
        if comm_dim > 0:
            self.comm_net = nn.Sequential(
                nn.LayerNorm(comm_dim),
                nn.Linear(comm_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            fused_dim = 256 + 128
        else:
            self.comm_net = None
            fused_dim = 256
            
        # 3. Fused Network
        self.fused_net = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = 512

    def forward(self, x):
        if self.comm_dim > 0:
            core_obs = x[:, :self.core_dim]
            comm_obs = x[:, self.core_dim:]
            
            core_feat = self.core_net(core_obs)
            comm_feat = self.comm_net(comm_obs)
            
            fused = torch.cat([core_feat, comm_feat], dim=1)
        else:
            fused = self.core_net(x)
            
        return self.fused_net(fused)

def build_vex_model(core_dim, comm_dim, action_dim, enable_communication=False):
    """
    Build a simple feedforward neural network for VEX robotics.
    
    This is a PURE PyTorch model - no RLlib dependencies!
    Modify this function to change your model architecture.
    
    Args:
        core_dim: Input observation dimension for the core environment features
        comm_dim: Input observation dimension for the communication features
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
    encoder = FusionEncoder(core_dim, comm_dim)
    prev_dim = encoder.output_dim
    
    # ATOC Heads
    # 1. Intention Head (The Policy)
    intention_head = nn.Linear(prev_dim, action_dim)
    
    # 2. Value Head
    value_head = nn.Linear(prev_dim, 1)
    
    # 3. Attention Unit (Should I communicate? / Communication Weight)
    # Output: 1 scalar (logit)
    # Only create if communication is enabled
    if enable_communication:
        attention_unit = nn.Linear(prev_dim, 1)
        # 4. Message Encoder (What to say)
        message_encoder = nn.Linear(prev_dim, MESSAGE_SIZE)
        # Message LogStd (Learnable parameter)
        message_log_std = nn.Parameter(torch.zeros(1, MESSAGE_SIZE))
    else:
        attention_unit = None
        message_encoder = None
        message_log_std = None
    
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
        if hasattr(self.observation_space, "spaces") and "observations" in self.observation_space.spaces:
            obs_dim = self.observation_space["observations"].shape[0]
        else:
            obs_dim = self.observation_space.shape[0]
        
        # Determine enable_communication from action space type.
        # When communication is enabled, action space is Tuple(Discrete, Box).
        # When disabled, action space is just Discrete.
        enable_communication = isinstance(self.action_space, Tuple)
        
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
            is_continuous = False
        elif isinstance(self.action_space, Box):
            action_dim = self.action_space.shape[0]
            is_continuous = True
        elif isinstance(self.action_space, Tuple):
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
        
        try:
            comm_mode_str = self.model_config.get("communication_mode", "none")
            from vex_core.config import CommunicationOption
            comm_mode = CommunicationOption(comm_mode_str)
        except Exception:
            from vex_core.config import CommunicationOption
            comm_mode = CommunicationOption.NONE

        if comm_mode == CommunicationOption.ATTENTION:
            from vex_core.base_env import MESSAGE_SIZE
            comm_dim = MESSAGE_SIZE
            core_dim = obs_dim - comm_dim
        elif comm_mode == CommunicationOption.COPY:
            num_teammates = 1  # Standard for VEX AI
            core_dim = obs_dim // (num_teammates + 1)
            comm_dim = obs_dim - core_dim
        else:
            core_dim = obs_dim
            comm_dim = 0
            
        # Build the actual model using our simple function
        self._encoder_net, self.pi, self.value_head, self.attention_unit, self.message_head, self.msg_log_std = build_vex_model(
            core_dim=core_dim,
            comm_dim=comm_dim,
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
                    if "observations" in x:
                        obs = x["observations"]
                    elif Columns.OBS in x:
                        obs = x[Columns.OBS]
                        if isinstance(obs, dict) and "observations" in obs:
                            obs = obs["observations"]
                    else:
                        obs = x
                else:
                    obs = x
                features = self.net(obs)
                return {ENCODER_OUT: features}
        
        self.encoder_bridge = EncoderBridge(self._encoder_net)

        # Set action distribution class for Tuple action spaces (Discrete + Box)
        # The default TorchRLModule only handles Discrete/Box, not Tuple.
        if enable_communication:
            from ray.rllib.core.models.catalog import Catalog
            import gymnasium as gym
            # Use a dummy obs space — Catalog can't handle Dict obs spaces,
            # but we only need it to resolve the action distribution class.
            catalog = Catalog(
                observation_space=gym.spaces.Box(-1.0, 1.0, (1,)),
                action_space=self.action_space,
                model_config_dict={},
            )
            self.action_dist_cls = catalog.get_action_dist_cls(framework="torch")

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        output = {}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT] 
        
        # Heads
        intention_logits = self.pi(features)
        
        # Action masking
        if isinstance(batch, dict):
            obs_dict = batch.get(Columns.OBS, batch)
            if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                action_mask = obs_dict["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=-1e8)
                intention_logits = intention_logits + inf_mask
        
        if self.attention_unit is not None and self.message_head is not None:
            # ATOC Outputs - Communication enabled (Tuple action space)
            attention_logits = self.attention_unit(features)
            msg_mean = self.message_head(features)
            
            # Gate message by attention (soft on/off for communication)
            gate = torch.sigmoid(attention_logits)  # (B, 1)
            msg_mean = msg_mean * gate               # (B, MESSAGE_SIZE) * (B, 1) → broadcast
            
            # Concat outputs for Tuple Distribution: [DiscreteLogits, BoxMean, BoxLogStd]
            batch_size = features.shape[0]
            gated_log_std = self.msg_log_std + torch.log(gate + 1e-8)
            msg_log_std_exp = gated_log_std.expand(batch_size, -1)
            
            dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp], dim=1)
            
            output["attention_logits"] = attention_logits
            output["comm_gate"] = gate
            output["message"] = msg_mean
        else:
            # Communication disabled (Discrete action space) - only action logits
            dist_inputs = intention_logits
        
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
        
        # Action masking
        if isinstance(batch, dict):
            obs_dict = batch.get(Columns.OBS, batch)
            if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                action_mask = obs_dict["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask + 1e-10), min=-1e8)
                intention_logits = intention_logits + inf_mask
        
        if self.attention_unit is not None and self.message_head is not None:
            # ATOC Outputs - Communication enabled (Tuple action space)
            attention_logits = self.attention_unit(features)
            msg_mean = self.message_head(features)
            
            # Gate message by attention (soft on/off for communication)
            gate = torch.sigmoid(attention_logits)  # (B, 1)
            msg_mean = msg_mean * gate               # (B, MESSAGE_SIZE) * (B, 1) → broadcast
            
            # Concat outputs for Tuple Distribution: [DiscreteLogits, BoxMean, BoxLogStd]
            batch_size = features.shape[0]
            gated_log_std = self.msg_log_std + torch.log(gate + 1e-8)
            msg_log_std_exp = gated_log_std.expand(batch_size, -1)
            
            dist_inputs = torch.cat([intention_logits, msg_mean, msg_log_std_exp], dim=1)
            
            output["attention_logits"] = attention_logits
            output["comm_gate"] = gate
            output["message"] = msg_mean
        else:
            # Communication disabled (Discrete action space) - only action logits
            dist_inputs = intention_logits
        
        output[Columns.ACTION_DIST_INPUTS] = dist_inputs
        
        return output


    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    def compute_values(self, batch, embeddings=None):
        if isinstance(batch, dict):
            obs = batch.get(Columns.OBS, batch)
            if isinstance(obs, dict) and "observations" in obs:
                obs = obs["observations"]
        else:
            obs = batch
        features = self._encoder_net(obs)
        return self.value_head(features).squeeze(-1)
    
    @property
    def encoder(self):
        return self.encoder_bridge

    @property
    def clean_encoder(self):
        """Returns the pure PyTorch encoder for TorchScript export."""
        return self._encoder_net