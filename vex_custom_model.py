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

class VexCustomPPO(DefaultPPOTorchRLModule):
    """
    Custom PPO RLModule with clean PyTorch architecture.
    """
    
    def setup(self):
        # 1. Get dimensions
        obs_dim = self.observation_space.shape[0]
        
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
            self.is_continuous = False
        elif isinstance(self.action_space, Box):
            action_dim = self.action_space.shape[0]
            self.is_continuous = True
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        # 2. Network Config
        hidden_layers = self.model_config.get("fcnet_hiddens", [64, 64])
        activation = self.model_config.get("fcnet_activation", "relu")
        
        # 3. Build CLEAN Encoder (The part we export)
        encoder_layers = []
        prev_dim = obs_dim
        
        for hidden_size in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_size))
            if activation == "relu":
                encoder_layers.append(nn.ReLU())
            elif activation == "tanh":
                encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(nn.ReLU())
            prev_dim = hidden_size
        
        # Store the clean neural network in a private variable
        self._encoder_net = nn.Sequential(*encoder_layers)
        
        # 4. BRIDGE: Create a wrapper that RLlib sees as "self.encoder"
        class EncoderBridge(nn.Module):
            def __init__(self, clean_net):
                super().__init__()
                self.net = clean_net
            
            def forward(self, x):
                # If input is a dict (RLlib training), extract OBS
                if isinstance(x, dict):
                    x = x[Columns.OBS]
                
                # Pass purely tensor to the clean net
                features = self.net(x)

                # We use the same features for both Actor and Critic (Shared Encoder)
                return {
                    ENCODER_OUT: features
                }

        self.encoder_bridge = EncoderBridge(self._encoder_net)
        
        # 5. Policy Head
        if self.is_continuous:
            self.pi = nn.Linear(prev_dim, action_dim * 2)
        else:
            self.pi = nn.Linear(prev_dim, action_dim)
        
        # 6. Value Head
        self.value_head = nn.Linear(prev_dim, 1)

    # ------------------------------------------------------------------------
    # We must Override _forward_train to handle the output structure manually
    # ------------------------------------------------------------------------
    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        """
        Custom forward_train to bypass default PPO expectations.
        """
        output = {}
        
        # 1. Run Encoder
        # The bridge returns {'encoder_out': features}
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT] 
        
        # 2. Compute Actions (Actor)
        output[Columns.ACTION_DIST_INPUTS] = self.pi(features)
        
        # 3. Compute Values (Critic)
        output[Columns.VF_PREDS] = self.value_head(features).squeeze(-1)
        
        # 4. Save Embeddings (Optional but good for debugging)
        output[Columns.EMBEDDINGS] = features

        return output

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        """
        Custom forward_inference for cleaner execution.
        """
        output = {}
        
        # 1. Run Encoder
        encoder_outs = self.encoder(batch)
        features = encoder_outs[ENCODER_OUT]
        
        # 2. Compute Actions
        output[Columns.ACTION_DIST_INPUTS] = self.pi(features)
        
        return output

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        """
        Re-use inference logic for exploration.
        """
        return self._forward_inference(batch, **kwargs)

    # Helper for value function computation
    def compute_values(self, batch, embeddings=None):
        obs = batch[Columns.OBS]
        features = self._encoder_net(obs)
        return self.value_head(features).squeeze(-1)
    
    @property
    def encoder(self):
        return self.encoder_bridge

    @property
    def clean_encoder(self):
        return self._encoder_net