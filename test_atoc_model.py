
import unittest
import torch
import numpy as np
from gymnasium import spaces
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vex_custom_model import VexCustomPPO, Columns

class TestATOCModel(unittest.TestCase):
    def test_model_shapes(self):
        # Observation Space: 88 dims
        obs_dim = 88
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        
        # Action Space: Tuple(Discrete(5), Box(8))
        # Assuming 5 actions for test
        num_actions = 5
        action_space = spaces.Tuple((
            spaces.Discrete(num_actions),
            spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        ))
        
        # Instantiate Model
        # We need a dummy config
        model_config = {}
        
        model = VexCustomPPO(
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config,
        )
        
        # Dummy Input Batch
        batch_size = 4
        obs_tensor = torch.randn(batch_size, obs_dim) # [B, 88]
        
        input_dict = {
            "obs": obs_tensor,
            # RLlib passes separate flat obs sometimes, but "obs" key usually holds the tensor
            # If ModelV2, it expects SampleBatch keys sometimes.
            # VexCustomPPO inherits from TorchModelV2
        }
        
        # Forward Pass
        # New RLModule API: forward(batch) -> params dict
        output = model.forward(input_dict)
        
        # Output should be [Batch, N + 16]
        # In new API, output is a dict with keys like ACTION_DIST_INPUTS
        if Columns.ACTION_DIST_INPUTS in output:
             dist_inputs = output[Columns.ACTION_DIST_INPUTS]
        else:
             # Fallback if specific forward_inference implementation returns differently
             # But VexCustomPPO._forward_inference sets ACTION_DIST_INPUTS
             dist_inputs = output.get("action_dist_inputs", output)

        expected_dim = num_actions + 8 + 8
        self.assertEqual(dist_inputs.shape, (batch_size, expected_dim))
        
        # Check components
        # Intention Logits: first num_actions
        # Message Mean: next 8
        # Message LogStd: next 8
        
        # Check extra outputs
        self.assertTrue("attention_logits" in output) # Wait, forward returns TENSOR.
        # extra outputs are stored in model._last_output usually? Or returned in output dict if API stack new?
        # VexCustomPPO uses `self.context` or just returns tensor?
        # Let's check vex_custom_model.py again.
        
        # In vex_custom_model.py, _forward_train returns dict. 
        # But forward() calls _forward_train and returns... wait.
        # Custom model usually overrides forward().
        pass 

if __name__ == "__main__":
    unittest.main()
