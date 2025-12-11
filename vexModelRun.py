import torch
import argparse
import os
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame

class ModelRunner:
    def __init__(self, model_path: str, env: VexGame):
        self.model_path = model_path
        self.env = env
        self.model = None

        # Load the TorchScript model
        try:
            self.model = torch.jit.load(self.model_path)
            self.model.eval()  # Set the model to evaluation mode
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def get_actions(self, observation: np.ndarray):
        """
        Loads a trained model and runs a simulation in the High_Stakes_Multi_Agent_Env.

        Args:
            observation: The current observation for the agent.
        """

        obs_np = observation # TODO: verify correct preprocessing
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.model(obs_tensor)
        # Sort actions by descending logit value (best first)
        sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
        # Find the first valid action
        action = None
        for candidate_action in sorted_actions:
            if self.env.is_valid_action(candidate_action, observation):
                action = candidate_action
                break
        
        if action is None:
            # Fallback: if no valid action found, pick top choice
            action = torch.argmax(action_logits, dim=1).item()

        return self.env.split_action(action, observation)
