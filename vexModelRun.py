import torch
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame

class ModelRunner:
    def __init__(self, model_path: str, game: VexGame):
        self.model_path: str = model_path
        self.game: VexGame = game
        self.model: torch.jit.ScriptModule = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load the TorchScript model
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            self.model = torch.jit.optimize_for_inference(self.model)
            
            print(f"Successfully loaded and optimized model from {self.model_path}")
            
            sample_agent = game.possible_agents[0]
            obs_shape = game.observation_space(sample_agent).shape
            dummy_input = torch.randn(1, *obs_shape, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("Model warmed up and ready for inference")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def get_actions(self, observation: np.ndarray):
        """
        Runs inference on the model to get actions for the given observation.

        Args:
            observation: The current observation for the agent.
        """
        # Zero-Copy Tensor Creation: Use from_numpy() instead of tensor()
        # Avoids memory copy if numpy array is already contiguous and correct dtype
        # This is faster and reduces memory usage
        obs_np = observation.astype(np.float32) if observation.dtype != np.float32 else observation
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits = self.model(obs_tensor)
        # Sort actions by descending logit value (best first)
        sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
        # Find the first valid action
        action = None
        for candidate_action in sorted_actions:
            if self.game.is_valid_action(candidate_action, observation):
                action = candidate_action
                break
        
        if action is None:
            # Fallback: if no valid action found, pick top choice
            action = torch.argmax(action_logits, dim=1).item()

        return self.game.split_action(action, observation)
