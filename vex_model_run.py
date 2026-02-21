import torch
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame
from typing import Dict

class VexModelRunner:
    def __init__(self, model_path: str, game: VexGame):
        self.model_path: str = model_path
        self.game: VexGame = game
        self.robot = game.robots[0]  # Use first (only) robot
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

    def get_prediction(self, observation: np.ndarray) -> int:
        """
        Runs inference on the model to get the predicted action for the given observation.
        
        Uses stochastic sampling (Categorical distribution) to match training behavior.
        During training, RLlib samples from the action distribution rather than
        taking the argmax, so we replicate that here.
        
        Handles both communication-enabled and communication-disabled models.

        Args:
            observation: The current observation for the agent.
        """
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        # Zero-Copy Tensor Creation: Use from_numpy() instead of tensor()
        # Avoids memory copy if numpy array is already contiguous and correct dtype
        # This is faster and reduces memory usage
        obs_np = observation.astype(np.float32) if observation.dtype != np.float32 else observation
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_output = self.model(obs_tensor)
        
        # Handle both Discrete (no communication) and Tuple (with communication) outputs
        # If communication is enabled, output is [ActionLogits, MessageMean, MessageLogStd]
        # If disabled, output is just ActionLogits
        # The number of action logits determines the action dim
        action_dim = self.game.action_space(self.game.possible_agents[0]).n if hasattr(self.game.action_space(self.game.possible_agents[0]), 'n') else model_output.shape[1]
        action_logits = model_output[:, :action_dim]
        
        # Use stochastic sampling like RLlib does during training
        # The model outputs logits; convert to probabilities and sample
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Sample action and retry if invalid (up to 20 attempts)
        action = None
        for _ in range(20):
            sampled_action = dist.sample().item()
            if self.game.is_valid_action(sampled_action, observation):
                action = sampled_action
                break
        
        # Fallback: if no valid action found via sampling, use highest valid logit
        if action is None:
            sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
            for candidate_action in sorted_actions:
                if self.game.is_valid_action(candidate_action, observation):
                    action = candidate_action
                    break
            if action is None:
                action = self.game.fallback_action()
        
        return action

    def get_inference(self, observation: np.ndarray):
        """Get action for the robot based on current observation.
        
        Returns:
            Tuple of (high_level_action: int, split_actions: List[str])
        """

        observation = self.game.update_observation_from_tracker(
            agent=self.robot.name,
            observation=observation
        )

        # Get action using current observation
        # Observation uses tracker fields (held_blocks, loaders_taken, goals_added)
        action = self.get_prediction(observation)
        
        # Get split actions
        split_actions = self.game.split_action(action, observation, self.robot)
        
        return action, split_actions  # Returns both action ID and low-level commands

    def run_action(self, action):
        """Called after robot successfully completes an action.
        
        Updates tracker only (no full simulation needed for inference).
        """
        # Update tracker fields based on completed action (uses game.state)
        self.game.update_tracker(agent=self.robot.name, action=action)


"""
Flow in USB script will be something like this:
while True:
    wait for usb message
    if message is action done:
        runner.run_action(action) assuming action was successful
    if message is ready for next action: (can happen immediately after action done)
        runner.get_inference(data)
        send action back over usb
    else:
        update data (position, blocks, etc)

"""
