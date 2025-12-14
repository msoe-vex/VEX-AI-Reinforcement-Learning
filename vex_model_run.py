import torch
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame, Robot
from typing import Dict

class VexModelRunner:
    def __init__(self, model_path: str, game: VexGame, robot: Robot):
        self.model_path: str = model_path
        self.game: VexGame = game
        self.robot: Robot = robot
        self.model: torch.jit.ScriptModule = None
        self.game_state: Dict = game.get_initial_state()
        self.observation: np.ndarray = game.get_observation(robot.name, self.game_state)
        
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
        sorted_actions = torch.argsort(action_logits, dim=1, descending=True).squeeze(0).tolist()
        # Find the first valid action
        action = None
        for candidate_action in sorted_actions:
            if self.game.is_valid_action(candidate_action, observation):
                action = candidate_action
                break
        if action is None:
            action = self.game.fallback_action()
        return action

    def get_inference(self, data):
        # Get data from robot brain and camera, TODO
        # This will probably be constantly updated in a separate thread
        # Will be passed in as 'data' argument for now

        # Update robot positions

        # Update block positions

        # That's probably it for the game state update
        # everything else can be implied through simulated action execution

        # Get action
        action = self.get_prediction(self.observation)

        # Get split actions
        split_actions = self.game.split_action(action, self.observation, self.robot)

        return split_actions # this will be sent to the robot

    def run_action(self, action):
        """
        Runs the inference loop until the game ends.
        """

        # This will run after a response from the robot is received
        # Assuming action execution is a success
        self.game.execute_action(agent=self.robot.name, action=action, state=self.game_state)
        
        # Update observation
        self.observation = self.game.get_observation(self.robot.name, self.game_state)


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
