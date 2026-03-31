import torch
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame
from vex_core.config import VexEnvConfig, CommunicationOption
from vex_core.base_env import MESSAGE_SIZE, VexMultiAgentEnv
from typing import Dict

class VexModelRunner:
    def __init__(self, model_path: str, game: VexGame, temperature: float = 1.0, agent_name: str = None, model: torch.jit.ScriptModule = None):
        self.model_path = model_path
        self.game = game
        if hasattr(self.game, "deterministic"):
            self.game.deterministic = True
            
        self.agent_name = agent_name if agent_name else game.possible_agents[0]
        self.robot = game.get_robot_for_agent(self.agent_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.temperature = max(1e-6, float(temperature))

        self.model = model

        # Load the TorchScript model
        if self.model is None and self.model_path:
            try:
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
                
                self.model = torch.jit.optimize_for_inference(self.model)
                
                print(f"Successfully loaded and optimized model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
                
        if self.model is not None:
            try:
                env_config = VexEnvConfig(
                    game_name=game.get_game_name(),
                    communication_mode=game.communication_mode,
                    render_mode='none',
                    experiment_path="",
                    randomize=False,
                    deterministic=False
                )
                dummy_env = VexMultiAgentEnv(game=game, config=env_config)
                
                obs_space = dummy_env.observation_space(self.agent_name)
                if hasattr(obs_space, "spaces") and "observations" in obs_space.spaces:
                    obs_shape = obs_space["observations"].shape
                else:
                    obs_shape = obs_space.shape
                comm_mode = getattr(game, "communication_mode", None)
                comm_mode_value = getattr(comm_mode, "value", str(comm_mode))
                print(f"Warmup observation shape: {obs_shape}, communication_mode: {comm_mode_value}")
                dummy_input = torch.randn(1, *obs_shape, device=self.device)
                # Determine action mask dim
                action_space = game.get_game_action_space(self.agent_name)
                if hasattr(action_space, 'spaces') and len(action_space.spaces) > 0 and hasattr(action_space.spaces[0], 'n'):
                    mask_dim = action_space.spaces[0].n
                elif hasattr(action_space, 'n'):
                    mask_dim = action_space.n
                else:
                    mask_dim = obs_shape[0]
                dummy_mask = torch.ones(1, mask_dim, device=self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input, dummy_mask)
                print(f"Model warmed up and ready for inference for {self.agent_name}")
            except Exception as e:
                print(f"Error warming up model: {e}")

    def get_prediction(self, observation: np.ndarray, action_mask: np.ndarray = None):
        """
        Runs inference on the model to get the predicted action and message vector.
        
        Uses stochastic sampling (Categorical distribution) to match training behavior.
        During training, RLlib samples from the action distribution rather than
        taking the argmax, so we replicate that here.
        
        Handles both communication-enabled and communication-disabled models.

        Args:
            observation: The current observation for the agent.
            action_mask: Optional pre-computed action mask.
            
        Returns:
            Tuple of (action: int, message_vector: np.ndarray or None)
        """
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        # Zero-Copy Tensor Creation: Use from_numpy() instead of tensor()
        # Avoids memory copy if numpy array is already contiguous and correct dtype
        # This is faster and reduces memory usage
        obs_np = observation.astype(np.float32) if observation.dtype != np.float32 else observation
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        # Compute action mask using is_valid_action for all actions if not provided
        if action_mask is None:
            action_space = self.game.get_game_action_space(self.agent_name)
            if hasattr(action_space, 'spaces') and len(action_space.spaces) > 0 and hasattr(action_space.spaces[0], 'n'):
                num_actions = action_space.spaces[0].n
            elif hasattr(action_space, 'n'):
                num_actions = action_space.n
            else:
                num_actions = 14  # fallback
            
            action_mask = np.array(
                [1.0 if self.game.is_valid_action(self.agent_name, i, observation) else 0.0 for i in range(num_actions)],
                dtype=np.float32
            )
            
        mask_tensor = torch.from_numpy(action_mask.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_output = self.model(obs_tensor, mask_tensor)
        
        # Handle both Discrete and Tuple action spaces.
        # Tuple structure is (Discrete(action), Box(message)).
        action_space = self.game.get_game_action_space(self.agent_name)
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
        elif hasattr(action_space, 'spaces') and len(action_space.spaces) > 0 and hasattr(action_space.spaces[0], 'n'):
            action_dim = action_space.spaces[0].n
        else:
            action_dim = model_output.shape[1]
        action_logits = model_output[:, :action_dim]
        
        # Extract message vector (MESSAGE_SIZE dims) if present after action logits
        message_vector = None
        remaining = model_output.shape[1] - action_dim
        if remaining >= MESSAGE_SIZE:
            # Message mean is the first MESSAGE_SIZE values after action logits
            message_mean = model_output[:, action_dim:action_dim + MESSAGE_SIZE]
            
            # If stochastic and log_std is available, sample from Normal distribution
            # Otherwise use mean.
            is_deterministic = hasattr(self.game, 'deterministic') and self.game.deterministic
            if not is_deterministic and remaining >= 2 * MESSAGE_SIZE:
                msg_log_std = model_output[:, action_dim + MESSAGE_SIZE:action_dim + 2 * MESSAGE_SIZE]
                msg_std = torch.exp(msg_log_std) * getattr(self, "temperature", 1.0)
                msg_dist = torch.distributions.Normal(message_mean, msg_std)
                message_vector = msg_dist.sample().cpu().numpy()[0]
            else:
                message_vector = message_mean.cpu().numpy()[0]
        
        # Use temperature-scaled sampling like RLlib does during training
        scaled_logits = action_logits / max(1e-6, getattr(self, "temperature", 1.0))
        probs = torch.softmax(scaled_logits, dim=-1)
        
        is_deterministic = hasattr(self.game, 'deterministic') and self.game.deterministic
        if not is_deterministic:
            action_probs = probs.squeeze(0).cpu().numpy().copy()
            
            prob_sum = action_probs.sum()
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
                action = np.random.choice(action_dim, p=action_probs)
            else:
                action = self.game.fallback_action()
                
            # If the model still somehow chooses an invalid action, fallback gracefully
            if not self.game.is_valid_action(self.agent_name, action, observation):
                action = self.game.fallback_action()
        else:
            sorted_actions = torch.argsort(scaled_logits, dim=1, descending=True).squeeze(0).tolist()
            action = None
            for candidate_action in sorted_actions:
                if self.game.is_valid_action(self.agent_name, candidate_action, observation):
                    action = candidate_action
                    break
            if action is None:
                action = self.game.fallback_action()
        
        return action, message_vector

    def get_inference(self, observation: np.ndarray):
        """Get action for the robot based on current observation.
        
        Returns:
            Tuple of (high_level_action: int, split_actions: List[str], message_vector: np.ndarray or None)
        """

        observation = self.game.update_observation_from_tracker(
            agent=self.agent_name,
            observation=observation
        )

        # Get action and message vector using current observation
        # Observation uses tracker fields (held_blocks, loaders_taken, goals_added)
        action, message_vector = self.get_prediction(observation)
        
        # Get split actions
        split_actions = self.game.split_action(action, observation, self.robot)
        
        return action, split_actions, message_vector

    def run_action(self, action):
        """Called after robot successfully completes an action.
        
        Updates tracker only (no full simulation needed for inference).
        """
        # Update tracker fields based on completed action (uses game.state)
        self.game.update_tracker(agent=self.agent_name, action=action)


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
