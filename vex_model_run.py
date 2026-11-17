import torch
import numpy as np

# Import from new modular architecture
from vex_core.base_game import VexGame
from vex_core.config import CommunicationOption
from vex_core.base_env import MESSAGE_SIZE, OPTION_TERMINATION_THRESHOLD
from typing import Optional

class VexModelRunner:
    def __init__(self, model_path: str, game: VexGame, temperature: float = 1.0, agent_name: str = None, model: torch.nn.Module = None):
        self.model_path = model_path
        self.game = game
        if hasattr(self.game, "deterministic"):
            self.game.deterministic = True
            
        self.agent_name = agent_name if agent_name else game.possible_agents[0]
        self.robot = game.get_robot_for_agent(self.agent_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.temperature = max(1e-6, float(temperature))
        self._base_obs_dim = 0
        self._expected_obs_dim = 0
        self._num_actions = int(self.game.num_actions)
        self._comm_mode_name = "none"

        self.model = model

        # Normalize communication mode to a lowercase string so this runner
        # remains stable even if CommunicationOption enums come from different modules.
        comm_mode = getattr(self.game, "communication_mode", CommunicationOption.NONE)
        self._comm_mode_name = str(getattr(comm_mode, "value", comm_mode)).lower()

        # Derive expected observation size directly from game space + comm mode.
        # This avoids fragile dependence on enum identity when constructing dummy envs.
        game_obs_space = self.game.get_game_observation_space(self.agent_name)
        if hasattr(game_obs_space, "shape") and len(game_obs_space.shape) > 0:
            self._base_obs_dim = int(game_obs_space.shape[0])
        else:
            self._base_obs_dim = 81

        if self._comm_mode_name == "attention":
            self._expected_obs_dim = self._base_obs_dim + MESSAGE_SIZE
        elif self._comm_mode_name == "copy":
            my_team = self.game.get_team_for_agent(self.agent_name)
            num_teammates = sum(
                1
                for a in self.game.possible_agents
                if self.game.get_team_for_agent(a) == my_team and a != self.agent_name
            )
            self._expected_obs_dim = self._base_obs_dim + (self._base_obs_dim * num_teammates)
        else:
            self._expected_obs_dim = self._base_obs_dim

        # Load the exported model
        if self.model is None and self.model_path:
            try:
                if self.model_path.endswith(".pt"):
                    raise RuntimeError(
                        f"{self.model_path} is a legacy TorchScript artifact. "
                        "Re-export the checkpoint with vex_model_compile.py to "
                        "create a .pt2 model compatible with torch.export."
                    )
                self.model = torch.export.load(self.model_path).module().to(self.device)
                print(f"Successfully loaded exported model from {self.model_path}")
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load exported model from {self.model_path}: {exc}"
                ) from exc

        if self.model is not None:
            try:
                obs_shape = (self._expected_obs_dim,)
                print(f"Warmup observation shape: {obs_shape}, communication_mode: {self._comm_mode_name}")
                dummy_input = torch.randn(1, *obs_shape, device=self.device)
                # Determine action mask dim
                action_space = self.game.get_game_action_space(self.agent_name)
                if hasattr(action_space, 'spaces') and len(action_space.spaces) > 0 and hasattr(action_space.spaces[0], 'n'):
                    mask_dims = [action_space.spaces[0].n]
                elif hasattr(action_space, 'n'):
                    mask_dims = [action_space.n]
                else:
                    mask_dims = [obs_shape[0]]

                # Older exported checkpoints can have a mismatched action-mask width.
                # Try the current env size first, then smaller legacy sizes until one works.
                if self.game.num_actions > 0:
                    mask_dims.append(int(self.game.num_actions))
                for dim in range(1, max(mask_dims) + 1):
                    if dim not in mask_dims:
                        mask_dims.append(dim)

                warmup_ok = False
                for mask_dim in mask_dims:
                    dummy_mask = torch.ones(1, mask_dim, device=self.device)
                    try:
                        with torch.no_grad():
                            _ = self.model(dummy_input, dummy_mask)
                        warmup_ok = True
                        break
                    except (RuntimeError, AssertionError) as exc:
                        message = str(exc)
                        action_mask_mismatch = (
                            "size of tensor a" in message
                            and "non-singleton dimension 1" in message
                        ) or (
                            "action_mask.size()" in message and "==" in message
                        )
                        if not action_mask_mismatch:
                            raise
                if warmup_ok:
                    print(f"Model warmed up and ready for inference for {self.agent_name}")
                else:
                    raise RuntimeError(f"Unable to warm up model for {self.agent_name} with any compatible action-mask width")
            except Exception as exc:
                raise RuntimeError(
                    f"Model for agent {self.agent_name} is incompatible with the "
                    f"{self._comm_mode_name} environment observation "
                    f"(expected {self._expected_obs_dim} features)."
                ) from exc

    def _prepare_model_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation length for exported model input."""
        obs = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        current_dim = int(obs.shape[0])

        if current_dim == self._expected_obs_dim:
            return obs

        if current_dim > self._expected_obs_dim:
            return obs[:self._expected_obs_dim]

        padded = np.zeros(self._expected_obs_dim, dtype=np.float32)
        padded[:current_dim] = obs
        return padded

    def get_prediction(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray = None,
        active_action: Optional[int] = None,
    ):
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
            Tuple of (action, message_vector, termination_probability)
        """
        # Zero-Copy Tensor Creation: Use from_numpy() instead of tensor()
        # Avoids memory copy if numpy array is already contiguous and correct dtype
        # This is faster and reduces memory usage
        obs_np = self._prepare_model_observation(observation)
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

        action_mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        mask_candidates = []
        env_action_dim = int(getattr(self.game, "num_actions", action_mask_np.size))
        for dim in [action_mask_np.size, env_action_dim]:
            if dim is not None and dim > 0:
                mask_candidates.append(int(dim))
        max_dim = max([action_mask_np.size, env_action_dim, 1])
        for dim in range(1, max_dim + 1):
            if dim not in mask_candidates:
                mask_candidates.append(dim)

        resolved_mask = None
        resolved_model_output = None
        for candidate_dim in mask_candidates:
            if candidate_dim <= 0:
                continue
            candidate_mask = action_mask_np[:candidate_dim]
            if candidate_dim > action_mask_np.size:
                candidate_mask = np.pad(candidate_mask, (0, candidate_dim - action_mask_np.size), mode='constant', constant_values=0.0)
            mask_tensor = torch.from_numpy(candidate_mask.astype(np.float32)).unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    resolved_model_output = self.model(obs_tensor, mask_tensor)
                resolved_mask = candidate_mask
                break
            except (RuntimeError, AssertionError) as exc:
                msg = str(exc)
                action_mask_mismatch = (
                    "size of tensor a" in msg
                    and "non-singleton dimension 1" in msg
                ) or (
                    "action_mask.size()" in msg and "==" in msg
                )
                if not action_mask_mismatch:
                    raise
                continue

        if resolved_model_output is None:
            raise RuntimeError(f"Unable to find an action-mask width compatible with model for agent {self.agent_name}")

        model_output = resolved_model_output
        action_mask_np = resolved_mask

        # The game exposes the discrete macro-action space; the exported
        # policy may additionally contain message and termination outputs.
        action_space = self.game.get_game_action_space(self.agent_name)
        output_dim = int(model_output.shape[1])
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
        elif hasattr(action_space, 'spaces') and len(action_space.spaces) > 0 and hasattr(action_space.spaces[0], 'n'):
            action_dim = action_space.spaces[0].n
        else:
            action_dim = output_dim

        # Some legacy checkpoints export the action logits without the full
        # mask width or with the message tail concatenated, so prefer the model's
        # actual output width when it is smaller than the env's nominal action count.
        message_output_dim = 2 * MESSAGE_SIZE if self._comm_mode_name == "attention" else 0
        has_termination_head = (
            output_dim >= action_dim + message_output_dim + self._num_actions
        )
        if not has_termination_head:
            if self._comm_mode_name == "attention" and output_dim >= 2 * MESSAGE_SIZE:
                action_dim = max(1, output_dim - 2 * MESSAGE_SIZE)
            elif output_dim > 0 and output_dim <= max(1, action_dim):
                action_dim = output_dim

        action_logits = model_output[:, :action_dim]

        message_vector = None
        if self._comm_mode_name == "attention" and output_dim >= action_dim + MESSAGE_SIZE:
            message_mean = model_output[:, action_dim:action_dim + MESSAGE_SIZE]
            if output_dim >= action_dim + 2 * MESSAGE_SIZE:
                msg_log_std = model_output[:, action_dim + MESSAGE_SIZE:action_dim + 2 * MESSAGE_SIZE]
                msg_std = torch.exp(msg_log_std) * getattr(self, "temperature", 1.0)
                msg_dist = torch.distributions.Normal(message_mean, msg_std)
                message_vector = msg_dist.sample().cpu().numpy()[0]
            else:
                message_vector = message_mean.cpu().numpy()[0]

        # Use temperature-scaled sampling like RLlib does during training
        scaled_logits = action_logits / max(1e-6, getattr(self, "temperature", 1.0))
        probs = torch.softmax(scaled_logits, dim=-1)
        action_probs = probs.squeeze(0).cpu().numpy().copy()

        prob_sum = action_probs.sum()
        if prob_sum > 0:
            action_probs = action_probs / prob_sum
            action = np.random.choice(action_dim, p=action_probs)
        else:
            action = self.game.fallback_action

        if not self.game.is_valid_action(self.agent_name, action, observation):
            action = self.game.fallback_action

        if not has_termination_head:
            raise RuntimeError(
                "The loaded model has no Option-Critic termination head. "
                "Train and export a checkpoint with the current model code."
            )
        termination_probability = 0.0
        if active_action is not None:
            active_action = int(active_action)
            start = action_dim + message_output_dim
            probabilities = model_output[0, start:start + self._num_actions]
            if active_action < 0 or active_action >= probabilities.numel():
                raise ValueError(f"active_action {active_action} is outside the policy action range")
            termination_probability = float(probabilities[active_action].item())

        return int(action), message_vector, termination_probability

    def evaluate_termination(
        self,
        observation: np.ndarray,
        active_action: int,
        action_mask: np.ndarray = None,
    ) -> float:
        """Return beta(s, active_action), the probability of terminating the option."""
        active_action = int(active_action)
        if active_action < 0 or active_action >= self._num_actions:
            raise ValueError(f"active_action {active_action} is outside the policy action range")

        obs_np = self._prepare_model_observation(observation)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        if action_mask is None:
            mask_np = np.ones(self._num_actions, dtype=np.float32)
        else:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            if mask_np.size != self._num_actions:
                raise ValueError(
                    f"action_mask must have {self._num_actions} entries, got {mask_np.size}"
                )
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(obs_tensor, mask_tensor)
        message_output_dim = 2 * MESSAGE_SIZE if self._comm_mode_name == "attention" else 0
        start = self._num_actions + message_output_dim
        end = start + self._num_actions
        if output.shape[-1] < end:
            raise RuntimeError(
                "The loaded model has no compatible Option-Critic termination head. "
                "Train and export a checkpoint with the current model code."
            )
        return float(output[0, start + active_action].item())

    def get_inference(
        self,
        observation: np.ndarray,
        robot_position: Optional[np.ndarray] = None,
        active_action: Optional[int] = None,
    ):
        """Get action for the robot based on current observation.

        Args:
            observation: Robot-relative policy observation with no absolute x/y.
            robot_position: Optional absolute world position used internally for
                action planning. It is not included in the policy observation.
            active_action: Policy action currently being executed, if any.
        
        Returns:
            Tuple of action, split actions, message, termination probability, and
            whether to preempt the active action.
        """
        if robot_position is not None:
            position = np.asarray(robot_position, dtype=np.float32)
            if position.shape != (2,) or not np.isfinite(position).all():
                raise ValueError("robot_position must contain two finite world coordinates")
            self.game.state["agents"][self.agent_name]["position"] = position.copy()
            self.game.update_robot_position(self.agent_name, position)

        observation = self.game.update_observation_from_tracker(
            agent=self.agent_name,
            observation=observation
        )

        # The observation uses tracker fields (held_blocks, loaders_taken, goals_added).
        action, message_vector, termination_probability = self.get_prediction(
            observation,
            active_action=active_action,
        )
        
        should_stop = (
            active_action is not None
            and action != int(active_action)
            and termination_probability >= OPTION_TERMINATION_THRESHOLD
        )
        split_actions = (
            self.game.split_action(action, observation, self.robot)
            if active_action is None or should_stop
            else []
        )
        return action, split_actions, message_vector, termination_probability, should_stop

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
        action, split_actions, message, beta, should_stop = runner.get_inference(
            data,
            active_action=current_action,
        )
        if current_action is None:
            send action back over usb
        elif should_stop:
            send PREEMPT_ACTION and then split_actions over usb
    else:
        update data (position, blocks, etc)

"""
