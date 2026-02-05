# attentional communication (ATOC) ref notes w/ code snippets

- ATOC paper: [https://arxiv.org/pdf/1805.07733](https://arxiv.org/pdf/1805.07733)
- DI-engine overview: [https://opendilab.github.io/DI-engine/12_policies/atoc.html](https://opendilab.github.io/DI-engine/12_policies/atoc.html)

## attention unit (decision gate)
```python
class AttentionUnit(nn.Module):
    """
    Attention unit for ATOC that decides when to communicate and how to integrate messages.
    """

    def __init__(self, hidden_dim: int, broadcast_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.broadcast_dim = broadcast_dim

        # Gate network: decides probability of initiating communication
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Query and key networks for attention mechanism
        self.query_net = nn.Linear(hidden_dim, broadcast_dim)
        self.key_net = nn.Linear(broadcast_dim, broadcast_dim)
        self.value_net = nn.Linear(broadcast_dim, broadcast_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        broadcasts: Optional[torch.Tensor] = None,
        return_gate_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_state: (batch_size, hidden_dim) - agent's internal representation
            broadcasts: (batch_size, n_other_agents, broadcast_dim) - received broadcast vectors
            return_gate_only: if True, only compute communication gate

        Returns:
            gate_prob: (batch_size, 1) - probability of initiating communication
            integrated_broadcast: (batch_size, broadcast_dim) - attention-weighted message integration
            attention_weights: (batch_size, n_other_agents) - attention over received messages
        """
        # Communication gate: should this agent broadcast?
        gate_prob = self.gate_net(hidden_state)

        if return_gate_only or broadcasts is None:
            return gate_prob, None, None

        # Attention over received broadcasts
        batch_size = hidden_state.shape[0]
        n_messages = broadcasts.shape[1] if len(broadcasts.shape) == 3 else 1

        # Query from own hidden state
        query = self.query_net(hidden_state)  # (batch, broadcast_dim)

        # Keys and values from received broadcasts
        if len(broadcasts.shape) == 2:
            broadcasts = broadcasts.unsqueeze(1)  # (batch, 1, broadcast_dim)

        keys = self.key_net(broadcasts)  # (batch, n_messages, broadcast_dim)
        values = self.value_net(broadcasts)  # (batch, n_messages, broadcast_dim)

        # Scaled dot-product attention
        scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, broadcast_dim)
            keys.transpose(1, 2),  # (batch, broadcast_dim, n_messages)
        ) / np.sqrt(
            self.broadcast_dim
        )  # (batch, 1, n_messages)

        attention_weights = F.softmax(scores, dim=-1)  # (batch, 1, n_messages)

        # Weighted sum of values
        integrated_broadcast = torch.bmm(
            attention_weights,  # (batch, 1, n_messages)
            values,  # (batch, n_messages, broadcast_dim)
        ).squeeze(
            1
        )  # (batch, broadcast_dim)

        return gate_prob, integrated_broadcast, attention_weights.squeeze(1)
```

## ATOC agent base class
```python
class ATOCAgent(nn.Module):
    """
    Base class for multi-agent RL with optional ATOC communication.

    This class can be extended to implement MAPPO, MAT, or other MARL algorithms.
    It handles the ATOC communication logic and provides hooks for different
    policy architectures.
    """

    def __init__(
        self,
        obs_dim: int = 5,
        action_dim: int = 2,
        hidden_dim: int = 128,
        broadcast_dim: int = 64,
        broadcast_idx: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.broadcast_dim = broadcast_dim
        self.extended_obs_dim = obs_dim + broadcast_dim
        self.hidden_dim = hidden_dim
        self.broadcast_dim = broadcast_dim
        self.broadcast_action = broadcast_idx

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ATOC attention unit (optional)
        self.attention_unit = AttentionUnit(hidden_dim, broadcast_dim)
        # Broadcast generator: creates message content from hidden state
        self.broadcast_generator = nn.Sequential(
            nn.Linear(hidden_dim, broadcast_dim), nn.Tanh()
        )
        # Fusion layer: combines own hidden state with integrated messages
        self.fusion_layer = nn.Linear(hidden_dim + broadcast_dim, hidden_dim)

        # Action head (to be used by subclasses)
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head (for actor-critic methods)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def decide_communication(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to communicate using ATOC gate.

        Args:
            hidden: (batch_size, hidden_dim)
            deterministic: if True, use gate_prob > 0.5 instead of sampling

        Returns:
            should_communicate: (batch_size,) - binary decision
            gate_probs: (batch_size,) - communication probability
        """

        gate_probs, _, _ = self.attention_unit(hidden, return_gate_only=True)
        gate_probs = gate_probs.squeeze(-1)

        if deterministic:  # use thresholding (used for eval)
            should_communicate = (gate_probs > 0.5).float()
        else:  # sample from Bernoulli distribution (used for training)
            should_communicate = torch.bernoulli(gate_probs)

        return should_communicate, gate_probs

    def integrate_communication(
        self,
        hidden: torch.Tensor,
        broadcasts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate received broadcasts using attention mechanism.

        Args:
            hidden: (batch_size, hidden_dim) - agent's hidden state
            broadcasts: (batch_size, n_messages, broadcast_dim) - received broadcasts
            valid_mask: (batch_size, n_messages) - mask for valid messages

        Returns:
            fused_hidden: (batch_size, hidden_dim) - hidden state after communication
            attention_weights: (batch_size, n_messages) - attention over messages
        """
        if broadcasts is None:
            return hidden, torch.zeros_like(broadcasts[:, :, 0])

        # Apply mask to broadcasts if provided (zero out invalid messages)
        if valid_mask is not None:
            broadcasts = broadcasts * valid_mask.unsqueeze(-1)

        _, integrated_broadcast, attention_weights = self.attention_unit(
            hidden, broadcasts
        )

        # Fuse own hidden state with integrated communication
        combined = torch.cat([hidden, integrated_broadcast], dim=-1)
        fused_hidden = self.fusion_layer(combined)

        return fused_hidden, attention_weights

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Per-agent forward pass. Designed for RLlib with independent agent calls.

        Args:
            obs: (batch_size, extended_obs_dim) where extended_obs_dim = obs_dim + (max_other_agents * broadcast_dim)
            deterministic: if True, use deterministic actions

        Returns:
            Dictionary containing:
                - action_logits: (batch_size, action_dim)
                - values: (batch_size,)
                - broadcast: (batch_size, broadcast_dim) - generated broadcast for broadcasting
                - comm_decision: (batch_size,) - whether to communicate (binary 0/1)
                - comm_prob: (batch_size,) - probability of communication
                - attention_weights: (batch_size, n_other_agents) - if using ATOC
        """
        # Ensure obs is 2D: (batch, obs_dim)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)  # (1, extended_obs_dim)

        # Split observation into base_obs and broadcasts along the feature dimension
        base_obs = obs[:, : self.obs_dim]  # (batch, obs_dim)
        broadcast_flat = obs[:, self.obs_dim :]
        batch_size = base_obs.shape[0]

        # Reshape flattened broadcasts into (batch, n_messages, broadcast_dim)
        broadcast = broadcast_flat
        if broadcast_flat.numel() > 0:
            broadcast = broadcast_flat.reshape(batch_size, -1, self.broadcast_dim)

        # Encode observation
        hidden = self.obs_encoder(base_obs)  # (batch, hidden_dim)

        # Generate broadcast message
        agent_broadcast = self.broadcast_generator(hidden)

        # Integrate received broadcasts
        attention_weights = torch.zeros(batch_size, 1, device=hidden.device)
        fused_hidden, attention_weights = self.integrate_communication(
            hidden, broadcast
        )
        hidden = fused_hidden

        # Decide whether to communicate
        comm_decision, comm_prob = self.decide_communication(hidden, deterministic)
        # Ensure these are 1D: (batch,)
        if len(comm_decision.shape) > 1:
            comm_decision = comm_decision.squeeze(-1)
        if len(comm_prob.shape) > 1:
            comm_prob = comm_prob.squeeze(-1)

        # Generate action logits
        action_logits = self.action_net(hidden)  # (batch, action_dim)

        # Generate value estimate
        values = self.value_net(hidden)  # (batch, 1)
        values = values.squeeze(-1)  # (batch,)

        return {
            "action_logits": action_logits,
            "values": values,
            "broadcast": agent_broadcast,
            "comm_decision": comm_decision,
            "comm_prob": comm_prob,
            "attention_weights": attention_weights,
        }

    def get_action_distribution(self, action_logits: torch.Tensor):
        """Get categorical distribution over actions."""
        return torch.distributions.Categorical(logits=action_logits)

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions with broadcasts embedded in obs.

        Args:
            obs: (batch_size, extended_obs_dim)
            deterministic: if True, use deterministic actions
        Returns:
        Dictionary containing:
            - actions: (batch_size,) - sampled actions
            - log_probs: (batch_size,) - log probabilities of sampled actions
            - values: (batch_size,) - value estimates
            - comm_decisions: (batch_size,) - binary communication decisions
            - comm_probs: (batch_size,) - probabilities of communication
            - broadcast: (batch_size, broadcast_dim) - broadcast message
        """
        output = self.forward(obs, deterministic=deterministic)
        action_logits = output["action_logits"].clone()

        # Adjust logits for broadcast action based on communication probability
        action_logits[..., self.broadcast_action] = action_logits[
            ..., self.broadcast_action
        ] + torch.log(output["comm_prob"] + 1e-8)

        action_dist = self.get_action_distribution(action_logits)

        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_probs = action_dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_probs,
            "value": output["values"],
            "comm_decision": output["comm_decision"],
            "comm_prob": output["comm_prob"],
            "broadcast": output["broadcast"],
        }
```

## broadcast message with delay
```python

class Broadcast:

    def __init__(
        self,
        message: Optional[np.ndarray] = None,
        delay: int = 0,
        broadcast_dim: int = 64,
    ):
        self.message_len = broadcast_dim
        self.message = (
            np.zeros(self.message_len, dtype=np.float32)
            if message is None
            else np.asarray(message, dtype=np.float32).reshape(-1)[: self.message_len]
        )
        if self.message.shape[0] < self.message_len:
            self.message = np.pad(
                self.message,
                (0, self.message_len - self.message.shape[0]),
                mode="constant",
            )
        self.delay = delay
        self.age = 0
        self.limit = delay * 2

    def get_message(self) -> np.ndarray:
        # message len is de
        return self.message if self.age >= self.delay else np.zeros(self.message_len)

    def update(self, message, delay: int = 0):
        self.age += 1
        if (message is not None) or (self.age >= self.limit):
            msg = np.asarray(message, dtype=np.float32).reshape(-1)
            if msg.shape[0] < self.message_len:
                msg = np.pad(msg, (0, self.message_len - msg.shape[0]), mode="constant")
            elif msg.shape[0] > self.message_len:
                msg = msg[: self.message_len]
            self.message = msg
            self.delay = delay
            self.age = 0
            self.limit = delay * 2
```

## broadcast buffer
```python
class BroadcastBuffer:
    def __init__(self, delay: int = 0, broadcast_dim: int = 64):
        self.delay = delay
        self.broadcast_dim = broadcast_dim
        self.buffer: dict[str, Broadcast] = {}

    def store_broadcast(self, message, agent_id: str, delay: int = 0):
        delay = max(delay, self.delay)
        self.buffer[agent_id] = Broadcast(
            message, delay, broadcast_dim=self.broadcast_dim
        )
        self.buffer[agent_id].update(message, delay)

    def get_broadcasts(self, agent_id: str) -> List[np.ndarray]:
        # Return messages from all *other* agents in deterministic key order
        return [
            self.buffer[k].get_message()
            for k in sorted(self.buffer.keys())
            if k != agent_id
        ]
```