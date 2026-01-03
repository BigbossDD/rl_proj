import torch
import torch.nn.functional as F
from torch.distributions import Categorical

 
class ActorCritic:
    """
    Wraps the PPO neural network and provides:
      - forward pass
      - action sampling
      - evaluating old actions (for PPO loss)
    """

    def __init__(self, model, device="cpu"):
        """
        Args:
            model: Your neural network with outputs:
                    actor_logits, value
            device (str): "cpu" or "cuda"
        """
        self.model = model.to(device)
        self.device = device

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): [batch, *state_shape]

        Returns:
            actor_logits (torch.Tensor)
            value (torch.Tensor)
        """
        state = state.to(self.device)
        actor_logits, value = self.model(state)
        return actor_logits, value

    def act(self, state):
        """
        Select an action for interacting with the environment.

        Args:
            state (torch.Tensor): single state

        Returns:
            action (int)
            log_prob (float)
            value (float)
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.squeeze(0).item()
        )

    def evaluate(self, states, actions):
        """
        Evaluate states & actions (used during PPO update).

        Args:
            states (torch.Tensor)
            actions (torch.Tensor)

        returns -->
            log_probs (torch.Tensor)
            entropy (torch.Tensor)
            values (torch.Tensor)
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        logits, values = self.forward(states)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze(-1)
