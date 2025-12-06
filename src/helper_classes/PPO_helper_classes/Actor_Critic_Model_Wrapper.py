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
        pass

    def forward(self, state):
        """
        Args:
            state (torch.Tensor): [batch, *state_shape]

        Returns:
            actor_logits (torch.Tensor)
            value (torch.Tensor)
        """
        pass

    def act(self, state):
        """
        Select an action for interacting with the environment.

        Args:
            state (torch.Tensor): single state

        Returns:
            action (int or np.ndarray)
            log_prob (float)
            value (float)
        """
        pass

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
        pass
