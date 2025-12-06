class RolloutBuffer:
    """
    Stores a full rollout (trajectory) of experience for PPO.

    After rollout is collected, compute:
      - advantages (using GAE)
      - returns (discounted rewards)
    """

    def __init__(self, buffer_size, state_shape, device="cpu"):
        """
        Args:
            buffer_size (int): Max steps in a rollout (T).
            state_shape (tuple): Shape of observations.
            device (str): "cpu" or "cuda".
        """
        pass

    def add(self, state, action, reward, done, log_prob, value):
        """
        Store a single transition.

        Args:
            state (np.ndarray / torch.Tensor)
            action (int or np.ndarray)
            reward (float)
            done (bool)
            log_prob (float): log π(a|s) of old policy
            value (float): V(s) from old policy
        
        returns -->
            None
        """
        pass

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value (float): V(s_T) from the critic
            gamma (float): discount factor
            lam (float): GAE λ parameter

        returns -->
            None (fills self.advantages and self.returns)
        """
        pass

    def get(self, batch_size):
        """
        Yields shuffled mini-batches for PPO training.

        Args:
            batch_size (int)

        returns -->
            Iterator of batches:
                states, actions, log_probs, values, advantages, returns
        """
        pass

    def clear(self):
        """
        Reset the buffer after PPO update.
        """
        pass
