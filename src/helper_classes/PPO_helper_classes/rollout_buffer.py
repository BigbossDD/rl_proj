import numpy as np
import torch
from helper_classes.PPO_helper_classes.GAE import GAE
from helper_classes.PPO_helper_classes.MiniBatch_generator import minibatch_generator


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
        self.buffer_size = buffer_size
        self.device = device

        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)

        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)

        self.ptr = 0

    # ------------------------------------------------------------------
    def add(self, state, action, reward, done, log_prob, value):
        """
        Store a single transition.
        """
        assert self.ptr < self.buffer_size, "RolloutBuffer overflow"

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """
        Compute GAE advantages and discounted returns.
        """
        values = np.append(self.values[:self.ptr], last_value)
        rewards = self.rewards[:self.ptr]
        dones = self.dones[:self.ptr]

        advantages, returns = GAE.compute(
            values=values[:-1],
            rewards=rewards,
            dones=dones,
            last_value=last_value,
            gamma=gamma,
            lam=lam
        )

        self.advantages[:self.ptr] = advantages
        self.returns[:self.ptr] = returns

        # Normalize advantages (PPO best practice)
        adv_mean = self.advantages[:self.ptr].mean()
        adv_std = self.advantages[:self.ptr].std() + 1e-8
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / adv_std

    # ------------------------------------------------------------------
    def get(self, batch_size):
        """
        Yields shuffled mini-batches for PPO training.
        """
        states = torch.tensor(self.states[:self.ptr], device=self.device)
        actions = torch.tensor(self.actions[:self.ptr], device=self.device)
        log_probs = torch.tensor(self.log_probs[:self.ptr], device=self.device)
        values = torch.tensor(self.values[:self.ptr], device=self.device)
        advantages = torch.tensor(self.advantages[:self.ptr], device=self.device)
        returns = torch.tensor(self.returns[:self.ptr], device=self.device)

        for batch in minibatch_generator(
            batch_size=self.ptr,
            mini_batch_size=batch_size,
            states=states,
            actions=actions,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns
        ):
            yield batch

    # ------------------------------------------------------------------
    def clear(self):
        """
        Reset the buffer after PPO update.
        """
        self.ptr = 0
