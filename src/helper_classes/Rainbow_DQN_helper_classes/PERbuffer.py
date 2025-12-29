import numpy as np
import torch


class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, device="cpu"):
        """
        Args --> 
            capacity --> max number of transitions
            alpha    --> how much prioritization is used (0 = uniform replay)
            beta     --> importance-sampling correction factor
            device   --> device for returning batches
        """

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device

        # Transition storage
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.uint8)

        # Priority storage
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Pointers
        self.ptr = 0
        self.size = 0

        # Small constant
        self.epsilon = 1e-6

    # ----------------------------------------------------------------------
    def add(self, state, action, reward, next_state, done):
        """
        Add one experience to the buffer
        """

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Assign max priority so new experience is sampled at least once
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_priority

        # Move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ----------------------------------------------------------------------
    def sample(self, batch_size):
        """
        Sample a batch according to priorities
        """

        # Use only valid priorities
        priorities = self.priorities[:self.size]

        # Compute probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Importance-sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize

        # Convert to tensors
        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device) / 255.0
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device) / 255.0
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    # ----------------------------------------------------------------------
    def update_priorities(self, indices, new_priorities):
        """
        Update priorities after learning step
        """

        new_priorities = new_priorities.detach().cpu().numpy()

        self.priorities[indices] = np.abs(new_priorities) + self.epsilon

    # ----------------------------------------------------------------------
    def __len__(self):
        return self.size
