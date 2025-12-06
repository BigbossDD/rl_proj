import numpy as np
import torch


#in this PER buffer , we will implement a Prioritized Experience Replay (PER) buffer without using a SumTree
# as to test and measure if it will take a long time as what people usually think of when not using a SumTree 
#

class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, device="cpu"):
        """
        Prioritized Experience Replay (PER) buffer.
        - Stores transitions: (state, action, reward, next_state, done)
        - Maintains priorities for sampling important transitions more often
        - Does NOT use a SumTree. Uses simple array-based sampling.

        Args:
            capacity : max number of transitions
            alpha    : how much prioritization is used (0 = uniform replay)
            beta     : importance-sampling correction factor
            device   : torch device for returning batches
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

        # Priority storage (initialized to small nonzero value)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Pointers
        self.ptr = 0      # position to write next transition
        self.size = 0     # number of valid transitions stored

        # Small constant to avoid zero priority
        self.epsilon = 1e-6


    # ----------------------------------------------------------------------
    def add(self, state, action, reward, next_state, done):
        """
        Add one experience to the buffer.

        Steps:
        1. Store transition in the ring buffer (overwrites oldest data)
        2. Assign priority = max_priority (new samples should be sampled at least once)
        """

        # Save transition data
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Assign maximum priority to new element
        # ensures new experiences are sampled at least once
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio

        # Move pointer forward
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    # ----------------------------------------------------------------------
    def sample(self, batch_size):
        """
        Sample a batch according to priorities.

        Steps:
        1. Convert priorities → probabilities (p_i = prio^alpha)
        2. Multinomial sample indices according to probabilities
        3. Compute importance-sampling weights
        4. Return tensors for training
        """

        # Compute sampling probabilities
        # Raise priorities to α (controls level of prioritization)
        scaled_prios = self.priorities[:self.size] ** self.alpha
        prob = scaled_prios / scaled_prios.sum()

        # Sample indices according to probability distribution
        indices = np.random.choice(self.size, batch_size, p=prob, replace=False)

        # Importance sampling weights:
        # Corrects the bias introduced by non-uniform sampling
        weights = (self.size * prob[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize for stability

        # Convert everything into torch tensors
        states = torch.tensor(self.states[indices], device=self.device, dtype=torch.float32) / 255.0
        next_states = torch.tensor(self.next_states[indices], device=self.device, dtype=torch.float32) / 255.0
        actions = torch.tensor(self.actions[indices], device=self.device)
        rewards = torch.tensor(self.rewards[indices], device=self.device)
        dones = torch.tensor(self.dones[indices], device=self.device, dtype=torch.float32)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights


    # ----------------------------------------------------------------------
    def update_priorities(self, indices, new_priorities):
        """
        Update the priorities of sampled transitions.

        Usually: new_priority = TD-error + epsilon

        Args:
            indices        : transitions that were sampled in the batch
            new_priorities : TD-error or loss per transition
        """
        new_priorities = new_priorities.squeeze().detach().cpu().numpy()

        # Add small epsilon to avoid zero priorities
        self.priorities[indices] = new_priorities + self.epsilon


    # ----------------------------------------------------------------------
    def __len__(self):
        return self.size
