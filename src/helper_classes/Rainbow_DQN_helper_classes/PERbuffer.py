import numpy as np
import random
from helper_classes.Rainbow_DQN_helper_classes.SumTree import SumTree
import torch


class PERBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Prioritized Experience Replay Buffer.

        Args:
            capacity (int): max number of transitions to store
            alpha (float): prioritization exponent (0 = uniform, 1 = fully prioritized)
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-6  # small amount to avoid zero priority

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer with max priority.

        Args:
            state (np.array)
            action (int)
            reward (float)
            next_state (np.array)
            done (bool)
        """
        max_priority = np.max(self.tree.tree[-self.capacity:])
        if max_priority == 0:
            max_priority = 1.0

        data = (state, action, reward, next_state, done)
        self.tree.add(max_priority, data)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences, weighted by priority.

        Args:
            batch_size (int)
            beta (float): importance-sampling exponent (0=no correction, 1=full correction)

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
            where:
                states, next_states are torch.FloatTensors (B, *state_shape)
                actions, rewards, dones are torch tensors
                indices are tree indices for priority update
                weights are importance-sampling weights
        """
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = (self.tree.size * sampling_probabilities) ** (-beta)
        weights /= weights.max()

        # Convert to torch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, idxs, errors):
        """
        Update priorities on the tree for sampled indices.

        Args:
            idxs (list): indices in the sum tree to update
            errors (torch.Tensor or np.array): TD errors or loss for those indices
        """
        errors = errors.detach().cpu().numpy()
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
