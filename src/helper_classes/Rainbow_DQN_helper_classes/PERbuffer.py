# PERbuffer.py (sample fixed with retry loop and size check)
import numpy as np
import random
from helper_classes.Rainbow_DQN_helper_classes.SumTree import SumTree
import torch

class PERBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = 1e-6  # small amount to avoid zero priority

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        data = (state, action, reward, next_state, done)
        self.tree.add(max_priority, data)

    def sample(self, batch_size, beta=0.4):
        # Reject sampling if buffer not full enough
        if self.tree.size < batch_size:
            return None  # Not enough data yet

        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Try-except in case data is None, retry sampling s
            for _ in range(5):  # try max 5 times per sample
                try:
                    idx, priority, data = self.tree.get(s)
                    if data is None:
                        raise RuntimeError("Got None data")
                    break
                except RuntimeError:
                    s = random.uniform(a, b)
            else:
                # Failed to get valid data after 5 tries, fallback to random data from buffer
                # Choose random idx from valid range
                valid_idx = random.randint(0, self.tree.size - 1)
                idx = valid_idx + self.capacity - 1
                priority = self.tree.tree[idx]
                data = self.tree.data[valid_idx]
                if data is None:
                    return None  # give up on this batch, caller should handle None

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
        errors = errors.detach().cpu().numpy()
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
