# SumTree.py (fixed)
import numpy as np
'''
this class is SumTree for Prioritized Experience Replay
'''

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0
    # Add a new priority and data point to the tree
    def add(self, priority, data):
        priority = max(priority, 1e-6)
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    # Update the priority of a given tree index
    def update(self, idx, priority):
        priority = max(priority, 1e-6)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
    # Get the index, priority, and data for a given cumulative priority s
    def get(self, s):
        s = np.clip(s, 0, self.total_priority)
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1

        # Guard: out of range
        if data_idx < 0 or data_idx >= self.capacity:
            raise RuntimeError("SumTree index out of bounds")

        data = self.data[data_idx]

        # Guard: None data means sampling before buffer ready
        if data is None:
            raise RuntimeError(
                "SumTree returned None data — sampling before buffer is ready"
            )

        return idx, self.tree[idx], data
    # Property to get the total priority
    @property
    def total_priority(self):
        return max(self.tree[0], 1e-6)
