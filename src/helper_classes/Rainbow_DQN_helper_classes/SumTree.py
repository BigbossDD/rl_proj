import numpy as np

class SumTree:
    """
    Binary SumTree for Prioritized Experience Replay
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    # -------------------------------------------------
    # Add a new priority + data
    # -------------------------------------------------
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # -------------------------------------------------
    # Update priority
    # -------------------------------------------------
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        # Propagate change up
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    # -------------------------------------------------
    # Get leaf based on cumulative sum
    # -------------------------------------------------
    def get(self, s):
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
        return idx, self.tree[idx], self.data[data_idx]

    # -------------------------------------------------
    # Total priority
    # -------------------------------------------------
    @property
    def total_priority(self):
        return self.tree[0]
