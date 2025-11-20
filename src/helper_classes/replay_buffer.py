import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, state_shape, device="cpu"):
        '''
        Buffer initialization
        
        '''
        self.capacity = capacity
        self.device = device
        # the buffer storage of experiences(state, action, reward, next_state, done)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.uint8)

        self.ptr = 0
        self.size = 0
######################################################################
    def push(self, state, action, reward, next_state, done):
        """
        
        will add a new exprince  into the buffer.
        
        """
        pass
######################################################################
    def sample(self, batch_size):
        """
        
        Uniformaly sample a batch from the buffer.
        returns: states, actions, rewards, next_states, dones
        
        """
        pass
######################################################################
    def __len__(self):
        
        
        return self.size