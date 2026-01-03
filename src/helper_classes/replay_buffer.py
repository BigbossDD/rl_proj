import random
import numpy as np
from collections import deque
'''
A simple Replay Buffer implementation for storing and sampling experiences.
used mostly in DQN  
'''

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
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
            
        
######################################################################
    def sample(self, batch_size):
        """
        
        Uniformaly sample a batch from the buffer.
        returns: states, actions, rewards, next_states, dones
        
        """
        idxs = np.random.choice(self.size, batch_size, replace=False)
        batch = {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs]
        }   
        return batch
        
######################################################################
    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        
        return self.size