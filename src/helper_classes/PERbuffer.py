import numpy as np
import torch


#in this PER buffer , we will implement a Prioritized Experience Replay (PER) buffer without using a SumTree
# as to test and measure if it will take a long time as what people usually think of when not using a SumTree 
#we using a simple numpy array to store priorities and sample from it using numpy functions

class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, device="cpu"):
        """
        
        Args --> 
            capacity --> max number of transitions
            alpha    --> how much prioritization is used (0 = uniform replay)
            beta     --> importance-sampling correction factor
            device   -->device for returning batches , for us we using cuda 
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
        Add one experience to the buffer , acounting fot priorities , and overwriting old data if full

        returns --> Nothing
        """

        pass
    # ----------------------------------------------------------------------
    def sample(self, batch_size):
        """
        Sample a batch according to priorities 
  
        return -->  states, actions, rewards, next_states, dones, indices, weights
        """

        pass
        


    # ----------------------------------------------------------------------
    def update_priorities(self, indices, new_priorities):
        """
        Update the priorities of sampled transitions.

        Usually: new_priority = TD-error + epsilon

        Args:
            indices        : transitions that were sampled in the batch
            new_priorities : TD-error or loss per transition
            
            returns--> Nothing
        """
        pass

    # ----------------------------------------------------------------------
    def __len__(self):
        return self.size
