import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noisy_linear import NoisyLinear  

   # This  Q-Network Combines --> 
   # 1. Noisy Linear Layers (Exploration)
   # 2. Dueling Architecture (Value + Advantage)
   # 3. Distributional Output (C51 Atoms)

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51):
        """
        Initialization
        Args:
            input_shape: Shape of the input state for us  (4, 84, 84)
            num_actions: Number of possible actions in the environment
            num_atoms: Number of discrete atoms for C51 distributional outputs
        """
        pass

    def _get_conv_out(self, x):
        """
        calculate the output size of the conv layers.
        Args:
            x: Input tensor to pass through conv layers
       
        returns -->  
        size of the flattened conv output
        
        """
        pass
    def forward(self, x):
        """
        Defines the "forward pass" of the network.
        
        Args:
            x: Input state tensor of shape [Batch, Channels, Height, Width]
        returns --> 
            Log-Probabilities tensor of shape [Batch, Actions, Atoms] 
            
        """
        pass