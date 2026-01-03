import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_classes.Rainbow_DQN_helper_classes.noisy_linear import NoisyLinear  # Uses your separate NoisyLinear file
"""
    The Rainbow Q-Network.
    Combines:
    1. Noisy Linear Layers (Exploration)
    2. Dueling Architecture (Value + Advantage)
    3. Distributional Output (C51 Atoms)
"""



class QNetwork(nn.Module):
   
    # Initialize the Q-Network
    def __init__(self, input_shape, num_actions, num_atoms=51):
        """
        Initializes the network layers.
        Args:
            num_atoms (int): The number of buckets (atoms) for the value distribution.
        """
        super(QNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        
        in_channels = input_shape[0]
    
        
        # Convolutional layers to process image input
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from conv layers to feed into fully connected layers
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)
        
        
        # 1. Value Stream: Estimates V(s) -> Output Shape [Batch, 1, Atoms]
        self.fc_val_hidden = NoisyLinear(conv_out_size, 512)
        self.fc_val_out = NoisyLinear(512, num_atoms)
        
        # 2. Advantage Stream: Estimates A(s, a) -> Output Shape [Batch, Actions, Atoms]
        self.fc_adv_hidden = NoisyLinear(conv_out_size, 512)
        self.fc_adv_out = NoisyLinear(512, num_actions * num_atoms)
        
    def _get_conv_out(self, x):
        """Helper function to calculate the output size of the conv layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the tensor, except for the batch dimension, to get the total size
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        """
        Defines the "forward pass" of the network.
        Returns Log-Probabilities for C51.
        """
        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        x = x / 255.0
        
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output from the conv layers into a 1D vector
        x = x.view(x.size(0), -1)
        
        # ### START CHANGE: Dueling + C51 Aggregation ###
        # 1. Process Value Stream
        val = F.relu(self.fc_val_hidden(x))
        val_logits = self.fc_val_out(val)
        # Reshape to [Batch, 1, Atoms] for broadcasting addition
        val_logits = val_logits.view(-1, 1, self.num_atoms)
        
        # 2. Process Advantage Stream
        adv = F.relu(self.fc_adv_hidden(x))
        adv_logits = self.fc_adv_out(adv)
        # Reshape to [Batch, Actions, Atoms]
        adv_logits = adv_logits.view(-1, self.num_actions, self.num_atoms)
        
        # 3. Aggregation: Q_logits(s,a) = V_logits(s) + (A_logits(s,a) - mean(A_logits(s,a)))
        # We calculate mean across the Action dimension (dim=1)
        adv_mean = adv_logits.mean(dim=1, keepdim=True)
        q_logits = val_logits + (adv_logits - adv_mean)
        
        # 4. Return Log Softmax (Distribution)
        # We apply log_softmax across the Atoms dimension (dim=2)
        return F.log_softmax(q_logits, dim=2)
      