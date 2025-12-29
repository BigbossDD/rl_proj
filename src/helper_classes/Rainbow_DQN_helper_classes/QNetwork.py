import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_classes.Rainbow_DQN_helper_classes.noisy_linear import NoisyLinear


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51):
        """
        Args:
            input_shape: (C, H, W) -> (4, 84, 84)
            num_actions: number of actions
            num_atoms: number of atoms for C51
        """
        super(QNetwork, self).__init__()

        self.num_actions = num_actions
        self.num_atoms = num_atoms

        c, h, w = input_shape

        # -----------------------
        # Convolutional Backbone
        # -----------------------
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # -----------------------
        # Value Stream
        # -----------------------
        self.value_fc = NoisyLinear(conv_out_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)

        # -----------------------
        # Advantage Stream
        # -----------------------
        self.advantage_fc = NoisyLinear(conv_out_size, 512)
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms)

    # -------------------------------------------------
    def _get_conv_out(self, shape):
        """
        Compute flattened size after conv layers
        """
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))

    # -------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            log_probs: [B, Actions, Atoms]
        """

        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)                  # [B, Atoms]
        value = value.view(-1, 1, self.num_atoms)

        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)      # [B, A * Atoms]
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)

        # Dueling combination
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Distributional output
        log_probs = F.log_softmax(q_atoms, dim=2)

        return log_probs
