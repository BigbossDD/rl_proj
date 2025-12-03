import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOActorCritic(nn.Module):
    """Shared CNN feature extractor + separate actor & critic heads."""

    def __init__(self, action_size):
        super().__init__()

        # CNN feature extractor (Atari-standard)
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Compute final feature size
        


class PPOAgent:
   def __init__(
        self,
        state_shape,
        action_size,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.1,
        epochs=4,
        batch_size=128,
        steps_per_update=2048
    ):
        
    pass