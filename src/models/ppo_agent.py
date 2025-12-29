import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO_PolicyNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[1]

        self.fc_actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.fc_critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Ensure input has batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x / 255.0
        features = self.conv(x)
        actor_logits = self.fc_actor(features)
        value = self.fc_critic(features).squeeze(-1)
        return actor_logits, value
