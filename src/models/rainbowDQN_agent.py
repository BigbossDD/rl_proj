import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# using -->  Dueling + Noisy + Categorical DQN + PER + N-step

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()
##########################################
    

####################################################################################
# -------------------------
# Categorical DQN Network
# -------------------------
class RainbowNetwork(nn.Module):
    def __init__(self, action_size, atom_size=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_size = action_size
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(v_min, v_max, atom_size)

        # CNN torso (Atari standard)
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_hidden = NoisyLinear(3136, 512)
        self.fc_value = NoisyLinear(512, atom_size)
        self.fc_advantage = NoisyLinear(512, action_size * atom_size)


####################################################################################
# -------------------------
# Prioritized Replay Buffer
# -------------------------
class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
##########################################
    

################################################################################################
# -------------------------
# Rainbow DQN Agent
# -------------------------
class RainbowDQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size

        self.v_min = -10
        self.v_max = 10
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)

        self.gamma = 0.99
        self.lr = 1e-4

        self.n_step = 3
        self.nstep_buffer = deque(maxlen=self.n_step)

        self.replay = PERBuffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RainbowNetwork(action_size).to(self.device)
        self.target = RainbowNetwork(action_size).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
##########################################
    
