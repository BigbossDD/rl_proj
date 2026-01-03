import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
    Noisy Linear Layer (Factorized Gaussian Noise)
    
    IMPORTANT(after multiple fixes and iterations):
    - Noise is NOT reset inside forward()
    - Noise is reset EXTERNALLY by the agent
    - No in-place modification during forward pass
"""

class NoisyLinear(nn.Module):
   

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # ------------------------------------------------------------
        # Learnable parameters (mu & sigma)
        # ------------------------------------------------------------
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()  # initial noise

    
    
    def reset_parameters(self):
        """
        Initialize learnable parameters.
        """
        mu_range = 1 / math.sqrt(self.in_features)

      
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))


    def _scale_noise(self, size):
        """
        Factorized Gaussian noise.
        """
        x = torch.randn(size, device=self.weight_mu.device)

    
        return x.sign() * torch.sqrt(torch.abs(x))


    def reset_noise(self):
        """
        Sample new noise.
        
        FIX 3:
        ------
        This method MUST be called:
        - once per training step
        - OUTSIDE forward()
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

     
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


    def forward(self, x):
        """
        Forward pass.
        
        """

        if self.training:
            
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Deterministic evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
