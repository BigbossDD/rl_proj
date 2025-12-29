import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# implementing a Noisy Linear layer with factorized Gaussian noise. 

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        '''
        Initialization
        Arguments -->
         in_features:  input features
         out_features: output features
         std_init: initial standard deviation for noise parameters
        '''
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Noise buffers (factorized)
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initializes learnable parameters  
        mean (mu) and standard deviation (sigma)
        """
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def scale_noise(self, size):
        """
        Factorized Gaussian Noise generation method
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """
        Samples fresh noise using factorized Gaussian noise
        """
        eps_in = self.scale_noise(self.in_features)
        eps_out = self.scale_noise(self.out_features)

        # Outer product for weight noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, input):
        """
        Forward pass through the Noisy Linear layer.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)
