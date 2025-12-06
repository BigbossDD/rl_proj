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

        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

       
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initializes learnable parameters  
        mean (bias_mu) and standard deviation (bias_sigma) weights 
        """
        pass

    def scale_noise(self, size):
        """
        Factorized Gaussian Noise generation method 
  
        returns --> 
                A noise vector of given size 
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """
        Samples fresh noise (random vectors) and will use scale_noise method to generate noise for weights and bias.
       
        returns -->  nothing 

            but it 
            Updates the weight_epsilon and bias_epsilon buffers with new noise values 
        """
        pass
    def forward(self, input):
        """
        Forward pass through the Noisy Linear layer.
        
        returns -->
             in training mode-->    Noisy output using sampled noise (input , weight, bias).
             in eval mode -->    Deterministic output using mean weights (input, weight_mu, bias_mu)
            
        """
        pass