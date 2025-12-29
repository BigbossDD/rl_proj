import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    A Linear layer with parametric noise for exploration.
    
    Standard Linear Layer:
        y = Wx + b
    
    Noisy Linear Layer:
        The weights and biases are perturbed by learnable noise.
        y = (W + sigma_W * epsilon_W)x + (b + sigma_b * epsilon_b)
    
    where:
        mu (u): The mean weight (learnable).
        sigma (o): The noise scale (learnable).
        epsilon (e): Random noise sampled from a Gaussian distribution (not learned).
        
    The actual parameters (W and b) are calculated on every forward pass as:
        W = mu_W + sigma_W * epsilon_W
        b = mu_b + sigma_b * epsilon_b
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # --- Learnable Parameters ---
        # Instead of one weight matrix W, we have two: mu_W and sigma_W.
        # Shape: [out_features, in_features]
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Same for bias: mu_b and sigma_b
        # Shape: [out_features]
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # --- Noise Buffers (Non-learnable) ---
        # These hold the random epsilon values sampled during forward().
        # We register them as buffers so they are saved with the model state
        # but not updated by the optimizer.
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initializes the learnable parameters (mu and sigma).
        """
        # 1. Initialize mu (mean) weights
        # We use a uniform distribution: U[-1/sqrt(p), 1/sqrt(p)]
        # where p is the number of inputs (in_features).
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # 2. Initialize sigma (noise scale) weights
        # Initialized to a constant small value: sigma_0 / sqrt(p)
        # This ensures the initial noise is small but present.
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def scale_noise(self, size):
        """
        Helper function for Factorized Gaussian Noise.
        To reduce computation, we use 'Factorized' noise rather than 'Independent' noise.
        
        Instead of generating a huge matrix of random numbers for the weights,
        we generate two smaller vectors:
            1. epsilon_in (size of input)
            2. epsilon_out (size of output)
            
        We then transform them using the function f(x) from the paper:
            f(x) = sgn(x) * sqrt(|x|)
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """
        Samples fresh noise (epsilon) for the current training step.
        This is called before every forward pass during training.
        """
        # 1. Generate random noise vectors for inputs and outputs
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        
        # 2. Calculate the final noise matrix for weights (epsilon_W)
        # Formula: epsilon_W = f(epsilon_out) outer_product f(epsilon_in)
        # This creates a [out, in] matrix from two vectors.
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        
        # 3. Calculate noise for bias (epsilon_b)
        # Formula: epsilon_b = f(epsilon_out)
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        """
        Forward pass of the layer.
        """
        if self.training:
            # A. TRAIN MODE: Sample fresh noise
            self.reset_noise()
            
            # B. Calculate Noisy Weights
            # W_noisy = mu_W + (sigma_W * epsilon_W)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            
            # C. Calculate Noisy Bias
            # b_noisy = mu_b + (sigma_b * epsilon_b)
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            
            # D. Standard Linear Transformation
            # y = W_noisy * x + b_noisy
            return F.linear(input, weight, bias)
        
        else:
            # E. EVAL MODE: No noise
            # We just use the mean values (mu) for deterministic action selection.
            # y = mu_W * x + mu_b
            return F.linear(input, self.weight_mu, self.bias_mu)