import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        pool_kernel_size (int): Kernel size for average pooling
        scale_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        
        # Calculate the pooled output size
        self.pooled_size = out_features // pool_kernel_size
        
        # Create a standard linear layer for proper initialization
        linear = nn.Linear(in_features, out_features)
        
        # Pre-compute pooled weights: reshape to [pooled_size, pool_kernel_size, in_features]
        # and average along the pool_kernel_size dimension
        w_reshaped = linear.weight.view(self.pooled_size, pool_kernel_size, in_features)
        pooled_weight = w_reshaped.mean(dim=1).contiguous()
        
        # Pre-compute pooled bias: reshape to [pooled_size, pool_kernel_size]
        # and average along the pool_kernel_size dimension
        b_reshaped = linear.bias.view(self.pooled_size, pool_kernel_size)
        pooled_bias = b_reshaped.mean(dim=1).contiguous()
        
        # Register the pooled parameters with standard names for potential optimization
        self.weight = nn.Parameter(pooled_weight)
        self.bias = nn.Parameter(pooled_bias)
        
        # Register scale factor as a buffer for efficient access
        self.register_buffer('scale', torch.tensor(scale_factor, dtype=torch.float32))
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Combined linear transformation and pooling using pre-computed weights and bias
        pooled = F.linear(x, self.weight, self.bias)
        
        # GELU activation
        activated = F.gelu(pooled)
        
        # In-place scaling to reduce memory allocation
        activated.mul_(self.scale)
        
        # Max reduction along dimension 1, directly accessing values
        result = torch.max(activated, dim=1).values
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, pool_kernel_size, scale_factor]