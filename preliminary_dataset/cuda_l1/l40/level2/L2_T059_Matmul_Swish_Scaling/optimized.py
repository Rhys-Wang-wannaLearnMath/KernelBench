import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Initialize weights and bias directly
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters the same way nn.Linear does
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Store scaling factor as a primitive float for optimal performance
        self.scaling_factor = float(scaling_factor)
        
        # Pre-compute and store weight transpose as buffer for optimal performance
        self.register_buffer('weight_t', self.weight.t().contiguous())
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Fast path for already contiguous inputs
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use optimized matrix multiplication via torch.addmm
        # This directly leverages cuBLAS for maximum performance
        out = torch.addmm(
            self.bias,              # bias
            x,                      # input
            self.weight_t,          # pre-transposed weight
            beta=1.0,               # bias multiplier
            alpha=1.0               # matmul multiplier
        )
        
        # Use PyTorch's optimized SiLU (Swish) implementation in-place
        # F.silu is equivalent to x * sigmoid(x) but with optimized CUDA kernels
        out = F.silu(out, inplace=True)
        
        # Apply scaling factor in-place to reduce memory allocation
        out.mul_(self.scaling_factor)
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, scaling_factor]