import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of Instance Normalization using PyTorch's functional API
    to minimize overhead while leveraging PyTorch's native optimized CUDA kernels.
    
    Args:
        num_features (int): Number of features in the input tensor.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        
        # Initialize parameters with the same default values as nn.InstanceNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Use the same default epsilon as nn.InstanceNorm2d
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor using PyTorch's
        optimized functional API.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).
            
        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # Ensure input is contiguous for optimal memory access
        x_cont = x if x.is_contiguous() else x.contiguous()
        
        # Use PyTorch's highly optimized functional interface
        # This bypasses nn.Module overhead while using the same optimized CUDA kernels
        return F.instance_norm(
            input=x_cont,
            running_mean=None,  # Instance norm doesn't use running stats
            running_var=None,   # Instance norm doesn't use running stats
            weight=self.weight,
            bias=self.bias,
            use_input_stats=True,  # Always use input stats for instance norm
            momentum=0.1,  # Default value (not used when running_mean is None)
            eps=self.eps
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]