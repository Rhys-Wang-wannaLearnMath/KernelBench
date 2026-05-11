import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    Optimized implementation that maintains identical functionality.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        groups (int): Number of blocked connections from input to output channels
        num_groups (int): Number of groups for GroupNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # Initialize the standard layers for parameter management
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Pre-cache parameters for faster access
        self.weight = self.conv_transpose.weight
        self.bias = self.conv_transpose.bias
        self.gn_weight = self.group_norm.weight
        self.gn_bias = self.group_norm.bias
        
        # Store hyperparameters
        self.stride = stride
        self.num_groups = num_groups
        
        # Enable cuDNN benchmark mode for potentially faster convolutions
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Step 1: Transposed Convolution - use functional interface directly with explicit parameters
        x = F.conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=0,  # Default in reference implementation
            output_padding=0,  # Default in reference implementation
            groups=1  # Default in reference implementation
        )
        
        # Step 2: GELU activation
        x = F.gelu(x)
        
        # Step 3: GroupNorm - use functional interface directly with explicit parameters
        x = F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.gn_weight,
            bias=self.gn_bias,
            eps=1e-5  # Default epsilon value
        )
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
groups = 8
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]