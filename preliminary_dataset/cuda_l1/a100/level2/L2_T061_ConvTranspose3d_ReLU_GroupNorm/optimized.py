import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    Optimized implementation using mathematical transformations and PyTorch's optimized primitives.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        groups (int): Number of groups for group normalization
        bias (bool): Whether to include bias in the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # Create the original modules for reference and parameter management
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)  # Use inplace ReLU to reduce memory usage
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        
        # Create optimized regular convolution equivalent to transposed convolution
        self.optimized_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias
        )
        
        # Pre-compute and optimize weight transformation
        with torch.no_grad():
            # Transform weights from ConvTranspose3d format to Conv3d format
            transposed_weight = self.conv_transpose.weight
            
            # Permute dimensions and flip kernel
            transformed_weight = transposed_weight.permute(1, 0, 2, 3, 4).flip(dims=[2, 3, 4])
            
            # Ensure optimal memory layout
            self.optimized_conv.weight.data.copy_(transformed_weight.contiguous())
            
            # Copy bias if present
            if bias and self.conv_transpose.bias is not None:
                self.optimized_conv.bias.data.copy_(self.conv_transpose.bias.data)
        
        # Pre-compute padding configuration for efficiency
        self.padding = (
            kernel_size - 1, kernel_size - 1,  # W dimension (left, right)
            kernel_size - 1, kernel_size - 1,  # H dimension (top, bottom)
            kernel_size - 1, kernel_size - 1   # D dimension (front, back)
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Apply padding with pre-computed configuration
        x_padded = F.pad(x, self.padding)
        
        # Apply optimized convolution
        x = self.optimized_conv(x_padded)
        
        # Apply ReLU in-place
        self.relu(x)  # in-place operation
        
        # Apply group normalization and return
        return self.group_norm(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]