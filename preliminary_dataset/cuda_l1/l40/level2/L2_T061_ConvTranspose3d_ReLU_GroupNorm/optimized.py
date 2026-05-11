import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    Optimized implementation using algorithmic transformations and PyTorch's optimized primitives.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        groups (int): Number of groups for group normalization
        bias (bool): Whether to include bias in the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # Create a temporary ConvTranspose3d to get the correct initial weights
        temp_conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        
        # Create a regular Conv3d that we'll use in our optimized implementation
        self.optimized_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias
        )
        
        # Transform the weights from the transposed convolution to our regular convolution
        with torch.no_grad():
            # For ConvTranspose3d -> Conv3d, we need to:
            # 1. Swap in_channels and out_channels dimensions
            # 2. Flip the kernel spatially
            # Perform both operations efficiently in a single chain
            self.optimized_conv.weight.data.copy_(
                temp_conv_transpose.weight.permute(1, 0, 2, 3, 4).flip(dims=[2, 3, 4]).contiguous()
            )
            
            # Copy bias if present
            if bias and temp_conv_transpose.bias is not None:
                self.optimized_conv.bias.data.copy_(temp_conv_transpose.bias.data)
        
        # Use inplace ReLU to reduce memory usage
        self.relu = nn.ReLU(inplace=True)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        
        # Pre-calculate padding needed for the regular convolution to mimic transposed convolution
        self.padding = kernel_size - 1
        
        # Clean up the temporary module to free memory
        del temp_conv_transpose
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        # Pad the input appropriately to mimic transposed convolution
        # Using a single padding value for all dimensions for simplicity and efficiency
        p = self.padding
        padded_input = F.pad(x, (p, p, p, p, p, p))
        
        # Apply convolution
        x = self.optimized_conv(padded_input)
        
        # Apply ReLU in-place
        self.relu(x)  # In-place operation
        
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