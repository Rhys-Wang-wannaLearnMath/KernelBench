import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of 2D Average Pooling.
    
    Args:
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
        padding (int, optional): Padding applied to the input tensor. Defaults to 0.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
        # Pre-compute the averaging kernel
        kernel_value = 1.0 / (kernel_size * kernel_size)
        self.register_buffer('kernel', torch.full((1, 1, kernel_size, kernel_size), 
                                                 kernel_value, dtype=torch.float32))
        
        # Cache for expanded kernel to avoid repeated operations
        self.cached_channels = None
        self.cached_expanded_kernel = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using optimized operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        if not x.is_cuda:
            # Fall back to PyTorch implementation for CPU tensors
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        
        # Get number of channels
        channels = x.size(1)
        
        # Check if we need to create a new expanded kernel
        if self.cached_channels != channels or self.cached_expanded_kernel is None:
            # Create expanded kernel directly with the right shape
            # Each channel gets its own kernel
            self.cached_expanded_kernel = self.kernel.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
            self.cached_channels = channels
        
        # Apply convolution with the averaging kernel
        # Using groups=channels ensures each channel is processed independently
        result = F.conv2d(
            x,
            self.cached_expanded_kernel,
            stride=self.stride, 
            padding=self.padding,
            groups=channels
        )
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]