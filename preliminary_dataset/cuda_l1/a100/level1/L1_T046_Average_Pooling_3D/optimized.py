import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of 3D Average Pooling using convolution.
    
    Args:
        kernel_size (int): Size of the kernel to apply pooling.
        stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
        padding (int, optional): Padding to apply before pooling. Defaults to 0.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
        # Pre-compute kernel value for maximum efficiency
        kernel_value = 1.0 / (kernel_size ** 3)
        
        # Create weight tensor with optimal dtype and register as buffer
        weight = torch.full((1, 1, kernel_size, kernel_size, kernel_size), 
                           kernel_value, dtype=torch.float32)
        self.register_buffer('weight', weight)
        
        # Pre-allocate expanded weight tensor for channels=32 (from hyperparameters)
        expanded_weight = weight.repeat(channels, 1, 1, 1, 1).contiguous()
        self.register_buffer('expanded_weight', expanded_weight)
        
        # Create standard avgpool for fallback
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        """
        Applies optimized Average Pooling to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
            
        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # Fast path for CUDA tensors
        if x.is_cuda:
            # Ensure input is contiguous for optimal memory access
            x_cont = x if x.is_contiguous() else x.contiguous()
            
            # Get the number of input channels
            in_channels = x_cont.size(1)
            
            # If input channels match our pre-allocated weight tensor (most common case)
            if in_channels == channels:
                # Use pre-allocated expanded weight (already on the correct device)
                return F.conv3d(
                    x_cont,
                    self.expanded_weight,
                    stride=self.stride,
                    padding=self.padding,
                    groups=in_channels
                )
            else:
                # For different channel counts, create weight tensor on the fly
                expanded_weight = self.weight.expand(in_channels, 1, self.kernel_size, 
                                                  self.kernel_size, self.kernel_size).contiguous()
                return F.conv3d(
                    x_cont,
                    expanded_weight,
                    stride=self.stride,
                    padding=self.padding,
                    groups=in_channels
                )
        
        # Fallback to standard implementation for non-CUDA tensors
        return self.avg_pool(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]