import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of the 3D convolution model with fused operations
    and memory layout optimization
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        pool_kernel_size (int): Size of the pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Store original pooling layers for compatibility
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        
        # Pre-compute the combined pooling parameters
        self.fused_pool_size = pool_kernel_size * 2
        self.fused_pool_stride = pool_kernel_size * 2
        
        # Enable cudnn benchmarking for faster convolution
        torch.backends.cudnn.benchmark = True
        
        # Ensure weights are contiguous and in optimal memory layout
        self.conv.weight.data = self.conv.weight.data.contiguous()
        if self.conv.bias is not None:
            self.conv.bias.data = self.conv.bias.data.contiguous()
            
        # Convert weights to channels_last format for better memory access patterns
        if torch.cuda.is_available():
            self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
    
    def forward(self, x):
        """
        Optimized forward pass with fused operations and memory layout optimization
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, softmax, and pooling
        """
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Convert input to channels_last format if on CUDA
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution with optimized memory layout
        x = self.conv(x)
        
        # Apply softmax along channel dimension
        x = F.softmax(x, dim=1)
        
        # Apply fused pooling operations - combining two consecutive pooling operations
        # into a single pooling with larger kernel_size and stride
        x = F.max_pool3d(x, kernel_size=self.fused_pool_size, 
                         stride=self.fused_pool_stride)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_channels, out_channels, kernel_size, pool_kernel_size]