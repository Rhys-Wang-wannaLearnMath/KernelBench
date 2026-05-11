import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, 
    multiplies by a scaling factor, and applies sigmoid.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        scaling_factor (float): Scaling factor to apply
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        # Create the convolution layer
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Initialize scaling factor and bias parameters
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Enable cuDNN benchmarking for optimal convolution performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Pre-convert weights to channels_last format for better performance
        if torch.cuda.is_available():
            self.conv.weight.data = self.conv.weight.data.to(
                memory_format=torch.channels_last_3d)
        
        # Create JIT-compiled versions of the operations for better performance
        try:
            @torch.jit.script
            def fused_ops(x, scaling_factor, bias):
                # Fuse operations to reduce memory traffic
                x = x * scaling_factor
                x = torch.tanh(x)
                x = x * bias
                x = torch.sigmoid(x)
                return x
            
            self.fused_ops = fused_ops
            self.use_jit = True
        except Exception:
            self.use_jit = False
            
        # Cache for optimized algorithms
        self._conv_algorithm_cache = {}

    def forward(self, x):
        # Convert to channels_last format for better memory access patterns if on CUDA
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d)
            
            # Ensure weights are in channels_last format
            if not self.conv.weight.is_contiguous(memory_format=torch.channels_last_3d):
                self.conv.weight.data = self.conv.weight.data.to(
                    memory_format=torch.channels_last_3d)
        
        # Select optimal convolution algorithm based on input dimensions
        key = (tuple(x.shape), x.device.index if x.is_cuda else None)
        if key not in self._conv_algorithm_cache and x.is_cuda:
            # Find the best algorithm for this specific input configuration
            with torch.no_grad():
                # Run once to trigger algorithm selection
                _ = self.conv(x[0:1].clone())
                # Cache that we've optimized for this configuration
                self._conv_algorithm_cache[key] = True
        
        # Perform convolution with optimized memory layout
        x = self.conv(x)
        
        # Use JIT-compiled operations if available
        if self.use_jit:
            return self.fused_ops(x, self.scaling_factor, self.bias)
        
        # Otherwise, use standard operations
        x = x * self.scaling_factor
        x = torch.tanh(x)
        x = x * self.bias
        x = torch.sigmoid(x)
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]