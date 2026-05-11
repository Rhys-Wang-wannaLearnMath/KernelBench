import torch
import torch.nn as nn

class MinTanhTanhModule(torch.nn.Module):
    """JIT-compilable module for min + double tanh operations"""
    def forward(self, x):
        # Fuse operations to minimize intermediate memory allocations
        return torch.tanh(torch.tanh(torch.min(x, dim=1, keepdim=True)[0]))

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Use PyTorch's optimized Conv2d implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Create and JIT compile the min-tanh-tanh module with optimization flags
        self.min_tanh_tanh = MinTanhTanhModule()
        if torch.cuda.is_available():
            try:
                # Disable profiling for more aggressive optimization during compilation
                with torch.jit.optimized_execution(True):
                    self.min_tanh_tanh = torch.jit.script(self.min_tanh_tanh)
            except Exception:
                pass
        
        # Create a dedicated CUDA stream for better overlapping
        self._stream = None
        if torch.cuda.is_available():
            try:
                self._stream = torch.cuda.Stream()
            except Exception:
                self._stream = None
    
    def forward(self, x):
        """
        Optimized forward pass (caching mechanism removed)
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, min operation, and double tanh
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Directly compute without caching
        conv_out = self.conv(x)
        return self.min_tanh_tanh(conv_out)
    
    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, '_stream') and self._stream is not None:
            try:
                del self._stream
            except Exception:
                pass

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size]