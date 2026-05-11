import torch
import torch.nn as nn

class MinTanhTanhModule(torch.nn.Module):
    """JIT-compilable module for min + double tanh operations"""
    def forward(self, x):
        # Combine min and double tanh operations for potential fusion
        min_val = torch.min(x, dim=1, keepdim=True)[0]
        return torch.tanh(torch.tanh(min_val))

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
        
        # Create and JIT compile the min-tanh-tanh module
        self.min_tanh_tanh = MinTanhTanhModule()
        if torch.cuda.is_available():
            try:
                self.min_tanh_tanh = torch.jit.script(self.min_tanh_tanh)
            except Exception:
                pass  # Fallback to non-JIT version if compilation fails
        
        # CUDA graph capture state
        self._graph = None
        self._static_input = None
        self._static_output = None
        self._input_shape = None
        self._warmup_done = False
    
    def _cleanup_graph(self):
        """Clean up CUDA graph resources"""
        if self._graph is not None:
            del self._graph
            self._graph = None
        if self._static_input is not None:
            del self._static_input
            self._static_input = None
        if self._static_output is not None:
            del self._static_output
            self._static_output = None
        self._warmup_done = False
    
    def _initialize_cuda_graph(self, x):
        """Initialize CUDA graph for faster repeated execution"""
        if not torch.cuda.is_available() or not x.is_cuda:
            return False
            
        try:
            # Clean up any existing graph resources
            self._cleanup_graph()
            
            # Record input shape
            self._input_shape = x.shape
            
            # Create static tensor for graph capture
            self._static_input = torch.zeros_like(x, memory_format=torch.contiguous_format)
            
            # Warmup runs to ensure GPU initialization
            for _ in range(5):
                _ = self.min_tanh_tanh(self.conv(self._static_input))
            
            torch.cuda.synchronize()
                
            # Capture the graph
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                conv_out = self.conv(self._static_input)
                self._static_output = self.min_tanh_tanh(conv_out)
                
            self._warmup_done = True
            return True
        except Exception:
            # Fall back to normal execution if CUDA graphs fail
            self._cleanup_graph()
            return False
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, min operation, and double tanh
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Check if input shape has changed or if this is first run
        if x.is_cuda and (self._input_shape is None or x.shape != self._input_shape or not self._warmup_done):
            # Initialize or reinitialize CUDA graph for new input shape
            self._initialize_cuda_graph(x)
        
        # Try to use CUDA graph if available
        if x.is_cuda and self._graph is not None and self._warmup_done:
            # Copy input to static tensor
            self._static_input.copy_(x)
            # Replay the graph
            self._graph.replay()
            # Return the output
            return self._static_output
        
        # Standard execution path (fallback)
        conv_out = self.conv(x)
        return self.min_tanh_tanh(conv_out)
    
    def __del__(self):
        """Clean up resources when the module is deleted"""
        self._cleanup_graph()

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