import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, group normalization, scaling, max pooling, and clamping.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        num_groups (int): Number of groups for group normalization
        scale_shape (tuple): Shape of the scaling parameter
        maxpool_kernel_size (int): Size of the max pooling kernel
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # For CUDA graph optimization
        self.use_cuda_graph = torch.cuda.is_available()
        self.cuda_graphs = {}  # Store multiple graphs for different input shapes
        self.static_inputs = {}
        self.static_outputs = {}
        
        # For TorchScript optimization
        self.use_script = torch.cuda.is_available()
        if self.use_script:
            try:
                # Create scripted version of the forward implementation
                self.scripted_forward = torch.jit.script(self._forward_impl)
            except Exception:
                self.use_script = False
        
        # For torch.compile optimization (PyTorch 2.0+)
        self.use_compile = hasattr(torch, 'compile') and torch.cuda.is_available()
        if self.use_compile:
            try:
                self.compiled_forward = torch.compile(self._forward_impl)
            except Exception:
                self.use_compile = False
    
    def _forward_impl(self, x):
        """
        Implementation of the forward pass for optimization
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x
    
    def _calculate_output_shape(self, input_shape):
        """
        Calculate the output shape based on the input shape
        
        Args:
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
        
        Returns:
            Tuple of (batch_size, out_channels, out_height, out_width)
        """
        batch_size, _, height, width = input_shape
        
        # Calculate convolution output dimensions
        conv_height = height - self.conv.kernel_size[0] + 1
        conv_width = width - self.conv.kernel_size[1] + 1
        
        # Calculate maxpool output dimensions
        out_height = conv_height // self.maxpool.kernel_size
        out_width = conv_width // self.maxpool.kernel_size
        
        return (batch_size, self.conv.out_channels, out_height, out_width)
    
    def _warmup(self, x, iterations=14):
        """
        Perform thorough warmup iterations to ensure CUDA kernels are compiled
        
        Args:
            x: Input tensor
            iterations: Number of warmup iterations
        """
        with torch.no_grad():
            # First run with synchronization to ensure initial compilation
            _ = self._forward_impl(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            
            # Multiple warmup runs with progressive synchronization
            # More frequent synchronization at the beginning, less frequent later
            for i in range(iterations):
                _ = self._forward_impl(x)
                # Synchronize more frequently in early iterations, less in later ones
                if i < 5 and i % 2 == 0 and x.is_cuda:
                    torch.cuda.synchronize()
                elif i >= 5 and i % 4 == 0 and x.is_cuda:
                    torch.cuda.synchronize()
            
            # Final synchronization to ensure all operations are complete
            if x.is_cuda:
                torch.cuda.synchronize()
    
    def forward(self, x):
        """
        Optimized forward pass using CUDA graph capture when possible
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use CUDA graph for static input shapes when possible
        if self.use_cuda_graph and x.is_cuda:
            # Create a key based on input shape and device
            shape_key = (x.shape, x.device.index if x.device.index is not None else 0)
            
            try:
                # If we don't have a graph for this shape yet, create one
                if shape_key not in self.cuda_graphs:
                    # Perform thorough warmup to ensure CUDA kernels are compiled
                    self._warmup(x)
                    
                    # Calculate output dimensions
                    output_shape = self._calculate_output_shape(x.shape)
                    
                    # Initialize static tensors for CUDA graph
                    # Use torch.empty instead of torch.zeros for potentially better performance
                    self.static_inputs[shape_key] = torch.empty_like(x, device=x.device)
                    self.static_outputs[shape_key] = torch.empty(
                        output_shape, 
                        device=x.device,
                        dtype=x.dtype
                    )
                    
                    # Copy input data to static input tensor
                    self.static_inputs[shape_key].copy_(x)
                    
                    # Capture the CUDA graph
                    graph = torch.cuda.CUDAGraph()
                    
                    with torch.cuda.graph(graph):
                        # Use the most optimized version available
                        if self.use_compile:
                            output = self.compiled_forward(self.static_inputs[shape_key])
                        elif self.use_script:
                            output = self.scripted_forward(self.static_inputs[shape_key])
                        else:
                            output = self._forward_impl(self.static_inputs[shape_key])
                        self.static_outputs[shape_key].copy_(output)
                    
                    self.cuda_graphs[shape_key] = graph
                
                # Copy input to static tensor and replay the graph
                self.static_inputs[shape_key].copy_(x)
                self.cuda_graphs[shape_key].replay()
                
                # Return a view instead of a clone to avoid memory allocation
                return self.static_outputs[shape_key].view_as(self.static_outputs[shape_key])
                
            except Exception:
                # If CUDA graph fails, fall back to other optimizations
                pass
        
        # Try torch.compile if available (prioritize this over TorchScript)
        if self.use_compile and x.is_cuda:
            try:
                return self.compiled_forward(x)
            except Exception:
                pass
        
        # Try TorchScript if available
        if self.use_script and x.is_cuda:
            try:
                return self.scripted_forward(x)
            except Exception:
                pass
        
        # Fallback to standard implementation
        return self._forward_impl(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]