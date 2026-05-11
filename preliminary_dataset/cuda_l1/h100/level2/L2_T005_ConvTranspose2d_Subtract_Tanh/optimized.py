import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        bias_shape (tuple): Shape of the bias tensor
        stride (int): Stride for the transposed convolution
        padding (int): Padding for the transposed convolution
        output_padding (int): Output padding for the transposed convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # For CUDA graph optimization
        self.use_cuda_graph = torch.cuda.is_available()
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.graph_ready = False
        
        # For tracking whether weights have changed (to invalidate CUDA graph)
        self._weight_version = None
        self._bias_version = None
        
        # JIT compile the fused bias and tanh operation for better performance
        self.fused_bias_tanh = None
        if torch.cuda.is_available():
            try:
                # Define a JIT function that fuses bias subtraction and tanh with in-place operations
                @torch.jit.script
                def fused_bias_tanh_inplace(x, bias):
                    return torch.tanh_(x.sub_(bias))
                
                self.fused_bias_tanh = fused_bias_tanh_inplace
                
                # Warm up the JIT compiler with a small tensor
                dummy_input = torch.zeros(1, out_channels, 4, 4, device='cuda')
                dummy_bias = torch.zeros_like(self.bias, device='cuda')
                _ = self.fused_bias_tanh(dummy_input.clone(), dummy_bias)
            except:
                # Fallback to non-inplace version if the inplace version fails
                try:
                    @torch.jit.script
                    def fused_bias_tanh(x, bias):
                        return torch.tanh(x - bias)
                    
                    self.fused_bias_tanh = fused_bias_tanh
                except:
                    pass  # Fallback to standard operations if JIT fails
        
        # Set to eval mode by default for better inference performance
        self.eval()
    
    def _check_weight_changed(self):
        """Check if weights or bias have changed since last call"""
        current_weight_version = self.conv_transpose.weight.data_ptr()
        current_bias_version = self.bias.data_ptr()
        
        if (self._weight_version != current_weight_version or 
            self._bias_version != current_bias_version):
            self._weight_version = current_weight_version
            self._bias_version = current_bias_version
            return True
        return False
    
    def forward(self, x):
        # Fast path for inference with CUDA graph
        if x.is_cuda and not self.training and self.use_cuda_graph:
            try:
                # Check if we need to rebuild the graph (input shape changed or weights changed)
                rebuild_graph = (not self.graph_ready or 
                                self.static_input is None or
                                self.static_input.shape != x.shape or
                                self._check_weight_changed())
                
                if rebuild_graph:
                    # Clean up previous graph resources if they exist
                    self.cuda_graph = None
                    
                    # Initialize or reinitialize static tensors
                    self.static_input = torch.zeros_like(x)
                    
                    # Create CUDA graph
                    self.cuda_graph = torch.cuda.CUDAGraph()
                    
                    # Record operations into the graph
                    with torch.cuda.graph(self.cuda_graph):
                        self.static_input.copy_(x)
                        # Perform transposed convolution
                        conv_out = self.conv_transpose(self.static_input)
                        
                        # Fused bias subtraction and tanh activation
                        if self.fused_bias_tanh is not None:
                            result = self.fused_bias_tanh(conv_out, self.bias)
                        else:
                            # Use in-place operations for better performance
                            result = torch.tanh_(conv_out.sub_(self.bias))
                        
                        # Store the result
                        self.static_output = result
                    
                    self.graph_ready = True
                
                # Execute the captured graph
                self.static_input.copy_(x)
                self.cuda_graph.replay()
                return self.static_output
            except Exception:
                # If graph capture or replay fails, fall back to regular execution
                self.graph_ready = False
        
        # Regular path (training or when CUDA graph fails)
        conv_out = self.conv_transpose(x)
        
        # Use fused operation if available (for CUDA)
        if x.is_cuda and self.fused_bias_tanh is not None:
            return self.fused_bias_tanh(conv_out, self.bias)
        
        # Standard path with in-place operations where possible
        if x.is_cuda:
            # Use aggressive in-place operations for CUDA
            return torch.tanh_(conv_out.sub_(self.bias))
        else:
            # CPU path - avoid in-place operations which might be slower on CPU
            x = conv_out - self.bias
            x = torch.tanh(x)
            return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]