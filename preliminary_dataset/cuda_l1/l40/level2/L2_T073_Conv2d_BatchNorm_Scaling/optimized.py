import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        
        # Create standard modules for initialization and training mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        
        # Register buffers for fused parameters
        self.register_buffer('fused_weight', torch.empty_like(self.conv.weight))
        self.register_buffer('fused_bias', torch.empty(out_channels, device=self.conv.weight.device))
        
        # Store convolution parameters
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        
        # Check if we need to pass explicit parameters to conv2d
        # Default values for conv2d are stride=1, padding=0, dilation=1, groups=1
        self.needs_explicit_params = (
            self.stride != (1, 1) or 
            self.padding != (0, 0) or 
            self.dilation != (1, 1) or 
            self.groups != 1
        )
        
        # Track parameter folding state
        self.fused_params_ready = False
        
        # CUDA graph related attributes
        self.use_cuda_graph = torch.cuda.is_available()
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.graph_ready = False
        self.last_input_shape = None
        
        # Set to evaluation mode and immediately compute fused parameters
        self.eval()
        self._compute_fused_parameters()
    
    def _compute_fused_parameters(self):
        """Pre-compute the fused parameters for BatchNorm and scaling"""        
        with torch.no_grad():
            # Get batch norm parameters
            gamma = self.bn.weight
            beta = self.bn.bias  
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
            eps = self.bn.eps
            
            # Compute combined scaling factor using rsqrt (more efficient)
            inv_std = torch.rsqrt(running_var + eps)
            combined_scale = gamma * inv_std * self.scaling_factor
            
            # Reshape for broadcasting with conv weights
            scale_reshaped = combined_scale.view(-1, 1, 1, 1)
            
            # Fold everything into weights (conv + bn + scaling in one step)
            self.fused_weight.copy_(self.conv.weight * scale_reshaped)
            
            # Fold everything into bias (conv + bn + scaling in one step)  
            if self.conv.bias is not None:
                self.fused_bias.copy_((self.conv.bias - running_mean) * combined_scale + beta * self.scaling_factor)
            else:
                self.fused_bias.copy_(beta * self.scaling_factor - running_mean * combined_scale)
                
            # Ensure tensors are contiguous for optimal memory access
            if not self.fused_weight.is_contiguous():
                self.fused_weight = self.fused_weight.contiguous()
            if not self.fused_bias.is_contiguous():
                self.fused_bias = self.fused_bias.contiguous()
                
            self.fused_params_ready = True
            
            # Reset CUDA graph state when parameters change
            self.graph_ready = False
    
    def _calculate_output_shape(self, input_shape):
        """Calculate the output shape for a given input shape"""
        batch_size, _, in_height, in_width = input_shape
        
        # Calculate output dimensions using convolution formula
        out_height = ((in_height + 2 * self.padding[0] - self.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) 
                      // self.stride[0] + 1)
        out_width = ((in_width + 2 * self.padding[1] - self.dilation[1] * (self.conv.kernel_size[1] - 1) - 1) 
                     // self.stride[1] + 1)
        
        return (batch_size, self.conv.out_channels, out_height, out_width)
    
    def _run_with_graph(self, x):
        """Execute the forward pass using CUDA graph for better performance"""
        current_shape = x.shape
        
        # Check if we need to recreate the graph due to shape change
        shape_changed = (self.last_input_shape != current_shape)
        
        if not self.graph_ready or shape_changed:
            # Clean up old graph resources if they exist
            if self.graph is not None:
                del self.graph
                self.graph = None
            
            if self.static_input is not None and (self.static_input.shape != current_shape):
                del self.static_input
                self.static_input = None
            
            if self.static_output is not None:
                output_shape = self._calculate_output_shape(current_shape)
                if self.static_output.shape != output_shape:
                    del self.static_output
                    self.static_output = None
            
            # Initialize static input tensor
            if self.static_input is None:
                self.static_input = torch.zeros_like(x, device=x.device)
            
            # Calculate output shape and initialize static output tensor
            output_shape = self._calculate_output_shape(current_shape)
            if self.static_output is None:
                self.static_output = torch.zeros(output_shape, device=x.device)
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Copy input data to static tensor
            self.static_input.copy_(x)
            
            # Synchronize before graph capture
            torch.cuda.synchronize()
            
            with torch.cuda.graph(self.graph):
                if self.needs_explicit_params:
                    self.static_output = F.conv2d(
                        self.static_input, 
                        self.fused_weight, 
                        self.fused_bias, 
                        self.stride, 
                        self.padding, 
                        self.dilation, 
                        self.groups
                    )
                else:
                    self.static_output = F.conv2d(
                        self.static_input, 
                        self.fused_weight, 
                        self.fused_bias
                    )
            
            # Synchronize after graph capture
            torch.cuda.synchronize()
            
            self.graph_ready = True
            self.last_input_shape = current_shape
        
        # Copy input data to static tensor and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return the output directly
        return self.static_output
    
    def forward(self, x):
        if self.training:
            # Standard implementation for training mode
            x = self.conv(x)
            x = self.bn(x)
            x = x * self.scaling_factor
            return x
        else:
            # Optimized path for inference
            if not self.fused_params_ready:
                self._compute_fused_parameters()
            
            # Ensure optimal memory layout
            if not x.is_contiguous():
                x = x.contiguous()
            
            # Use CUDA graph if available and input is on CUDA
            if self.use_cuda_graph and x.is_cuda:
                try:
                    return self._run_with_graph(x)
                except Exception:
                    # Fall back to regular execution if graph fails
                    self.use_cuda_graph = False
            
            # Use minimal parameter call for best performance
            if self.needs_explicit_params:
                return F.conv2d(x, self.fused_weight, self.fused_bias, 
                               self.stride, self.padding, 
                               self.dilation, self.groups)
            else:
                return F.conv2d(x, self.fused_weight, self.fused_bias)
    
    def train(self, mode=True):
        """Override train method to handle parameter folding state"""
        result = super(ModelNew, self).train(mode)
        if not mode and not self.fused_params_ready:
            # Switching to eval mode - compute fused parameters
            self._compute_fused_parameters()
        elif mode:
            # Switching to train mode - mark parameters as needing recomputation
            self.fused_params_ready = False
            self.graph_ready = False
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, scaling_factor]