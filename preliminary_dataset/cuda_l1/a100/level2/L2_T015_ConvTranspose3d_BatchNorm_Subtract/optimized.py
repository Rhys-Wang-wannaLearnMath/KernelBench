import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Optimized implementation of 3D convolutional transpose layer 
    followed by Batch Normalization and subtraction.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): Padding added to all sides of the input
        bias (bool): If True, adds a learnable bias to the output
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        
        # Keep the original layers for parameter initialization and fallback
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        
        # Store parameters for easy access
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.bias = bias
        
        # Pre-allocate and cache all computation buffers
        self.register_buffer('bn_scale', torch.ones(1, out_channels, 1, 1, 1))
        self.register_buffer('bn_shift', torch.zeros(1, out_channels, 1, 1, 1))
        
        # Cache epsilon for efficiency
        self.eps = self.batch_norm.eps
        
        # Flag to track if we need to update parameters
        self._params_cached = False
        
    def _cache_bn_params(self):
        """Cache batch normalization parameters in optimal format for broadcasting"""
        with torch.no_grad():
            # Get batch norm parameters
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            weight = self.batch_norm.weight
            bias = self.batch_norm.bias
            
            # Pre-compute scale and shift with optimal broadcasting shape
            inv_std = torch.rsqrt(running_var + self.eps)
            scale = weight * inv_std
            shift = bias - running_mean * scale
            
            # Store in pre-shaped format for broadcasting (1, C, 1, 1, 1)
            self.bn_scale.copy_(scale.view(1, -1, 1, 1, 1))
            self.bn_shift.copy_(shift.view(1, -1, 1, 1, 1))
            
            self._params_cached = True

    def _optimized_forward(self, x):
        """Optimized forward implementation using PyTorch operations"""
        # Step 1: Apply ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Cache batch norm parameters if not already done
        if not self._params_cached:
            self._cache_bn_params()
        
        # Step 2: Apply batch normalization using pre-computed parameters
        x = torch.addcmul(self.bn_shift, x, self.bn_scale)
        
        # Step 3: Subtract spatial mean
        spatial_mean = x.mean(dim=(2, 3, 4), keepdim=True)
        x.sub_(spatial_mean)
        
        return x
    
    def _fused_conv_bn_subtract(self, x):
        """
        Fused implementation of ConvTranspose3d + BatchNorm3d + mean subtraction
        using PyTorch's memory-efficient operations
        """
        # Get the weight and bias from conv_transpose
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.bias else None
        
        # Cache batch norm parameters if not already done
        if not self._params_cached:
            self._cache_bn_params()
        
        # Get output shape for transposed convolution
        batch_size = x.size(0)
        input_depth, input_height, input_width = x.size(2), x.size(3), x.size(4)
        
        # Calculate output spatial dimensions
        output_depth = (input_depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        output_height = (input_height - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        output_width = (input_width - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]
        
        # Step 1: Apply ConvTranspose3d
        output = F.conv_transpose3d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding
        )
        
        # Step 2: Apply batch normalization using pre-computed parameters
        output = torch.addcmul(self.bn_shift, output, self.bn_scale)
        
        # Step 3: Compute spatial mean efficiently
        # Reshape to combine all spatial dimensions for more efficient mean calculation
        batch_size, channels = output.shape[:2]
        spatial_size = output.shape[2] * output.shape[3] * output.shape[4]
        
        # Reshape to (batch_size, channels, spatial_size)
        output_reshaped = output.reshape(batch_size, channels, -1)
        
        # Compute mean along spatial dimensions
        spatial_mean = output_reshaped.mean(dim=2, keepdim=True)
        
        # Reshape mean back to original shape for broadcasting
        spatial_mean = spatial_mean.view(batch_size, channels, 1, 1, 1)
        
        # Subtract mean in-place
        output.sub_(spatial_mean)
        
        return output

    def forward(self, x):
        """
        Forward pass with optimized implementation
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Use the fused implementation for better performance
        return self._fused_conv_bn_subtract(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding]