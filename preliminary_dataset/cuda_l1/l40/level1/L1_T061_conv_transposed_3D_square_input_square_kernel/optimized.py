import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the original transposed convolution layer for reference and fallback
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
        # Store parameters for optimization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Pre-compute padding values for direct convolution
        self.pad_depth = kernel_size - 1 - padding
        self.pad_height = kernel_size - 1 - padding
        self.pad_width = kernel_size - 1 - padding
        
        # Pre-compute common padding tuple for the specific case
        self.common_padding = (
            self.pad_width, self.pad_width + output_padding,
            self.pad_height, self.pad_height + output_padding,
            self.pad_depth, self.pad_depth + output_padding
        )
        
        # Pre-compute transformed weights for direct convolution
        with torch.no_grad():
            # Get the original weights
            weight = self.conv_transpose3d.weight
            
            # Flip the weights in all spatial dimensions
            weight = weight.flip(2, 3, 4)
            
            # Swap input and output channels for direct convolution
            if groups > 1:
                # For group convolution
                transformed_weight = weight.clone()
            else:
                # For standard convolution
                transformed_weight = weight.permute(1, 0, 2, 3, 4).contiguous()
        
        # Register the transformed weight as a buffer (not a parameter)
        self.register_buffer('transformed_weight', transformed_weight)
        
        # Flag to use optimized implementation
        self.use_optimized = True
        
        # Flag to track if weights need updating (for training)
        self.weights_updated = True
        
        # Flag to determine if channels_last format should be used
        self.use_channels_last = hasattr(torch, 'channels_last_3d')
    
    def _update_transformed_weight(self):
        """Update the transformed weight buffer from the current weights"""
        with torch.no_grad():
            # Get the current weights
            weight = self.conv_transpose3d.weight
            
            # Flip the weights in all spatial dimensions
            weight = weight.flip(2, 3, 4)
            
            # Swap input and output channels for direct convolution
            if self.groups > 1:
                # For group convolution
                self.transformed_weight.copy_(weight)
            else:
                # For standard convolution
                self.transformed_weight.copy_(weight.permute(1, 0, 2, 3, 4).contiguous())
            
            self.weights_updated = True
    
    def _optimized_forward_specialized(self, x):
        """Specialized implementation for the specific hyperparameters"""
        # Update transformed weights if training (weights might have changed)
        if self.training and not self.weights_updated:
            self._update_transformed_weight()
        
        # Apply padding - for kernel_size=3, padding=0, we need pad=2
        x_padded = F.pad(x, self.common_padding)
        
        # Ensure the input is contiguous for better memory access
        if not x_padded.is_contiguous():
            x_padded = x_padded.contiguous()
            
        # Ensure the weights are contiguous for better memory access
        weights = self.transformed_weight
        if not weights.is_contiguous():
            weights = weights.contiguous()
        
        # Use channels_last memory format if available and dimensions are suitable
        if (self.use_channels_last and x_padded.shape[2] >= 8 and 
            x_padded.shape[3] >= 8 and x_padded.shape[4] >= 8):
            x_padded = x_padded.to(memory_format=torch.channels_last_3d)
            weights = weights.to(memory_format=torch.channels_last_3d)
        
        # Perform the convolution with optimized settings
        output = F.conv3d(
            x_padded, weights, 
            bias=self.conv_transpose3d.bias, 
            stride=1, padding=0, dilation=1, groups=self.groups
        )
        
        return output
    
    def _optimized_forward_stride1(self, x):
        """Optimized implementation for stride=1 case"""
        # Update transformed weights if training (weights might have changed)
        if self.training and not self.weights_updated:
            self._update_transformed_weight()
        
        # Apply padding
        x_padded = F.pad(x, self.common_padding)
        
        # Ensure the input is contiguous for better memory access
        if not x_padded.is_contiguous():
            x_padded = x_padded.contiguous()
        
        # Use channels_last memory format if available and dimensions are suitable
        if (self.use_channels_last and x_padded.shape[2] >= 8 and 
            x_padded.shape[3] >= 8 and x_padded.shape[4] >= 8):
            x_padded = x_padded.to(memory_format=torch.channels_last_3d)
            weights = self.transformed_weight.to(memory_format=torch.channels_last_3d)
        else:
            weights = self.transformed_weight
            
        # Use direct convolution with the transformed weights
        output = F.conv3d(
            x_padded, weights, 
            bias=self.conv_transpose3d.bias, 
            stride=1, padding=0, dilation=1, groups=self.groups
        )
        
        return output
    
    def _optimized_forward_striden(self, x):
        """Optimized implementation for stride>1 case"""
        # Update transformed weights if training (weights might have changed)
        if self.training and not self.weights_updated:
            self._update_transformed_weight()
        
        batch_size, in_channels, depth, height, width = x.shape
        
        # For stride > 1, we need to insert zeros between input elements
        if self.stride > 1:
            # Create a tensor of zeros with the shape needed for the dilated input
            dilated_shape = (batch_size, in_channels, 
                            depth + (depth - 1) * (self.stride - 1),
                            height + (height - 1) * (self.stride - 1),
                            width + (width - 1) * (self.stride - 1))
            dilated_input = torch.zeros(dilated_shape, dtype=x.dtype, device=x.device)
            
            # Place the original input values at stride intervals
            dilated_input[:, :, ::self.stride, ::self.stride, ::self.stride] = x
            
            # Update input for the next step
            x = dilated_input
        
        # Apply padding
        x_padded = F.pad(x, self.common_padding)
        
        # Ensure the input is contiguous for better memory access
        if not x_padded.is_contiguous():
            x_padded = x_padded.contiguous()
        
        # Use channels_last memory format if available and dimensions are suitable
        if (self.use_channels_last and x_padded.shape[2] >= 8 and 
            x_padded.shape[3] >= 8 and x_padded.shape[4] >= 8):
            x_padded = x_padded.to(memory_format=torch.channels_last_3d)
            weights = self.transformed_weight.to(memory_format=torch.channels_last_3d)
        else:
            weights = self.transformed_weight
            
        # Use direct convolution with the transformed weights
        output = F.conv3d(
            x_padded, weights, 
            bias=self.conv_transpose3d.bias, 
            stride=1, padding=0, dilation=1, groups=self.groups
        )
        
        return output
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution with optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Mark weights as potentially changed when in training mode
        if self.training:
            self.weights_updated = False
        
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use optimized implementation if enabled and on CUDA
        if self.use_optimized and x.is_cuda:
            try:
                # Use specialized implementation for the specific hyperparameters
                if (self.kernel_size == 3 and self.in_channels == 3 and 
                    self.out_channels == 64 and self.stride == 1):
                    return self._optimized_forward_specialized(x)
                elif self.stride == 1:
                    return self._optimized_forward_stride1(x)
                else:
                    return self._optimized_forward_striden(x)
            except Exception:
                # Fallback to PyTorch implementation if our optimization fails
                self.use_optimized = False
                return self.conv_transpose3d(x)
        else:
            # Use PyTorch's implementation
            return self.conv_transpose3d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization