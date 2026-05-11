import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

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
        
        # Create a standard ConvTranspose2d layer to manage parameters
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, groups=groups, bias=bias
        )
        
        # Store parameters for our optimized implementation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Pre-compute values for the optimized implementation
        self.conv_padding = kernel_size - 1 - padding
        
        # Register buffers for transformed weights to avoid recomputing
        self.register_buffer('weight_transposed', None)
        self.register_buffer('bias_reshaped', None)
        self.weight_prepared = False
        
        # Enable cudnn benchmarking for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Prepare weights immediately if we're in eval mode
        if not self.training:
            self._prepare_weights()
    
    def _prepare_weights(self):
        """Prepare weights for efficient computation"""
        if not self.weight_prepared:
            with torch.no_grad():
                # Get weights from the standard layer
                weight = self.conv_transpose2d.weight
                
                # Flip the weights for transposed convolution
                flipped_weight = weight.flip(2, 3)
                
                # Transpose input/output channels and make contiguous
                self.weight_transposed = flipped_weight.transpose(0, 1).contiguous()
                
                # Pre-reshape bias for efficient broadcasting if present
                if self.conv_transpose2d.bias is not None:
                    self.bias_reshaped = self.conv_transpose2d.bias.view(1, -1, 1, 1).contiguous()
                
                self.weight_prepared = True
    
    def _add_bias(self, output):
        """Add bias to output if present"""
        if self.bias_reshaped is not None:
            output.add_(self.bias_reshaped)  # In-place addition
        return output
    
    def _apply_output_padding(self, output):
        """Apply output padding if needed"""
        if self.output_padding > 0:
            output = F.pad(output, [0, self.output_padding, 0, self.output_padding])
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure input is contiguous for better performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Prepare weights if not already done
        if not self.weight_prepared:
            self._prepare_weights()
        
        # For stride=1, use optimized implementation with conv2d
        if self.stride == 1:
            # Use regular convolution with transposed weights
            output = F.conv2d(
                x, 
                self.weight_transposed, 
                bias=None,  # We'll add bias separately for better performance
                padding=self.conv_padding, 
                groups=self.groups
            )
            
            # Add bias if present and apply output padding
            output = self._add_bias(output)
            output = self._apply_output_padding(output)
            
            return output
        
        # For stride > 1, use optimized approach
        else:
            batch_size, _, height_in, width_in = x.shape
            
            # Calculate output dimensions
            height_out = (height_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
            width_out = (width_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
            
            # Dynamic threshold based on input size and channel dimensions
            input_size = height_in * width_in
            channel_factor = (self.in_channels * self.out_channels) / 2048  # Normalize by reference channel product
            size_threshold = int(17000 * channel_factor)  # Empirically tuned threshold
            
            # For larger inputs, use dilated approach
            if input_size > size_threshold:
                # Calculate dilated dimensions
                dilated_height = height_in + (height_in - 1) * (self.stride - 1)
                dilated_width = width_in + (width_in - 1) * (self.stride - 1)
                
                # Create dilated input tensor filled with zeros
                dilated_input = torch.zeros(
                    batch_size, self.in_channels, dilated_height, dilated_width, 
                    device=x.device, dtype=x.dtype
                )
                
                # Fill in the values from the original input - this is the key operation for transposed conv
                dilated_input[:, :, ::self.stride, ::self.stride] = x
                
                # Use regular convolution with transposed weights
                output = F.conv2d(
                    dilated_input, 
                    self.weight_transposed, 
                    bias=None,
                    padding=self.conv_padding, 
                    groups=self.groups
                )
                
                # Add bias if present
                output = self._add_bias(output)
                
                # Apply output padding if needed
                output = self._apply_output_padding(output)
            else:
                # For smaller inputs, use PyTorch's optimized implementation
                # but handle output padding and bias separately for better control
                
                # Use F.conv_transpose2d directly with original weights
                output = F.conv_transpose2d(
                    x,
                    self.conv_transpose2d.weight,
                    bias=None,  # We'll add bias separately for better performance
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=0,  # Handle output padding separately for better control
                    groups=self.groups
                )
                
                # Add bias if present
                output = self._add_bias(output)
                
                # Apply output padding if needed
                output = self._apply_output_padding(output)
            
            return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization