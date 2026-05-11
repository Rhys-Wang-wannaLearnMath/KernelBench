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
        self.has_bias = bias
        
        # Pre-compute values for the optimized implementation
        self.conv_padding = kernel_size - 1 - padding
        
        # Register buffer for transformed weights to avoid recomputing
        self.register_buffer('weight_transformed', None)
        self.register_buffer('weight_transposed', None)
        self.register_buffer('bias_reshaped', None)
        self.weights_prepared = False
        
        # Enable cudnn benchmarking for better performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            
        # Thresholds for different implementation paths - fine-tuned based on empirical testing
        self.small_input_threshold = 4096  # H*W < this value
        self.medium_input_threshold = 16384  # H*W < this value
        
    def _prepare_weights(self):
        """Prepare weights for efficient computation"""
        if not self.weights_prepared:
            with torch.no_grad():
                # Get weights from the standard layer
                weight = self.conv_transpose2d.weight
                
                # Flip the weights for transposed convolution
                flipped_weight = weight.flip(2, 3)
                
                # Store both versions for different implementations
                self.weight_transposed = flipped_weight.transpose(0, 1).contiguous()
                self.weight_transformed = flipped_weight.contiguous()
                
                # Pre-reshape bias for efficient broadcasting if present
                if self.has_bias and self.conv_transpose2d.bias is not None:
                    self.bias_reshaped = self.conv_transpose2d.bias.view(1, -1, 1, 1).contiguous()
                
                self.weights_prepared = True
    
    def _get_output_shape(self, input_shape):
        """Calculate output shape for transposed convolution"""
        batch_size, _, height_in, width_in = input_shape
        
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        return (batch_size, self.out_channels, height_out, width_out)
    
    def _stride1_implementation(self, x):
        """Optimized implementation for stride=1"""
        # Use regular convolution with transposed weights and adjusted padding
        output = F.conv2d(
            x, 
            self.weight_transposed, 
            bias=None,  # We'll add bias separately for better performance
            padding=self.conv_padding, 
            groups=self.groups
        )
        
        # Add bias if present
        if self.has_bias and self.bias_reshaped is not None:
            output.add_(self.bias_reshaped)  # In-place addition
        
        # Apply output padding if needed
        if self.output_padding > 0:
            output = F.pad(output, [0, self.output_padding, 0, self.output_padding])
        
        return output
    
    def _dilated_implementation(self, x):
        """Implementation using dilated input for stride>1"""
        batch_size, in_channels, height_in, width_in = x.shape
        
        # Create dilated input by inserting zeros between elements
        dilated_height = (height_in - 1) * self.stride + 1
        dilated_width = (width_in - 1) * self.stride + 1
        
        # Create dilated input tensor filled with zeros
        dilated_input = torch.zeros(
            batch_size, in_channels, dilated_height, dilated_width, 
            device=x.device, dtype=x.dtype
        )
        
        # Fill in the values from the original input
        dilated_input[:, :, ::self.stride, ::self.stride] = x
        
        # Use regular convolution with properly transposed weights
        output = F.conv2d(
            dilated_input, 
            self.weight_transposed,
            bias=None,
            padding=self.kernel_size - 1 - self.padding, 
            groups=self.groups
        )
        
        # Add bias if present
        if self.has_bias and self.bias_reshaped is not None:
            output.add_(self.bias_reshaped)  # In-place addition
        
        # Apply output padding if needed
        if self.output_padding > 0:
            output = F.pad(output, [0, self.output_padding, 0, self.output_padding])
        
        return output
    
    def _blockwise_implementation(self, x):
        """Memory-efficient implementation for large inputs with stride>1"""
        batch_size, _, height_in, width_in = x.shape
        out_shape = self._get_output_shape(x.shape)
        
        # Process each channel group separately for better memory efficiency
        channels_per_group = self.in_channels // self.groups
        out_channels_per_group = self.out_channels // self.groups
        
        # Pre-allocate output tensor
        output = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
        
        # Process in batches to improve memory efficiency
        batch_size_per_iter = min(4, batch_size)  # Process up to 4 batches at a time
        
        for batch_start in range(0, batch_size, batch_size_per_iter):
            batch_end = min(batch_start + batch_size_per_iter, batch_size)
            batch_slice = slice(batch_start, batch_end)
            
            for g in range(self.groups):
                # Get input and weight for this group
                x_g = x[batch_slice, g*channels_per_group:(g+1)*channels_per_group]
                
                # Use PyTorch's native implementation for each group separately
                # This is more memory-efficient than creating a full dilated tensor
                out_g = F.conv_transpose2d(
                    x_g,
                    self.conv_transpose2d.weight[g*out_channels_per_group:(g+1)*out_channels_per_group],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    groups=1  # We're already handling groups manually
                )
                
                # Add to output tensor
                output[batch_slice, g*out_channels_per_group:(g+1)*out_channels_per_group] = out_g
        
        # Add bias if present
        if self.has_bias and self.bias_reshaped is not None:
            output.add_(self.bias_reshaped)  # In-place addition
            
        return output
    
    def _stride_gt_1_implementation(self, x):
        """Optimized implementation for stride > 1"""
        batch_size, _, height_in, width_in = x.shape
        input_size = height_in * width_in
        
        # For small inputs or complex group configurations, use PyTorch's native implementation
        if input_size < self.small_input_threshold or self.groups > 4:
            return self.conv_transpose2d(x)
        
        # For medium-sized inputs, use dilated approach
        if input_size < self.medium_input_threshold:
            return self._dilated_implementation(x)
        
        # For larger inputs, use blockwise implementation
        return self._blockwise_implementation(x)
    
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
        if not self.weights_prepared:
            self._prepare_weights()
        
        # Choose implementation based on stride
        if self.stride == 1:
            return self._stride1_implementation(x)
        else:
            return self._stride_gt_1_implementation(x)

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