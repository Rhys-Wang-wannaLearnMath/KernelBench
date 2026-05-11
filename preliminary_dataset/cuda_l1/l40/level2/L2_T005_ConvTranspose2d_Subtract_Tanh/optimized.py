import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of a model that performs a transposed convolution,
    subtracts a bias term, and applies tanh activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        bias_shape (tuple): Shape of the bias tensor
        stride (int): Stride of the convolution (default: 2)
        padding (int): Padding added to input (default: 1)
        output_padding (int): Additional padding for output (default: 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        # Initialize the transposed convolution layer with the same parameters
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        
        # Initialize the bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store parameters for optimization
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        
        # Cache for output dimensions
        self.output_shape_cache = {}

    def forward(self, x):
        """
        Optimized forward pass using in-place operations and direct functional calls.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, bias subtraction and tanh activation
        """
        # Get input shape
        input_shape = (x.shape[2], x.shape[3])
        
        # Step 1: Apply transposed convolution directly using functional API
        # This avoids the overhead of the nn.Module wrapper
        output = F.conv_transpose2d(
            x, 
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )
        
        # Step 2: Subtract bias (in-place)
        output.sub_(self.bias)
        
        # Step 3: Apply tanh activation (in-place)
        output.tanh_()
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, bias_shape]