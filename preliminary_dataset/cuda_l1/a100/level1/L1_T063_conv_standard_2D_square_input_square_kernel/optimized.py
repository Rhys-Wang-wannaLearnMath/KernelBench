import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store convolution parameters directly as attributes for faster access
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Configure cuDNN for maximum performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Set workspace limit for cuDNN
            if hasattr(torch.backends.cudnn, 'workspace_limit_in_bytes'):
                torch.backends.cudnn.workspace_limit_in_bytes = 1024 * 1024 * 1024  # 1GB
        
        # Create temporary conv layer for weight initialization
        temp_conv = nn.Conv2d(
            in_channels, out_channels, (kernel_size, kernel_size),
            stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias
        )
        
        # Pre-convert weights to channels_last format during initialization
        weight_cl = temp_conv.weight.to(memory_format=torch.channels_last)
        self.register_parameter('weight', nn.Parameter(weight_cl))
        
        # Handle bias
        if bias:
            self.register_parameter('bias', nn.Parameter(temp_conv.bias))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Apply memory format optimization only for CUDA tensors
        if x.is_cuda:
            # Convert input to channels_last for optimal memory access
            x = x.to(memory_format=torch.channels_last)
        
        # Direct convolution with pre-optimized weights
        out = F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        
        return out

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization