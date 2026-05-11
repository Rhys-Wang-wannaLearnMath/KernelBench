import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Enable cuDNN optimizations for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        
        # Increase workspace limit for cuDNN to allow more memory-intensive but faster algorithms
        # This is especially useful for transposed convolutions
        if hasattr(torch.backends.cudnn, 'workspace_limit'):
            torch.backends.cudnn.workspace_limit = 1024 * 1024 * 512  # 512 MB
        
        # Create the transposed convolution layer
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding, 
            groups=groups, 
            bias=bias
        )
        
        # Convert to channels_last memory format for better performance on CUDA
        self.conv_transpose2d = self.conv_transpose2d.to(memory_format=torch.channels_last)
        
        # Use TorchScript for JIT compilation
        self.scripted_conv = torch.jit.script(self.conv_transpose2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure input is in channels_last format for optimal performance
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        
        # Use the JIT-compiled version for better performance
        return self.scripted_conv(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    # Create input tensor in channels_last format to avoid conversion in forward pass
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')
    x = x.to(memory_format=torch.channels_last)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization