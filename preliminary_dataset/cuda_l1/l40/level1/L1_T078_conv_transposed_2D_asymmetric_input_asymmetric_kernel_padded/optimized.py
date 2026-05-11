import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Initialize the transposed convolution layer
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        
        # Enable cuDNN benchmarking for algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Check if we have a GPU that supports tensor cores (Volta or newer)
        self.has_tensor_cores = (torch.cuda.is_available() and 
                                torch.cuda.get_device_capability()[0] >= 7)
        
        # Use channels_last format on CUDA devices
        self.use_channels_last = torch.cuda.is_available()
        
        # Enable mixed precision only if we have tensor core support
        self.use_amp = self.has_tensor_cores and hasattr(torch.cuda, 'amp')
        
        # Pre-convert weights to channels_last format if applicable
        if self.use_channels_last:
            self.conv_transpose2d.weight.data = self.conv_transpose2d.weight.data.to(
                memory_format=torch.channels_last
            )
            
        # JIT compile the convolution module for better performance
        if torch.cuda.is_available():
            self.conv_transpose2d = torch.jit.script(self.conv_transpose2d)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Only apply optimizations if on CUDA
        if x.is_cuda:
            # Store original dtype
            original_dtype = x.dtype
            
            # Convert to channels_last format if on CUDA
            if self.use_channels_last:
                x = x.to(memory_format=torch.channels_last)
            
            # Use mixed precision if available and beneficial
            if self.use_amp and original_dtype == torch.float32:
                with torch.cuda.amp.autocast():
                    output = self.conv_transpose2d(x)
                    
                    # Convert back to original dtype if needed
                    if output.dtype != original_dtype:
                        output = output.to(dtype=original_dtype)
                    
                    return output
            else:
                # Use the pre-optimized convolution
                return self.conv_transpose2d(x)
        else:
            # Fallback for CPU execution
            return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]