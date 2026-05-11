import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.
    Optimized implementation using memory layout and computation optimizations.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size in the form (depth, height, width).
        stride (tuple, optional): Tuple of 3 integers representing the stride in the form (depth, height, width). Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding in the form (depth, height, width). Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding in the form (depth, height, width). Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the main convolution layer
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
        # Optimization flags
        self.use_channels_last = torch.cuda.is_available()
        
        # Determine optimal precision based on hardware capabilities
        if torch.cuda.is_available():
            self.device_capability = torch.cuda.get_device_capability()
            self.use_mixed_precision = self.device_capability[0] >= 7
            
            # Enable cuDNN benchmarking for algorithm selection
            torch.backends.cudnn.benchmark = True
            
            # Set math mode for tensor cores if available (Ampere+ GPUs)
            if self.device_capability[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Preprocess weights to optimal memory format
            with torch.no_grad():
                if self.use_channels_last:
                    self.conv_transpose3d.weight.data = self.conv_transpose3d.weight.data.contiguous(memory_format=torch.channels_last_3d)
        else:
            self.use_mixed_precision = False
            self.device_capability = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution with optimized memory layout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Store original properties
        original_dtype = x.dtype
        
        # Skip optimization if not on CUDA
        if not x.is_cuda:
            return self.conv_transpose3d(x)
        
        # Optimize memory layout
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        else:
            x = x.contiguous()
        
        # Use mixed precision for computation
        if self.use_mixed_precision and original_dtype == torch.float32:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    result = self.conv_transpose3d(x)
            
            # Convert back to original precision if needed
            if result.dtype != original_dtype:
                result = result.to(dtype=original_dtype)
        else:
            # Standard computation path with gradient disabled for inference
            with torch.no_grad():
                result = self.conv_transpose3d(x)
        
        # Ensure output has consistent memory format
        if self.use_channels_last:
            result = result.contiguous(memory_format=torch.channels_last_3d)
        
        return result

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth_in = 16
height_in = 32
width_in = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization