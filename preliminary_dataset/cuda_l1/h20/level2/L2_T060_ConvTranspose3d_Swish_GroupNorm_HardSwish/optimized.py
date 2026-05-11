import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        
        # Enable cuDNN benchmarking for faster convolution
        torch.backends.cudnn.benchmark = True
        
        # Pre-convert weights to channels_last format if on CUDA
        if torch.cuda.is_available():
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                memory_format=torch.channels_last_3d
            )
        
        # JIT compile the group norm for better performance
        self.scripted_group_norm = torch.jit.script(self.group_norm)

    def forward(self, x):
        # Convert to channels_last_3d for better memory locality if on CUDA
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last_3d)
            
        # Use mixed precision where supported
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Apply transposed convolution
            x = self.conv_transpose(x)
            
            # Apply Swish activation using SiLU (which is equivalent but faster)
            x = F.silu(x)
            
            # Apply group normalization using the JIT compiled version
            x = self.scripted_group_norm(x)
            
            # Apply HardSwish activation
            x = F.hardswish(x)
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]