import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor,
                 cudnn_enabled=None, cudnn_benchmark=None,
                 cudnn_deterministic=None, cudnn_allow_tf32=None):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        # Store cuDNN flags to be used in the forward pass.
        # None values will be ignored, preserving the global setting.
        self.cudnn_flags = {
            'enabled': cudnn_enabled,
            'benchmark': cudnn_benchmark,
            'deterministic': cudnn_deterministic,
            'allow_tf32': cudnn_allow_tf32,
        }

    def forward(self, x):
        # Filter out flags that are None to avoid overriding global settings unnecessarily.
        active_flags = {k: v for k, v in self.cudnn_flags.items() if v is not None}
        with torch.backends.cudnn.flags(**active_flags):
            x = self.conv(x)
            x = x / self.divisor
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]