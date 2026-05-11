import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        # Initialize cudnn flags from the current global settings.
        # These can be modified on the model instance after creation.
        self.cudnn_enabled = torch.backends.cudnn.enabled
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.allow_tf32 = torch.backends.cudnn.allow_tf32

    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=self.cudnn_enabled, benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic, allow_tf32=self.allow_tf32):
            x = self.conv(x)
        x = torch.min(x, torch.tensor(self.constant_value))
        x = x + self.bias
        x = x * self.scaling_factor
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]