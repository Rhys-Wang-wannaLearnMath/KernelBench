import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # CUDNN Backend Flags
        self.cudnn_enabled = torch.backends.cudnn.enabled
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.allow_tf32 = torch.backends.cudnn.allow_tf32

    def forward(self, x):
        with torch.backends.cudnn.flags(
            enabled=self.cudnn_enabled,
            benchmark=self.cudnn_benchmark,
            deterministic=self.cudnn_deterministic,
            allow_tf32=self.allow_tf32,
        ):
            x = self.conv(x)
            x = torch.relu(x)
            x = x + self.bias
            return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]