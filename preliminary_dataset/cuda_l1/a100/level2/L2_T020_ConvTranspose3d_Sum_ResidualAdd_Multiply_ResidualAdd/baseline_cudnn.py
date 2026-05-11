import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, cudnn_benchmark=False, cudnn_deterministic=False, cudnn_allow_tf32=True):
        super(Model, self).__init__()
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_allow_tf32 = cudnn_allow_tf32
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        with torch.backends.cudnn.flags(benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic, allow_tf32=self.cudnn_allow_tf32):
            x = self.conv_transpose(x)
            original_x = x.clone().detach()
            x = x + self.bias
            x = x + original_x
            x = x * original_x
            x = x + original_x
            return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]