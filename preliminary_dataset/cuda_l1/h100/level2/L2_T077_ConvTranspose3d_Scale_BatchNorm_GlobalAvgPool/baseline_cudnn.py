import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1, cudnn_flags=None):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cudnn_flags = cudnn_flags

    def forward(self, x):
        def _forward_impl(input_tensor):
            out = self.conv_transpose(input_tensor)
            out = out * self.scale_factor
            out = self.batch_norm(out)
            out = self.global_avg_pool(out)
            return out

        if self.cudnn_flags:
            with torch.backends.cudnn.flags(**self.cudnn_flags):
                return _forward_impl(x)
        else:
            return _forward_impl(x)

batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]