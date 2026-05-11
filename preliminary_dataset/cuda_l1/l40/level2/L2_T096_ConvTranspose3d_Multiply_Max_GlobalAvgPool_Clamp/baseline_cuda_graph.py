import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        
        # CUDA graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                static_y = self.conv_transpose(self.static_input)
                static_y = static_y * self.scale
                static_y = self.maxpool(static_y)
                static_y = self.global_avg_pool(static_y)
                self.static_output = torch.clamp(static_y, min=self.clamp_min, max=self.clamp_max)

        # For all runs, copy the current input to the static tensor and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]