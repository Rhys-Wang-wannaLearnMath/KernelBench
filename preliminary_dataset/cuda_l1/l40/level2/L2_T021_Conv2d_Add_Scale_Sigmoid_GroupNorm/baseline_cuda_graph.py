import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first forward pass, capture the graph
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                # The model's operations are recorded using static tensors
                static_out = self.conv(self.static_input)
                static_out = static_out + self.bias
                static_out = static_out * self.scale
                static_out = torch.sigmoid(static_out)
                self.static_output = self.group_norm(static_out)
        
        # For all subsequent calls, copy input data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]