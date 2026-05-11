import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling
    3. Addition
    4. Softmax
    5. Tanh activation
    6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first run, capture the graph.
            # The `torch.cuda.graph` context manager handles using a non-default
            # stream for capture automatically, which is a requirement.
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.graph):
                y = self.conv_transpose(self.static_input)
                y = torch.mean(y, dim=1, keepdim=True)
                y = y + self.bias
                y = torch.softmax(y, dim=1)
                y = torch.tanh(y)
                y = y * self.scaling_factor
                self.static_output = y

        # For every run (including the first), copy the input data and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the static output to avoid user modification of the graph's memory.
        return self.static_output.clone()

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor]