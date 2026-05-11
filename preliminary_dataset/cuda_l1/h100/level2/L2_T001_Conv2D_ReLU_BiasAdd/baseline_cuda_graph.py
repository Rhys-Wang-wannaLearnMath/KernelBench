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
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            # Create a static input tensor for graph capture
            self.static_input = x.clone()

            # Instantiate and capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The graph needs its own static output tensor
                y = self.conv(self.static_input)
                y = torch.relu(y)
                self.static_output = y + self.bias

        # Copy the new input data into the static tensor
        self.static_input.copy_(x)

        # Replay the graph
        self.graph.replay()

        # Return a clone of the graph's output tensor
        return self.static_output.clone()

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