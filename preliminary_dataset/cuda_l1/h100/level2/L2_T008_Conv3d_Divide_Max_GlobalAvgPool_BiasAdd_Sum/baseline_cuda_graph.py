import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

        # CUDA graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the graph
        if self.graph is None:
            # Create static placeholders for inputs and outputs
            self.static_input = torch.empty_like(x)

            # Create the graph object
            g = torch.cuda.CUDAGraph()

            # Capture the model's operations in the graph
            with torch.cuda.graph(g):
                y = self.conv(self.static_input)
                y = y / self.divisor
                y = self.max_pool(y)
                y = self.global_avg_pool(y)
                y = y + self.bias
                self.static_output = torch.sum(y, dim=self.sum_dim)

            self.graph = g

        # Copy the current input data to the static placeholder
        self.static_input.copy_(x)

        # Replay the captured graph
        self.graph.replay()

        # Return a clone of the graph's output
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]