import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        if self.graph is None:
            # First run: perform a standard eager execution to get the correct output.
            eager_output = self.conv(x)
            eager_output = self.group_norm(eager_output)
            eager_output = eager_output * self.scale
            eager_output = self.maxpool(eager_output)
            eager_output = torch.clamp(eager_output, self.clamp_min, self.clamp_max)

            # In parallel, capture the graph for all subsequent runs.
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                graph_output = self.conv(self.static_input)
                graph_output = self.group_norm(graph_output)
                graph_output = graph_output * self.scale
                graph_output = self.maxpool(graph_output)
                self.static_output = torch.clamp(graph_output, self.clamp_min, self.clamp_max)
            
            return eager_output

        # Subsequent runs: copy new data to the static input and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]