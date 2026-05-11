import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first forward pass, we perform two actions:
            # 1. A standard forward pass to compute the correct output and update stateful layers.
            y = self.conv_transpose(x)
            y = self.batch_norm(y)
            y = self.tanh(y)
            y = self.max_pool(y)
            out = self.group_norm(y)

            # 2. Capture the graph for all subsequent runs.
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                y_graph = self.conv_transpose(self.static_input)
                y_graph = self.batch_norm(y_graph)
                y_graph = self.tanh(y_graph)
                y_graph = self.max_pool(y_graph)
                self.static_output = self.group_norm(y_graph)

            # Return the result from the initial standard forward pass.
            return out

        # For subsequent runs, replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
groups = 8
num_groups = 4
height, width = 32, 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]