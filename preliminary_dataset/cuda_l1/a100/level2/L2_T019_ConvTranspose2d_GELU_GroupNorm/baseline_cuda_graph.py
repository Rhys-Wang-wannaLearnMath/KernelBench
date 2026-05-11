import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph.
        if self.graph is None:
            # Create a static input tensor that will be used for all subsequent runs.
            self.static_input = torch.empty_like(x)
            
            # Define and capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Run the model's operations with the static input.
                # The resulting tensor is designated as the static output for the graph.
                graph_out = self.conv_transpose(self.static_input)
                graph_out = torch.nn.functional.gelu(graph_out)
                self.static_output = self.group_norm(graph_out)
        
        # For every run (including the first), copy the current input to the static input tensor
        # and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
groups = 8
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]