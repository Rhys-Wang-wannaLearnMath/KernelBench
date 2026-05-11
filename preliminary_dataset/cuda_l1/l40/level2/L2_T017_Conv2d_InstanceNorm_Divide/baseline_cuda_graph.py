import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, the graph is not yet captured
        if self.graph is None:
            # Create a CUDA graph object
            g = torch.cuda.CUDAGraph()

            # Create static input tensor with the same properties as the real input
            self.static_input = torch.empty_like(x)

            # Begin graph capture using a context manager
            with torch.cuda.graph(g):
                # Run the model's operations with the static input
                # The result is a static output tensor that's part of the graph
                static_y = self.conv(self.static_input)
                static_y = self.instance_norm(static_y)
                self.static_output = static_y / self.divide_by
            
            # Save the captured graph for future runs
            self.graph = g

        # For every run (including the first), copy the current input data
        # into the static input buffer used by the graph
        self.static_input.copy_(x)

        # Replay the captured graph operations. This will update self.static_output
        # with the new result
        self.graph.replay()

        # Return a clone of the static output tensor. Cloning is necessary to
        # prevent the user from accidentally modifying the graph's output buffer
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]