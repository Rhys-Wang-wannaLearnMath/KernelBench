import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()
        
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # First call: capture the graph
            self.static_input = x
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The model's forward pass logic is placed inside the capture block
                # It operates on a static version of the input tensor
                out = self.conv(self.static_input)
                out = out * self.multiplier
                out = self.leaky_relu(out)
                out = torch.nn.functional.gelu(out)
                self.static_output = out

        # Copy the new input data into the static tensor used by the graph
        self.static_input.copy_(x)
        
        # Replay the captured graph with the new input data
        self.graph.replay()
        
        # Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]