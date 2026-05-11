import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, the graph is not yet captured.
        if self.graph is None:
            # Create a graph instance
            g = torch.cuda.CUDAGraph()

            # Create a static input tensor with the same properties as the input.
            self.static_input = torch.empty_like(x)
            
            # Begin graph capture
            with torch.cuda.graph(g):
                # Run the forward pass with the static input
                y = self.conv(self.static_input)
                y = torch.tanh(y)
                y = y * self.scaling_factor
                y = y + self.bias
                y = self.max_pool(y)
                # The output of the graph is also made static
                self.static_output = y
            
            # Save the graph for subsequent runs
            self.graph = g

        # Copy the current input data to the static input tensor
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return the static output
        return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]