import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = torch.empty_like(x)
            
            with torch.cuda.graph(self.graph):
                # The model's forward pass is recorded
                y = self.conv(self.static_input)
                y = y - self.subtract_value
                y = torch.nn.functional.hardswish(y)
                y = self.pool(y)
                self.static_output = torch.nn.functional.mish(y)
        
        # Copy the current input to the static placeholder
        self.static_input.copy_(x)
        # Replay the captured graph
        self.graph.replay()
        # Return the result from the static output tensor
        return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]