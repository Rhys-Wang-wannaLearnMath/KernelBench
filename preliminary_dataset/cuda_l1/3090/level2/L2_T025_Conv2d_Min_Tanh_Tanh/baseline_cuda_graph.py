import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                # The model's operations are recorded in the graph
                static_y = self.conv(self.static_input)
                static_y = torch.min(static_y, dim=1, keepdim=True)[0]
                static_y = torch.tanh(static_y)
                self.static_output = torch.tanh(static_y)
        
        # Copy the current input to the static memory buffer
        self.static_input.copy_(x)
        
        # Replay the graph
        self.graph.replay()
        
        # Return a clone of the output to avoid user modification of graph's static memory
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]