import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _forward_impl(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            # Create a static input tensor with the same properties as the input
            self.static_input = x.clone()

            # Create the graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Capture the graph using a context manager
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        # Copy the current input data to the static input tensor
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return a clone of the static output
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]