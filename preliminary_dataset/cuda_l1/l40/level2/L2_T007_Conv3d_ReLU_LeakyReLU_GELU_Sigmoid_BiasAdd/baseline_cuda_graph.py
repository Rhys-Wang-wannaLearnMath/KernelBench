import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # On the first pass, capture the graph.
        if self.graph is None:
            # The capture must be done on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                
                # Create static tensors for inputs and outputs.
                # They must be on the same device as the input.
                self.static_input = x.clone()

                self.graph.capture_begin()
                
                # The sequence of operations to be captured
                y = self.conv(self.static_input)
                y = torch.relu(y)
                y = torch.nn.functional.leaky_relu(y, negative_slope=0.01)
                y = torch.nn.functional.gelu(y)
                y = torch.sigmoid(y)
                self.static_output = y + self.bias
                
                self.graph.capture_end()
        
        # For every pass (including the first), update the input tensor's data and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]