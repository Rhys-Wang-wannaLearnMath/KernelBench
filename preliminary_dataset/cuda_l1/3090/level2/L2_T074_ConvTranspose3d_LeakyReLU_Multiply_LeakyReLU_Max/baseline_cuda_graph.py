import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first call, the graph is not yet captured.
        if self.graph is None:
            # We need to create static placeholders for the graph's inputs and outputs.
            # We use the first input `x` to determine the shape and device.
            self.static_input = x.clone()

            # To determine the output shape, we perform an initial, eager-mode forward pass.
            # The result of this pass is used to create a correctly-sized static output tensor.
            with torch.no_grad():
                y_temp = self.conv_transpose(self.static_input)
                y_temp = self.leaky_relu(y_temp)
                y_temp = y_temp * self.multiplier
                y_temp = self.leaky_relu(y_temp)
                y_temp = self.max_pool(y_temp)
            self.static_output = torch.empty_like(y_temp)

            # Now, capture the sequence of operations into a CUDA graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                y_graph = self.conv_transpose(self.static_input)
                y_graph = self.leaky_relu(y_graph)
                y_graph = y_graph * self.multiplier
                y_graph = self.leaky_relu(y_graph)
                y_graph = self.max_pool(y_graph)
                self.static_output.copy_(y_graph)
        
        # For every run (including the first), copy the input data and replay the graph.
        # This ensures consistent execution path and behavior.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]