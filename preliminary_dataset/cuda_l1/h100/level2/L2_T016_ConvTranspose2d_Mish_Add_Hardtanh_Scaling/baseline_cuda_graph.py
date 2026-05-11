import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first run, create static tensors and capture the graph.
            # A static tensor is created to hold the input for graph replay.
            self.static_input = torch.empty_like(x)
            
            # Create the graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Capture the graph definition.
            with torch.cuda.graph(self.graph):
                # The model's operations are defined using the static input tensor.
                y = self.conv_transpose(self.static_input)
                y = torch.nn.functional.mish(y) # Mish activation
                y = y + self.add_value
                y = torch.nn.functional.hardtanh(y, min_val=-1, max_val=1) # Hardtanh activation
                y = y * self.scale # Scaling
                # The result of the graphed operations becomes the static output.
                self.static_output = y

        # For every run (including the first), copy the new input data to the static
        # input tensor and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]