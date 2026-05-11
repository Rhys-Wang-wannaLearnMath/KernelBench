import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # If the graph is not yet captured, it's the first run.
        if self.graph is None:
            # Create static tensors for the graph. These have fixed memory addresses.
            # We use empty_like to create placeholder tensors with the correct properties (shape, dtype, device).
            self.static_input = torch.empty_like(x)
            
            # Initialize and capture the CUDA graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Define the sequence of operations within the graph using the static placeholders.
                # The capture process allocates a static output tensor (`self.static_output`).
                y = self.conv_transpose(self.static_input)
                y = torch.clamp(y, min=self.min_value)
                self.static_output = y / self.divisor
        
        # For every run (including the first), copy the current input data into the static input tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations with high efficiency,
        # using the data in `self.static_input` and updating `self.static_output` in-place.
        self.graph.replay()
        
        # Return the static output tensor.
        return self.static_output

batch_size = 16
in_channels = 32
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]