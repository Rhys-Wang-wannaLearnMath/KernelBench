import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # Create persistent buffers for input and output. It's essential to
            # clone the input to ensure a stable memory address for the graph.
            self.static_input = x.clone()

            # Instantiate the CUDA graph.
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the model's operations. The code inside the 'with'
            # block is recorded into the graph and executed once.
            with torch.cuda.graph(self.graph):
                # The forward pass logic is captured using the static input tensor.
                y = self.conv_transpose(self.static_input)
                y = y * self.multiplier
                y = torch.mean(y, dim=[2, 3], keepdim=True)  # First global average pooling
                y = torch.mean(y, dim=[2, 3], keepdim=True)  # Second global average pooling
                # The final result tensor's memory becomes the graph's static output buffer.
                self.static_output = y

        # For every run (including the first, after capture), copy the new input
        # data into the static input buffer and replay the graph. This ensures
        # a consistent execution path.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the static output. This is crucial to prevent the
        # caller from holding a reference to the internal buffer, which will be
        # overwritten in subsequent calls.
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]